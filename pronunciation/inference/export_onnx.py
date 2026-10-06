"""Export scratch raw-waveform inference and benchmark the actual CPU artifacts.

Outputs use Modal frame_matrix semantics: phone is joint log probability
(includes blank), nonblank is sigmoid probability, other factors are categorical
probabilities. No language input: all factor heads are emitted. Metadata records
ordered labels; consumers select relevant factors exactly as for frame_matrix.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from src.factorized_ctc import FactorizedCTCModel
from src.validation_metrics import collapse_phones


class FrameMatrix(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head_names = sorted(model.language_head_specs)

    def forward(self, waveform, lengths):
        mask = torch.arange(waveform.shape[1], device=waveform.device)[None] < lengths[:, None]
        out = self.model(waveform, attention_mask=mask, active_language_heads=self.head_names)
        return (out["log_probs"], out["nonblank_logit"].sigmoid().unsqueeze(-1),
                out["stress_logits"].softmax(-1),
                *(out["language_head_logits"][name].softmax(-1) for name in self.head_names))


def greedy(values, blank_id, *, with_stress=False):
    phone, nonblank = values[:2]
    scores = phone[0].copy()
    scores[:, blank_id] = -np.inf
    ids = np.where(nonblank[0, :, 0] > .5, scores.argmax(-1), blank_id).tolist()
    stress = values[2][0].argmax(-1).tolist() if with_stress else None
    phones, starts = collapse_phones(ids, blank_id, stress, return_starts=True)
    return [[phone, stress[start]] for phone, start in zip(phones, starts)] if with_stress else phones


def session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-clips", type=Path,
                   help="JSON list of real held-out wav paths to compare and decode")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(1)
    model = FactorizedCTCModel.load_from_dir(args.checkpoint).eval()
    if not (args.checkpoint / "scratch_backbone.pt").exists():
        raise ValueError("This exporter is for scratch checkpoints")
    wrapper = FrameMatrix(model).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["phone", "nonblank", "stress", *wrapper.head_names]
    vocab = json.loads((args.checkpoint / "vocab.json").read_text())
    labels = [t for t, i in sorted(vocab.items(), key=lambda item: item[1])]
    heads = {
        "phone": dict(labels=labels, value_semantics="joint_log_probability", blank_id=model.blank_id),
        "nonblank": dict(labels=["nonblank"], value_semantics="sigmoid_probability"),
        "stress": dict(labels=["none", "primary", "secondary"], value_semantics="probability"),
    }
    for name in wrapper.head_names:
        spec = model.language_head_specs[name]
        factor_labels = {
            "tha_tone": ["not_bearer", "mid", "low", "falling", "high", "rising"],
            "zho_hans_tone": ["not_bearer", "high_level", "rising", "dipping", "falling", "neutral"],
            "jpn_pitch_accent": ["not_bearer", "low", "high"],
        }
        heads[name] = dict(labels=spec.get("labels", factor_labels[name]),
                           value_semantics="probability", language=spec["lang"], target=spec["target"])
    metadata = dict(schema_version=1, sample_rate=16000, frame_rate_ms=20,
                    frame_center_samples=200, heads=heads,
                    lengths="Output frame count per clip = (lengths - 400) // 320 + 1; inputs >=400 samples")
    (args.output / "frame_matrix.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    fp32, int8 = args.output / "model.onnx", args.output / "model.int8.onnx"
    batch_dim = torch.export.Dim("batch", min=1)
    sample_dim = torch.export.Dim("samples", min=400)
    torch.onnx.export(wrapper, (torch.randn(2, 16000), torch.tensor([16000, 12000])), str(fp32),
        input_names=["waveform", "lengths"], output_names=names,
        dynamic_shapes=({0: batch_dim, 1: sample_dim}, {0: batch_dim}),
        opset_version=18, dynamo=True, external_data=False)
    import onnx
    onnx.checker.check_model(str(fp32))
    from onnxruntime.quantization import quantize_dynamic, QuantType
    # Quantize learned linear weights, not the fixed mel integration. Dynamic
    # activation quantization before log-mel destroys quiet spectral bins.
    graph = onnx.load(str(fp32))
    frontend_nodes = [node.name for node in graph.graph.node
                      if any("frontend.mel" in name for name in node.input)]
    assert frontend_nodes, "Exporter lost the fixed mel matrix identity; do not quantize it blindly"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=True,
                     nodes_to_exclude=frontend_nodes, extra_options={"MatMulConstBOnly": True},
                     reduce_range=True)  # avoid U8S8 pairwise saturation on AVX2 CPUs
    result = dict(parameters=sum(p.numel() for p in model.parameters()), artifacts={}, parity=[])
    sessions = {"fp32": session(fp32), "int8": session(int8)}
    rng = np.random.default_rng(42)
    # Execute both dynamic axes, including odd lengths and unequal padding.
    result["dynamic_shapes"] = []
    for lengths_list in ([8017], [5101, 19003]):
        lengths = np.array(lengths_list, dtype=np.int64)
        x = rng.normal(0, .1, (len(lengths), int(lengths.max()))).astype(np.float32)
        with torch.no_grad():
            expected = [v.numpy() for v in wrapper(torch.from_numpy(x), torch.from_numpy(lengths))]
        for precision, sess in sessions.items():
            actual = sess.run(None, dict(waveform=x, lengths=lengths))
            assert actual[0].shape[:2] == (len(lengths), (int(lengths.max())-400)//320+1)
            error = 0.
            for i, length in enumerate(lengths):
                frames = (int(length)-400)//320+1
                for a, b in zip(actual, expected):
                    error = max(error, float(np.abs(a[i, :frames]-b[i, :frames]).max()))
            if precision == "fp32":
                assert error < .005, error
            result["dynamic_shapes"].append(dict(lengths=lengths_list, precision=precision, max_abs=error))
    for precision, path in [("fp32", fp32), ("int8", int8)]:
        timings = {}
        for seconds in (5, 15):
            x = rng.normal(0, .1, (1, 16000*seconds)).astype(np.float32)
            inputs = dict(waveform=x, lengths=np.array([x.shape[1]], dtype=np.int64))
            sessions[precision].run(None, inputs)
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                sessions[precision].run(None, inputs)
                samples.append(time.perf_counter()-start)
            timings[str(seconds)] = dict(seconds=float(np.median(samples)),
                realtime_factor=float(np.median(samples)/seconds), trials=samples)
        result["artifacts"][precision] = dict(bytes=path.stat().st_size, benchmarks=timings)
    clips = json.loads(args.validation_clips.read_text()) if args.validation_clips else []
    import soundfile as sf
    for clip in clips:
        x, sr = sf.read(clip, dtype="float32")
        assert sr == 16000 and x.ndim == 1
        x = x[None]
        lengths = np.array([x.shape[1]], dtype=np.int64)
        with torch.no_grad():
            expected = [v.numpy() for v in wrapper(torch.from_numpy(x), torch.from_numpy(lengths))]
        ref = greedy(expected, model.blank_id)
        row = dict(file=clip, samples=x.shape[1], pytorch_decode=ref, comparisons={})
        for precision, sess in sessions.items():
            actual = sess.run(None, dict(waveform=x, lengths=lengths))
            errors = {name: float(np.max(np.abs(a-b))) for name, a, b in zip(names, actual, expected)}
            # Also compare categorical factor log probabilities, not just their
            # bounded probabilities; phone already is a log probability.
            log_errors = {name: float(np.max(np.abs(np.log(np.maximum(a, 1e-30)) -
                         np.log(np.maximum(b, 1e-30))))) for name, a, b in zip(names[1:], actual[1:], expected[1:])}
            decoded = greedy(actual, model.blank_id)
            if precision == "fp32":
                assert max(errors.values()) < .005, errors
                assert decoded == ref, f"fp32 greedy mismatch on {clip}"
            row["comparisons"][precision] = dict(max_abs=errors, factor_logprob_max_abs=log_errors,
                identical_greedy=decoded == ref, decode=decoded,
                identical_stress_greedy=(greedy(actual, model.blank_id, with_stress=True) ==
                                         greedy(expected, model.blank_id, with_stress=True)))
        result["parity"].append(row)
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
