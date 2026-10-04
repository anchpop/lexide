"""Verify single-sentence ONNX inference against CPU fp32 predict_joint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from data_prep import apply_script
from export_joint_onnx import ENCODER_INPUTS, HEAD_OUTPUTS
from mst import single_root_mst
from predict_joint import load_checkpoint, predict_batch, spans_from_char_labels, span_tensors


class OnnxJoint:
    def __init__(self, path, threads=4, int8=False):
        path = Path(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.encoder = ort.InferenceSession(str(path / ("encoder.int8.onnx" if int8 else "encoder.onnx")), options,
                                           providers=["CPUExecutionProvider"])
        self.heads = ort.InferenceSession(str(path / "heads.onnx"), options, providers=["CPUExecutionProvider"])

    def predict(self, record, batch, vocab):
        feed = {k: batch[k if k != "lang_id" else "lang_ids"].numpy() for k in ENCODER_INPUTS}
        boundary, chars, sub = self.encoder.run(None, feed)
        spans = spans_from_char_labels(record["text"], boundary[0].argmax(-1))
        starts, ends, _ = span_tensors([spans], "cpu")
        outputs = self.heads.run(None, dict(chars=chars, sub_at_char=sub, starts=starts.numpy(), ends=ends.numpy()))
        pos, lemma, arcs, rel = outputs
        parents = single_root_mst(arcs[0, :len(spans), :len(spans) + 1])
        tokens = [dict(start=s, end=e, pos=vocab["pos"][pos[0, i].argmax()],
                       lemma=apply_script(record["text"][s:e], vocab["lemma_scripts"][lemma[0, i].argmax()]),
                       head=parents[i], dep=vocab["dep"][rel[0, i, parents[i]].argmax()])
                  for i, (s, e) in enumerate(spans)]
        return dict(lang=record["lang"], text=record["text"], tokens=tokens), (boundary, chars, sub), outputs, (starts, ends)


def sample_records(records, per_lang):
    groups = defaultdict(list)
    for record in records:
        groups[record["lang"]].append(record)
    # Length-spread examples include each language's longest sentence.
    return [rows[i] for lang, group in sorted(groups.items())
            for rows in [sorted(group, key=lambda r: len(r["text"]))]
            for i in np.linspace(0, len(rows) - 1, min(per_lang, len(rows)), dtype=int)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--per-lang", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, vocab = load_checkpoint(args.checkpoint)
    runner = OnnxJoint(args.onnx, args.threads)
    records = sample_records([json.loads(line) for line in Path(args.test).read_text().splitlines()], args.per_lang)
    assert {r["lang"] for r in records} == set(vocab["langs"]), "Verification must cover every model language"
    diffs = defaultdict(float)
    mismatches = []
    start = time.perf_counter()
    for i, record in enumerate(records):
        ref, encoded, batch = predict_batch(model, tokenizer, vocab, [record])
        got, enc, heads, spans = runner.predict(record, batch, vocab)
        with torch.inference_mode():
            refheads = model.word_heads(encoded, *spans, torch.ones_like(spans[0], dtype=torch.bool))
        for name, expected, actual in [
            *[(k, encoded[k], v) for k, v in zip(("boundary_logits", "chars", "sub_at_char"), enc)],
            *[(k, refheads[k], v) for k, v in zip(HEAD_OUTPUTS, heads)],
        ]:
            expected = expected.numpy()
            finite = np.isfinite(expected)
            assert np.array_equal(finite, np.isfinite(actual)), name
            diffs[name] = max(diffs[name], float(np.abs(expected[finite] - actual[finite]).max(initial=0)))
        if ref[0] != got:
            mismatches.append(dict(index=i, reference=ref[0], onnx=got))
        print(f"{i + 1}/{len(records)} {record['lang']} mismatches={len(mismatches)}", flush=True)
    report = dict(sentences=len(records), seconds=time.perf_counter() - start,
                  max_abs_diff=dict(diffs), mismatches=mismatches)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    assert max(diffs.values(), default=0) < 1e-3, f"Numerical parity exceeded 1e-3; see {args.report}"
    assert not mismatches, f"{len(mismatches)} final predictions differ; see {args.report}"


if __name__ == "__main__":
    main()
