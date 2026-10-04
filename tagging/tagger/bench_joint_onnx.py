"""Measure fp32 versus dynamic-int8 ONNX on the full joint test corpus (never ship int8)."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from dataset_joint import collate, encode_records
from eval_e2e import score
from verify_joint_onnx import OnnxJoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    path, out = Path(args.onnx), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(path / "encoder.onnx"), str(path / "encoder.int8.onnx"),
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     use_external_data_format=True)
    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint) / "tokenizer", local_files_only=True)
    vocab = json.loads((path / "vocab.json").read_text())
    records = [json.loads(line) for line in Path(args.test).read_text().splitlines()]
    report = {}
    for precision in ("fp32", "int8"):
        runner = OnnxJoint(path, args.threads, int8=precision == "int8")
        predictions = []
        start = time.perf_counter()
        with (out / f"{precision}.jsonl").open("w") as f:
            for i, record in enumerate(records):
                batch = collate(encode_records([record], tokenizer, vocab, training=False), tokenizer.pad_token_id)
                prediction, _, _, _ = runner.predict(record, batch, vocab)
                predictions.append(prediction)
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                if (i+1) % 1000 == 0:
                    print(precision, i+1, time.perf_counter()-start, flush=True)
        elapsed = time.perf_counter()-start
        names = ["encoder.onnx", "encoder.onnx.data"] if precision == "fp32" else ["encoder.int8.onnx", "encoder.int8.onnx.data"]
        report[precision] = dict(seconds=elapsed, sentences_per_second=len(records)/elapsed,
                                 encoder_bytes=sum((path / name).stat().st_size for name in names if (path / name).exists()),
                                 metrics=score(records, predictions))
        del runner
        (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
