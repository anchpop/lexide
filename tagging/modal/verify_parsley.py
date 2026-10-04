"""Verify deployed parsley against direct L4 predict_joint, with the same batch shapes.

Run with modal run modal/verify_parsley.py. No deployment or persistent app is created.
"""
import json
from pathlib import Path
import urllib.request

import modal
from modal_serve_tagger import image, SOURCE, MODEL

app = modal.App("parsley-release-check")
TAG_URL = "https://anchpop--lexide-parsley-parsley-tag.modal.run"
SEGMENT_URL = "https://anchpop--lexide-parsley-parsley-segment.modal.run"


def post(url, request):
    req = urllib.request.Request(url, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=600)
def direct(groups):
    import os
    import sys
    import torch
    sys.path.insert(0, SOURCE)
    from predict_joint import load_checkpoint, predict_batch
    torch.set_num_threads(4)
    model, tokenizer, vocab = load_checkpoint(Path(MODEL) / os.environ["JOINT_HF_PATH"], "cuda")
    results = []
    for lang, texts in groups:
        predictions, _, _ = predict_batch(model, tokenizer, vocab, [dict(lang=lang, text=t) for t in texts], "cuda")
        results.append([[{**t, "text": p["text"][t["start"]:t["end"]]} for t in p["tokens"]] for p in predictions])
    return results


@app.local_entrypoint()
def main():
    from collections import defaultdict
    records = [json.loads(line) for line in (Path(__file__).resolve().parents[1] / "data/processed-joint/test.jsonl").read_text().splitlines()]
    samples = defaultdict(list)
    for record in records:
        if len(samples[record["lang"]]) < 4:
            samples[record["lang"]].append(record["text"])
    groups = sorted(samples.items())
    references = direct.remote(groups)
    for (lang, texts), reference in zip(groups, references):
        actual = post(TAG_URL, dict(lang=lang, sentences=texts))["results"]
        assert actual == reference, f"GPU mismatch for {lang}: {actual!r} != {reference!r}"
        print(f"{lang}: {len(texts)} exact direct-GPU matches")
    assert post(TAG_URL, dict(lang="eng", sentences=["", "   "])) == {"results": [[], []]}
    print("48 sentences match direct L4 bf16 predict_joint; empty inputs OK")
