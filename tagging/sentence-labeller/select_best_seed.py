"""Pick the best seed-sweep segmenter (used by sky_byte_models.yaml).

Scans output/segmenter-s*/ for patterns.json (written by eval_patterns.py --json-out)
and meta.json (written by train_segmenter.py), ranks by (pattern pass count, val F1),
and copies the winner to output/segmenter/ for upload.
"""
import json
import os
import shutil

best, key = None, None
for d in sorted(os.listdir("output")):
    pj = os.path.join("output", d, "patterns.json")
    if not d.startswith("segmenter-s") or not os.path.exists(pj):
        continue
    pat = json.load(open(pj))
    meta = json.load(open(os.path.join("output", d, "meta.json")))
    k = (pat["pass"], meta["metrics"]["f1"])
    print(f"{d}: patterns {pat['pass']}/{pat['total']}, val f1 {meta['metrics']['f1']}")
    if key is None or k > key:
        best, key = d, k
assert best is not None, "no seed-sweep outputs found under output/"
print("winner:", best, key)
shutil.copytree(os.path.join("output", best), "output/segmenter", dirs_exist_ok=True)
