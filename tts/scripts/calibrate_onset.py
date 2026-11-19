"""Measure where the phonetizer's CTC emissions sit inside a phone (adapted from
tiny-voca's calibrate_aligner_timing.py).

For transitions that must flip the pitch tracker's voicing (vowel -> voiceless
fricative and back), compare the flip with the start of the next phone's first
emission frame. The two directions bracket the true onset (pYIN widens voiced
regions about one frame each side); their mean is the suggested
``AlignmentSettings.onset_offset_seconds``.

    PYTHONPATH=. python scripts/calibrate_onset.py --features features/eng_achernar --frames data/eng_achernar/frames
"""
import argparse, collections, json
from pathlib import Path
import numpy as np
import torch

from voca.align import ALIGNER_RATE, ALIGNER_STRIDE, token_spans
from voca.build import FrameMatrixRecognizer
from voca.config import Config
from voca.text import PHONE_SET, SYMBOLS, VOWEL_SET

FRICATIVES, STOPS = {"f", "θ", "s", "ʃ"}, {"p", "t", "k"}
parser = argparse.ArgumentParser()
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("--frames", type=Path, required=True)
parser.add_argument("--count", type=int, default=400)
args = parser.parse_args()
c = Config()
rows = [json.loads(l) for l in open(args.features / "train.jsonl", encoding="utf-8")][:args.count]
offsets, runs = collections.defaultdict(list), collections.Counter()
for row in rows:
    stored = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)
    voiced = stored["prosody"][:, 1].numpy() > 0.5
    symbols = [SYMBOLS[i] for i in row["tokens"]]
    position = [i for i, s in enumerate(symbols) if s in PHONE_SET]
    names = [symbols[i] for i in position]
    recognizer = FrameMatrixRecognizer(json.loads((args.frames / f"{row['id']}.json").read_text()))
    spans = token_spans(recognizer.matrix, [recognizer.label(s) for s in names], recognizer.blank)
    runs.update(last - first + 1 for first, last in spans)
    flips = [(f, bool(voiced[f - 1])) for f in range(1, len(voiced)) if voiced[f - 1] != voiced[f]]
    for k in range(len(names) - 1):
        if position[k + 1] != position[k] + 1:
            continue
        a, b = names[k], names[k + 1]
        kind = ("vowel>fricative" if a in VOWEL_SET and b in FRICATIVES else "vowel>stop" if a in VOWEL_SET and b in STOPS
                else "fricative>vowel" if a in FRICATIVES and b in VOWEL_SET else "stop>vowel" if a in STOPS and b in VOWEL_SET else None)
        if kind is None:
            continue
        before = kind.startswith("vowel")
        emission = spans[k + 1][0] * ALIGNER_STRIDE / ALIGNER_RATE
        near = [t for t in ((f - 0.5) * c.hop / c.sample_rate - emission for f, was in flips if was == before) if abs(t) < 0.15]
        offsets[kind].append(min(near, key=abs) if near else None)
report = {"utterances": len(rows), "emission_run_lengths": dict(sorted(runs.items())[:8]), "transitions": {}}
for kind, values in sorted(offsets.items()):
    found = np.array([v for v in values if v is not None]) * 1000
    report["transitions"][kind] = {"count": len(values), "flip_found": round(len(found) / len(values), 3),
                                   "median_ms": round(float(np.median(found)), 1), "std_ms": round(float(found.std()), 1)}
clean = [report["transitions"][k]["median_ms"] for k in ("vowel>fricative", "fricative>vowel") if k in report["transitions"]]
report["suggested_onset_offset_seconds"] = round(float(np.mean(clean)) / 1000, 4)
print(json.dumps(report, indent=1))
