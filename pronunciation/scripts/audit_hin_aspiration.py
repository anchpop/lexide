#!/usr/bin/env python3
"""Hindi aspiration listen-audit (Gemini forced-choice, per word).

The film eval showed the model rarely emits the aspiration the hin reference
predicts (`h` missing x166, standalone `ʰ` x112 vs espeak). This audit
decides between the two explanations — the reference over-predicts
aspiration vs the model under-hears it — by asking a listener (Gemini) to
judge, word by word, whether aspiration marked in the ORTHOGRAPHY is
actually realized. Devanagari marks it explicitly (ख घ छ झ ठ ढ थ ध फ भ,
plus ह), so no phoneme-to-word alignment is needed, and unaspirated
counterpart words (क ग च ज ट ड त द प ब) are included as bias controls: a
listener that "hears" aspiration on those is guessing.

Run:  scripts/py-linux.sh scripts/audit_hin_aspiration.py [--clips 36]
Writes .work/hin_aspiration_audit.jsonl (resumable) and prints rates.
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "data" / "audio" / "hin"
OUT = REPO / ".work" / "hin_aspiration_audit.jsonl"

ASPIRATED = set("खघछझठढथधफभ")
UNASPIRATED = set("कगचजटडतदपब")

PROMPT = """You are a Hindi phonetician auditing a speech dataset. The \
attached clip is claimed to contain this Hindi sentence:

  "{sentence}"

For each numbered item below, listen closely to how the SPECIFIC consonant \
of the marked letter is pronounced in that word, and judge its aspiration. \
Native connected speech often weakens aspiration — judge what you HEAR, not \
what the spelling implies.

{items}

Answer in strict JSON (no markdown): a list, one object per item, in order:
[{{"item": <n>, "word": "<the word>", "letter": "<the letter>",
   "verdict": "clearly_aspirated" | "weakly_aspirated" | "unaspirated" | \
"not_audible",
   "note": "<short, only if noteworthy>"}}]

For ह items, "clearly_aspirated" means a clearly audible [ɦ], \
"weakly_aspirated" a reduced breathy trace, "unaspirated" means the [ɦ] is \
deleted entirely."""


def call_gemini(model, key, wav, prompt):
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/wav",
                             "data": base64.b64encode(wav.read_bytes()).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0,
                             "response_mime_type": "application/json"},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            return json.loads(
                resp["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return [{"item": -1, "verdict": "error", "note": str(e)[:200]}]
            time.sleep(5 * (attempt + 1))


def targets_in(sentence):
    """(word, letter, kind) items: aspirated letters, ह, and controls."""
    items = []
    for word in re.findall(r"[ऀ-ॿ]+", sentence):
        for ch in word:
            if ch in ASPIRATED:
                items.append((word, ch, "aspirated_letter"))
            elif ch == "ह":
                items.append((word, ch, "h_letter"))
            elif ch in UNASPIRATED:
                items.append((word, ch, "control_unaspirated"))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=36)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')

    done = set()
    if OUT.exists():
        for line in OUT.open():
            done.add(json.loads(line)["file"])

    rows = [json.loads(l) for l in (AUDIO / "phonemes.jsonl").open()]
    by_src = {}
    for r in rows:
        if targets_in(r["sentence"]):
            by_src.setdefault(r.get("source"), []).append(r)
    rng = random.Random(args.seed)
    per_src = max(1, args.clips // max(1, len(by_src)))
    picked = []
    for src, rs in sorted(by_src.items()):
        picked.extend(rng.sample(rs, min(per_src, len(rs))))
    picked = [r for r in picked if r["file"] not in done][:args.clips]
    print(f"{len(picked)} clips to audit across {len(by_src)} sources")

    with OUT.open("a") as out:
        for i, r in enumerate(picked):
            items = targets_in(r["sentence"])
            # Cap the aspirated/h items; keep at most 2 controls.
            asp = [x for x in items if x[2] != "control_unaspirated"][:5]
            ctl = [x for x in items if x[2] == "control_unaspirated"][:2]
            chosen = asp + ctl
            listing = "\n".join(
                f"{n+1}. In the word “{w}” — the consonant of the "
                f"letter “{c}”."
                for n, (w, c, _) in enumerate(chosen))
            verdicts = call_gemini(args.model, key, AUDIO / r["file"],
                                   PROMPT.format(sentence=r["sentence"],
                                                 items=listing))
            rec = {"file": r["file"], "source": r.get("source"),
                   "sentence": r["sentence"],
                   "items": [{"word": w, "letter": c, "kind": k}
                             for w, c, k in chosen],
                   "verdicts": verdicts}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i+1}/{len(picked)}] {r.get('source')}/{r['file']}",
                  flush=True)

    # Aggregate.
    import collections
    tally = collections.defaultdict(collections.Counter)
    for line in OUT.open():
        rec = json.loads(line)
        for item, v in zip(rec["items"], rec["verdicts"]):
            if isinstance(v, dict) and v.get("verdict") not in (None, "error"):
                tally[item["kind"]][v["verdict"]] += 1
    print("\n=== aspiration realization by letter class ===")
    for kind, c in sorted(tally.items()):
        total = sum(c.values())
        parts = ", ".join(f"{k}: {n} ({n/total:.0%})"
                          for k, n in c.most_common())
        print(f"{kind:22} n={total}  {parts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
