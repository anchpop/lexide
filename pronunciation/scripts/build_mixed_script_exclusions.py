#!/usr/bin/env python3
"""Deterministic exclusions for expansion-language label holes (no API calls).

Two classes, both found by the 2026-08-24 data audit:

1. **Mixed-script rows** (hin/jpn/zho-hans): sentences containing Latin-script
   words. The g2p backends either silently drop the Latin span (g2pM for
   Mandarin, the Hindi backend) or letter-name it (pyopenjtalk), while the
   audio speaks it naturally ("Alice太太呢?" is labeled as just 太太呢;
   Pimsleur's "SATOUSANTO" letter-named e-su-e-... while the audio says
   Satō-san to). Either way the audio<->label pair is broken. Data is
   abundant; drop them all (~1% of those corpora).

2. **Whisper-hallucination rows** (jpn Pimsleur): canonical YouTube-outro
   hallucinations ("ご視聴ありがとうございました" on 0.5s clips at 40+ phones/s).
   whisper_no_speech_prob is 0 and avg_logprob is above the -0.7 floor, so
   the trainer's whisper filters do NOT catch these.

Writes train/mixed_script_exclusions.jsonl in the load_asr_audit_exclusions
schema (per=1.0 rows; the loader re-checks expected_sha256 so a later label
fix is never punished by a stale row). train.sh passes it via --audit-path.
"""

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "train" / "mixed_script_exclusions.jsonl"

LATIN = re.compile(r"[A-Za-z]{3,}")
HALLUCINATION = re.compile("ご視聴|チャンネル登録|ご清聴")
MIXED_LANGS = ["hin", "jpn", "zho-hans"]


def main() -> None:
    rows = []
    for lang in MIXED_LANGS:
        man = REPO / "data" / "audio" / lang / "manifest.jsonl"
        for line in man.open():
            r = json.loads(line)
            reason = None
            if LATIN.search(r["sentence"]):
                reason = "mixed_script_label_hole"
            elif lang == "jpn" and HALLUCINATION.search(r["sentence"]):
                reason = "whisper_hallucination"
            if reason:
                rows.append({
                    "lang": lang, "file": r["file"], "ok": True, "per": 1.0,
                    "expected_sha256":
                        hashlib.sha256(r["sentence"].encode()).hexdigest(),
                    "reason": reason,
                    "expected": r["sentence"][:120],
                })
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[(r["lang"], r["reason"])] = by.get((r["lang"], r["reason"]), 0) + 1
    print(f"wrote {len(rows)} exclusions to {OUT}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:9} {k[1]:26} {v}")


if __name__ == "__main__":
    main()
