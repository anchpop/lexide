#!/usr/bin/env python3
"""Verify the installed g2p build (our espeak-ng fork) reproduces the existing
corpus labels.

CLAUDE.md mandates this before regenerating labels with any new g2p/espeak
build: re-phonemize a sample of the existing phonemes.jsonl through the WHOLE
path — phonemize() then validate_phonemes() (vocab + LANG_PHONEME_REMAP), using
each row's own espeak_voice — and require byte-identical phoneme output.

Run:  scripts/py-linux.sh scripts/verify_espeak_build.py [--per-lang N] [--langs rus,ita]

Exit 0 = every sampled espeak-labeled row reproduces exactly.
Rows labeled by a non-espeak backend (phoneme_backend != "espeak") are skipped.
Stress is compared too, except on rows with a fra stress override (those rows'
stored stress came from the override, not espeak).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "train" / "scripts"))

# Load .env (per-machine) so a G2P_BIN override there is honoured.
for env_file in (REPO / ".env",):
    if env_file.exists():
        import os
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

from preprocess import LANG_TO_ESPEAK, phonemize, validate_phonemes  # noqa: E402


def sample(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=300)
    ap.add_argument("--langs", type=str, default=None,
                    help="comma-separated subset; default = all with phonemes.jsonl")
    ap.add_argument("--show", type=int, default=3, help="mismatch examples to print per lang")
    args = ap.parse_args()

    audio = REPO / "data" / "audio"
    langs = (args.langs.split(",") if args.langs else
             sorted(d.name for d in audio.iterdir()
                    if (d / "phonemes.jsonl").exists() and d.name in LANG_TO_ESPEAK))

    any_bad = False
    for lang in langs:
        lang_dir = audio / lang
        voice_by_file = {}
        manifest = lang_dir / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.open():
                r = json.loads(line)
                if r.get("espeak_voice"):
                    voice_by_file[r["file"]] = r["espeak_voice"]

        override_files = set()
        so = lang_dir / "stress_overrides.jsonl"
        if so.exists():
            for line in so.open():
                override_files.add(json.loads(line)["file"])

        rows = [json.loads(l) for l in (lang_dir / "phonemes.jsonl").open()]
        rows = [r for r in rows if r.get("phoneme_backend", "espeak") == "espeak"]
        picked = sample(rows, args.per_lang)

        ph_bad = st_bad = 0
        examples = []
        for r in picked:
            voice = voice_by_file.get(r["file"]) or LANG_TO_ESPEAK[lang]
            ph, st, _spans = phonemize(r["sentence"], voice)
            ph, st, _unknown = validate_phonemes(ph, st, lang)
            if ph != r["phonemes"]:
                ph_bad += 1
                if len(examples) < args.show:
                    examples.append((r["file"], r["sentence"],
                                     " ".join(r["phonemes"]), " ".join(ph)))
            elif st != r["stress"] and r["file"] not in override_files:
                st_bad += 1
        status = "OK " if ph_bad == 0 and st_bad == 0 else "FAIL"
        print(f"{status} {lang}: {len(picked)} sampled, "
              f"{ph_bad} phoneme mismatches, {st_bad} stress mismatches "
              f"({len(rows)} espeak rows total)")
        for f, sent, want, got in examples:
            print(f"    {f}: {sent}")
            print(f"      corpus: {want}")
            print(f"      build : {got}")
        if ph_bad or st_bad:
            any_bad = True
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
