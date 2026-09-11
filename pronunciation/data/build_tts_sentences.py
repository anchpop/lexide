#!/usr/bin/env python3
"""Build TTS sentence pools for languages without a tagging corpus.

generate_tts.py reads its text from ``tagging/train/data/cleaned_<lang>.jsonl``;
the Pimsleur-only ride-along languages (ara/ces/dan) have no tagging corpus, so
this script derives a pool from the Tatoeba text export instead (text only —
these languages have no Tatoeba *audio*). Output goes to
``data/tts_sentences/<lang>.jsonl`` with the same ``{"sentence": ...}`` shape,
plus the Tatoeba sentence id for CC-BY attribution.

Rows are filtered to the language's own script (a code-switched row would make
the TTS voice improvise on foreign text), length-bounded, deduplicated, and
written in ascending sentence-id order so a rebuild is byte-identical against
the same export. generate_tts.py applies its own seeded shuffle on top.

Fetch the export first (same file download_tatoeba.py uses):
    https://downloads.tatoeba.org/exports/sentences.tar.bz2
"""

import argparse
import json
import unicodedata
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "tts_sentences"

MIN_CHARS = 10
MAX_CHARS = 140


def script_of(ch: str) -> str:
    """Coarse script class via unicodedata name prefixes."""
    if not ch.isalpha():
        return "other"
    name = unicodedata.name(ch, "")
    for script in ("LATIN", "ARABIC", "CYRILLIC", "CJK", "HIRAGANA",
                   "KATAKANA", "HANGUL", "THAI", "DEVANAGARI", "GREEK",
                   "HEBREW"):
        if name.startswith(script):
            return script
    return "other"


# Internal lang id -> (Tatoeba lang id, required script). A sentence must
# contain the required script and no letters from any other listed script.
LANG_SCRIPT = {
    "ara": ("ara", "ARABIC"),
    "ces": ("ces", "LATIN"),
    "dan": ("dan", "LATIN"),
}


def acceptable(text: str, required: str) -> bool:
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return False
    scripts = {script_of(ch) for ch in text if ch.isalpha()}
    scripts.discard("other")
    return scripts == {required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-csv", type=Path,
                        default=Path("/tmp/tat_audio/sentences.csv"))
    parser.add_argument("--langs", nargs="+", default=sorted(LANG_SCRIPT),
                        choices=sorted(LANG_SCRIPT))
    args = parser.parse_args()
    if not args.sentences_csv.exists():
        raise SystemExit(f"{args.sentences_csv} missing — fetch sentences.tar.bz2 "
                         f"from https://downloads.tatoeba.org/exports/ first.")

    tatoeba_to_lang = {LANG_SCRIPT[l][0]: l for l in args.langs}
    pools: dict[str, dict[str, int]] = {l: {} for l in args.langs}  # text -> id
    with open(args.sentences_csv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[1] not in tatoeba_to_lang:
                continue
            lang = tatoeba_to_lang[parts[1]]
            text = parts[2].strip()
            if not acceptable(text, LANG_SCRIPT[lang][1]):
                continue
            sid = int(parts[0])
            # Dedupe on exact text; keep the lowest id for determinism.
            prev = pools[lang].get(text)
            if prev is None or sid < prev:
                pools[lang][text] = sid

    OUT_DIR.mkdir(exist_ok=True)
    for lang in args.langs:
        rows = sorted(pools[lang].items(), key=lambda kv: kv[1])
        out = OUT_DIR / f"{lang}.jsonl"
        with open(out, "w") as f:
            for text, sid in rows:
                f.write(json.dumps(
                    {"sentence": text, "tatoeba_id": sid,
                     "license": "CC BY 2.0 FR",
                     "attribution_url": f"https://tatoeba.org/sentences/show/{sid}"},
                    ensure_ascii=False) + "\n")
        print(f"{lang}: {len(rows)} sentences -> {out}")


if __name__ == "__main__":
    main()
