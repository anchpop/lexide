#!/usr/bin/env python3
"""Backfill `espeak_voice` into Pimsleur manifest rows extracted before
download_pimsleur.py was taught to store it.

Why this exists
---------------
Several Pimsleur languages share a 3-letter lang code across multiple
dialect-distinct courses with different espeak voices:

  por: Brazilian Portuguese (pt-br)  +  European Portuguese (pt)
  hye: Armenian Eastern (hy)         +  Western Armenian (hyw)
  spa: Castilian Spanish (es)        +  Latin American Spanish (es-419)

Manifest rows produced by the old code don't record which voice was used
at extraction time. preprocess.py therefore falls back to
LANG_TO_ESPEAK[lang], which is a single voice per lang — and that
silently mislabels every clip whose dialect differs from the canonical
voice. Most acute: European Portuguese audio getting Brazilian phonemes.

This script rebuilds the lookup by walking the Pimsleur archive, mapping
each MP3's `bare-stem` to the espeak voice declared in
PIMSLEUR_FOLDER_TO_LANG for its parent folder. Then it rewrites each
lang's manifest.jsonl in place, adding `espeak_voice` to any Pimsleur
row that lacks it.

Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
from download_pimsleur import PIMSLEUR_FOLDER_TO_LANG  # noqa: E402


def build_stem_to_voice(pimsleur_root: Path) -> dict[tuple[str, str], str]:
    """Map (3-letter lang, bare-stem) -> espeak voice, by scanning the archive.

    Keyed by (lang, stem) instead of just stem so cross-language stem
    collisions (theoretically possible — e.g. a hypothetical "Unit 01.mp3"
    that exists in both French and German trees) can't mass-mislabel one
    language with another's voice. Same-language voice collisions (a stem
    that maps to two different espeak voices within the same lang, the
    actual dialect ambiguity case) are dropped entirely: we refuse to
    backfill when the source dialect can't be determined unambiguously,
    so preprocess.py falls back to LANG_TO_ESPEAK[lang] for those rows
    rather than guessing.
    """
    voices_seen: dict[tuple[str, str], set[str]] = {}
    for folder in pimsleur_root.iterdir():
        if not folder.is_dir():
            continue
        mapping = PIMSLEUR_FOLDER_TO_LANG.get(folder.name)
        if mapping is None:
            continue
        lang, espeak_voice = mapping
        if espeak_voice is None:
            continue
        for mp3 in folder.rglob("*.mp3"):
            voices_seen.setdefault((lang, mp3.stem), set()).add(espeak_voice)
    out: dict[tuple[str, str], str] = {}
    ambiguous: list[tuple[str, str, set[str]]] = []
    for key, voices in voices_seen.items():
        if len(voices) == 1:
            out[key] = next(iter(voices))
        else:
            ambiguous.append((key[0], key[1], voices))
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} (lang, stem) pairs are ambiguous "
              f"(same stem mapped to >1 espeak voice within the same lang); "
              f"these rows will be left without espeak_voice and will fall "
              f"back to LANG_TO_ESPEAK[lang] in preprocess.py.")
        for lang, stem, voices in ambiguous[:10]:
            print(f"  ({lang}) {stem}: {sorted(voices)}")
    return out


def stem_from_filename(file_field: str) -> str | None:
    """`pimsleur_<lesson_id>_<seg:04d>.wav` -> lesson_id.

    Handles both the old bare-stem scheme (lesson_id = mp3.stem) and the
    new path-derived scheme (lesson_id may contain `__`). For the new
    scheme this stem doesn't match the archive directly — those rows are
    expected to already have espeak_voice and don't need backfill.
    """
    if not file_field.startswith("pimsleur_") or not file_field.endswith(".wav"):
        return None
    inner = file_field[len("pimsleur_"):-len(".wav")]
    # last `_NNNN` is the segment index
    return inner.rsplit("_", 1)[0]


def backfill_manifest(
    lang: str, manifest: Path,
    stem_to_voice: dict[tuple[str, str], str],
) -> tuple[int, int, int]:
    """Returns (rewritten_rows, already_had_voice, unmatched_pimsleur_rows)."""
    if not manifest.exists():
        return (0, 0, 0)
    rows = []
    with manifest.open() as f:
        for line in f:
            rows.append(json.loads(line))

    rewritten = 0
    already = 0
    unmatched = 0
    for rec in rows:
        if rec.get("source") != "pimsleur":
            continue
        if rec.get("espeak_voice"):
            already += 1
            continue
        stem = stem_from_filename(rec["file"])
        if stem is None:
            unmatched += 1
            continue
        voice = stem_to_voice.get((lang, stem))
        if voice is None:
            unmatched += 1
            continue
        rec["espeak_voice"] = voice
        rewritten += 1

    if rewritten:
        tmp = manifest.with_suffix(manifest.suffix + ".tmp")
        with tmp.open("w") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(manifest)

    return rewritten, already, unmatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pimsleur-root", type=Path,
                        default=Path("/Volumes/T7/p_rty/Pimsleur Complete Collection"))
    parser.add_argument("--audio-root", type=Path,
                        default=REPO_ROOT / "data" / "audio")
    args = parser.parse_args()

    if not args.pimsleur_root.exists():
        sys.exit(f"Pimsleur root not found: {args.pimsleur_root}")

    print(f"Scanning {args.pimsleur_root}...")
    stem_to_voice = build_stem_to_voice(args.pimsleur_root)
    print(f"  built lookup for {len(stem_to_voice)} mp3 stems\n")

    grand_rewrite = 0
    grand_already = 0
    grand_unmatched = 0
    for lang_dir in sorted(args.audio_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        manifest = lang_dir / "manifest.jsonl"
        rw, al, um = backfill_manifest(lang_dir.name, manifest, stem_to_voice)
        if rw or al or um:
            print(f"  {lang_dir.name}: rewrote={rw} already_set={al} unmatched={um}")
        grand_rewrite += rw
        grand_already += al
        grand_unmatched += um
    print(f"\nTotal: rewrote={grand_rewrite} already_set={grand_already} "
          f"unmatched={grand_unmatched}")


if __name__ == "__main__":
    main()
