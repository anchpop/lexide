"""Stage a MUSAN subset as the real-noise pool for degradation augmentation.

Downloads MUSAN (openslr/17: music, noise, and speech recordings, 16 kHz mono
WAV) and copies a size-budgeted subset into ``data/audio/_noise/{music,noise,
speech}/``, where the training collate's NoisePool (train/src/dataset.py)
random-crops it for the music / babble / ambience degradations. The dir rides
along in the dataset tar automatically (build_dataset_tar packs all of
data/audio); preprocess and the trainer both skip it (no manifest.jsonl /
phonemes.jsonl).

Budgets: everything a category offers is pointless — each training clip mixes
in only a few seconds, so a few hours of unique material per category is
plenty and keeps the tar small. Defaults: all of noise/ (~0.7 GB), 1.5 GB of
music, 0.75 GB of speech (babble source). Selection is seeded, so re-runs pick
the same files.

The full archive (~11 GB) and its extraction are cached under .work/ so
re-subsetting with different budgets never re-downloads.

Usage (from the pronunciation/ root, any python with stdlib only):
    scripts/py-linux.sh data/download_noise.py
    scripts/py-linux.sh data/download_noise.py --music-gb 2.5 --force
"""

import argparse
import random
import shutil
import subprocess
import sys
import tarfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / ".work"
ARCHIVE = WORK / "musan.tar.gz"
EXTRACTED = WORK / "musan"
DEST = REPO / "data" / "audio" / "_noise"
URL = "https://www.openslr.org/resources/17/musan.tar.gz"
SEED = 0


def download() -> None:
    if EXTRACTED.is_dir():
        print(f"using cached extraction: {EXTRACTED}")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {ARCHIVE} (resumable)")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "-o", str(ARCHIVE), URL],
        check=True,
    )
    print(f"extracting -> {EXTRACTED.parent}")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(EXTRACTED.parent)  # archive root is musan/
    if not EXTRACTED.is_dir():
        raise SystemExit(f"archive did not extract to expected {EXTRACTED}")


def wav_ok(path: Path) -> bool:
    """The pool assumes 16 kHz mono; MUSAN is, but verify per file rather than
    trusting the archive — a violating file would silently mix wrong-speed noise."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1
    except (wave.Error, EOFError, OSError):
        return False


def select(category: str, budget_gb: float | None) -> list[Path]:
    """Seeded shuffle, then take files until the byte budget (None = all)."""
    files = sorted((EXTRACTED / category).rglob("*.wav"))
    if not files:
        raise SystemExit(f"no wavs under {EXTRACTED / category} — bad extraction?")
    random.Random(f"{SEED}/{category}").shuffle(files)
    if budget_gb is None:
        return files
    picked, total = [], 0
    for f in files:
        size = f.stat().st_size
        if total + size > budget_gb * 1e9:
            continue
        picked.append(f)
        total += size
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--music-gb", type=float, default=1.5)
    ap.add_argument("--noise-gb", type=float, default=None,
                    help="default: keep all of noise/ (~0.7 GB)")
    ap.add_argument("--speech-gb", type=float, default=0.75)
    ap.add_argument("--force", action="store_true",
                    help="rebuild data/audio/_noise even if it exists")
    args = ap.parse_args()

    if DEST.is_dir() and any(DEST.rglob("*.wav")) and not args.force:
        n = sum(1 for _ in DEST.rglob("*.wav"))
        raise SystemExit(f"{DEST} already holds {n} wavs — pass --force to rebuild")

    download()

    budgets = {"music": args.music_gb, "noise": args.noise_gb, "speech": args.speech_gb}
    if DEST.is_dir():
        shutil.rmtree(DEST)
    total_bytes = 0
    bad = 0
    for category, budget in budgets.items():
        picked = select(category, budget)
        for src in picked:
            if not wav_ok(src):
                bad += 1
                continue
            rel = src.relative_to(EXTRACTED)
            dst = DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copy, NOT copy2: build_dataset_tar's freshness check compares the
            # tar's mtime against the newest file under data/audio, so staged
            # files must carry their staging time, not MUSAN's archive dates.
            shutil.copy(src, dst)
            total_bytes += src.stat().st_size
        print(f"{category}: staged {len(picked)} files")
    # MUSAN's attribution/license metadata travels with the subset.
    for meta in EXTRACTED.rglob("LICENSE*"):
        dst = DEST / meta.relative_to(EXTRACTED)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meta, dst)
    if bad:
        print(f"skipped {bad} files that were not 16 kHz mono", file=sys.stderr)
    print(f"wrote {total_bytes / 1e9:.2f} GB to {DEST}")


if __name__ == "__main__":
    main()
