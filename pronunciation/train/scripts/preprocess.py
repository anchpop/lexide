"""Convert text transcripts to IPA using phonemizer and write JSONL manifests.

Uses load_dataset (which hits HF cache from the download step) to load each
dataset, then runs eSpeak-ng G2P to produce IPA labels.

Usage:
    python scripts/preprocess.py --cache_dir data/raw --output_dir data/manifests --max_hours 2000
"""

import argparse
import json
import os
import random
import traceback
from pathlib import Path

import datasets
from datasets import load_dataset, DatasetDict, Audio
from phonemizer.backend import EspeakBackend
from tqdm import tqdm

# Language -> eSpeak language code
LANG_TO_ESPEAK = {
    "en": "en-us",
    "de": "de",
    "es": "es",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "ru": "ru",
    "ko": "ko",
    "zh": "cmn",
    "ja": "ja",
    "el": "el",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
}

MLS_LANGUAGES = {
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
}

INDIC_LANG_NAMES = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "bn": "bengali",
}

# Each entry: (hf_repo, config_or_none, text_key, split_mapping, extra_kwargs)
# split_mapping: {our_split: hf_split}
DATASET_CONFIGS = {}

def _std_splits():
    return {"train": "train", "val": "validation", "test": "test"}

def _train_test():
    return {"train": "train", "test": "test"}

def build_dataset_configs():
    """Build the mapping of lang -> list of (repo, config, text_key, splits, kwargs)."""
    configs = {}

    # English — LibriSpeech
    configs["en"] = [(
        "openslr/librispeech_asr", "all", "text",
        {"train": "train.clean.360", "val": "validation.clean", "test": "test.clean"},
        {},
    )]

    # MLS languages
    for lang, config in MLS_LANGUAGES.items():
        configs[lang] = [(
            "facebook/multilingual_librispeech", config, "transcript",
            {"train": "train", "val": "dev", "test": "test"},
            {},
        )]

    # Russian — Golos
    configs["ru"] = [(
        "bond005/sberdevices_golos_10h_crowd", None, "text",
        {"train": "train", "test": "test"},
        {},
    )]

    # Korean — Zeroth
    configs["ko"] = [(
        "kresnik/zeroth_korean", None, "text",
        _train_test(),
        {},
    )]

    # Mandarin — AISHELL-1
    configs["zh"] = [(
        "AISHELL/AISHELL-1", None, "text",
        {"train": "train", "val": "validation", "test": "test"},
        {},
    )]

    # Japanese — ReazonSpeech
    configs["ja"] = [(
        "reazon-research/reazonspeech", "small", "transcription",
        {"train": "train"},
        {"trust_remote_code": True},
    )]

    # Greek — Common Voice (try mozilla first, then fsicoli)
    configs["el"] = [(
        "mozilla-foundation/common_voice_17_0", "el", "sentence",
        {"train": "train", "val": "validation", "test": "test"},
        {"token": os.environ.get("HF_TOKEN")},
    )]

    # Indic languages — Kathbath + Shrutilipi
    for lang, lang_name in INDIC_LANG_NAMES.items():
        configs[lang] = [
            (
                "ai4bharat/Kathbath", lang_name, "text",
                {"train": "train", "val": "valid", "test": "test"},
                {},
            ),
            (
                "ai4bharat/Shrutilipi", lang_name, "text",
                {"train": "train"},
                {},
            ),
        ]

    # Bengali supplement — FLEURS (replaced OpenSLR 53 which needs loading scripts)
    configs["bn"].append((
        "google/fleurs", "bn_in", "transcription",
        {"train": "train", "val": "validation", "test": "test"},
        {},
    ))

    return configs


def phonemize_batch(texts: list[str], lang: str) -> list[str]:
    espeak_lang = LANG_TO_ESPEAK[lang]
    backend = EspeakBackend(
        language=espeak_lang,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )
    return backend.phonemize(texts, strip=True)


def _write_one_flac(args):
    """Worker function: decode audio bytes and write FLAC. Returns example dict or None."""
    import io
    import soundfile as sf

    idx, audio_bytes, audio_path_hint, text, audio_dir, min_dur, max_dur = args

    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        return None

    duration = len(data) / sr
    if not (min_dur <= duration <= max_dur):
        return None

    ext = Path(audio_path_hint).stem if audio_path_hint else str(idx)
    flac_path = audio_dir / f"{ext}.flac"
    if not flac_path.exists():
        sf.write(str(flac_path), data, sr, format="FLAC")

    return {"audio_path": str(flac_path), "text": text, "duration_s": duration}


def extract_and_save_audio(
    ds,
    text_key: str,
    lang: str,
    output_dir: str,
    split: str,
    min_dur: float = 0.5,
    max_dur: float = 30.0,
    num_workers: int = 32,
) -> list[dict]:
    """Extract audio files from HF dataset and save as FLAC files.

    Uses multiprocessing to parallelize decode+write across CPUs.
    """
    from multiprocessing import Pool

    audio_dir = Path(output_dir) / "audio" / lang / split
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Disable HF audio decoding — we'll decode manually via soundfile
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    # Collect work items (read from HF dataset is sequential)
    work_items = []
    for idx, row in enumerate(tqdm(ds, desc="  Reading dataset")):
        text = (row.get(text_key) or "").strip()
        if not text:
            continue

        audio = row.get("audio")
        if audio is None:
            continue

        audio_bytes = None
        audio_path_hint = ""
        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes")
            audio_path_hint = audio.get("path") or ""

        if audio_bytes is None:
            continue

        work_items.append((idx, audio_bytes, audio_path_hint, text, audio_dir, min_dur, max_dur))

    print(f"  Writing {len(work_items)} FLAC files with {num_workers} workers...")

    # Parallel decode + write
    examples = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(_write_one_flac, work_items, chunksize=64), total=len(work_items), desc="  Writing FLAC"):
            if result is not None:
                examples.append(result)

    return examples


def process_split(
    examples: list[dict],
    lang: str,
    output_path: str,
    max_hours: float | None,
    batch_size: int = 1000,
) -> dict:
    total_hours = sum(e["duration_s"] for e in examples) / 3600

    if max_hours and total_hours > max_hours:
        ratio = max_hours / total_hours
        random.seed(42)
        examples = random.sample(examples, int(len(examples) * ratio))
        print(f"  Subsampled from {total_hours:.0f}h to ~{max_hours}h ({len(examples)} utterances)")

    all_texts = [e["text"] for e in examples]
    all_ipa: list[str] = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc=f"  Phonemizing {lang}"):
        batch = all_texts[i : i + batch_size]
        all_ipa.extend(phonemize_batch(batch, lang))

    written = 0
    total_dur = 0.0
    with open(output_path, "w") as f:
        for example, ipa in zip(examples, all_ipa):
            if not ipa.strip():
                continue
            entry = {
                "audio_path": example["audio_path"],
                "text": example["text"],
                "ipa": ipa,
                "lang": lang,
                "duration_s": example["duration_s"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            total_dur += example["duration_s"]

    return {"utterances": written, "hours": total_dur / 3600}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/raw",
                        help="HF datasets cache directory (same as download_data.py --output_dir)")
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    parser.add_argument("--max_hours", type=float, default=2000.0)
    parser.add_argument("--languages", type=str, default="all")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_configs = build_dataset_configs()

    if args.languages == "all":
        langs = list(LANG_TO_ESPEAK.keys())
    else:
        langs = args.languages.split(",")

    all_stats = []

    for lang in langs:
        if lang not in dataset_configs:
            print(f"No config for {lang}, skipping.")
            continue

        for repo, config, text_key, split_map, extra_kwargs in dataset_configs[lang]:
            print(f"\n{'='*40}")
            print(f"{lang}: {repo} (config={config})")
            print(f"{'='*40}")

            # Determine cache subdir — match what download_data.py used
            if "multilingual_librispeech" in repo:
                cache_sub = f"mls/{lang}"
            elif "librispeech_asr" in repo:
                cache_sub = "librispeech"
            elif "golos" in repo:
                cache_sub = "golos"
            elif "zeroth" in repo:
                cache_sub = "zeroth_korean"
            elif "AISHELL" in repo:
                cache_sub = "aishell1"
            elif "reazonspeech" in repo:
                cache_sub = "reazonspeech"
            elif "common_voice" in repo:
                cache_sub = "common_voice_el"
            elif "Kathbath" in repo:
                cache_sub = f"kathbath/{lang}"
            elif "Shrutilipi" in repo:
                cache_sub = f"shrutilipi/{lang}"
            elif "fleurs" in repo:
                cache_sub = "fleurs_bn"
            else:
                cache_sub = lang

            cache_dir = str(Path(args.cache_dir) / cache_sub)

            try:
                load_args = [repo]
                load_kwargs = {"cache_dir": cache_dir, **extra_kwargs}
                if config:
                    load_args.append(config)

                ds_dict = load_dataset(*load_args, **load_kwargs)
            except Exception:
                print(f"  Failed to load {repo}:")
                traceback.print_exc()
                continue

            if not isinstance(ds_dict, DatasetDict):
                # Single split dataset
                ds_dict = DatasetDict({"train": ds_dict})

            print(f"  Splits available: {list(ds_dict.keys())}")

            for our_split, hf_split in split_map.items():
                if hf_split not in ds_dict:
                    print(f"  Split '{hf_split}' not found, skipping {our_split}.")
                    continue

                output_path = output_dir / f"manifest_{lang}_{our_split}.jsonl"
                # If manifest already exists (from another dataset for same lang),
                # append to it
                mode = "a" if output_path.exists() else "w"

                print(f"\n  {our_split} (hf: {hf_split})...")
                try:
                    examples = extract_and_save_audio(
                        ds_dict[hf_split], text_key, lang,
                        str(output_dir), our_split,
                    )
                    print(f"  Extracted {len(examples)} examples")

                    if not examples:
                        continue

                    max_hours = args.max_hours if our_split == "train" else None
                    stats = process_split(examples, lang, str(output_path), max_hours)
                    all_stats.append({"lang": lang, "repo": repo, "split": our_split, **stats})
                    print(f"  -> {stats['utterances']} utterances, {stats['hours']:.1f}h")
                except Exception:
                    print(f"  ERROR processing {lang}/{our_split}:")
                    traceback.print_exc()

    print(f"\n{'='*40}")
    print("Summary")
    print(f"{'='*40}")
    for s in all_stats:
        print(f"  {s['lang']}/{s['split']} ({s['repo']}): {s['utterances']} utt, {s['hours']:.1f}h")


if __name__ == "__main__":
    main()
