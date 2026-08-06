#!/usr/bin/env python3
"""Generate TTS audio from tagging sentences using Google Chirp3-HD voices."""

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import texttospeech
from tqdm import tqdm

LANG_CONFIG = {
    "eng": "en-US",
    "deu": "de-DE",
    "fra": "fr-FR",
    "ita": "it-IT",
    "por": "pt-BR",
    "spa": "es-ES",
    "rus": "ru-RU",
    "tha": "th-TH",
    "zho-hans": "cmn-CN",
    "hin": "hi-IN",
    "jpn": "ja-JP",
}

TAGGING_DATA = Path(__file__).resolve().parent.parent.parent / "tagging" / "train" / "data"


def make_client() -> texttospeech.TextToSpeechClient:
    """Authenticate with GOOGLE_CLOUD_API_KEY (env or repo .env), else ADC.

    The API key path is what works on the Linux box, which has no gcloud
    application-default credentials.
    """
    key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    if not key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GOOGLE_CLOUD_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if key:
        return texttospeech.TextToSpeechClient(client_options={"api_key": key})
    return texttospeech.TextToSpeechClient()


def get_chirp3_voices(client: texttospeech.TextToSpeechClient, language_code: str) -> list[str]:
    """Get available Chirp3-HD voice names for a language."""
    response = client.list_voices(language_code=language_code)
    return [v.name for v in response.voices if "Chirp3-HD" in v.name]


def load_sentences(lang: str) -> list[str]:
    """Load sentences from tagging data."""
    path = TAGGING_DATA / f"cleaned_{lang}.jsonl"
    sentences = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            sentences.append(record["sentence"])
    return sentences


def sentence_hash(sentence: str) -> str:
    """First 16 hex chars of SHA256."""
    return hashlib.sha256(sentence.encode()).hexdigest()[:16]


def synthesize_one(client, sentence, voice_name, language_code, audio_config, out_dir):
    """Synthesize a single sentence. Returns (hash, record) or None on error."""
    h = sentence_hash(sentence)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )
    synthesis_input = texttospeech.SynthesisInput(text=sentence)

    for attempt in range(5):
        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            print(f"Error synthesizing '{sentence[:60]}...': {e}")
            return None

    wav_path = out_dir / f"{h}.wav"
    wav_path.write_bytes(response.audio_content)

    return {
        "file": f"{h}.wav",
        "sentence": sentence,
        "source": "tts",
        "voice": voice_name,
    }


def generate_for_language(
    lang: str,
    max_sentences: int | None,
    output_root: Path,
    seed: int,
    workers: int,
    rps: float,
    extremes: int,
):
    client = make_client()
    language_code = LANG_CONFIG[lang]
    voices = get_chirp3_voices(client, language_code)
    if not voices:
        print(f"No Chirp3-HD voices found for {language_code}, skipping {lang}")
        return

    print(f"{lang}: found {len(voices)} Chirp3-HD voices, using {workers} workers")

    all_sentences = load_sentences(lang)
    rng = random.Random(seed)
    shuffled = all_sentences.copy()
    rng.shuffle(shuffled)

    if max_sentences:
        sentences = shuffled[:max_sentences]
    else:
        sentences = shuffled

    # Add N longest and N shortest sentences for variation
    if extremes > 0:
        by_length = sorted(all_sentences, key=len)
        shortest = by_length[:extremes]
        longest = by_length[-extremes:]
        # Dedupe via hash (also dedupes against the random sample)
        seen = {sentence_hash(s) for s in sentences}
        for s in shortest + longest:
            h = sentence_hash(s)
            if h not in seen:
                sentences.append(s)
                seen.add(h)

    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Load existing hashes to allow resuming
    existing = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                existing.add(json.loads(line)["file"].removesuffix(".wav"))

    # Filter to only sentences we still need
    todo = []
    for sentence in sentences:
        h = sentence_hash(sentence)
        if h not in existing:
            voice_name = rng.choice(voices)
            todo.append((sentence, voice_name))
        else:
            # Advance RNG to keep determinism
            rng.choice(voices)

    if not todo:
        print(f"{lang}: all sentences already generated, skipping")
        return

    chars = sum(len(sentence) for sentence, _ in todo)
    print(f"{lang}: {len(todo)} sentences to generate ({len(existing)} already done), "
          f"{chars:,} chars ≈ ${chars / 1e6 * 30:.2f} at Chirp3-HD list price")

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    manifest_lock = threading.Lock()
    # Token bucket rate limiter to stay under quota
    rate_lock = threading.Lock()
    last_request_time = [0.0]  # mutable container for closure

    def throttled_synthesize(*args):
        # Enforce minimum interval between requests globally
        with rate_lock:
            now = time.monotonic()
            min_interval = 1.0 / rps
            wait = last_request_time[0] + min_interval - now
            if wait > 0:
                time.sleep(wait)
            last_request_time[0] = time.monotonic()
        return synthesize_one(*args)

    with open(manifest_path, "a") as manifest:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    throttled_synthesize, client, sentence, voice_name,
                    language_code, audio_config, out_dir,
                ): sentence
                for sentence, voice_name in todo
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=lang):
                record = future.result()
                if record is None:
                    continue
                record["lang"] = lang
                with manifest_lock:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()


def main():
    parser = argparse.ArgumentParser(description="Generate TTS dataset from tagging sentences")
    parser.add_argument("--langs", nargs="+", default=list(LANG_CONFIG.keys()),
                        choices=list(LANG_CONFIG.keys()),
                        help="Languages to generate (default: all)")
    parser.add_argument("--max-sentences", type=int, default=5000,
                        help="Max sentences per language. The default keeps a "
                             "4-language run around $30 at Chirp3-HD list "
                             "price and near source-parity with the core "
                             "languages' tts split; pass 0 for all.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "audio",
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--rps", type=float, default=5, help="Max requests per second")
    parser.add_argument("--extremes", type=int, default=0,
                        help="Also include N longest and N shortest sentences per language")
    args = parser.parse_args()

    for lang in args.langs:
        generate_for_language(
            lang, args.max_sentences, args.output, args.seed,
            args.workers, args.rps, args.extremes,
        )


if __name__ == "__main__":
    main()
