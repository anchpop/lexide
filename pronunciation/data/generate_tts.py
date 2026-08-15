#!/usr/bin/env python3
"""Generate TTS audio from tagging sentences.

Two backends, both writing into the same ``data/audio/<lang>/`` layout:

``chirp3``
    Google Cloud Text-to-Speech Chirp3-HD voices. One synthetic speaker per
    request, drawn from the language's Chirp3-HD roster.

``gemini``
    Gemini's TTS model (``gemini-3.1-flash-tts-preview``) over the Generative
    Language REST API. Its ~29 prebuilt voices are *not* language-scoped — the
    same "Kore" speaks every language — so the recorded ``voice`` string is
    scoped by language anyway. Within-speaker acoustic normalization keys on
    ``speaker_cluster or voice``, and per the clustering rules splitting one
    speaker is harmless while merging two corrupts the baseline.

Because a Gemini TTS model is an LLM, it can in principle editorialize rather
than read. Audit its output the same way the human corpora are audited
(``scripts/audit_asr_groq.py --source tts``) before training on it.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from wav_utils import repair_streamed_wav_header

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

TARGET_SAMPLE_RATE = 16000

GEMINI_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Verified against the live model roster: every name below returned audio.
# "Callirrhoe" is in the published prebuilt list but timed out on probe, so it
# is left out rather than allowed to fail mid-run.
GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina",
    "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam",
    "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

# The TTS models take natural-language style direction in the prompt and speak
# only the content. Keep the direction minimal and neutral: this corpus wants
# ordinary citation-register speech, not performance. Read-it-verbatim wording
# matters more than tone — an LLM asked to "say" something may paraphrase.
GEMINI_STYLE = (
    "Read the following text aloud exactly as written, at a natural pace "
    "in a neutral speaking voice. Do not add, omit, or reword anything.\n\n"
)


def read_env_key(name: str) -> str | None:
    """Read a secret from the environment, falling back to the repo .env."""
    key = os.environ.get(name)
    if key:
        return key
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def make_client():
    """Authenticate with GOOGLE_CLOUD_API_KEY (env or repo .env), else ADC.

    The API key path is what works on the Linux box, which has no gcloud
    application-default credentials.
    """
    from google.cloud import texttospeech

    key = read_env_key("GOOGLE_CLOUD_API_KEY")
    if key:
        return texttospeech.TextToSpeechClient(client_options={"api_key": key})
    return texttospeech.TextToSpeechClient()


def get_chirp3_voices(client, language_code: str) -> list[str]:
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


def gemini_clip_hash(lang: str, voice: str, sentence: str) -> str:
    """Filename key for a Gemini clip.

    Deliberately *not* ``sentence_hash``: both backends write into one
    ``data/audio/<lang>/`` directory, and the two roster picks for a sentence
    are different recordings. Keying on backend+voice keeps them from
    overwriting each other and leaves every pre-existing Chirp3 filename
    (which is the bare sentence hash) untouched.
    """
    return hashlib.sha256(f"gemini:{lang}:{voice}:{sentence}".encode()).hexdigest()[:16]


def parse_pcm_mime(mime: str) -> int:
    """Pull the sample rate out of a ``audio/l16; rate=24000; channels=1`` mime.

    The rate is documented as 24 kHz but is declared per-response, so read it
    rather than assume — a silent rate change would resample every clip wrong.
    """
    for part in mime.split(";"):
        part = part.strip()
        if part.startswith("rate="):
            return int(part.split("=", 1)[1])
    raise ValueError(f"no sample rate in TTS mime type {mime!r}")


def write_pcm_as_wav(pcm: bytes, source_rate: int, path: Path) -> float:
    """Resample raw mono s16le PCM to the corpus rate and write a WAV.

    Returns the duration in seconds. 24 kHz -> 16 kHz is an exact 2/3 ratio, so
    polyphase resampling is exact-rational and introduces no drift.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if source_rate != TARGET_SAMPLE_RATE:
        gcd = np.gcd(source_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // gcd, source_rate // gcd)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / TARGET_SAMPLE_RATE


def synthesize_one_gemini(api_key, sentence, voice, lang, model, style, out_dir):
    """Synthesize one sentence with a Gemini TTS voice.

    Returns (record, audio_tokens) or (None, 0). Uses the REST API directly:
    the payload is three fields and this avoids adding an SDK dependency to a
    script whose only other client is google-cloud-texttospeech.
    """
    body = json.dumps({
        "contents": [{"parts": [{"text": style + sentence}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}},
            },
        },
    }).encode()
    url = GEMINI_ENDPOINT.format(model=model) + f"?key={api_key}"

    payload = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url, body, {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as e:
            # 429 (quota) and 5xx (transient) are worth retrying; a 400 is a
            # bad request that will fail identically forever.
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(2 ** attempt, 30))
                continue
            print(f"Gemini HTTP {e.code} on '{sentence[:50]}...': {e.read()[:200]!r}")
            return None, 0
        except Exception as e:
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))
                continue
            print(f"Gemini error on '{sentence[:50]}...': {e}")
            return None, 0

    try:
        part = payload["candidates"][0]["content"]["parts"][0]
        inline = part["inlineData"]
    except (KeyError, IndexError):
        # A refusal or a text reply instead of audio — the model declined to
        # read this one. Drop the clip rather than write a broken pair.
        finish = (payload.get("candidates") or [{}])[0].get("finishReason")
        print(f"Gemini returned no audio ({finish}) for '{sentence[:50]}...'")
        return None, 0

    pcm = base64.b64decode(inline["data"])
    rate = parse_pcm_mime(inline["mimeType"])
    if not pcm:
        print(f"Gemini returned empty audio for '{sentence[:50]}...'")
        return None, 0

    h = gemini_clip_hash(lang, voice, sentence)
    duration = write_pcm_as_wav(pcm, rate, out_dir / f"{h}.wav")

    audio_tokens = 0
    for detail in payload.get("usageMetadata", {}).get("candidatesTokensDetails", []):
        if detail.get("modality") == "AUDIO":
            audio_tokens += int(detail.get("tokenCount", 0))

    return {
        "file": f"{h}.wav",
        "sentence": sentence,
        "source": "tts",
        # Language-scoped: a Gemini voice name is shared across languages, and
        # the speaker key (`speaker_cluster or voice`) is global.
        "voice": f"gemini:{lang}:{voice}",
        "tts_backend": "gemini",
        "tts_model": model,
        "duration_sec": round(duration, 3),
    }, audio_tokens


def synthesize_one(client, sentence, voice_name, language_code, audio_config, out_dir):
    """Synthesize a single sentence. Returns (hash, record) or None on error."""
    from google.cloud import texttospeech

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
    # Some TTS responses carry ffmpeg-style sentinel (0xffffffff) chunk sizes;
    # patch them to the real lengths so `wave`-based tooling sees the true
    # duration. No-op on well-formed files, lossless (no re-encode).
    repair_streamed_wav_header(wav_path)

    return {
        "file": f"{h}.wav",
        "sentence": sentence,
        "source": "tts",
        "voice": voice_name,
    }


def select_sentences(
    lang: str, max_sentences: int | None, offset: int, extremes: int, seed: int,
) -> tuple[list[str], random.Random]:
    """Pick this run's sentences and return the RNG that drew them.

    The returned RNG is then used for voice assignment, so the shuffle and the
    voice draws stay on one deterministic stream — re-running with the same
    seed reproduces the same (sentence, voice) pairs and therefore resumes
    cleanly. ``offset`` slices deeper into the same shuffle without disturbing
    that stream, which is how a second backend gets sentences disjoint from
    what the first one already recorded.
    """
    all_sentences = load_sentences(lang)
    rng = random.Random(seed)
    shuffled = all_sentences.copy()
    rng.shuffle(shuffled)

    window = shuffled[offset:] if offset else shuffled
    sentences = window[:max_sentences] if max_sentences else window

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

    return sentences, rng


def load_existing(manifest_path: Path) -> set[str]:
    """Filename stems already recorded, for resume."""
    existing = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                existing.add(json.loads(line)["file"].removesuffix(".wav"))
    return existing


def run_pool(todo, work, manifest_path, lang, workers, rps, on_result=None):
    """Run `work(*item)` over `todo` under a global rate limit, appending records.

    `work` returns a manifest record (or None to drop the clip). Records are
    written and flushed as they land so a killed run resumes from where it got
    to rather than losing the batch.
    """
    manifest_lock = threading.Lock()
    # Token bucket rate limiter to stay under quota
    rate_lock = threading.Lock()
    last_request_time = [0.0]  # mutable container for closure

    def throttled(*args):
        # Enforce minimum interval between requests globally
        with rate_lock:
            now = time.monotonic()
            min_interval = 1.0 / rps
            wait = last_request_time[0] + min_interval - now
            if wait > 0:
                time.sleep(wait)
            last_request_time[0] = time.monotonic()
        return work(*args)

    written = 0
    with open(manifest_path, "a") as manifest:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(throttled, *item) for item in todo]
            for future in tqdm(as_completed(futures), total=len(futures), desc=lang):
                result = future.result()
                if on_result is not None:
                    result = on_result(result)
                if result is None:
                    continue
                result["lang"] = lang
                with manifest_lock:
                    manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                    manifest.flush()
                    written += 1
    return written


def generate_chirp3(
    lang: str,
    max_sentences: int | None,
    output_root: Path,
    seed: int,
    workers: int,
    rps: float,
    extremes: int,
    offset: int,
):
    from google.cloud import texttospeech

    client = make_client()
    language_code = LANG_CONFIG[lang]
    voices = get_chirp3_voices(client, language_code)
    if not voices:
        print(f"No Chirp3-HD voices found for {language_code}, skipping {lang}")
        return

    print(f"{lang}: found {len(voices)} Chirp3-HD voices, using {workers} workers")

    sentences, rng = select_sentences(lang, max_sentences, offset, extremes, seed)
    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = load_existing(manifest_path)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SAMPLE_RATE,
    )

    # Filter to only sentences we still need.
    #
    # `queued` guards against a duplicated sentence in the pool: the filename is
    # the hash of the sentence, so synthesizing it twice writes one wav but two
    # manifest rows, and preprocess rejects the manifest outright. (cleaned_tha
    # has 35 such pairs.) Dedupe here rather than in select_sentences so the RNG
    # stream — and therefore every already-generated filename — is untouched.
    todo = []
    queued = set()
    for sentence in sentences:
        h = sentence_hash(sentence)
        voice_name = rng.choice(voices)  # drawn either way, to keep determinism
        if h not in existing and h not in queued:
            queued.add(h)
            todo.append((client, sentence, voice_name, language_code, audio_config, out_dir))

    if not todo:
        print(f"{lang}: all sentences already generated, skipping")
        return

    chars = sum(len(item[1]) for item in todo)
    print(f"{lang}: {len(todo)} sentences to generate ({len(existing)} already done), "
          f"{chars:,} chars ≈ ${chars / 1e6 * 30:.2f} at Chirp3-HD list price")

    run_pool(todo, synthesize_one, manifest_path, lang, workers, rps)


def generate_gemini(
    lang: str,
    max_sentences: int | None,
    output_root: Path,
    seed: int,
    workers: int,
    rps: float,
    extremes: int,
    offset: int,
    model: str,
    style: str,
    price_per_mtok: float,
):
    api_key = read_env_key("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (checked env and ../.env)")

    sentences, rng = select_sentences(lang, max_sentences, offset, extremes, seed)
    out_dir = output_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = load_existing(manifest_path)

    # `queued` dedupes a sentence that appears twice in the pool — see the note
    # in generate_chirp3. A Gemini filename also folds in the voice, so a repeat
    # would usually get a different one, but two draws can coincide.
    todo = []
    queued = set()
    for sentence in sentences:
        voice = rng.choice(GEMINI_VOICES)  # drawn either way, to keep determinism
        h = gemini_clip_hash(lang, voice, sentence)
        if h not in existing and h not in queued:
            queued.add(h)
            todo.append((api_key, sentence, voice, lang, model, style, out_dir))

    if not todo:
        print(f"{lang}: all Gemini clips already generated, skipping")
        return

    print(f"{lang}: {len(todo)} sentences via {model} "
          f"({len(GEMINI_VOICES)} voices, {workers} workers, {rps} rps)")

    tokens = [0]
    dropped = [0]

    def account(result):
        if result is None:
            dropped[0] += 1
            return None
        record, audio_tokens = result
        if record is None:
            dropped[0] += 1
            return None
        tokens[0] += audio_tokens
        return record

    written = run_pool(
        todo, synthesize_one_gemini, manifest_path, lang, workers, rps,
        on_result=account,
    )
    print(f"{lang}: wrote {written}, dropped {dropped[0]}, "
          f"{tokens[0]:,} audio tokens ≈ ${tokens[0] / 1e6 * price_per_mtok:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS dataset from tagging sentences")
    parser.add_argument("--backend", choices=["chirp3", "gemini"], default="chirp3",
                        help="Which TTS service to synthesize with")
    parser.add_argument("--langs", nargs="+", default=list(LANG_CONFIG.keys()),
                        choices=list(LANG_CONFIG.keys()),
                        help="Languages to generate (default: all)")
    parser.add_argument("--max-sentences", type=int, default=5000,
                        help="Max sentences per language. The default keeps a "
                             "4-language run around $30 at Chirp3-HD list "
                             "price and near source-parity with the core "
                             "languages' tts split; pass 0 for all.")
    parser.add_argument("--sentence-offset", type=int, default=0,
                        help="Skip this many sentences of the shuffle before "
                             "taking --max-sentences. Use it to give a second "
                             "backend text the first one has not already "
                             "recorded (same seed => same shuffle order).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "audio",
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--rps", type=float, default=5, help="Max requests per second")
    parser.add_argument("--extremes", type=int, default=0,
                        help="Also include N longest and N shortest sentences per language")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL,
                        help="Gemini TTS model id")
    parser.add_argument("--gemini-style", default=GEMINI_STYLE,
                        help="Style/direction prefix prepended to each sentence")
    parser.add_argument("--gemini-price-per-mtok", type=float, default=20.0,
                        help="$ per 1M output audio tokens, for the cost readout only "
                             "(list price as of 2026-08; audio bills at ~25 tok/s)")
    args = parser.parse_args()

    for lang in args.langs:
        if args.backend == "chirp3":
            generate_chirp3(
                lang, args.max_sentences, args.output, args.seed,
                args.workers, args.rps, args.extremes, args.sentence_offset,
            )
        else:
            generate_gemini(
                lang, args.max_sentences, args.output, args.seed,
                args.workers, args.rps, args.extremes, args.sentence_offset,
                args.gemini_model, args.gemini_style, args.gemini_price_per_mtok,
            )


if __name__ == "__main__":
    main()
