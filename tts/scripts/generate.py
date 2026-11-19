"""Synthesize a script as 24 kHz mono WAV with a Google voice.

    # Gemini TTS, called the way yap's app calls it (GEMINI_API_KEY)
    python generate.py data/eng_script.jsonl data/eng_gemini_achernar --provider gemini \
        --language-code en-US --voice Achernar --style "warm and welcoming"
    # Cloud TTS (GOOGLE_CLOUD_API_KEY)
    python generate.py data/eng_script.jsonl data/eng_achernar --provider cloud \
        --language-code en-US --voice en-US-Chirp3-HD-Achernar

Resumable: clips already on disk are skipped. Writes <out>/manifest.jsonl.
"""
import argparse, base64, concurrent.futures, io, json, os, pathlib, time, wave
import requests

parser = argparse.ArgumentParser()
parser.add_argument("script")
parser.add_argument("out")
parser.add_argument("--provider", choices=("gemini", "cloud"), required=True)
parser.add_argument("--language-code", required=True)
parser.add_argument("--voice", required=True)
parser.add_argument("--style", default="", help="Gemini delivery direction (yap: Language::tts_direction)")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--limit", type=int)
args = parser.parse_args()

out = pathlib.Path(args.out)
(out / "wav").mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(args.script)][: args.limit]


def post(url, body, **kwargs):
    for attempt in range(12):
        r = requests.post(url, json=body, timeout=180, **kwargs)
        if r.status_code not in (429, 500, 503):
            break
        time.sleep(min(60, 2 ** attempt))
    if r.status_code != 200:  # not raise_for_status: its message can carry the key
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:300]}")
    return r.json()


def cloud(text):
    body = {"input": {"text": text}, "voice": {"languageCode": args.language_code, "name": args.voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}
    reply = post("https://texttospeech.googleapis.com/v1/text:synthesize", body,
                 params={"key": os.environ["GOOGLE_CLOUD_API_KEY"]})
    return base64.b64decode(reply["audioContent"])  # LINEAR16 comes as a WAV file


def gemini(text):
    """Same request as yap's google-speech GeminiTtsRequest. Raw PCM only: the
    model's default once became WAV with a trailing C2PA chunk."""
    content = {"type": "text", "text": text}
    if args.style:
        content["annotations"] = [{"type": "speech_metadata", "style": args.style}]
    body = {"model": "gemini-3.8-flash-tts", "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
            "generation_config": {"speech_config": [{"voice": args.voice, "language": args.language_code}]}}
    for _ in range(4):   # the API sometimes answers 200 with no audio
        reply = post("https://generativelanguage.googleapis.com/v1beta/interactions", body,
                     headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        parts = [c for s in reply.get("steps", []) for c in s.get("content", []) if c.get("type") == "audio"]
        assert all(p["mime_type"].startswith("audio/l16") and p["sample_rate"] == 24000 for p in parts), parts
        pcm = b"".join(base64.b64decode(p["data"]) for p in parts)
        if pcm:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(pcm)
            return buffer.getvalue()
    raise RuntimeError("no audio in four replies")


def synthesize(row):
    path = out / "wav" / f"{row['id']}.wav"
    if not path.exists():
        try:
            audio = (gemini if args.provider == "gemini" else cloud)(row["text"])
        except RuntimeError as error:
            raise RuntimeError(f"{row['id']}: {error}") from None
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(audio)
        tmp.rename(path)
    return {**row, "wav": f"wav/{row['id']}.wav", "voice": args.voice, "provider": args.provider, "style": args.style}


with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, open(out / "manifest.jsonl", "w") as manifest:
    for i, row in enumerate(pool.map(synthesize, rows)):
        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 100 == 0:
            print(i, flush=True)
print("done", len(rows))
