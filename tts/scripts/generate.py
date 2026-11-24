"""Synthesize a script as 24 kHz mono WAV with a Google voice.

    # Gemini TTS, called the way yap's app calls it (GEMINI_API_KEY)
    python generate.py data/scripts/deu_kore.jsonl data/deu_gemini_kore --provider gemini --language deu --voice Kore
    # Cloud TTS (GOOGLE_CLOUD_API_KEY)
    python generate.py data/eng_script.jsonl data/eng_achernar --provider cloud --language eng --voice en-US-Chirp3-HD-Achernar

``--language`` is a yap language code. It picks the locale and, for Gemini, the
delivery direction yap's app gives (``Language::tts_locale`` and
``Language::tts_direction``: "warm and welcoming" written in the language
itself, since an English direction pulls Gemini toward an English reading,
plus the accent where yap names one).

Resumable: clips already on disk are skipped. Writes <out>/manifest.jsonl.
"""
import argparse, base64, concurrent.futures, io, json, os, pathlib, time, wave
import requests

# yap language: (locale, Gemini style direction), from yap's language-utils.
LANGUAGES = {
    "eng": ("en-US", "warm and welcoming"),
    "deu": ("de-DE", "warm und einladend"),
    "fra": ("fr-FR", "chaleureux et accueillant"),
    "hin": ("hi-IN", "गर्मजोशी और अपनेपन से"),
    "ita": ("it-IT", "caloroso e accogliente"),
    "jpn": ("ja-JP", "温かく親しみやすく"),
    "kor": ("ko-KR", "따뜻하고 다정하게"),
    "por": ("pt-BR", "caloroso e acolhedor, com sotaque brasileiro"),
    "por-pt": ("pt-PT", "caloroso e acolhedor, com sotaque de Portugal"),
    "rus": ("ru-RU", "тепло и приветливо"),
    "spa": ("es-US", "cálido y acogedor, con acento latinoamericano"),
    "spa-es": ("es-ES", "cálido y acogedor, con acento de España"),
    "tha": ("th-TH", "อบอุ่นและเป็นกันเอง"),
    "zho-hans": ("cmn-CN", "温暖亲切"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("out")
    parser.add_argument("--provider", choices=("gemini", "cloud"), required=True)
    parser.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    locale, style = LANGUAGES[args.language]
    style = style if args.provider == "gemini" else ""

    out = pathlib.Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.script, encoding="utf-8")][: args.limit]


    def post(url, body, **kwargs):
        for attempt in range(12):
            try:
                r = requests.post(url, json=body, timeout=180, **kwargs)
            except requests.RequestException:     # reset or timed-out connections, also mid-reply
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code not in (429, 500, 503):
                break
            time.sleep(min(60, 2 ** attempt))
        if r.status_code != 200:  # not raise_for_status: its message can carry the key
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:300]}")
        return r.json()


    def cloud(text):
        body = {"input": {"text": text}, "voice": {"languageCode": locale, "name": args.voice},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}
        reply = post("https://texttospeech.googleapis.com/v1/text:synthesize", body,
                     params={"key": os.environ["GOOGLE_CLOUD_API_KEY"]})
        return base64.b64decode(reply["audioContent"])  # LINEAR16 comes as a WAV file


    def gemini(text):
        """Same request as yap's google-speech GeminiTtsRequest. Raw PCM only: the
        model's default once became WAV with a trailing C2PA chunk."""
        content = {"type": "text", "text": text, "annotations": [{"type": "speech_metadata", "style": style}]}
        body = {"model": "gemini-3.8-flash-tts", "input": [{"type": "user_input", "content": [content]}],
                "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
                "generation_config": {"speech_config": [{"voice": args.voice, "language": locale}]}}
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
                if "prohibited_content" in str(error):   # Gemini refuses this sentence: leave it out of the set
                    print("blocked", row["id"], flush=True)
                    return None
                raise RuntimeError(f"{row['id']}: {error}") from None
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(audio)
            tmp.rename(path)
        return {**row, "wav": f"wav/{row['id']}.wav", "language": args.language, "voice": args.voice, "provider": args.provider,
                "locale": locale, "style": style}


    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, open(out / "manifest.jsonl", "w", encoding="utf-8") as manifest:
        for i, row in enumerate(pool.map(synthesize, rows)):
            if row:
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(i, flush=True)
    print("done", len(rows))
