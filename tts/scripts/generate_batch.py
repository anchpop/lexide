"""Gemini TTS for whole scripts through the Batch API: half of generate.py's price, the same model, voice and direction.

    set -a; . /data/coding/yap/.env; set +a
    python scripts/generate_batch.py hin                    # submit what isn't recorded yet, wait, write the WAVs
    python scripts/language_data.py hin --stages generate   # the few the batch missed; writes the manifests

Every data/scripts/<language>_<voice>.jsonl plus data/scripts/<language>_words.jsonl (Achernar, into
data/<language>_gemini_achernar_words). Rerunning resumes: the submitted batches are listed in
data/<language>_batches.jsonl, clips already on disk are skipped. (Batch takes generateContent requests; the
Interactions API, which generate.py and yap use, has no batch.)
"""
import argparse, base64, io, json, os, pathlib, time, wave
import requests

from generate import LANGUAGES
from language_data import VOICES, clips, script

API = "https://generativelanguage.googleapis.com"
MODEL = "models/gemini-3.8-flash-tts"
session = requests.Session()
session.headers["x-goog-api-key"] = os.environ["GEMINI_API_KEY"]


def call(method, url, **kwargs):
    for attempt in range(10):
        try:
            r = session.request(method, url, timeout=600, **kwargs)
            if r.status_code < 500 and r.status_code != 429:
                break
        except requests.RequestException:
            r = None
        time.sleep(min(300, 5 * 2 ** attempt))
    if r is None or r.status_code >= 400:      # not raise_for_status: its message can carry the key
        raise RuntimeError("no reply" if r is None else f"HTTP {r.status_code} {r.text[:400]}")
    return r


def jobs(language, part):
    """(set directory, voice, row) of every clip. ``part``: a later addition ("more": data/scripts/eng_more_achernar.jsonl,
    data/eng_gemini_more_achernar, ...)."""
    if part:
        sets = [(script(language, f"{part}_{v}"), clips(language, f"{part}_{v}"), v) for v in VOICES]
        sets.append((script(language, f"{part}_words"), clips(language, f"{part}_words"), "Achernar"))
    else:
        sets = [(script(language, v), clips(language, v), v) for v in VOICES]
        sets.append((script(language, "words"), clips(language, "achernar_words"), "Achernar"))
    return [(out, voice, json.loads(line)) for path, out, voice in sets if path.exists() for line in open(path, encoding="utf-8")]


def request(language, voice, text):
    locale, style = LANGUAGES[language]
    return {"contents": [{"role": "user", "parts": [{"text": text, "speechMetadata": {"style": style}}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "responseFormat": {"audio": {"mimeType": "AUDIO_L16", "sampleRate": 24000}},
                                 "speechConfig": {"languageCode": locale, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}


def submit(language, chunk, name):
    lines = "\n".join(json.dumps({"key": f"{out.name}/{row['id']}", "request": request(language, voice, row["text"])}, ensure_ascii=False)
                      for out, voice, row in chunk).encode()
    start = call("POST", f"{API}/upload/v1beta/files", json={"file": {"display_name": name}},
                 headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                          "X-Goog-Upload-Header-Content-Length": str(len(lines)), "X-Goog-Upload-Header-Content-Type": "application/jsonl"})
    file = call("POST", start.headers["x-goog-upload-url"], data=lines,
                headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"}).json()["file"]["name"]
    reply = call("POST", f"{API}/v1beta/{MODEL}:batchGenerateContent",
                 json={"batch": {"display_name": name, "input_config": {"file_name": file}}}).json()
    return reply["name"]


def save(key, response):
    parts = [p["inlineData"] for c in response.get("candidates", []) for p in c.get("content", {}).get("parts", []) if "inlineData" in p]
    pcm = b"".join(base64.b64decode(p["data"]) for p in parts)
    if not pcm:
        return False
    assert all(p["mimeType"].lower().startswith("audio/l16") and "24000" in p["mimeType"] for p in parts), [p["mimeType"] for p in parts]
    out, clip = key.split("/")
    path = pathlib.Path("data") / out / "wav" / f"{clip}.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(pcm)
    if path.exists():
        return True
    path.with_suffix(".tmp").write_bytes(buffer.getvalue())
    path.with_suffix(".tmp").rename(path)
    return True


def collect(name):
    """Saves a finished batch's audio; False while it runs."""
    meta = call("GET", f"{API}/v1beta/{name}").json()
    state = meta.get("metadata", meta).get("state", "")
    if state.endswith(("PENDING", "RUNNING")):
        return False
    batch = meta.get("metadata", meta)
    file = (batch.get("output") or meta.get("response") or {}).get("responsesFile")
    if not file:
        print(name, state, json.dumps(meta.get("error") or {})[:300], flush=True)
        return True
    saved, failed = 0, []
    with call("GET", f"{API}/download/v1beta/{file}:download?alt=media", stream=True) as reply:
        for line in reply.iter_lines(chunk_size=1 << 20):
            row = json.loads(line)
            if save(row["key"], row.get("response") or {}):
                saved += 1
            else:
                failed.append(json.dumps(row.get("error") or row.get("response"))[:150])
    print(name, state, saved, "saved", len(failed), "failed", failed[:3], flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("language", choices=sorted(LANGUAGES))
    parser.add_argument("--part", help="an addition to the language's first scripts (see jobs)")
    parser.add_argument("--size", type=int, default=1000, help="requests per batch (its answer is ~300 kB per clip)")
    parser.add_argument("--limit", type=int, help="clips at most (a test)")
    args = parser.parse_args()
    ledger = pathlib.Path(f"data/{args.language}_batches.jsonl")
    submitted = [json.loads(x) for x in open(ledger)] if ledger.exists() else []
    queued = {k for b in submitted for k in b["keys"]}
    todo = []
    for out, voice, row in jobs(args.language, args.part):
        (out / "wav").mkdir(parents=True, exist_ok=True)
        if f"{out.name}/{row['id']}" not in queued and not (out / "wav" / f"{row['id']}.wav").exists():
            todo.append((out, voice, row))
    todo = todo[: args.limit]
    for i in range(0, len(todo), args.size):
        chunk = todo[i:i + args.size]
        name = submit(args.language, chunk, f"{args.language}-{len(submitted)}")
        submitted.append({"name": name, "keys": [f"{o.name}/{r['id']}" for o, _, r in chunk]})
        with open(ledger, "a") as fh:
            fh.write(json.dumps(submitted[-1]) + "\n")
        print("submitted", name, len(chunk), flush=True)
    waiting = [b["name"] for b in submitted if any(not (pathlib.Path("data") / k.split("/")[0] / "wav" / (k.split("/")[1] + ".wav")).exists() for k in b["keys"])]
    while waiting := [n for n in waiting if not collect(n)]:
        time.sleep(120)
    print("done", flush=True)
