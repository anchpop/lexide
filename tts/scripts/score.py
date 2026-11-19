"""Score a directory of synthesized clips (voca.speak output) two ways:

* Whisper (Groq whisper-large-v3) word error rate against the sentence text.
* The lexide phonetizer: per-clip CTC log-likelihood of the intended phone
  sequence (``target_score``) and greedy phone error rate, via the batch
  endpoint.

    GROQ_API_KEY=... bash ../pronunciation/scripts/py-linux.sh scripts/score.py runs/x/samples-validation --language eng
"""
import argparse, base64, io, json, pathlib, re
import numpy as np
import requests

from voca.text import bare_phones
import soundfile as sf

BATCH = "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict-batch.modal.run"
parser = argparse.ArgumentParser()
parser.add_argument("clips", type=pathlib.Path)
parser.add_argument("--language", required=True)
parser.add_argument("--whisper-language", default="en")
args = parser.parse_args()

import os
rows = [json.loads(l) for l in open(args.clips / "manifest.jsonl", encoding="utf-8")]


def words(text):
    return re.sub(r"[^\w' ]+", " ", text.lower()).split()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def whisper(path):
    response = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                             headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                             files={"file": (path.name, path.read_bytes(), "audio/wav")},
                             data={"model": "whisper-large-v3", "language": args.whisper_language, "temperature": "0"}, timeout=120)
    response.raise_for_status()
    return response.json()["text"]


targets, phone_errors, phone_total, word_errors, word_total = [], 0, 0, 0, 0
results = []
for start in range(0, len(rows), 64):
    batch = rows[start:start + 64]
    payload = []
    for row in batch:
        audio, rate = sf.read(args.clips / row["wav"], dtype="float32")
        phones = bare_phones(row["grams"])
        row["_phones"] = phones
        payload.append({"audio_f32_b64": base64.b64encode(audio.astype("<f4").tobytes()).decode(), "sample_rate": rate,
                        "language": args.language, "target_phonemes": phones})
    response = requests.post(BATCH, json={"requests": payload}, timeout=900)
    response.raise_for_status()
    for row, result in zip(batch, response.json()["results"]):
        heard = [p["phoneme"].lstrip("ˈˌ") for p in result.get("phonemes", [])]
        errors = edit_distance(row["_phones"], heard)
        phone_errors += errors
        phone_total += len(row["_phones"])
        transcript = whisper(args.clips / row["wav"])
        werr = edit_distance(words(row["text"]), words(transcript))
        word_errors += werr
        word_total += len(words(row["text"]))
        results.append({"id": row["id"], "text": row["text"], "whisper": transcript, "word_errors": werr,
                        "phone_errors": errors, "phones": len(row["_phones"]), "target_score": result.get("target_score")})
(args.clips / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
summary = {"clips": len(results), "wer": round(word_errors / word_total, 4), "per": round(phone_errors / phone_total, 4)}
(args.clips / "scores-summary.json").write_text(json.dumps(summary, indent=1))
print(summary)
for r in sorted(results, key=lambda r: -r["word_errors"])[:8]:
    if r["word_errors"]:
        print(f"  {r['text']!r} -> {r['whisper']!r}")
