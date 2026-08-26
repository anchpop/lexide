#!/usr/bin/env python3
"""Perceptual audio<->label spot-audit via Gemini multimodal listening.

Stratified-samples clips per (lang, source) from the manifests, sends each
wav + its claimed sentence to Gemini, and asks whether the audio actually
says that sentence in that language. Catches wrong-audio pairings, heavy
truncation, wrong-language clips, TTS refusals/paraphrases — the failure
modes a text-only pipeline can't see.

Run:  scripts/py-linux.sh scripts/audit_gemini_listen.py [--per-cell 3] [--model ...]
Writes verdicts to .work/gemini_listen_audit.jsonl (resumable: skips files
already present in the output).
"""

import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "data" / "audio"
OUT = REPO / ".work" / "gemini_listen_audit.jsonl"

LANG_NAMES = {
    "ara": "Arabic", "ces": "Czech", "dan": "Danish", "deu": "German",
    "eng": "English", "fas": "Persian", "fra": "French", "hin": "Hindi",
    "ita": "Italian", "jpn": "Japanese", "por": "Portuguese",
    "rus": "Russian", "spa": "Spanish", "tha": "Thai",
    "zho-hans": "Mandarin Chinese",
}

PROMPT = """You are auditing a speech dataset. The attached audio clip is \
claimed to contain this sentence, spoken in {lang_name}:

  "{sentence}"

Listen carefully and answer in strict JSON (no markdown):
{{
  "verdict": "match" | "partial" | "mismatch" | "empty_or_noise",
  "spoken_language": "<language you actually hear, or null>",
  "heard": "<your best transcription of what is actually said>",
  "issues": ["<zero or more of: truncated_start, truncated_end, extra_speech, \
wrong_language, background_speech, heavy_noise, synthesis_artifact, \
silence, paraphrased, mispronunciation>"],
  "note": "<one short sentence, only if something is off>"
}}

"match" = the sentence is spoken essentially verbatim (minor accent/prosody \
variation is fine). "partial" = clearly the same text but words missing, \
added, or altered. "mismatch" = different text entirely. Judge the TEXT, \
not the audio quality, except where quality makes words unrecoverable."""


def call_gemini(model: str, key: str, wav: Path, sentence: str, lang: str):
    audio_b64 = base64.b64encode(wav.read_bytes()).decode()
    body = {
        "contents": [{"parts": [
            {"text": PROMPT.format(lang_name=LANG_NAMES.get(lang, lang),
                                   sentence=sentence)},
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
        ]}],
        "generationConfig": {"temperature": 0.0,
                             "response_mime_type": "application/json"},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:  # noqa: BLE001 — retry then surface
            if attempt == 3:
                return {"verdict": "error", "note": str(e)[:200]}
            time.sleep(5 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3,
                    help="clips per (lang, source) cell")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--langs", default=None,
                    help="comma-separated subset of langs")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        print("no GEMINI_API_KEY", file=sys.stderr)
        return 1

    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["lang"], r["file"]))

    rng = random.Random(args.seed)
    langs = args.langs.split(",") if args.langs else sorted(LANG_NAMES)
    cells = {}
    for lang in langs:
        man = AUDIO / lang / "manifest.jsonl"
        if not man.exists():
            continue
        for line in man.open():
            r = json.loads(line)
            src = r.get("source") or "?"
            if src == "tts" and r.get("tts_backend"):
                src = f"tts-{r['tts_backend']}"
            cells.setdefault((lang, src), []).append(r)

    tasks = []
    for (lang, src), rows in sorted(cells.items()):
        for r in rng.sample(rows, min(args.per_cell, len(rows))):
            if (lang, r["file"]) not in done:
                tasks.append((lang, src, r))
    print(f"{len(tasks)} clips to audit "
          f"({len(cells)} cells x {args.per_cell})")

    with OUT.open("a") as out:
        for i, (lang, src, r) in enumerate(tasks):
            wav = AUDIO / lang / r["file"]
            if not wav.exists():
                verdict = {"verdict": "error", "note": "file missing on disk"}
            else:
                verdict = call_gemini(args.model, key, wav,
                                      r["sentence"], lang)
            rec = {"lang": lang, "source": src, "file": r["file"],
                   "sentence": r["sentence"], **verdict}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            flag = "" if verdict.get("verdict") == "match" else "  <-- " + \
                str(verdict.get("verdict"))
            print(f"[{i+1}/{len(tasks)}] {lang}/{src} {r['file']}{flag}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
