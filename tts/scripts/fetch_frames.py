"""Fetch the phonetizer's frame matrix for every clip of a generated set.

Uses the production batch endpoint (64 clips per request; see
pronunciation/modal/PRONUNCIATION_BATCHING.md). Writes <set>/frames/<id>.json
holding the phone head (joint log-probabilities, fp16, zlib+base64) and the
endpoint's identity. Resumable.

    bash ../pronunciation/scripts/py-linux.sh scripts/fetch_frames.py data/eng_achernar --language eng
"""
import argparse, base64, concurrent.futures, json, pathlib
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import soundfile as sf

URL = "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict-batch.modal.run"
parser = argparse.ArgumentParser()
parser.add_argument("set", type=pathlib.Path)
parser.add_argument("--language", required=True)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--parallel", type=int, default=4)
args = parser.parse_args()

out = args.set / "frames"
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(args.set / "manifest.jsonl")]
todo = [r for r in rows if not (out / f"{r['id']}.json").exists()]
print(f"{len(rows) - len(todo)} cached, {len(todo)} to fetch")


session = requests.Session()   # the endpoint sometimes answers a 502 while it scales
session.mount("https://", HTTPAdapter(max_retries=Retry(6, backoff_factor=5, status_forcelist=(500, 502, 503, 504), allowed_methods=None)))


def fetch(batch):
    requests_ = []
    for row in batch:
        audio, rate = sf.read(args.set / row["wav"], dtype="float32")
        requests_.append({"audio_f32_b64": base64.b64encode(audio.astype("<f4").tobytes()).decode(), "sample_rate": rate,
                          "language": args.language, "return_frame_matrix": True})
    response = session.post(URL, json={"requests": requests_}, timeout=900)
    response.raise_for_status()
    body = response.json()
    identity = {k: body[k] for k in ("deploy_marker", "model_id", "model_revision", "decoder_version")}
    for row, result in zip(batch, body["results"]):
        if "error" in result:
            print(row["id"], result["error"])
            continue
        matrix = result["frame_matrix"]
        record = {"identity": identity, "trained_against_g2p": matrix["trained_against_g2p"],
                  "frame_rate_ms": matrix["frame_rate_ms"], "sample_rate": matrix["sample_rate"], "phone": matrix["heads"]["phone"]}
        (out / f"{row['id']}.json").write_text(json.dumps(record, ensure_ascii=False))
    return len(batch)


batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
done = 0
with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
    for n in pool.map(fetch, batches):
        done += n
        print(f"{done}/{len(todo)}", flush=True)
