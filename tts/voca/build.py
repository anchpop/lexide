"""Training set from a generated clip set: tokens, acoustic features, durations.

    python -m voca.build --set data/eng_achernar --output features/eng_achernar --workers 32

Per clip:
1. Tokens from the clip's pack grams (``voca.text.tokenize``).
2. Acoustic features (``voca.prepare.extract``): waveform, log-mel, pYIN pitch
   and voicing, energy, on the 256-sample frame grid.
3. Durations: CTC Viterbi forced alignment of the phone sequence against the
   lexide phonetizer's frame matrix (fetched by ``scripts/fetch_frames.py``),
   turned into per-token frame counts by ``voca.align.assemble``.

Writes ``<output>/{train,validation}.jsonl`` (rows ``voca.data.AnnotatedDataset``
reads), ``rejected.jsonl``, ``annotations.json`` and ``features/<id>.pt``.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
import soundfile as sf
import torch

from .align import AlignmentSettings, align_utterance
from .config import Config
from .prepare import extract
from .text import frontend_fingerprint, tokenize

# An alignment this unsure means the clip does not say what the text says
# (a misread, a skipped word): leave it out rather than teach it.
MIN_TOKEN_PROBABILITY = 0.3
MAX_WEAK_PHONE_FRACTION = 0.15


class FrameMatrixRecognizer:
    """``align_utterance``'s recogniser, backed by a fetched frame matrix."""

    def __init__(self, record):
        head = record["phone"]
        frames, width = head["shape"]
        values = np.frombuffer(zlib.decompress(base64.b64decode(head["data"])), dtype="<f2").reshape(frames, width)
        self.matrix = np.maximum(values.astype(np.float32), -1e4)   # masked columns are -inf
        self.index = {label: i for i, label in enumerate(head["labels"])}
        self.blank = int(head["blank_id"])

    def label(self, symbol):
        if symbol in self.index:
            return self.index[symbol]
        # Geminates (ðː, nː) are one phone the model hears as its base consonant.
        if symbol.endswith("ː") and symbol[:-1] in self.index:
            return self.index[symbol[:-1]]
        raise KeyError(f"the phonetizer has no label for {symbol!r}")

    def log_probs(self, wave, sample_rate):
        return self.matrix


def build_one(job):
    row, set_dir, output, config = job
    c = config
    try:
        phonemized = tokenize(row["grams"])
        wave, rate = sf.read(set_dir / row["wav"], dtype="float32")
        if rate != c.sample_rate or wave.ndim != 1:
            raise ValueError(f"expected mono {c.sample_rate} Hz audio, got {rate} Hz")
        features = extract(wave, c)
        recognizer = FrameMatrixRecognizer(json.loads((set_dir / "frames" / f"{row['id']}.json").read_text()))
        durations, qc = align_utterance(recognizer, phonemized, features["wave"].numpy(), features["prosody"][:, 1].numpy(),
                                        c, AlignmentSettings())
        if qc["token_probability_mean"] < MIN_TOKEN_PROBABILITY or qc["weak_phone_fraction"] > MAX_WEAK_PHONE_FRACTION:
            raise ValueError(f"alignment too unsure: {qc['token_probability_mean']} mean, {qc['weak_phone_fraction']} weak")
    except (ValueError, KeyError, FileNotFoundError) as error:
        return {"id": row["id"], "text": row["text"], "reason": str(error)}
    path = Path("features") / f"{row['id']}.pt"
    torch.save(features, output / path)
    return {"id": row["id"], "split": row["split"], "path": str(path), "text": row["text"],
            "text_hash": hashlib.sha256(row["text"].strip().encode()).hexdigest(), "speaker": row["speaker"],
            "tokens": list(phonemized.ids), "stress": list(phonemized.stress), "durations": durations,
            "frames": int(features["mel"].shape[0]), "qc": qc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, help="Directory with manifest.jsonl, wav/ and frames/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker", default="achernar")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    c = Config()
    (args.output / "features").mkdir(parents=True, exist_ok=True)
    rows = [{**json.loads(line), "speaker": args.speaker} for line in open(args.set / "manifest.jsonl", encoding="utf-8")]
    frontend = frontend_fingerprint()
    annotation = hashlib.sha256(json.dumps({"rows": [r["id"] for r in rows], "frontend": frontend, "acoustic": c.acoustic_fingerprint,
                                            "alignment": AlignmentSettings().to_dict()}, sort_keys=True).encode()).hexdigest()
    accepted, rejected = [], []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(build_one, [(r, args.set, args.output, c) for r in rows], chunksize=4)):
            (rejected if "reason" in result else accepted).append(result)
            if i % 100 == 0:
                print(f"{i}/{len(rows)}: {len(rejected)} rejected", flush=True)
    for split in ("train", "validation"):
        with open(args.output / f"{split}.jsonl", "w", encoding="utf-8") as stream:
            for row in accepted:
                if row["split"] == split:
                    stream.write(json.dumps({**row, "frontend": frontend, "acoustic": c.acoustic_fingerprint,
                                             "annotation": annotation}, ensure_ascii=False) + "\n")
    with open(args.output / "rejected.jsonl", "w", encoding="utf-8") as stream:
        for row in rejected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "annotations.json").write_text(json.dumps({"fingerprint": annotation, "complete": True,
                                                              "acoustic_root": "."}, indent=1))
    hours = sum(r["frames"] for r in accepted) * c.hop / c.sample_rate / 3600
    print(f"{len(accepted)} accepted ({hours:.2f} h), {len(rejected)} rejected")


if __name__ == "__main__":
    main()
