"""Training set from generated clip sets (one per voice): tokens, acoustic features, durations.

    VOCA_LANGUAGE=deu python -m voca.build --set data/deu_gemini_achernar data/deu_gemini_kore ... \
        --output features/deu_gemini_multi_v1 --workers 22

The clips are in the run's language (``VOCA_LANGUAGE``); each clip's speaker is its
manifest row's ``voice``, lowercased, and its ID is ``<clip set>/<clip>``: a sentence
several voices read is one row per voice.

Per clip:
1. Tokens and their features (stress, tone, accent, part of speech, rarity, boundary type) from the clip's pack
   grams (``voca.text.tokenize``).
2. Acoustic features (``voca.prepare.extract``): waveform, log-mel, pYIN pitch
   and voicing, a Praat pitch track to check it against, energy, on the
   256-sample frame grid.
3. Durations: CTC Viterbi forced alignment of the phone sequence against the
   lexide phonetizer's frame matrix (fetched by ``scripts/fetch_frames.py``),
   turned into per-token frame counts by ``voca.align.assemble``.

Writes ``<output>/{train,validation}.jsonl`` (rows ``voca.data.AnnotatedDataset``
reads), ``rejected.jsonl``, ``annotations.json`` and ``features/<clip set>/<clip>.pt``.

``--reuse`` names earlier builds whose acoustic features (step 2, the slow one) a clip may take instead of
recomputing them: the same clip set and clip, built with the same acoustic settings. Relabelling (new grams,
new alignment settings) then costs only the alignment.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import unicodedata
import zlib
import numpy as np
import soundfile as sf
import torch

from .align import AlignmentSettings, align_utterance
from .config import Config
from .prepare import extract
from .text import LANGUAGE, PHONE_SET, UNK, frontend_fingerprint, lost_punctuation, spells, tokenize

# An alignment this unsure means the clip does not say what the text says
# (a misread, a skipped word): leave it out rather than teach it.
MIN_TOKEN_PROBABILITY = 0.3
MAX_WEAK_PHONE_FRACTION = 0.15
# The recognizer, left to itself, hearing this many times as many phones as the script says: the sentence read
# twice, or extra words (the forced alignment hides them inside one phone or the edge).
MAX_HEARD_PHONES = 1.3
# Shorter clips are too short for the waveform critics' windows (eng v10 crashed at step 350 on 35-frame words).
MIN_FRAMES = 80


class FrameMatrixRecognizer:
    """``align_utterance``'s recogniser, backed by a fetched frame matrix."""

    def __init__(self, record):
        head = record["phone"]
        frames, width = head["shape"]
        values = np.frombuffer(zlib.decompress(base64.b64decode(head["data"])), dtype="<f2").reshape(frames, width)
        self.matrix = np.maximum(values.astype(np.float32), -1e4)   # masked columns are -inf
        self.index = {label: i for i, label in enumerate(head["labels"])}
        # The phonetizer's labels spell some phones several ways, from its different languages and from before
        # today's g2p: decomposed (Portuguese ẽ), tied affricates (Hindi d͡ʒʱ, beside English dʒ), glides in
        # Mandarin's diphthongs (ai̯ for aɪ). Whichever it emits, the phone was heard: their probabilities add up.
        self.spelled = {}
        for label, i in self.index.items():
            self.spelled.setdefault(spelling(label), []).append(i)
        for members in self.spelled.values():
            if len(members) > 1:
                self.matrix[:, members] = np.logaddexp.reduce(self.matrix[:, members], axis=1, keepdims=True)
        self.blank = int(head["blank_id"])

    def label(self, symbol):
        for candidate in (symbol, symbol[:-1] if symbol.endswith("ː") else None):  # geminates: heard as their base phone
            if candidate in self.index:
                return self.index[candidate]
            if candidate and spelling(candidate) in self.spelled:
                return self.spelled[spelling(candidate)][0]
        raise KeyError(f"the phonetizer has no label for {symbol!r}")

    def log_probs(self, wave, sample_rate):
        return self.matrix

    def heard(self):
        """How many phones the recognizer hears without the script (likely ones, runs collapsed)."""
        best = self.matrix.argmax(1)
        starts = (best != self.blank) & np.r_[True, best[1:] != best[:-1]]
        return int((np.exp(self.matrix[np.arange(len(best)), best]) > 0.5)[starts].sum())


def spelling(phone):
    """A phone's letters without spelling choices: decomposed, without tie bars, glides of diphthongs as vowels."""
    return unicodedata.normalize("NFD", phone).replace("͡", "").replace("i̯", "ɪ").replace("u̯", "ʊ")


def build_one(job):
    row, set_dir, output, config, reuse = job
    c = config
    try:
        if not spells(row["grams"], row["text"]):
            raise ValueError("the grams don't spell the recorded text")
        phonemized = tokenize(row["grams"])
        stored = next((p for p in (r / "features" / set_dir.name / f"{row['clip']}.pt" for r in reuse) if p.exists()), None)
        if stored:
            features = torch.load(stored, weights_only=True)
        else:
            wave, rate = sf.read(set_dir / row["wav"], dtype="float32")
            if rate != c.sample_rate or wave.ndim != 1:
                raise ValueError(f"expected mono {c.sample_rate} Hz audio, got {rate} Hz")
            features = extract(wave, c)
        recognizer = FrameMatrixRecognizer(json.loads((set_dir / "frames" / f"{row['clip']}.json").read_text()))
        durations, qc = align_utterance(recognizer, phonemized, features["wave"].numpy(), features["prosody"][:, 1].numpy(),
                                        c, AlignmentSettings())
        if qc["token_probability_mean"] < MIN_TOKEN_PROBABILITY or qc["weak_phone_fraction"] > MAX_WEAK_PHONE_FRACTION:
            raise ValueError(f"alignment too unsure: {qc['token_probability_mean']} mean, {qc['weak_phone_fraction']} weak")
        spoken = sum(s in PHONE_SET or s == UNK for s in phonemized.symbols)
        if recognizer.heard() > MAX_HEARD_PHONES * spoken:
            raise ValueError(f"more speech than the script: {recognizer.heard()} phones heard, {spoken} written")
        if features["mel"].shape[0] < MIN_FRAMES:
            raise ValueError(f"shorter than {MIN_FRAMES} frames")
        if lost_punctuation(row["grams"]):
            raise ValueError("punctuation inside a gram is lost")
    except (ValueError, KeyError, FileNotFoundError) as error:
        return {"id": row["id"], "text": row["text"], "reason": str(error)}
    path = Path("features") / set_dir.name / f"{row['clip']}.pt"
    torch.save(features, output / path)
    return {"id": row["id"], "split": row["split"], "path": str(path), "text": row["text"],
            "text_hash": hashlib.sha256(row["text"].strip().encode()).hexdigest(), "speaker": row["voice"].lower(),
            "role": row.get("role"), "tokens": list(phonemized.ids), "lexical": phonemized.lexical(),
            "durations": durations, "frames": int(features["mel"].shape[0]), "qc": qc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, nargs="+", required=True,
                        help="Clip sets (scripts/generate.py output: manifest.jsonl, wav/, frames/), e.g. one per voice")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse", type=Path, nargs="*", default=[], help="Earlier builds (outputs of this) to take acoustic features from")
    args = parser.parse_args()
    c = Config()
    roots = []
    for build in args.reuse:
        settings = {json.loads(line)["acoustic"] for split in ("train", "validation") for line in open(build / f"{split}.jsonl")}
        if settings != {c.acoustic_fingerprint}:
            raise SystemExit(f"{build} was built with other acoustic settings; it can't lend features")
        roots.append(build / json.loads((build / "annotations.json").read_text())["acoustic_root"])
    jobs = []
    for clips in args.set:
        (args.output / "features" / clips.name).mkdir(parents=True, exist_ok=True)
        for line in open(clips / "manifest.jsonl", encoding="utf-8"):
            row = json.loads(line)
            if row["language"] != LANGUAGE:
                raise SystemExit(f"{clips} is {row['language']}, this build tokenizes {LANGUAGE} (set VOCA_LANGUAGE)")
            # A sentence every voice reads has one clip per voice: the clip set names the row.
            jobs.append(({**row, "id": f"{clips.name}/{row['id']}", "clip": row["id"]}, clips, args.output, c, roots))
    frontend = frontend_fingerprint()
    annotation = hashlib.sha256(json.dumps({"rows": [r["id"] for r, *_ in jobs], "frontend": frontend, "acoustic": c.acoustic_fingerprint,
                                            "alignment": AlignmentSettings().to_dict()}, sort_keys=True).encode()).hexdigest()
    accepted, rejected = [], []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(build_one, jobs, chunksize=4)):
            (rejected if "reason" in result else accepted).append(result)
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)}: {len(rejected)} rejected", flush=True)
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
    for voice in sorted({r["speaker"] for r in accepted}):
        mine = [r for r in accepted if r["speaker"] == voice]
        print(f"  {voice}: {len(mine)} clips, {sum(r['frames'] for r in mine) * c.hop / c.sample_rate / 3600:.2f} h, "
              f"{sum(r['split'] == 'validation' for r in mine)} validation")


if __name__ == "__main__":
    main()
