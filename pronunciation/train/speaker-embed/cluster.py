"""Cluster cached speaker embeddings into pseudo-speaker labels (speaker_cluster).

The voice=null sources (FLEURS, Pimsleur) get per-clip ECAPA embeddings from
embed.py; this assigns each clip a speaker id so the narrowing analysis can group
a speaker's clips and anchor vowels within-speaker.

  cluster.py --langs spa [--sources fleurs pimsleur] [--validate-gender] [--write]

Two regimes, validated by ear:
  - FLEURS  : per-language agglomerative @ cosine 0.15 (10s clips group tightly).
  - Pimsleur: per-COURSE agglomerative @ cosine 0.45 (1-3s clips need a looser
    bar to group a speaker across varied short utterances; course scoping keeps
    speakers from merging across unrelated recordings).
We lean slightly over-segmented on purpose: merging two speakers corrupts
within-speaker normalization, while splitting one is harmless.

Silence handling: we cluster only clips present in the lang's phonemes.jsonl
(the clean training set). preprocess.py drops silent/empty recordings there, so
this automatically excludes them — without it, FLEURS-spa's 490 silent clips
collapse into one degenerate "mega-cluster" (near-constant embedding).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed as E  # noqa: E402  (cache paths + audio root)

AUDIO = E.AUDIO


def phonemes_fileset(lang: str) -> set[str] | None:
    """Files surviving preprocess (clean training set). None if not yet built —
    callers then fall back to all clips, but lose the silent-clip exclusion."""
    p = AUDIO / lang / "phonemes.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["file"] for l in p.read_text().splitlines() if l.strip()}


def load_clips(lang: str, source: str, valid: set[str] | None):
    """(files, L2-normalized X, rows) for cached clips of lang/source that are in
    `valid` (the phonemes fileset). valid=None means no filter (with a warning)."""
    mf = AUDIO / lang / "manifest.jsonl"
    files, vecs, rows = [], [], {}
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("source") != source or not E.is_cached(lang, d["file"]):
            continue
        if valid is not None and d["file"] not in valid:
            continue  # dropped by preprocess (silent/empty) — skip
        files.append(d["file"])
        vecs.append(E.load_embedding(lang, d["file"]))
        rows[d["file"]] = d
    if not files:
        return [], np.zeros((0, 192), np.float32), {}
    X = np.vstack(vecs).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # L2 -> euclidean≈cosine
    return files, X, rows


def agglom(X: np.ndarray, threshold: float) -> np.ndarray:
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(X)


def course_of(fn: str) -> str | None:
    """Pimsleur course id from a clip filename (strip the _Unit/_Lesson/seq tail)."""
    m = re.match(r"pimsleur_(.+?)(_Unit|_Lesson|_\d{4}\b)", fn)
    return m.group(1) if m else None


def gender_purity(files, labels, rows) -> tuple[float, int] | None:
    """Mean per-cluster gender purity using the manifest's per-clip gender
    (FLEURS stores it). A real speaker cluster is single-gender -> ~1.0."""
    by = defaultdict(Counter)
    for f, l in zip(files, labels):
        g = rows[f].get("gender")
        if g:
            by[l][g] += 1
    pur = [max(c.values()) / sum(c.values()) for c in by.values() if sum(c.values())]
    if not pur:
        return None
    return float(np.mean(pur)), len(pur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=["fleurs", "pimsleur"])
    ap.add_argument("--threshold", type=float, default=0.15, help="FLEURS cosine threshold")
    ap.add_argument("--pimsleur-threshold", type=float, default=0.45,
                    help="Pimsleur per-course cosine threshold")
    ap.add_argument("--validate-gender", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for lang in args.langs:
        valid = phonemes_fileset(lang)
        if valid is None:
            print(f"{lang}: WARNING — no phonemes.jsonl; clustering ALL clips "
                  f"(silent clips not excluded). Run preprocess first.")
        assignments: dict[str, str] = {}

        if "fleurs" in args.sources:
            files, X, rows = load_clips(lang, "fleurs", valid)
            if files:
                labels = agglom(X, args.threshold)
                for f, l in zip(files, labels):
                    assignments[f] = f"{lang}:fleurs:s{int(l):03d}"
                n_clusters = len(set(labels))
                msg = f"{lang} fleurs: {len(files)} clips → {n_clusters} speakers @ {args.threshold}"
                if args.validate_gender:
                    gp = gender_purity(files, labels, rows)
                    if gp:
                        msg += f"  | gender purity {gp[0]:.3f} ({gp[1]} clusters)"
                print(msg)

        if "pimsleur" in args.sources:
            files, X, rows = load_clips(lang, "pimsleur", valid)
            if files:
                by_course: dict[str, list[int]] = defaultdict(list)
                for i, f in enumerate(files):
                    by_course[course_of(f) or "_unknown"].append(i)
                total_speakers = 0
                for course, idxs in by_course.items():
                    sub = X[idxs]
                    labels = agglom(sub, args.pimsleur_threshold)
                    for j, l in zip(idxs, labels):
                        assignments[files[j]] = f"{lang}:pims:{course}:s{int(l):03d}"
                    total_speakers += len(set(labels))
                print(f"{lang} pimsleur: {len(files)} clips across {len(by_course)} courses "
                      f"→ {total_speakers} speakers @ {args.pimsleur_threshold}")

        if args.write and assignments:
            p = AUDIO / lang / "manifest.jsonl"
            out = []
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                # Rewrite speaker_cluster fresh each run (drop stale labels, e.g.
                # on clips now excluded as silent).
                d.pop("speaker_cluster", None)
                if d["file"] in assignments:
                    d["speaker_cluster"] = assignments[d["file"]]
                out.append(json.dumps(d, ensure_ascii=False))
            p.write_text("\n".join(out) + "\n")
            print(f"   wrote speaker_cluster for {len(assignments)} clips to {p}")


if __name__ == "__main__":
    main()
