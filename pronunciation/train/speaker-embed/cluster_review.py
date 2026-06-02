"""Cluster + pull audio from random clusters for human (ear) validation.

For each sampled cluster, copies a few clips (different recordings → the same
voice should be heard saying different things) into a review dir so you can judge
"do these all sound like one person?". Clusters in memory; does NOT write
manifests (validate first). FLEURS = per-language agglomerative (+gender check);
Pimsleur = per-course agglomerative (bounded n, scalable).
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed as E  # noqa: E402

THR = 0.15
REVIEW = HERE.parent.parent / "espeak_audit" / "out" / "cluster_review"
random.seed(0)


def cluster(X: np.ndarray, thr: float = THR) -> np.ndarray:
    # FLEURS (10s clips) groups tightly at ~0.15; Pimsleur (1-3s clips) needs a
    # looser threshold (~0.45) to group a speaker across varied short utterances
    # rather than just near-duplicate drill repeats.
    return AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                   metric="cosine", linkage="average").fit_predict(X)


def load(lang, predicate):
    """Return (files, normalized X) for cached clips matching predicate(rec)."""
    files, vecs = [], []
    for line in (E.AUDIO / lang / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not predicate(d) or not E.is_cached(lang, d["file"]):
            continue
        files.append(d["file"]); vecs.append(E.load_embedding(lang, d["file"]))
    if not files:
        return [], np.zeros((0, 192))
    X = np.vstack(vecs).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return files, X


def sample(tag, lang, files, labels, n_clusters, clips_per, manifest_idx, min_size):
    by = defaultdict(list)
    for f, l in zip(files, labels):
        by[l].append(f)
    # Sample from clusters big enough to be a real speaker grouping (across
    # varied utterances); pull a few random clips out of each (no upper cap).
    cands = [c for c, fs in by.items() if len(fs) >= min_size]
    random.shuffle(cands)
    rows = []
    for ci, c in enumerate(cands[:n_clusters]):
        letter = chr(ord("A") + ci)
        picks = random.sample(by[c], clips_per)
        for i, f in enumerate(picks, 1):
            dest = REVIEW / f"{tag}_clusterX{letter}_{i}.wav"
            shutil.copy(E.AUDIO / lang / f, dest)
            rec = manifest_idx.get(f, {})
            rows.append({"tag": tag, "cluster": f"{letter}", "n_in_cluster": len(by[c]),
                         "clip": dest.name, "gender": rec.get("gender"),
                         "sentence": rec.get("sentence", "")[:55]})
    return rows, len(by)


def manifest_index(lang):
    return {json.loads(l)["file"]: json.loads(l)
            for l in (E.AUDIO / lang / "manifest.jsonl").read_text().splitlines() if l.strip()}


def main():
    if REVIEW.exists():
        shutil.rmtree(REVIEW)
    REVIEW.mkdir(parents=True)
    all_rows = []

    # 1) FLEURS Spanish — per-language, gender-checkable
    lang = "spa"
    idx = manifest_index(lang)
    files, X = load(lang, lambda d: d.get("source") == "fleurs")
    labels = cluster(X)
    rows, nclu = sample("fleurs-spa", lang, files, labels, n_clusters=3, clips_per=3, manifest_idx=idx, min_size=4)
    print(f"FLEURS spa: {len(files)} clips → {nclu} clusters; sampled 3")
    all_rows += rows

    # 2) Pimsleur — pick one moderate-size course, cluster within it
    import re
    def course_of(fn):
        m = re.match(r"pimsleur_(.+?)(_Unit|_Lesson|_\d{4}\b)", fn)
        return m.group(1) if m else None
    plang = "eng"
    pidx = manifest_index(plang)
    courses = defaultdict(int)
    for f, d in pidx.items():
        if d.get("source") == "pimsleur":
            c = course_of(f)
            if c:
                courses[c] += 1
    course = next((c for c, n in sorted(courses.items()) if 120 <= n <= 350), None)
    if course:
        files, X = load(plang, lambda d: d.get("source") == "pimsleur" and course_of(d["file"]) == course)
        labels = cluster(X, thr=0.45)
        rows, nclu = sample(f"pims-{course[:18]}", plang, files, labels, n_clusters=3, clips_per=3, manifest_idx=pidx, min_size=15)
        print(f"Pimsleur eng course {course!r}: {len(files)} clips → {nclu} clusters; sampled 3")
        all_rows += rows

    print(f"\n=== review clips ({len(all_rows)}) — clips in the SAME clusterX{{A,B,C}} should be one person ===")
    for r in all_rows:
        print(f"  {r['clip']:42s} [{r['tag']} clu{r['cluster']}, size {r['n_in_cluster']}] "
              f"gender={r['gender']} | {r['sentence']!r}")
    print(f"\nclips in: {REVIEW}")


if __name__ == "__main__":
    main()
