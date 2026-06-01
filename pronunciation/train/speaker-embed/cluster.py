"""Cluster cached speaker embeddings into pseudo-speaker labels.

Count-agnostic (HDBSCAN on L2-normalized embeddings — no fixed k, handles many
speakers per source and labels outliers as noise). Reads the per-clip cache
written by embed.py.

  cluster.py --langs spa --sources fleurs [--min-cluster-size 5]
             [--validate-gender] [--write]

--validate-gender (FLEURS only): cross-checks clusters against the gender column
in the FLEURS TSV (recoverable via sentence-join) — a real speaker cluster should
be single-gender, so mean per-cluster gender purity ~1.0 is the sanity signal.
--write: add a `speaker_cluster` field to the manifest rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embed as E  # noqa: E402  (cache paths + audio root)

AUDIO = E.AUDIO


def load_lang_source(lang: str, sources: set[str]):
    """Return (files, embeddings, rows_by_file) for cached clips of this lang/source."""
    mf = AUDIO / lang / "manifest.jsonl"
    files, vecs, rows = [], [], {}
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("source") not in sources:
            continue
        if not E.is_cached(lang, d["file"]):
            continue
        files.append(d["file"])
        vecs.append(E.load_embedding(lang, d["file"]))
        rows[d["file"]] = d
    if not files:
        return [], np.zeros((0, 192)), {}
    X = np.vstack(vecs).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # L2 -> euclidean≈cosine
    return files, X, rows


def fleurs_gender(lang: str) -> dict:
    """sentence -> gender from the FLEURS TSV (same join verified earlier)."""
    import csv, glob
    from huggingface_hub import hf_hub_download
    code = {"eng": "en_us", "deu": "de_de", "fra": "fr_fr", "ita": "it_it",
            "por": "pt_br", "spa": "es_419", "rus": "ru_ru"}[lang]
    g = {}
    for split in ("train", "dev", "test"):
        try:
            tsv = hf_hub_download("google/fleurs", f"data/{code}/{split}.tsv", repo_type="dataset")
        except Exception:
            continue
        with open(tsv, newline="") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) != 7:
                    continue
                s = row[2].strip()
                if len(s) >= 2 and s[0] == s[-1] == '"':
                    s = s[1:-1]
                g[s] = row[6].strip()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=["fleurs", "pimsleur"])
    # Agglomerative + cosine threshold beats HDBSCAN here (HDBSCAN under-segments
    # → gender-mixed blobs). Lower threshold = more, purer clusters; we lean
    # slightly over-segmented because merging two speakers corrupts within-speaker
    # normalization, while splitting one is harmless.
    ap.add_argument("--method", choices=["agglom", "hdbscan"], default="agglom")
    ap.add_argument("--threshold", type=float, default=0.15, help="cosine distance (agglom)")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="hdbscan")
    ap.add_argument("--validate-gender", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for lang in args.langs:
        files, X, rows = load_lang_source(lang, set(args.sources))
        if len(files) == 0:
            print(f"{lang}: no cached embeddings"); continue
        if args.method == "agglom":
            if len(files) > 12000:
                print(f"   note: {len(files)} clips — agglomerative is O(n²) in memory; "
                      f"for big sources cluster per-course or use --method hdbscan.")
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=args.threshold,
                metric="cosine", linkage="average").fit_predict(X)
        else:
            labels = HDBSCAN(min_cluster_size=args.min_cluster_size,
                             metric="euclidean").fit_predict(X)
        n_clusters = len({l for l in labels if l != -1})
        n_noise = int((labels == -1).sum())
        print(f"{lang}: {len(files)} clips → {n_clusters} speaker clusters "
              f"({n_noise} noise/outlier) [{args.method}]")

        if args.validate_gender and "fleurs" in args.sources:
            g = fleurs_gender(lang)
            by_cluster = defaultdict(Counter)
            for f, l in zip(files, labels):
                if l == -1:
                    continue
                gen = g.get(rows[f].get("sentence"))
                if gen:
                    by_cluster[l][gen] += 1
            purities = [max(c.values()) / sum(c.values()) for c in by_cluster.values() if sum(c.values())]
            if purities:
                print(f"   gender purity: mean {np.mean(purities):.3f} "
                      f"(1.0 = every cluster single-gender; {len(purities)} clusters checked)")

        if args.write:
            tag_of = {f: (f"{lang}:spk{int(l):03d}" if l != -1 else None)
                      for f, l in zip(files, labels)}
            p = AUDIO / lang / "manifest.jsonl"
            lines = []
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["file"] in tag_of and tag_of[d["file"]] is not None:
                    d["speaker_cluster"] = tag_of[d["file"]]
                lines.append(json.dumps(d, ensure_ascii=False))
            p.write_text("\n".join(lines) + "\n")
            print(f"   wrote speaker_cluster to {p}")


if __name__ == "__main__":
    main()
