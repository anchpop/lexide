"""Leak-free joint corpus. SQLite keeps identities/payloads off the Python heap."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from data_prep import LANGS, UPOS, COPY_SCRIPT, iter_records, normalize_sentence, lemma_script


def identity(text):
    return hashlib.sha256(text.encode()).digest()


def texts(obj):
    """Raw and reconstructed spellings both block held-out leakage."""
    out = set()
    for key in ("sentence", "text"):
        if isinstance(obj.get(key), str):
            out.add(obj[key])
    norm = normalize_sentence(obj)
    if norm:
        out.add(norm[0])
    return out


def build(args):
    root, out = Path(args.input_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Corpus storage may be spinning disks. Keep the random-access scratch DB
    # in the OS temporary directory (TMPDIR can select fast local scratch).
    scratch = tempfile.TemporaryDirectory(prefix="parsley-joint-")
    dbpath = Path(scratch.name) / "prep.sqlite"
    db = sqlite3.connect(dbpath)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-524288;
        CREATE TABLE seen (id BLOB PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE gold (lang TEXT, id BLOB, PRIMARY KEY(lang,id)) WITHOUT ROWID;
        CREATE TABLE records (lang TEXT, id BLOB, raw BLOB, payload TEXT, kind TEXT,
                              PRIMARY KEY(lang,id)) WITHOUT ROWID;
        CREATE TABLE splits (id BLOB PRIMARY KEY, split TEXT) WITHOUT ROWID;
    """)
    big, gold = sorted(Path(args.big_dir).rglob("*.jsonl")), sorted(Path(args.gold_dir).glob("cleaned_*.jsonl"))
    # An empty glob would make every sentence look unseen by v1 and leak into test.
    assert big and gold, f"v1 corpora not found under {args.big_dir} / {args.gold_dir}"
    old = big + gold
    for path in old:
        for obj in iter_records(path):
            db.executemany("INSERT OR IGNORE INTO seen VALUES (?)", [(identity(t),) for t in texts(obj)])
        db.commit()
        print(f"[seen] {path}", flush=True)
    for lang in LANGS:
        for path in (root / f"cleaned_{lang}.jsonl",):
            if path.exists():
                for obj in iter_records(path):
                    ids = [(lang, identity(t)) for t in texts(obj)]
                    db.executemany("INSERT OR IGNORE INTO gold VALUES (?,?)", ids)
                    # New gold may be held out; only the v1 sources block eligibility.
    counts = Counter()
    for lang in LANGS:
        for suffix in ("", "_augmented"):
            path = root / lang / f"target_language_sentences_tokenization{suffix}.jsonl"
            if not path.exists():
                continue
            for obj in iter_records(path):
                counts["scanned"] += 1
                norm = normalize_sentence(obj)
                if not norm:
                    counts["skipped"] += 1
                    continue
                text, tokens = norm
                key = identity(text)
                raw = identity(obj.get("sentence", text))
                kind = "gold" if db.execute("SELECT 1 FROM gold WHERE lang=? AND id IN (?,?)", (lang, key, raw)).fetchone() else "silver"
                rec = {"lang": lang, "kind": kind, "text": text, "tokens": tokens}
                result = db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)", (lang, key, raw, json.dumps(rec, ensure_ascii=False), kind))
                counts["dedup"] += not result.rowcount
            db.commit()
            print(f"[ingest] {lang}{suffix} {dict(counts)}", flush=True)
    # A normalized text is eligible only if *every* language/raw alias is unseen.
    db.executescript("""
        CREATE INDEX record_id ON records(id);
        CREATE TABLE blocked (id BLOB PRIMARY KEY) WITHOUT ROWID;
        INSERT OR IGNORE INTO blocked SELECT r.id FROM records r
          WHERE EXISTS (SELECT 1 FROM seen s WHERE s.id=r.id OR s.id=r.raw);
    """)
    candidates, split_counts = {}, Counter()
    for lang in LANGS:
        candidates[lang] = db.execute("SELECT count(*) FROM records r WHERE lang=? AND NOT EXISTS (SELECT 1 FROM blocked b WHERE b.id=r.id)", (lang,)).fetchone()[0]
        for (key,) in db.execute("SELECT id FROM records r WHERE lang=? AND NOT EXISTS (SELECT 1 FROM blocked b WHERE b.id=r.id) ORDER BY id", (lang,)):
            if split_counts[(lang, "test")] >= args.test_per_lang and split_counts[(lang, "val")] >= args.val_per_lang:
                break
            assigned = db.execute("SELECT split FROM splits WHERE id=?", (key,)).fetchone()
            if assigned:
                continue
            split = "test" if split_counts[(lang, "test")] < args.test_per_lang else "val"
            cap = args.test_per_lang if split == "test" else args.val_per_lang
            if split_counts[(lang, split)] >= cap:
                continue
            # Respect caps for all languages sharing this text.
            peers = [row[0] for row in db.execute("SELECT lang FROM records WHERE id=?", (key,))]
            if any(split_counts[(peer, split)] >= cap for peer in peers):
                continue
            db.execute("INSERT INTO splits VALUES (?,?)", (key, split))
            for peer in peers:
                split_counts[(peer, split)] += 1
    db.commit()
    dep, scripts = Counter(), Counter()
    # Vocab from training only: held-out labels never influence the model's label space.
    for (payload,) in db.execute("SELECT payload FROM records r WHERE NOT EXISTS (SELECT 1 FROM splits s WHERE s.id=r.id)"):
        for tk in json.loads(payload)["tokens"]:
            dep[tk["dep"]] += 1
            scripts[lemma_script(tk["form"], tk["lemma"])] += 1
    selected = [s for s, _ in scripts.most_common(args.max_lemma_scripts)]
    selected.append(COPY_SCRIPT)
    vocab = {"pos": UPOS, "dep": [d for d, _ in dep.most_common()], "lemma_scripts": selected, "langs": LANGS}
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    script_ids = {s: i for i, s in enumerate(selected)}
    coverage = sum(scripts[s] for s in selected) / max(1, sum(scripts.values()))
    emitted = Counter()
    with (out / "train.jsonl").open("w") as train, (out / "val.jsonl").open("w") as val, (out / "test.jsonl").open("w") as test:
        writers = {"train": train, "val": val, "test": test}
        for payload, split in db.execute("SELECT payload, COALESCE(s.split,'train') FROM records r LEFT JOIN splits s ON s.id=r.id ORDER BY r.lang,r.id"):
            rec = json.loads(payload)
            for tk in rec["tokens"]:
                tk["lemma_script"] = script_ids.get(lemma_script(tk["form"], tk["lemma"]), script_ids[COPY_SCRIPT])
                if tk["pos"] not in UPOS:
                    tk["pos"] = "X"
                if split == "train":
                    del tk["form"], tk["lemma"]
                elif tk["lemma"] is None:
                    tk["lemma"] = tk["form"]
            writers[split].write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            emitted[(rec["lang"], split)] += 1
            emitted[(rec["lang"], rec["kind"])] += 1
    report = {"input": dict(counts), "lemma_coverage": coverage, "dep_labels": len(dep), "lemma_scripts": len(selected), "languages": {lang: {**{sp: emitted[(lang, sp)] for sp in ("train", "val", "test", "gold", "silver")}, "unseen_candidates": candidates[lang], "shortage": max(0, args.test_per_lang + args.val_per_lang - emitted[(lang, "test")] - emitted[(lang, "val")])} for lang in LANGS}}
    (out / "prep_counts.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    db.close()
    scratch.cleanup()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data/export-2026-10")
    ap.add_argument("--big-dir", default="data/big")
    ap.add_argument("--gold-dir", default="train/data")
    ap.add_argument("--out-dir", default="data/processed-joint")
    ap.add_argument("--max-lemma-scripts", type=int, default=4000)
    ap.add_argument("--test-per-lang", type=int, default=1000)
    ap.add_argument("--val-per-lang", type=int, default=400)
    build(ap.parse_args())
