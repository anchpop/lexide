"""Fast, offline regression tests: python -m unittest discover -s tagger -p test_joint.py."""
import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from data_prep import LANGS, POS_TAGS
from data_prep_joint import build
from dataset_joint import IndexedCorpus, collate, prepare
from eval_e2e import score
from model import JointTagger
from mst import single_root_mst
from predict_joint import spans_from_char_labels


def valid_tree(heads):
    if heads.count(0) != 1:
        return False
    for word in range(1, len(heads) + 1):
        seen = set()
        while word:
            if word in seen:
                return False
            seen.add(word)
            word = heads[word - 1]
    return True


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cuda_guard(self):
        from unittest.mock import patch
        from train_joint import train
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(AssertionError, "CUDA unavailable"):
                train(argparse.Namespace(cpu_ok=False))

    def test_mst_brute_force(self):
        rng = np.random.default_rng(123)
        for n in range(1, 5):
            for _ in range(15):
                arcs = rng.normal(size=(n, n + 1))
                heads = single_root_mst(arcs)
                self.assertTrue(valid_tree(heads))
                best = max(sum(arcs[i, h] for i, h in enumerate(candidate))
                           for candidate in itertools.product(range(n + 1), repeat=n) if valid_tree(list(candidate)))
                self.assertAlmostEqual(sum(arcs[i, h] for i, h in enumerate(heads)), best)
        self.assertEqual(single_root_mst(np.empty((0, 1))), [])

    def test_scoring_head_spans_and_missing(self):
        gold = [{"lang": "eng", "text": "a bc", "tokens": [
            {"start": 0, "end": 1, "pos": "NOUN", "lemma": "a", "head": 2, "dep": "obj"},
            {"start": 2, "end": 4, "pos": "VERB", "lemma": "bc", "head": 0, "dep": "root"}]}]
        self.assertEqual(score(gold, gold)["macro"]["las"]["f1"], 1)
        pred = json.loads(json.dumps(gold))
        pred[0]["tokens"][1]["end"] = 3
        result = score(gold, pred)["macro"]
        self.assertEqual(result["token"]["f1"], .5)
        self.assertEqual(result["uas"]["f1"], 0)
        # A new token changes head indices, not the referred-to head span.
        shifted = json.loads(json.dumps(gold))
        shifted[0]["tokens"][0]["head"] = 3
        shifted[0]["tokens"].insert(1, {"start": 1, "end": 2, "pos": "SPACE", "lemma": " ", "head": 0, "dep": "root"})
        self.assertEqual(score(gold, shifted)["macro"]["uas"]["f1"], .8)
        self.assertEqual(score(gold, [])["macro"]["token"]["recall"], 0)
        with self.assertRaises(ValueError):
            score(gold, gold + gold)

    def test_bio_unicode(self):
        # Labels move boundaries but never drop text: an orphan I or stray O stays covered.
        self.assertEqual(spans_from_char_labels("你好吗 x", [2, 1, 2, 0, 1]), [(0, 1), (1, 3), (4, 5)])
        self.assertEqual(spans_from_char_labels("ab cd", [1, 2, 0, 0, 0]), [(0, 2), (3, 5)])
        self.assertEqual(spans_from_char_labels("New York", [1, 2, 2, 2, 2, 2, 2, 2]), [(0, 8)])
        self.assertEqual(spans_from_char_labels("ab cd", [1, 2, 2, 1, 2]), [(0, 2), (3, 5)])
        self.assertEqual(spans_from_char_labels("你好", [1]), [(0, 2)])
        self.assertEqual(spans_from_char_labels("", []), [])

    def test_char_model_shared_piece_empty_truncation(self):
        vocab = {"pos": POS_TAGS, "dep": ["root", "obj"], "lemma_scripts": ["COPY"]}
        record = {"lang": "jpn", "text": "你好", "tokens": [
            {"start": 0, "end": 1, "pos": "NOUN", "dep": "root", "head": 0},
            {"start": 1, "end": 2, "pos": "NOUN", "dep": "obj", "head": 1}]}
        enc = {"input_ids": [0, 4, 2], "offset_mapping": [(0, 0), (0, 2), (0, 0)]}
        item = prepare(record, enc, vocab, 64, False)
        empty = prepare({"lang": "eng", "text": "", "tokens": []},
                        {"input_ids": [0, 2], "offset_mapping": [(0, 0), (0, 0)]}, vocab, 64, False)
        batch = collate([item, empty], 1)
        model = JointTagger("unused", 18, 2, 1, encoder_layers=2, char_dim=8, char_hidden=8,
                            word_dim=16, char_buckets=64, arc_dim=8, rel_dim=8, dropout=0,
                            encoder_config={"model_type": "xlm-roberta", "vocab_size": 50,
                                            "hidden_size": 16, "intermediate_size": 32,
                                            "num_attention_heads": 2, "num_hidden_layers": 2,
                                            "max_position_embeddings": 10, "pad_token_id": 1})
        out = model(batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.equal(out["sub_at_char"][0, 0], out["sub_at_char"][0, 1]))
        self.assertFalse(torch.equal(out["chars"][0, 0], out["chars"][0, 1]))
        self.assertFalse(torch.equal(out["pos_logits"][0, 0], out["pos_logits"][0, 1]))
        out["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed = model(batch)
        mixed["loss"].backward()
        self.assertTrue(torch.isfinite(mixed["loss"]))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        partial = prepare(record, {"input_ids": [0, 4, 2], "offset_mapping": [(0, 0), (0, 1), (0, 0)]}, vocab, 64, True)
        self.assertEqual(partial["starts"], [0])
        dangling = json.loads(json.dumps(record))
        dangling["tokens"][0]["head"] = 2
        partial = prepare(dangling, {"input_ids": [0, 4, 2], "offset_mapping": [(0, 0), (0, 1), (0, 0)]}, vocab, 64, True)
        self.assertEqual(partial["head"], [-100])
        partial_word = {"lang": "jpn", "text": "你好", "tokens": [{"start": 0, "end": 2, "pos": "NOUN", "dep": "root", "head": 0}]}
        partial = prepare(partial_word, {"input_ids": [0, 4, 2], "offset_mapping": [(0, 0), (0, 1), (0, 0)]}, vocab, 64, True)
        self.assertEqual(partial["starts"], [])
        self.assertEqual(partial["boundary"], [-100])
        # Reject over-limit sentences rather than silently windowing the encoder.
        long = prepare({"lang": "eng", "text": "abcdefghijk", "tokens": []},
                       {"input_ids": [0] + [4] * 11 + [2],
                        "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(11)] + [(0, 0)]}, vocab, 64, False)
        with self.assertRaisesRegex(ValueError, "maximum is 8"):
            model.encode_chars(collate([long, empty], 1))
        model.eval()
        single = collate([item], 1)
        alone = model.encode_chars(single)["chars"]
        unpacked = model.encode_chars(single, unpadded=True)["chars"]
        self.assertTrue(torch.allclose(alone, unpacked, atol=1e-6))
        medium = prepare({"lang": "eng", "text": "abcde", "tokens": []},
                         {"input_ids": [0] + [4] * 5 + [2],
                          "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(5)] + [(0, 0)]}, vocab, 64, False)
        padded = model.encode_chars(collate([item, medium], 1))["chars"]
        self.assertTrue(torch.allclose(alone[0, :2], padded[0, :2], atol=1e-6))
        reload = JointTagger("unused", 18, 2, 1, encoder_layers=2, char_dim=8, char_hidden=8,
                             word_dim=16, char_buckets=64, arc_dim=8, rel_dim=8, dropout=0,
                             encoder_config=model.encoder.config.to_dict())
        reload.load_state_dict(model.state_dict())
        model.eval()
        reload.eval()
        self.assertTrue(torch.equal(model(batch)["pos_logits"], reload(batch)["pos_logits"]))

    def test_epoch_sampling_preserves_gold_over_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            records = [{"lang": lang, "text": str(i), "kind": "gold" if i < 3 else "silver"}
                       for lang in ("eng", "jpn") for i in range(30)]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            corpus = IndexedCorpus(path)
            ids, counts = corpus.sample(1, 10)
            self.assertEqual(counts["eng"], 3)
            self.assertEqual(counts["jpn"], 3)
            self.assertEqual(set(ids), set(np.flatnonzero(corpus.gold)))
            first, _ = corpus.sample(8, 10)
            second, _ = corpus.sample(8, 11)
            self.assertNotEqual(set(first), set(second))
            self.assertTrue(set(ids).issubset(first) and set(ids).issubset(second))
            corpus.data.close()
            corpus.file.close()

    def test_split_dedup_gold_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, old, gold, out = [root / name for name in ("source", "old", "gold", "out")]
            for p in (source, old, gold):
                p.mkdir()
            def raw(text):
                return {"sentence": text, "tokens": [{"text": text, "whitespace": "", "pos": "NOUN", "lemma": text, "dep": "root", "head": 0}]}
            (old / "anything.jsonl").write_text(json.dumps(raw("old")) + "\n")
            (gold / "cleaned_eng.jsonl").write_text(json.dumps(raw("older")) + "\n")
            for lang in ("eng", "jpn"):
                (source / lang).mkdir()
                records = [raw(t) for t in ("old", "shared", "new1", "new2", "new3", "new4")]
                for suffix in ("", "_augmented"):
                    (source / lang / f"target_language_sentences_tokenization{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                (source / f"cleaned_{lang}.jsonl").write_text(json.dumps(raw("new4")) + "\n")
            args = argparse.Namespace(input_dir=str(source), big_dir=str(old), gold_dir=str(gold), out_dir=str(out), max_lemma_scripts=10, test_per_lang=1, val_per_lang=1)
            with contextlib.redirect_stdout(io.StringIO()):
                report = build(args)
            rows = {sp: [json.loads(l) for l in (out / f"{sp}.jsonl").read_text().splitlines()] for sp in ("train", "test", "val")}
            sets = {sp: {r["text"] for r in recs} for sp, recs in rows.items()}
            self.assertFalse(sets["train"] & sets["test"] or sets["train"] & sets["val"] or sets["val"] & sets["test"])
            self.assertNotIn("old", sets["test"] | sets["val"])
            self.assertEqual(report["languages"]["eng"]["unseen_candidates"], 5)
            self.assertEqual(report["languages"]["eng"]["gold"], 1)
            self.assertEqual(report["languages"]["jpn"]["test"], 1)
            corpus = IndexedCorpus(out / "train.jsonl")
            for seed in (1, 2):
                ids, counts = corpus.sample(1, seed)
                self.assertTrue(set(np.flatnonzero(corpus.gold)).issubset(ids))
            saved = {sp: (out / f"{sp}.jsonl").read_bytes() for sp in rows}
            with contextlib.redirect_stdout(io.StringIO()):
                build(args)
            self.assertEqual(saved, {sp: (out / f"{sp}.jsonl").read_bytes() for sp in rows})
            corpus.data.close()
            corpus.file.close()


if __name__ == "__main__":
    unittest.main()
