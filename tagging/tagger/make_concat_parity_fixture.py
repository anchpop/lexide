"""Regenerate lexide/tests/fixtures/concat_parity.{safetensors,json}.

The released char-tokenizer fixtures come from whatever checkpoint shipped last, so they
cannot cover a code path no shipped model uses yet. This makes a small randomly-initialised
*concat*-mode model and records PyTorch's logits for it, so the Rust reimplementation's
concat path is checked from the day it is written rather than the day it ships.

Weights are drawn with a wide spread on purpose: near-zero weights would let an indexing
bug produce nearly-correct numbers.

    LD_LIBRARY_PATH=<gcc-lib> .venv-seg/bin/python tagger/make_concat_parity_fixture.py
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dataset import BOS_BYTE, CHAR_VOCAB_SIZE, EOS_BYTE, LANG_BOS  # noqa: E402
from model import CharBoundaryTagger  # noqa: E402
from prior import PRIOR_VOCAB, prior_ids_for  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "lexide", "tests", "fixtures")
# one no-space language, one long compound, one plain ASCII, one Japanese
TEXTS = [("kor", "나는 밥을 먹었어요"),
         ("deu", "Der Kraftfahrzeug-Haftpflichtversicherung."),
         ("eng", "ab cd"),
         ("jpn", "これはペンです")]


def main():
    torch.manual_seed(7)
    model = CharBoundaryTagger(vocab_size=CHAR_VOCAB_SIZE, emb_dim=16, hidden_dim=12,
                               layers=2, prior_vocab=PRIOR_VOCAB, prior_mode="concat",
                               prior_dim=5)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.4)
    model.eval()
    assert model.layers[0].fwd.to_z.in_features == 16 + 5, "not concat mode"

    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(OUT, "concat_parity.safetensors"))

    cases = []
    with torch.no_grad():
        for lang, text in TEXTS:
            for use_lang in (None, lang):
                ids = [LANG_BOS.get(use_lang, BOS_BYTE)] + list(text.encode()) + [EOS_BYTE]
                # the prior always knows the language, even when the lang token is dropped
                prior = prior_ids_for(text, lang, max_bytes=len(ids), wordbanks={})
                assert len(prior) == len(ids)
                logits = model(torch.tensor([ids]), torch.tensor([prior]))[0]
                cases.append({"text": text, "lang": use_lang, "prior_ids": prior,
                              "logits": [[round(v, 6) for v in row]
                                         for row in logits.tolist()]})
    with open(os.path.join(OUT, "concat_parity.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
