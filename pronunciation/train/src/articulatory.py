"""Articulatory feature decomposition for the espeak phoneme vocab.

We use panphon's Hayes-style 24-feature schema. Each phoneme maps to a
24-dimensional vector of ternary values {-1, 0, +1}; we re-encode those
as categorical indices {0, 1, 2} for use with 3-way feature heads.

Multi-segment tokens (diphthongs like 'aɪ', affricates like 'tʃ') get the
*element-wise max* across their constituent segments' feature vectors.
That preserves information from BOTH segments — `tʃ` ends up with `+cont`
(from ʃ) on top of `t`'s stop features, so `t` and `tʃ` get different
signatures instead of colliding.

Special tokens (`<pad>`, `<s>`, `</s>`, `<unk>`) get an all-unknown feature
vector. Their signatures don't matter because callers must mask them to
-inf in the phoneme logits — they're never legitimate targets and shouldn't
appear in greedy-decoded output.

`detect_special_token_ids` returns the ids of those special tokens so the
model can mask them downstream.
"""

from pathlib import Path

import torch

NUM_FEATURES = 24   # panphon's Hayes-style feature set
NUM_VALUES = 3      # {-1, 0, +1} encoded as {0, 1, 2}
UNKNOWN_VALUE = 1   # corresponds to 0 (feature not applicable / unknown)


def is_special_token(tok: str) -> bool:
    """A token is "special" if it's not a real phoneme — bracket-wrapped sentinels
    used by HuggingFace tokenizers (<pad>, <s>, </s>, <unk>, etc.).
    """
    return len(tok) >= 2 and tok.startswith("<") and tok.endswith(">")


def detect_special_token_ids(vocab_tokens: list[str]) -> list[int]:
    return [i for i, t in enumerate(vocab_tokens) if is_special_token(t)]


def build_feature_table(vocab_tokens: list[str]):
    """Build (V, NUM_FEATURES) tensor mapping each token to feature indices.

    Tokens are ordered by id (vocab_tokens[i] is the token with id i).
    Returns (table, stats) where table is a long tensor of values in {0, 1, 2}.
    """
    import panphon
    ft = panphon.FeatureTable()

    feature_names = [name for name, _ in ft.fts("a").items()]
    assert len(feature_names) == NUM_FEATURES, f"expected {NUM_FEATURES} features, got {len(feature_names)}"

    V = len(vocab_tokens)
    table = torch.full((V, NUM_FEATURES), UNKNOWN_VALUE, dtype=torch.long)

    n_covered = 0
    n_multi = 0
    n_empty = 0
    n_special = 0

    for idx, tok in enumerate(vocab_tokens):
        # Don't pass special tokens to panphon at all — it'd extract features
        # from substrings (e.g. `<s>` would parse as `s`, colliding with the
        # real /s/ phoneme). Special tokens get the all-unknown signature;
        # they must be masked to -inf in downstream phoneme logits.
        if is_special_token(tok):
            n_special += 1
            continue

        try:
            vecs = ft.word_to_vector_list(tok, numeric=True)
        except Exception:
            vecs = []

        if len(vecs) == 0:
            n_empty += 1
            continue
        if len(vecs) > 1:
            n_multi += 1
            # Element-wise max across segments. For ternary {-1, 0, +1}, this
            # preserves any "+" feature that appears in any segment, so
            # affricates / diphthongs inherit features from BOTH parts.
            # Crucially: `t` (-cont) and `tʃ` (+cont from /ʃ/) end up different.
            stacked = torch.tensor(vecs, dtype=torch.long)
            vec = stacked.max(dim=0).values.tolist()
        else:
            n_covered += 1
            vec = vecs[0]

        # Encode ternary {-1, 0, +1} → {0, 1, 2}.
        for i, v in enumerate(vec):
            table[idx, i] = v + 1

    # Report uniqueness — high collision rate indicates the table is broken.
    unique_sigs = len({tuple(table[i].tolist()) for i in range(V)})

    return table, {
        "covered": n_covered,
        "multi_segment": n_multi,
        "empty": n_empty,
        "special": n_special,
        "unique_signatures": unique_sigs,
        "feature_names": feature_names,
    }


def main():
    """CLI: build the feature table and save to disk for training to load."""
    import argparse
    from transformers import Wav2Vec2CTCTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        help="HF tokenizer name (the source for our vocab).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("feature_table.pt"),
        help="Where to save the resulting table.",
    )
    args = parser.parse_args()

    tok = Wav2Vec2CTCTokenizer.from_pretrained(args.tokenizer)
    vocab = tok.get_vocab()
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]
    print(f"Loaded vocab: {len(tokens)} tokens")

    table, stats = build_feature_table(tokens)
    print(f"  cleanly covered:    {stats['covered']}")
    print(f"  multi-segment:      {stats['multi_segment']} (element-wise max combined)")
    print(f"  empty (no segments):{stats['empty']}")
    print(f"  special (<...>):    {stats['special']}")
    print(f"  Unique signatures:  {stats['unique_signatures']} / {len(tokens)}")
    print(f"  Feature schema ({NUM_FEATURES} features):")
    for name in stats["feature_names"]:
        print(f"    {name}")

    torch.save({
        "table": table,           # (V, NUM_FEATURES) long tensor in {0, 1, 2}
        "feature_names": stats["feature_names"],
        "tokens": tokens,
        "num_features": NUM_FEATURES,
        "num_values": NUM_VALUES,
    }, args.output)
    print(f"\nSaved feature table to {args.output}  shape={tuple(table.shape)}")


if __name__ == "__main__":
    main()
