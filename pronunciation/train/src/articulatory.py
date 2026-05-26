"""Articulatory feature decomposition for the espeak phoneme vocab.

We use panphon's Hayes-style 24-feature schema. Each phoneme maps to a
24-dimensional vector of ternary values {-1, 0, +1}; we re-encode those
as categorical indices {0, 1, 2} for use with 3-way feature heads.

For multi-segment tokens (diphthongs like 'aɪ', affricates like 'tʃ') we
take the *first* segment's features — losing some information, but
keeping the table 1:1 with the vocab and avoiding hand-crafted exceptions
for the long tail. If empirically the diphthong/affricate distinction
matters, we revisit.

Special tokens (`<pad>`, `<s>`, `</s>`, `<unk>`) and any token panphon
can't parse get an all-zero feature vector (encoded as all-1 since
0 maps to index 1 in {-1: 0, 0: 1, +1: 2}). They become indistinguishable
in articulatory space — fine for `<pad>` (blank head handles it
separately); the others rarely appear as targets anyway.
"""

from pathlib import Path
import torch

NUM_FEATURES = 24  # panphon's Hayes-style feature set
NUM_VALUES = 3     # {-1, 0, +1} encoded as {0, 1, 2}
UNKNOWN_VALUE = 1  # corresponds to 0 (feature not applicable / unknown)


def build_feature_table(vocab_tokens: list[str]) -> torch.Tensor:
    """Build (V, NUM_FEATURES) tensor mapping each token to feature indices.

    Tokens are ordered by id (vocab_tokens[i] is the token with id i).
    Returns long tensor of values in {0, 1, 2}.
    """
    import panphon
    ft = panphon.FeatureTable()

    # Order matters: panphon's `fts(seg)` returns OrderedDict of feature names.
    # Pull the canonical order once from a known segment.
    feature_names = [name for name, _ in ft.fts("a").items()]
    assert len(feature_names) == NUM_FEATURES, f"expected {NUM_FEATURES} features, got {len(feature_names)}"

    V = len(vocab_tokens)
    table = torch.full((V, NUM_FEATURES), UNKNOWN_VALUE, dtype=torch.long)

    n_covered = 0
    n_multi = 0
    n_empty = 0

    for idx, tok in enumerate(vocab_tokens):
        # Special tokens, punctuation, or things panphon can't parse → leave at unknown.
        try:
            vecs = ft.word_to_vector_list(tok, numeric=True)
        except Exception:
            vecs = []

        if len(vecs) == 0:
            n_empty += 1
            continue
        if len(vecs) > 1:
            n_multi += 1
            # Use first segment — see module docstring for rationale.
            vec = vecs[0]
        else:
            n_covered += 1
            vec = vecs[0]

        # Encode ternary {-1, 0, +1} → {0, 1, 2}.
        for i, v in enumerate(vec):
            table[idx, i] = v + 1  # -1 → 0, 0 → 1, +1 → 2

    return table, {
        "covered": n_covered,
        "multi_segment": n_multi,
        "empty_or_special": n_empty,
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
    print(f"  multi-segment:      {stats['multi_segment']} (first segment used)")
    print(f"  empty/special/UNK:  {stats['empty_or_special']} (all-unknown vector)")
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
