"""Frozen model vocabulary shared by preprocessing and training."""

import json
from functools import cache
from pathlib import Path

def _load_training_labels() -> dict:
    """Read the model vocabulary used by the training pipeline.

    Sky stages only pronunciation/, with an explicit file_mount of the canonical
    artifact beside this script. A full checkout reads it from the Rust crate.
    """
    script = Path(__file__).resolve()
    path = script.parents[3] / "tagging/lexide/data/training_labels.json"
    if not path.is_file():
        path = script.with_name("training_labels.json")
    return json.loads(path.read_text(encoding="utf-8"))


# Frozen model vocabulary; g2p owns pronunciation and phone segmentation.
_TRAINING_LABELS = _load_training_labels()
TOKENIZER_NAME = _TRAINING_LABELS["provenance"]["tokenizer_name"]
VOCAB_EXTENSIONS: set[str] = set(_TRAINING_LABELS["vocab_extensions"])


@cache
def _tokenizer_vocab() -> set[str]:
    """Frozen base accepted set; independent of network/tokenizer revisions."""
    return set(_TRAINING_LABELS["base_vocab"])


def check_training_label_vocab(model_name: str, vocab: set[str]) -> None:
    """Reject default tokenizer drift; custom processor sources stay independent."""
    if model_name != TOKENIZER_NAME:
        return
    expected = _tokenizer_vocab()
    if vocab != expected:
        raise ValueError(
            f"Training-label vocabulary drift for {model_name}: "
            f"missing={sorted(expected - vocab)}, unexpected={sorted(vocab - expected)}. "
            "Review the shared training_labels.json contract before training."
        )


def unknown_phonemes(phonemes: list[str]) -> set[str]:
    """Report unsupported model labels without rewriting g2p's output."""
    return set(phonemes) - (_tokenizer_vocab() | VOCAB_EXTENSIONS)


