"""Curated segmental vocabulary for fresh models and preprocessing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_training_labels() -> dict:
    """Sky stages the canonical artifact beside this script."""
    script = Path(__file__).resolve()
    path = script.parents[3] / "tagging/lexide/data/training_labels.json"
    if not path.is_file():
        path = script.with_name("training_labels.json")
    return json.loads(path.read_text(encoding="utf-8"))


_TRAINING_LABELS = _load_training_labels()
PHONEMES = frozenset(_TRAINING_LABELS["phonemes"])


def unknown_phonemes(phonemes: list[str]) -> set[str]:
    """Reject unsupported labels and controls without rewriting the sequence."""
    return set(phonemes) - PHONEMES


def new_tokenizer():
    """Fresh IDs only. Existing checkpoints must load their saved tokenizer."""
    from transformers import Wav2Vec2CTCTokenizer

    # Blank and the tokenizer's unknown sentinel are structural, never targets.
    # The model masks both out of the conditional phoneme logits. Alias the
    # unused word delimiter to blank: transformers serializes None as "None"
    # and otherwise invents an extra token on reload.
    vocab = {token: i for i, token in enumerate(["<pad>", "<unk>", *sorted(PHONEMES)])}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "vocab.json"
        path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        return Wav2Vec2CTCTokenizer(
            str(path), pad_token="<pad>", unk_token="<unk>",
            bos_token=None, eos_token=None, word_delimiter_token="<pad>",
            do_lower_case=False,
        )
