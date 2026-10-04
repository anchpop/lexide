"""French rhythmic-group stress from the LLM sidecar."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preprocess_support import STRESS_NONE, STRESS_PRIMARY, apply_stress_override


def test_sentence_final_word_is_always_stressed():
    phonemes = ["a", "v", "ɛ", "k", "k", "i"]
    spans = [(0, 4), (4, 6)]
    final_only = [STRESS_NONE] * 5 + [STRESS_PRIMARY]
    # An empty answer means final stress only, never no stress at all.
    assert apply_stress_override(phonemes, spans, "Avec qui.", []) == final_only
    assert apply_stress_override(phonemes, spans, "Avec qui.", ["qui."]) == final_only
    # A listed non-final word adds its own group end; the final word stays stressed.
    both = [STRESS_NONE, STRESS_NONE, STRESS_PRIMARY, STRESS_NONE, STRESS_NONE, STRESS_PRIMARY]
    assert apply_stress_override(phonemes, spans, "Avec qui.", ["Avec"]) == both
    # A word the sentence does not contain still fails the override.
    assert apply_stress_override(phonemes, spans, "Avec qui.", ["moi"]) is None
