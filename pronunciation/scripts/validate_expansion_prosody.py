#!/usr/bin/env python3
"""Run small, linguistically interpretable prosody regression checks.

These checks do not prove corpus-wide acoustic correctness. They protect the
meaning and alignment of the tone/accent/stress metadata while larger audio
audits remain separate.
"""

from __future__ import annotations

import argparse

from audit_g2p_backends import (
    _g2pm_ipa,
    _hindi_surface_stress,
    _pyopenjtalk,
    _vachana_thai,
)
from build_external_phoneme_sidecars import (
    japanese_labels,
    mandarin_labels,
    thai_labels,
)


def fake_record(sentence: str) -> dict:
    return {"file": "prosody-regression.wav", "sentence": sentence}


def check_thai() -> None:
    # Orthographic minimal series: mid, low, falling, high, rising.
    sentence = "กา ก่า ก้า ก๊า ก๋า"
    output = _vachana_thai(sentence)
    labels = thai_labels(fake_record(sentence), {"output": output})
    assert [tone for tone in labels["tone"] if tone is not None] == [1, 2, 3, 4, 5]
    assert sum(value == 1 for value in labels["stress"]) == 5


def check_mandarin() -> None:
    # Same segment /ma/, all five citation-tone categories.
    sentence = "妈麻马骂吗"
    output = _g2pm_ipa(sentence)
    labels = mandarin_labels(fake_record(sentence), {"output": output})
    assert [tone for tone in labels["tone"] if tone is not None] == [1, 2, 3, 4, 5]
    assert len(labels["phonemes"]) == len(labels["tone"])


def check_japanese() -> None:
    # Tokyo Japanese /haɕi/: initial-accent, final-accent, and heiban. This is
    # the minimal pair the accent factor exists to separate, so pin both the
    # nucleus the frontend reports and the per-mora contour we train on.
    #
    # 橋 (odaka) and 端 (heiban) share a contour on purpose: spoken in
    # isolation they *are* the same, and they diverge only on the particle
    # that follows. Asserting that keeps anyone from "fixing" the encoding
    # into claiming a distinction this audio cannot carry.
    expected = {
        "箸": (1, [1, 0]),   # HL — accented on mora 1
        "橋": (2, [0, 1]),   # LH — accented on mora 2, falls onto the particle
        "端": (0, [0, 1]),   # LH — heiban, no fall at all
    }
    for word, (nucleus, levels) in expected.items():
        output = _pyopenjtalk(word)
        labels = japanese_labels(fake_record(word), {"output": output})
        borne = [item for item in labels["pitch_accent"] if item is not None]
        observed = {item["nucleus"] for item in borne}
        assert observed == {nucleus}, (word, observed)
        assert [item["level"] for item in borne] == levels, (word, borne)
        assert [item["mora"] for item in borne] == [1, 2], (word, borne)


def check_hindi() -> None:
    # Nasalization must not hide the vowel nucleus from syllable/stress logic.
    result = _hindi_surface_stress(["m", "eː̃"])
    assert result["syllables"] == [
        {"start": 0, "end": 2, "nucleus": 1, "moras": 2, "stress": 0}
    ]
    assert len(result["stress"]) == 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs", nargs="+", choices=("tha", "zho-hans", "jpn", "hin"),
        default=("tha", "zho-hans", "jpn", "hin"),
    )
    args = parser.parse_args()
    checks = {
        "tha": check_thai, "zho-hans": check_mandarin,
        "jpn": check_japanese, "hin": check_hindi,
    }
    for lang in args.langs:
        checks[lang]()
    print("prosody regression checks passed: " + ", ".join(args.langs))


if __name__ == "__main__":
    main()
