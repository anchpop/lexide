"""Frozen request/projection bodies from audit_g2p_backends.py at 67983cc.

Only docstrings are omitted. Keep independent of the refactored helper so the
shadow checks request options, refusals, field order, and projection failures.
"""

from typing import Any


def _g2p_zho(text: str) -> dict[str, Any]:
    import g2p_client

    try:
        r = g2p_client.request(text=text, lang="zho-hans")
    except g2p_client.Unlabelable as exc:
        return {"exclude_reason": exc.reason, "g2p": g2p_client.identity()}
    return {
        "phonemes": r["phonemes"], "stress": r["stress"], "tone": r["tone"],
        "pinyin": r["raw"], "g2p": g2p_client.identity(),
    }


def _g2p_jpn(text: str) -> dict[str, Any]:
    import g2p_client

    try:
        r = g2p_client.request(text=text, lang="jpn")
    except g2p_client.Unlabelable as exc:
        return {"exclude_reason": exc.reason, "g2p": g2p_client.identity()}
    return {
        "phonemes": r["phonemes"], "stress": r["stress"],
        "pitch_accent": [None if p is None else {**p, "source": "openjtalk-citation"}
                         for p in r.get("pitch", [])],
        "pitch_accent_exclude_reason": r.get("accent_withheld"),
        "phones": r["raw"].split(), "g2p": g2p_client.identity(),
    }


def _g2p_tha(text: str) -> dict[str, Any]:
    import g2p_client

    try:
        r = g2p_client.request(text=text, lang="tha")
    except g2p_client.Unlabelable as exc:
        return {"exclude_reason": exc.reason, "g2p": g2p_client.identity()}
    return {
        "phonemes": r["phonemes"], "stress": r["stress"], "tone": r.get("tone", []),
        "ipa_with_tone": r["raw"], "g2p": g2p_client.identity(),
    }


def _g2p_kor(text: str) -> dict[str, Any]:
    import g2p_client

    try:
        r = g2p_client.request(text=text, lang="kor")
    except g2p_client.Unlabelable as exc:
        return {"exclude_reason": exc.reason, "g2p": g2p_client.identity()}
    return {
        "phonemes": r["phonemes"], "stress": r["stress"],
        "tone": [None] * len(r["phonemes"]),
        "hangul_pronunciation": r["raw"], "g2p": g2p_client.identity(),
    }
