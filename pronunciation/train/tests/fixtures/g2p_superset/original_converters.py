"""Frozen production converters from 290702d, before superset unification.
Do not refresh with the adapter under test. Japanese intentionally accepts empty phones.
"""

JAPANESE_ACCENT_MIN_WHISPER_LOGPROB = -0.35


def g2p_hindi_labels(rec: dict, audit: dict) -> dict:
    """Adapt absolute g2p Hindi labels without rebuilding word-local arrays.

    Only the sidecar's syllable stress/word/provenance fields differ from the
    wire format. Keep the historical field order too: the shadow gate compares
    serialized sidecar bytes, not just equivalent dictionaries.
    """
    out = audit["output"]
    if out.get("exclude_reason"):
        return {"exclude_reason": out["exclude_reason"]}
    phonemes, stress = out["phonemes"], out["stress"]
    if len(phonemes) != len(stress):
        raise ValueError(f"Hindi word stress misalignment in {rec['file']}")
    native_syllables = out.get("syllables", [])
    syllables = []
    phone_cursor = 0
    syllable_index = 0
    for word_index, (start, end) in enumerate(out["word_spans"]):
        if start != phone_cursor or not start < end <= len(phonemes):
            raise ValueError(f"Hindi word span invariant failed in {rec['file']}")
        syllable_cursor = start
        while (syllable_index < len(native_syllables)
               and native_syllables[syllable_index]["start"] < end):
            s = native_syllables[syllable_index]
            syllable_stress = int(s["stressed"])
            if (s["start"] != syllable_cursor
                    or not s["start"] <= s["nucleus"] < s["end"] <= end
                    or stress[s["nucleus"]] != syllable_stress):
                raise ValueError(f"Hindi syllable invariant failed in {rec['file']}")
            syllables.append({
                "start": s["start"], "end": s["end"], "nucleus": s["nucleus"],
                "moras": s["moras"], "stress": syllable_stress, "word": word_index,
                "source": "roy-2017-rules-on-schwa-hin",
            })
            syllable_cursor = s["end"]
            syllable_index += 1
        if syllable_cursor != end:
            raise ValueError(f"Hindi syllables do not cover word in {rec['file']}")
        phone_cursor = end
    if phone_cursor != len(phonemes) or syllable_index != len(native_syllables):
        raise ValueError(f"Hindi word/syllable coverage failed in {rec['file']}")
    if not phonemes:
        return {"exclude_reason": "hindi_no_devanagari_phones"}
    return {
        "phonemes": phonemes, "stress": stress, "syllables": syllables,
        "stress_source": "roy-2017-rules-on-schwa-hin",
    }


def g2p_flat_labels(rec: dict, audit: dict) -> dict:
    """Providers whose g2p output is already in sidecar shape (phonemes,
    stress, tone), or an exclusion."""
    out = audit["output"]
    if out.get("exclude_reason"):
        return {"exclude_reason": out["exclude_reason"]}
    if len(out["phonemes"]) != len(out["stress"]) or len(out["phonemes"]) != len(out["tone"]):
        raise ValueError(f"g2p label misalignment in {rec['file']}")
    if not out["phonemes"]:
        return {"exclude_reason": "g2p_no_phonemes"}
    return {"phonemes": out["phonemes"], "stress": out["stress"], "tone": out["tone"]}


def g2p_japanese_labels(rec: dict, audit: dict) -> dict:
    """g2p's Japanese output (phones, pitch factor, and the parse-derived
    withhold reasons) plus the one check that needs the manifest row: a
    Whisper transcript below the accent bar may have guessed the kanji, and
    pitch accent is lexical, so the accent factor is withheld."""
    out = audit["output"]
    if out.get("exclude_reason"):
        return {"exclude_reason": out["exclude_reason"]}
    result = {"phonemes": out["phonemes"], "stress": out["stress"]}
    withhold = out.get("pitch_accent_exclude_reason")
    if withhold is None:
        logprob = rec.get("whisper_avg_logprob")
        if logprob is not None and logprob < JAPANESE_ACCENT_MIN_WHISPER_LOGPROB:
            withhold = "japanese_accent_low_asr_confidence"
    if withhold is not None:
        result["pitch_accent_exclude_reason"] = withhold
    else:
        result["pitch_accent"] = out["pitch_accent"]
    return result
