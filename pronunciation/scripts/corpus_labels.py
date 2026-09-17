"""Corpus metadata and storage around g2p's language-independent response."""


def _variety_for_record(rec: dict, lang: str) -> str:
    """Read dataset variety, including historical manifest voice metadata.

    Only Spanish and Portuguese had non-default varieties in the old corpus.
    This decodes saved metadata; it never chooses a phonemization engine.
    """
    if rec.get("variety"):
        return rec["variety"]
    legacy_voice = rec.get("espeak_voice")
    if lang == "spa":
        if legacy_voice:
            return {"es": "european", "es-419": "latin_american"}[legacy_voice]
        if rec.get("source") == "fleurs":
            return "latin_american"
        if rec.get("source") == "tts" and rec.get("tts_backend") in (None, "chirp3"):
            voice = rec.get("voice") or ""
            if voice.startswith("es-US-Chirp3-HD-"):
                return "latin_american"
            if voice.startswith("es-ES-Chirp3-HD-"):
                return "european"
    if lang == "por" and legacy_voice:
        return {"pt": "european", "pt-br": "brazilian"}[legacy_voice]
    return "default"


def language_for_record(rec: dict, lang: str) -> str:
    """Select g2p's combined language from recording metadata.

    Historical voice/variety fields are decoded here, never sent to g2p.
    Ordinary language codes already match the shared enum's wire values.
    """
    if rec.get("g2p_language"):
        return rec["g2p_language"]
    variety = _variety_for_record(rec, lang)
    if lang == "spa":
        return {"default": "spa-ES", "european": "spa-ES",
                "latin_american": "spa-419"}[variety]
    if lang == "por":
        return {"default": "por-BR", "brazilian": "por-BR",
                "european": "por-PT"}[variety]
    if variety != "default":
        raise ValueError(f"unsupported historical variety {variety!r} for {lang}")
    return lang


def training_fields(out: dict, rec: dict) -> dict:
    """Translate shared response fields to the training file's schema.

    No dispatch by language/provider. Acoustic and transcription-confidence
    gates are recording facts and stay on this side of the API.
    """
    phones = out["phonemes"]
    if len(out["stress"]) != len(phones):
        raise ValueError(f"unaligned g2p stress for {rec['file']}")
    result = {"stress_source": "g2p"}
    for field in ("tone", "pitch"):
        if field in out and len(out[field]) != len(phones):
            raise ValueError(f"unaligned g2p {field} for {rec['file']}")
    if "tone" in out:
        result["tone"] = out["tone"]
    if "pitch" in out or "accent_withheld" in out:
        reason = out.get("accent_withheld")
        logprob = rec.get("whisper_avg_logprob")
        if reason is None and logprob is not None and logprob < -0.35:
            reason = "pitch_accent_low_asr_confidence"
        if reason is not None:
            result["pitch_accent_exclude_reason"] = reason
        elif "pitch" in out:
            result["pitch_accent"] = [
                None if pitch is None else {**pitch, "source": "g2p"}
                for pitch in out["pitch"]
            ]
    if "syllables" in out:
        syllables = []
        spans = out["word_spans"]
        word = 0
        cursor = 0
        for syllable in out["syllables"]:
            start, end = syllable["start"], syllable["end"]
            while word < len(spans) and spans[word][1] <= start:
                word += 1
            if (word == len(spans) or start != cursor
                    or not spans[word][0] <= start <= syllable["nucleus"] < end <= spans[word][1]
                    or int(syllable["stressed"]) != out["stress"][syllable["nucleus"]]):
                raise ValueError(f"invalid g2p syllable coverage for {rec['file']}")
            syllables.append({
                "start": start, "end": end, "nucleus": syllable["nucleus"],
                "moras": syllable["moras"], "stress": int(syllable["stressed"]),
                "word": word, "source": "g2p",
            })
            cursor = end
        if cursor != len(phones):
            raise ValueError(f"incomplete g2p syllable coverage for {rec['file']}")
        result["syllables"] = syllables
    return result
