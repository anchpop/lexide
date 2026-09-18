"""Corpus metadata and storage around g2p's language-independent response."""


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
