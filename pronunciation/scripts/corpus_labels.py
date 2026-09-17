"""Corpus metadata and storage around g2p's language-independent response."""

import json
import sqlite3

import g2p_client


def variety_for_record(rec: dict, lang: str) -> str:
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


class LabelCache:
    """Cache raw labels and explicit refusals, never infrastructure failures.

    Exact text, language, variety and build identity bind each result to its
    request. Recording-specific quality gates run after reading the cache.
    """

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.build = g2p_client.identity()
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS labels (
            build TEXT, lang TEXT, variety TEXT, text TEXT, response TEXT,
            PRIMARY KEY (build, lang, variety, text))""")
        self.db.commit()

    def close(self):
        self.db.close()

    def phonemize(self, text, lang, *, variety="default"):
        key = (self.build, lang, variety, text)
        cached = self.db.execute(
            "SELECT response FROM labels WHERE build=? AND lang=? AND variety=? AND text=?",
            key,
        ).fetchone()
        if cached:
            result = json.loads(cached[0])
        else:
            try:
                result = g2p_client.phonemize(text, lang, variety=variety)
            except g2p_client.Unlabelable as error:
                result = {"exclude_reason": error.reason}
            with self.db:
                self.db.execute("INSERT INTO labels VALUES (?, ?, ?, ?, ?)",
                                (*key, json.dumps(result, ensure_ascii=False)))
        return result


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
