#!/usr/bin/env python3
"""Convert vetted native-backend audits into hash-bound training sidecars.

This converter is deliberately conservative. It emits a disposition for every
manifest row and uses ``exclude_reason`` when a backend is known not to have
transcribed part of the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "audio"
TONE_MARKS = {"\u0300": 2, "\u0302": 3, "\u0301": 4, "\u030c": 5}
PITCH_LETTERS = set("˥˦˧˨˩")

THAI_PHONES = tuple(sorted({
    "tɕʰ", "tɕ", "pʰ", "tʰ", "kʰ",
    "iə", "ɯə", "uə",
    "a", "aː", "i", "iː", "u", "uː", "ɯ", "ɯː", "e", "eː",
    "ɛ", "ɛː", "o", "oː", "ɔ", "ɔː", "ə", "əː",
    "b", "p", "m", "f", "d", "t", "n", "s", "r", "l", "k",
    "ŋ", "w", "j", "h", "ʔ",
}, key=len, reverse=True))

# OpenJTalk phones that a sokuon (っ, emitted as the closure phone "cl") can
# actually geminate: the obstruents. Japanese gemination is a held closure, so
# it needs a consonant with a closure to hold — vowels, glides, /r/ and the
# nasals cannot take one (a "long n" is the moraic nasal ん, phone "N", which is
# its own segment and never written っ). A closure sitting in front of anything
# outside this set is therefore not a geminate at all but a glottal cutoff, as
# in あっ、 — see the exclusion below.
JAPANESE_GEMINABLE = {
    "k", "ky", "g", "gy", "s", "sh", "z", "j", "t", "ty", "ch", "ts",
    "d", "dy", "h", "hy", "f", "b", "by", "p", "py", "v",
}

JAPANESE_PHONE_MAP = {
    "a": "a", "i": "i", "I": "i̥", "u": "ɯᵝ", "U": "ɯ̥ᵝ",
    "e": "e", "o": "o", "k": "k", "ky": "kʲ", "g": "ɡ",
    "gy": "ɡʲ", "s": "s", "sh": "ɕ", "z": "z", "j": "dʑ",
    "t": "t", "ty": "tʲ", "ch": "tɕ", "ts": "ts", "d": "d",
    "dy": "dʲ", "n": "n", "ny": "ɲ", "h": "h", "hy": "ç",
    "f": "ɸ", "b": "b", "by": "bʲ", "p": "p", "py": "pʲ",
    "m": "m", "my": "mʲ", "y": "j", "r": "ɾ", "ry": "ɾʲ",
    "w": "w", "N": "ɴ", "v": "v",
}


# Mora-bearing Japanese phones. Every mora carries a pitch target, so the
# accent factor labels all of them — the five vowels (including the devoiced
# I/U, whose mapped forms still start with the plain vowel letter) and the
# moraic nasal ん, which OpenJTalk gives its own mora index. The sokuon is the
# one mora with no segment of its own: gemination is encoded as "ː" on the
# following consonant, so there is no token to hang a label on.
JAPANESE_MORA_VOWELS = "aiɯeo"
JAPANESE_MORAIC_NASAL = "ɴ"

# IPADIC parts of speech that cannot begin an utterance. A clip whose first
# content word is one of these was cut out of the middle of a longer phrase —
# Pimsleur's backward-buildup drills ("…ません" / "…かりません" / "わかりません")
# are full of them. The phones are still what was said; the *accent* is not,
# because OpenJTalk computes an accent phrase for the fragment as if it were a
# standalone word.
JAPANESE_DEPENDENT_POS = {"助詞", "助動詞"}

# Accent supervision also needs the transcript to name the right words. Pitch
# accent is lexical, and Japanese homophones are exactly where it contrasts
# (箸 HL vs 橋 LH), so a Whisper transcript that guessed the kanji from its
# language-model prior yields an accent label for a word that may not have
# been said. This bar is stricter than the corpus-wide --min-whisper-logprob
# (-0.7) because that one only has to protect the phone labels.
JAPANESE_ACCENT_MIN_WHISPER_LOGPROB = -0.35


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def tokyo_pitch_level(mora: int, nucleus: int, phrase_moras: int) -> int:
    """Realized pitch of one mora of a Tokyo-Japanese accent phrase: 0=L, 1=H.

    Tokyo accent is fully determined by the nucleus position plus the
    initial-lowering rule (the first mora is low unless the accent is on it,
    because the phrase must *rise* onto mora 2):

        heiban   (nucleus 0)  L H H H …          — no fall
        atamadaka(nucleus 1)  H L L L …
        nakadaka (nucleus a)  L H…H L L …        — falls after mora a
        odaka    (nucleus = N) L H H…H           — falls onto the next phrase

    Note odaka and heiban produce an identical in-clip contour, and that is
    correct rather than a loss: they differ only in what happens on the
    following particle. Encoding the nucleus position instead would force a
    distinction the audio of this phrase does not carry, which is exactly the
    kind of cross-clip fact the labels are not allowed to assert.
    """
    if phrase_moras <= 1:
        return 1 if nucleus == 1 else 0
    if nucleus == 1:
        return 1 if mora == 1 else 0
    if nucleus == 0:
        return 0 if mora == 1 else 1
    return 1 if 2 <= mora <= nucleus else 0


def japanese_accent_withhold_reason(rec: dict, features: list[dict]) -> str | None:
    """Why this row's citation accent is not usable as a training target.

    Returns None when the accent labels can be trusted. The phones are
    unaffected either way — this only withholds the accent factor.
    """
    first = next((f for f in features if f["pos"] != "記号" and int(f["mora_size"]) > 0), None)
    if first is None:
        return "japanese_accent_no_content_word"
    if first["pos"] in JAPANESE_DEPENDENT_POS or first["pos_group1"] == "接尾":
        return "japanese_accent_head_truncated"
    logprob = rec.get("whisper_avg_logprob")
    if logprob is not None and logprob < JAPANESE_ACCENT_MIN_WHISPER_LOGPROB:
        return "japanese_accent_low_asr_confidence"
    return None


def thai_labels(rec: dict, audit: dict) -> dict:
    source_latin = re.findall(r"[A-Za-z]+", rec["sentence"])
    raw = audit["output"]["ipa_with_tone"]
    if source_latin:
        # The frontend has no English/name G2P path and usually copies these
        # spans. Exclude every mixed-script sentence, including ambiguous
        # one-letter abbreviations, rather than guessing which happened to be
        # nativized by a lexicon entry.
        return {"exclude_reason": "thai_mixed_latin_script:" + ",".join(source_latin)}

    text = unicodedata.normalize("NFD", raw)
    phonemes: list[str] = []
    tone: list[int | None] = []
    stress: list[int] = []
    word_vowels: list[int] = []

    def finish_word() -> None:
        if word_vowels:
            # Standard Thai lexical words carry primary stress on their final
            # syllable. Vachana/TLTK preserves its tokenizer's word boundaries
            # as spaces, so attach stress to that word's final vowel-bearing
            # phone rather than suppressing the existing stress head.
            stress[word_vowels[-1]] = 1
            word_vowels.clear()

    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace() or unicodedata.category(char).startswith("P") or char.isdigit():
            finish_word()
            pos += 1
            continue
        phone = None
        embedded_tone = None
        # Vachana writes the tone mark immediately after the first vowel
        # character (îː, ìə), i.e. inside our multi-codepoint phone token.
        for candidate in THAI_PHONES:
            if candidate[0] not in "aiuɯeɛoɔə":
                continue
            if text.startswith(candidate, pos):
                phone = candidate
                break
            mark_pos = pos + 1
            if (text.startswith(candidate[0], pos) and mark_pos < len(text)
                    and text[mark_pos] in TONE_MARKS
                    and text.startswith(candidate[1:], mark_pos + 1)):
                phone = candidate
                embedded_tone = TONE_MARKS[text[mark_pos]]
                break
        if phone is None:
            phone = next((p for p in THAI_PHONES if text.startswith(p, pos)), None)
        if phone is None:
            raise ValueError(f"unparsed Thai IPA at {text[pos:pos+20]!r} in {rec['file']}")
        pos += len(phone) + (embedded_tone is not None)
        label = embedded_tone if embedded_tone is not None else (
            1 if phone[0] in "aiuɯeɛoɔə" else None
        )
        while pos < len(text) and text[pos] in TONE_MARKS:
            label = TONE_MARKS[text[pos]]
            pos += 1
        phonemes.append(phone)
        tone.append(label)
        stress.append(0)
        if phone[:1] in "aiuɯeɛoɔə":
            word_vowels.append(len(phonemes) - 1)
    finish_word()
    return {
        "phonemes": phonemes, "stress": stress, "tone": tone,
        "stress_source": "thai-final-syllable-per-tltk-word",
    }


def mandarin_labels(rec: dict, audit: dict) -> dict:
    phonemes: list[str] = []
    tone: list[int | None] = []
    for syllable in audit["output"]["syllables"]:
        if syllable is None:
            continue
        pinyin = syllable["pinyin"]
        tone_number = int(pinyin[-1])
        variant = syllable["ipa_variants"][0]
        assigned = False
        for raw_phone in variant:
            has_pitch = any(c in PITCH_LETTERS for c in raw_phone)
            phone = "".join(c for c in raw_phone if c not in PITCH_LETTERS)
            phonemes.append(phone)
            tone.append(tone_number if has_pitch and not assigned else None)
            assigned |= has_pitch
        if not assigned:
            # Neutral tone has no contour in pinyin-to-ipa; attach it to the
            # first vowel-like phone in this syllable.
            start = len(phonemes) - len(variant)
            for index in range(start, len(phonemes)):
                # ɚ is here for neutral-tone erhua (哪儿 -> na3 + er5). Toned
                # erhua never reaches this branch: er2 carries a contour, so
                # the pitch-letter pass above already claimed it. Only the
                # toneless suffix lands here, and a rhotic schwa is as
                # vowel-like a tone bearer as the plain ə beside it.
                if phonemes[index][:1] in "iyɨɯuɪʊeɤoəɚɛaɑɔɻɹzʐ":
                    tone[index] = tone_number
                    assigned = True
                    break
        if not assigned:
            raise ValueError(f"no Mandarin tone bearer in {pinyin!r}")
    return {"phonemes": phonemes, "stress": [0] * len(phonemes), "tone": tone}


def hindi_labels(rec: dict, audit: dict) -> dict:
    # g2p refuses text it cannot fully label (digits, Latin script) instead of
    # emitting labels with a hole where the audio has speech.
    if audit["output"].get("exclude_reason"):
        return {"exclude_reason": audit["output"]["exclude_reason"]}
    phonemes: list[str] = []
    stress: list[int] = []
    syllables: list[dict] = []
    phone_offset = 0
    for word_index, word in enumerate(audit["output"]["words"]):
        word_phones = word["phonemes"]
        word_stress = word["stress"]
        if len(word_phones) != len(word_stress):
            raise ValueError(f"Hindi word stress misalignment in {rec['file']}")
        phonemes.extend(word_phones)
        stress.extend(word_stress)
        syllable_cursor = 0
        for syllable in word["syllables"]:
            if (syllable["start"] != syllable_cursor
                    or not syllable["start"] <= syllable["nucleus"] < syllable["end"]
                    or syllable["end"] > len(word_phones)
                    or word_stress[syllable["nucleus"]] != syllable["stress"]):
                raise ValueError(f"Hindi syllable invariant failed in {rec['file']}")
            syllables.append({
                **syllable,
                "word": word_index,
                "start": syllable["start"] + phone_offset,
                "end": syllable["end"] + phone_offset,
                "nucleus": syllable["nucleus"] + phone_offset,
                "source": "roy-2017-rules-on-schwa-hin",
            })
            syllable_cursor = syllable["end"]
        if word_phones and syllable_cursor != len(word_phones):
            raise ValueError(f"Hindi syllables do not cover word in {rec['file']}")
        phone_offset += len(word_phones)
    if not phonemes:
        return {"exclude_reason": "hindi_no_devanagari_phones"}
    return {
        "phonemes": phonemes, "stress": stress, "syllables": syllables,
        "stress_source": "roy-2017-rules-on-schwa-hin",
    }


def japanese_labels(rec: dict, audit: dict) -> dict:
    native = audit["output"]["phones"]
    contexts = audit["output"]["fullcontext"]
    features = audit["output"].get("njd_features")
    if features is None:
        raise ValueError(f"Japanese audit schema predates NJD features: {rec['file']}")
    accent_phrases: list[dict] = []
    feature_boundary = True
    for feature in features:
        moras = int(feature["mora_size"])
        if moras <= 0:
            # Punctuation can force a JPCommon short pause even when the next
            # dictionary word retains chain_flag=1.
            feature_boundary = True
            continue
        if feature_boundary or not accent_phrases or int(feature["chain_flag"]) != 1:
            accent_phrases.append({"nucleus": int(feature["acc"]), "moras": moras})
        else:
            # JPCommon keeps the first word's accent for a chained phrase and
            # appends subsequent word moras to that same accent phrase.
            accent_phrases[-1]["moras"] += moras
        feature_boundary = False
    context_phones = [re.search(r"\-([^+]+)\+", label).group(1) for label in contexts]
    if context_phones[1:-1] != native:
        raise ValueError(f"OpenJTalk phone/context misalignment in {rec['file']}")
    phonemes: list[str] = []
    pitch_accent: list[dict | None] = []
    geminate = False
    phrase = 0
    breath_group = 0
    current_phrase_key = None
    for native_index, phone in enumerate(native):
        if phone in {"pau", "sil"}:
            breath_group += 1
            current_phrase_key = None
            continue
        if phone == "cl":
            if geminate:
                raise ValueError(f"double OpenJTalk closure in {rec['file']}")
            geminate = True
            continue
        mapped = JAPANESE_PHONE_MAP.get(phone)
        if mapped is None:
            raise ValueError(f"unknown OpenJTalk phone {phone!r} in {rec['file']}")
        if geminate:
            if phone not in JAPANESE_GEMINABLE:
                # The closure has nothing it can lengthen, so it's a glottal
                # cutoff — same unlabelable case as a sokuon at the very end of
                # an utterance. Note this is NOT simply "a pause intervened":
                # 'かっ た' has a spurious space, and its closure really does
                # geminate the following /t/, so a pause-based rule would throw
                # away a correct row. Keying on the phone keeps that one and
                # drops only 'あっ、日本語…', where the closure would otherwise
                # have produced "nː" — a long nasal, which Japanese does not
                # have and the tokenizer vocab does not contain.
                return {"exclude_reason": "japanese_ungeminable_closure"}
            mapped += "ː"
            geminate = False
        label = contexts[native_index + 1]
        a_match = re.search(r"/A:-?\d+\+(\d+)\+\d+", label)
        f_match = re.search(r"/F:(\d+)_(\d+)#", label)
        f_position = re.search(r"/F:\d+_\d+#.*?@(\d+)_\d+\|", label)
        if not a_match or not f_match or not f_position:
            raise ValueError(f"missing OpenJTalk accent fields in {rec['file']}")
        mora = int(a_match.group(1))
        phrase_key = (breath_group, int(f_position.group(1)))
        if phrase_key != current_phrase_key:
            phrase += 1
            current_phrase_key = phrase_key
        phonemes.append(mapped)
        if not 1 <= phrase <= len(accent_phrases):
            raise ValueError(f"OpenJTalk accent-phrase mismatch in {rec['file']}")
        phrase_info = accent_phrases[phrase - 1]
        bears_mora = (
            mapped[:1] in JAPANESE_MORA_VOWELS or mapped == JAPANESE_MORAIC_NASAL
        )
        pitch_accent.append({
            "phrase": phrase,
            "mora": mora,
            "phrase_moras": phrase_info["moras"],
            # Preserve the NJD value: unlike HTS's F field, 0 remains
            # heiban and is distinguishable from a final-mora accent.
            "nucleus": phrase_info["nucleus"],
            # The trained target. `nucleus` is kept alongside it so the
            # acoustic validator can reason about where the fall should be.
            "level": tokyo_pitch_level(
                mora, phrase_info["nucleus"], phrase_info["moras"],
            ),
            "source": "openjtalk-citation",
        } if bears_mora else None)
    if geminate:
        # An utterance-final sokuon with nothing after it to geminate. This is
        # real colloquial Japanese, not corrupt input — Pimsleur dialogue has
        # 早っ / 遅っ / さらっ, where the っ is an abrupt glottal cutoff rather
        # than the first half of a long consonant. We have no phone for that:
        # gemination is encoded as "ː" on the FOLLOWING consonant, and there
        # isn't one. Inventing a glottal stop would put a symbol in the labels
        # that the tokenizer vocab never sees elsewhere, so exclude the row and
        # say why, the same way Thai mixed-script and Hindi non-Devanagari rows
        # are handled. Four rows across the corpus at time of writing, all of
        # them sub-second interjections that min_whisper_logprob would drop
        # anyway.
        return {"exclude_reason": "japanese_terminal_sokuon"}
    if phrase != len(accent_phrases):
        # NJD and HTS disagreeing on how many accent phrases an utterance has
        # is the same class of problem as them disagreeing on the values inside
        # one (handled just below), so treat it the same way instead of failing
        # the whole build. The phones are unaffected — only which phrase a mora
        # belongs to is in doubt — and phone supervision is the signal we
        # actually train on. Trailing chōonpu is the usual trigger: 'あいにくー。'
        # gets one phrase from HTS and two from NJD. Four rows in 36k.
        phrase_count_mismatch = True
    else:
        phrase_count_mismatch = False
    result = {
        "phonemes": phonemes,
        "stress": [0] * len(phonemes),
    }
    inconsistent = [
        item for item in pitch_accent
        if item is not None and not (
            1 <= item["mora"] <= item["phrase_moras"]
            and 0 <= item["nucleus"] <= item["phrase_moras"]
        )
    ]
    if inconsistent or phrase_count_mismatch:
        # Some punctuation/foreign-name chains make NJD and HTS disagree on
        # phrase boundaries.  Phones remain valid, but an accent label from
        # either interpretation would be invented.  Preserve partial phone
        # supervision and explicitly withhold only this factor.
        result["pitch_accent_exclude_reason"] = "njd_hts_phrase_boundary_mismatch"
        return result

    # The parse is self-consistent; the remaining question is whether the text
    # it parsed is the whole utterance and the right words.
    withhold = japanese_accent_withhold_reason(rec, features)
    if withhold is not None:
        result["pitch_accent_exclude_reason"] = withhold
    else:
        result["pitch_accent"] = pitch_accent
    return result


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


CONFIG = {
    # The g2p crate's port of schwa-stress-hin plus the audited corrections;
    # `schwa-stress-hin` (the Python original) stays in PROVIDERS for
    # reproducing the 2026-08 labels.
    "hin": ("g2p-hin", hindi_labels),
    # The same vachana-thai, run by the g2p crate as a pinned uv project, with
    # thai_labels' parsing in Rust (`vachana-thai` stays in PROVIDERS).
    "tha": ("g2p-tha", g2p_flat_labels),
    # The g2p crate's port of g2pM + pinyin_to_ipa (`g2pm-ipa` stays in
    # PROVIDERS; labels are identical where both label a row).
    "zho-hans": ("g2p-zho", g2p_flat_labels),
    # OpenJTalk via jpreprocess inside the g2p crate (`pyopenjtalk` stays in
    # PROVIDERS; 99.2% of rows identical, see _g2p_jpn).
    "jpn": ("g2p-jpn", g2p_japanese_labels),
    # g2pk2 + mecab-ko inside the g2p crate (see _g2p_kor). No Korean audio
    # has been collected yet; this is the label chain for when it is.
    "kor": ("g2p-kor", g2p_flat_labels),
}


def build_sidecar(lang: str, *, data_root: Path = DATA_ROOT,
                  output: Path | None = None) -> Path:
    """Convert the language's canonical audit into its hash-bound sidecar."""
    provider, convert = CONFIG[lang]
    lang_dir = data_root / lang
    manifest = read_jsonl(lang_dir / "manifest.jsonl")
    audits = {r["file"]: r for r in read_jsonl(lang_dir / f"g2p_audit_{provider}.jsonl")}
    output = output or lang_dir / f"phoneme_backend_{provider}.jsonl"
    dispositions = []
    excluded = 0
    for rec in manifest:
        audit = audits.get(rec["file"])
        digest = hashlib.sha256(rec["sentence"].encode()).hexdigest()
        if audit is None or audit["sentence_sha256"] != digest or audit.get("error"):
            raise ValueError(f"missing/stale/failed audit for {lang}/{rec['file']}")
        labels = convert(rec, audit)
        excluded += "exclude_reason" in labels
        dispositions.append({
            "file": rec["file"], "sentence_sha256": digest,
            "backend": provider, **labels,
        })
    with output.open("w") as target:
        for disposition in dispositions:
            target.write(json.dumps(disposition, ensure_ascii=False) + "\n")
    print(f"{lang}: wrote {len(dispositions)} dispositions to {output}; excluded={excluded}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", required=True, choices=sorted(CONFIG))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_sidecar(args.lang, data_root=args.data_root, output=args.output)


if __name__ == "__main__":
    main()
