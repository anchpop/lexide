"""The frontend: yap's sequence format (language_utils::sentence_phonemes), punctuation and word occurrences.

    PYTHONPATH=. python -m pytest tests

The non-English sentences in tests/data are excerpts of yap pack dumps
(dump_sentence_phonemes: jpn_for_eng, zho-hans_for_eng, tha_for_eng). They are
tokenized in a subprocess whose frontend is their language's (VOCA_LANGUAGE).
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from voca.text import (BETWEEN_GRAMS, NUCLEUS, PHRASE_START, POS_ID, POS_OTHER, PUNCTUATION, RARITY_BUCKETS, SP, SPECIALS, WITHIN_GRAM,
                       frontend_fingerprint, parse_phone, punctuation_of, rarity_bucket, tokenize)

DATA = Path(__file__).with_name("data")
ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("token, parsed", [
    ("a", ("a", 0, 0, False)),
    ("ˈaɪ", ("aɪ", 1, 0, False)),
    ("ˌʌ", ("ʌ", 2, 0, False)),
    ("a3", ("a", 0, 3, False)),
    ("ˈa5", ("a", 1, 5, False)),          # Thai: stress and tone
    ("ɻ̩2", ("ɻ̩", 0, 2, False)),          # a combining mark before the tone
    ("aꜜ", ("a", 0, 0, True)),            # Japanese accent nucleus
    ("ɯ̥ᵝ", ("ɯ̥ᵝ", 0, 0, False)),         # devoiced: part of the phoneme
    ("kʰ", ("kʰ", 0, 0, False)),
])
def test_parse_phone(token, parsed):
    assert parse_phone(token) == parsed


def gram(words, phonemes, pos=None, count=1):
    """A pack gram; every word a noun (punctuation PUNCT) unless ``pos`` says otherwise."""
    return {"words": words, "phonemes": phonemes, "count": count,
            "pos": pos or ["NOUN" if any(c.isalnum() for c in w) else "PUNCT" for w in words]}


def english(*grams):
    return tokenize([gram(*g) for g in grams])


def test_punctuation_inside_a_gram_falls_on_its_word_break():
    p = english((["look", ",", "I", "'m"], "l ˈʊ k # aɪ m"), (["tired"], "t ˈaɪ ɚ d"), (["."], ""))
    assert " ".join(p.symbols) == "<bos> l ʊ k , aɪ m <sp> t aɪ ɚ d . <eos>"
    assert p.words == ("look", "i'm", "tired")


def test_clitics_join_the_word_before_them():
    p = english((["you", "'ll", "see"], "j uː l # s ˈiː"), (["I", "do", "n't"], "aɪ # d ˈoʊ n t"))
    assert p.words == ("you'll", "see", "i", "don't")
    assert p.word == (-1, 0, 0, 0, -1, 1, 1, -1, 2, -1, 3, 3, 3, 3, -1)


def test_a_hyphen_is_not_a_boundary():
    assert english((["t", "-", "shirt"], "t ˈiː ʃ ˈɜː t")).words == ("t-shirt",)
    p = english((["part", "-", "time", "job"], "p ˈɑːɹ tː ˈaɪ m # dʒ ˈɑː b"))
    assert p.words == ("part-time", "job") and "—" not in p.symbols


def test_unmatched_groups_are_not_named():
    p = english((["she", "can", "not"], "ʃ iː # k æ n ˈɑː t"), ([","], ""), (["really"], "ɹ ˈiː l i"))
    assert p.words == (None, None, "really")
    assert p.word_names()[1:3] == [None, None] and "," in p.symbols


def test_repeated_words_are_distinct_occurrences():
    p = english((["over", "and", "over"], "ˈoʊ v ɚ # æ n d # ˈoʊ v ɚ"))
    assert p.words == ("over", "and", "over") and max(p.word) == 2


def test_english_has_no_tone_or_accent():
    for line in open(ROOT / "data/eng_script.jsonl", encoding="utf-8"):
        p = tokenize(json.loads(line)["grams"])
        assert not any(p.tone) and not any(p.accent)


def serialize(symbol, stress, tone, accent):
    """The sequence-format token back from a phone and its features (PhonemeToken's Display)."""
    return ("% " if accent & PHRASE_START else "") + ("", "ˈ", "ˌ")[stress] + symbol + (str(tone) if tone else "") + ("ꜜ" if accent & NUCLEUS else "")


def in_language(language, script, *args):
    """Run ``script`` in a process whose frontend is ``language``'s (VOCA_LANGUAGE); its stdout as JSON."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "VOCA_LANGUAGE": language}
    result = subprocess.run([sys.executable, "-c", script, *map(str, args)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def other_languages():
    """Each language's sentences tokenized by its own frontend (its inventory in voca/inventories)."""
    script = ("import json, sys; from voca.text import ALWAYS_VOICED, VOWELS, tokenize\n"
              "rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]\n"
              "print(json.dumps({'vowels': VOWELS, 'always_voiced': ALWAYS_VOICED,\n"
              "                  'rows': [{**r, 'tokens': tokenize(r['grams']).__dict__} for r in rows]}, ensure_ascii=False))")
    return {name: in_language(language, script, DATA / f"{name}_sentences.jsonl")
            for name, language in (("jpn", "jpn"), ("cmn", "zho-hans"), ("tha", "tha"))}


def test_each_language_has_its_own_frontend():
    """One inventory per language: its own phones, and a fingerprint no other language's model shares."""
    script = "import json; from voca.text import LANGUAGE, PHONES, frontend_fingerprint; print(json.dumps([LANGUAGE, PHONES, frontend_fingerprint()]))"
    english, japanese = in_language("eng", script), in_language("jpn", script)
    assert english[0] == "eng" and japanese[0] == "jpn"
    assert "ð" in english[1] and "ð" not in japanese[1] and "ɯ̥ᵝ" in japanese[1]
    assert english[2] != japanese[2]


WORDS = ("import json, sys; from voca.text import POS_ID, tokenize\n"
         "p = tokenize([{'words': w, 'pos': t, 'count': 1, 'phonemes': ph} for w, t, ph in json.loads(sys.argv[1])])\n"
         "print(json.dumps([p.words, [POS_ID[t] for _, ts, _ in json.loads(sys.argv[1]) for t in ts], sorted(set(p.pos) - {0})]))")


@pytest.mark.parametrize("language, grams, words, tag", [
    ("fra", [(["je", "l'", "ai"], ["PRON", "PRON", "AUX"], "ʒ ə # l e")], ["je", "l'ai"], "AUX"),
    ("fra", [(["c'", "est", "bien"], ["PRON", "VERB", "ADV"], "s ɛ # b j ˈɛ̃")], ["c'est", "bien"], "VERB"),
    ("ita", [(["l'", "uomo"], ["DET", "NOUN"], "l w ˈɔ m o")], ["l'uomo"], "NOUN"),
    # Korean particles and the copula are written on the word before...
    ("kor", [(["갈", "곳", "은"], ["VERB", "NOUN", "ADP"], "k a l # k͈ o s ɯ n")], ["갈", "곳은"], "NOUN"),
    # ...but a spaced auxiliary is its own word
    ("kor", [(["좋지", "않아"], ["ADJ", "AUX"], "tɕ o tɕʰ i # a n a")], ["좋지", "않아"], "AUX"),
])
def test_words_written_as_one(language, grams, words, tag):
    """Words yap splits that the language writes and g2p speaks as one are one named word."""
    names, _, tags = in_language(language, WORDS, json.dumps(grams, ensure_ascii=False))
    assert names == words
    assert POS_ID[tag] in tags


def test_english_keeps_a_trailing_apostrophe_word_apart():
    p = tokenize([{"words": ["the", "students'", "books"], "pos": ["DET", "NOUN", "NOUN"], "count": 1,
                   "phonemes": "ð ə # s t ˈuː d ə n t s # b ˈʊ k s"}])
    assert p.words == ("the", "students'", "books")


SYMBOLS_OF = ("import json, sys; from voca.text import tokenize\n"
              "p = tokenize([{'words': w, 'pos': t, 'count': 1, 'phonemes': ph} for w, t, ph in json.loads(sys.argv[1])])\n"
              "print(json.dumps([p.symbols, p.words]))")


@pytest.mark.parametrize("language, grams, symbols", [
    # a quote as its own gram, at the start and before a comma: «Pour demain», répondis-je.
    ("fra", [(["«"], ["PUNCT"], ""), (["pour", "demain"], ["ADP", "ADV"], "p u ʁ # d ə m ɛ̃"), (["»"], ["PUNCT"], ""), ([","], ["PUNCT"], ""),
             (["répondis", "-", "je"], ["VERB", "PUNCT", "PRON"], "ʁ e p ɔ̃ d i ʒ ə"), (["."], ["PUNCT"], "")],
     ["<bos>", '"', "p", "u", "ʁ", "<sp>", "d", "ə", "m", "ɛ̃", '"', ",", "ʁ", "e", "p", "ɔ̃", "d", "i", "ʒ", "ə", ".", "<eos>"]),
    # a quote inside a gram, between its words: “เข้าใจแล้ว” ฉันพูด
    ("tha", [(["“"], ["PUNCT"], ""), (["เข้าใจ"], ["VERB"], "kʰ a w3 # tɕ a j1"), (["แล้ว", "”", "ฉัน", "พูด"], ["ADV", "PUNCT", "PRON", "VERB"],
                                                                                  "l ɛː w4 # tɕʰ a n5 # pʰ uː t3")],
     ["<bos>", '"', "kʰ", "a", "w", "<sp>", "tɕ", "a", "j", "<sp>", "l", "ɛː", "w", '"', "tɕʰ", "a", "n", "<sp>", "pʰ", "uː", "t", "<eos>"]),
])
def test_quotes(language, grams, symbols):
    """Every language's quotation marks are the quote token, where yap writes them: as a gram of their own or inside one."""
    assert in_language(language, SYMBOLS_OF, json.dumps(grams, ensure_ascii=False))[0] == symbols


@pytest.mark.parametrize("mark, alias", [("।", "."), ("„", '"'), ("―", "—"), ("・・・", "…"), ("．", "."), ("『", '"'), ("«", '"'), ("»", '"'),
                                         ("“", '"'), ("”", '"'), ("「", '"'), ("」", '"'), ("』", '"'), ('"', '"')])
def test_punctuation_aliases(mark, alias):
    assert punctuation_of([mark]) == [alias]


@pytest.mark.parametrize("language", ["jpn", "cmn", "tha"])
def test_sequences_round_trip(other_languages, language):
    """Every phone keeps its stress, tone, accent nucleus and accent-phrase start: serialized back,
    the tokens give the pack's sequence (with ``#`` as the word-boundary token)."""
    for row in other_languages[language]["rows"]:
        t = row["tokens"]
        phones = [serialize(s, *f) for s, *f in zip(t["symbols"], t["stress"], t["tone"], t["accent"]) if s not in PUNCTUATION_OR_SPECIAL]
        expected = [x for g in row["grams"] for x in g["phonemes"].split() if x != "#"]
        assert " ".join(phones).split() == expected, row["text"]
        breaks = sum(g["phonemes"].split().count("#") for g in row["grams"])
        assert t["symbols"].count(SP) >= breaks


def test_japanese_accent(other_languages):
    first = other_languages["jpn"]["rows"][0]["tokens"]     # 私は | トラクター: % w a t a ɕ i w a % t o ɾ aꜜ ...
    assert first["symbols"][1:4] == ["w", "a", "t"] and first["accent"][1:4] == [PHRASE_START, 0, 0]
    nucleus = first["symbols"].index("ɾ") + 1
    assert first["symbols"][nucleus] == "a" and first["accent"][nucleus] == NUCLEUS
    # "% aꜜ ɾ ɯᵝ": a phrase whose first mora is its nucleus
    second = other_languages["jpn"]["rows"][1]["tokens"]
    assert PHRASE_START | NUCLEUS in second["accent"]
    # 、 ends a phrase as a comma does
    assert "," in other_languages["jpn"]["rows"][2]["tokens"]["symbols"]


def test_japanese_devoiced_vowels_are_not_forced_voiced(other_languages):
    inventory = other_languages["jpn"]
    assert "ɯ̥ᵝ" in inventory["vowels"] and "ɯ̥ᵝ" not in inventory["always_voiced"] and "ɯᵝ" in inventory["always_voiced"]


def test_tones(other_languages):
    first = other_languages["cmn"]["rows"][0]["tokens"]     # ʂ ɻ̩2 tɕʰ i1 # s w eɪ4 | l ɤ5
    phones = [(s, tone) for s, tone in zip(first["symbols"], first["tone"]) if s not in PUNCTUATION_OR_SPECIAL]
    assert phones == [("ʂ", 0), ("ɻ̩", 2), ("tɕʰ", 0), ("i", 1), ("s", 0), ("w", 0), ("eɪ", 4), ("l", 0), ("ɤ", 5)]
    thai = other_languages["tha"]["rows"][0]["tokens"]      # w ˈa5 ŋ # w ˈaː3 ...
    assert (thai["symbols"][2], thai["stress"][2], thai["tone"][2]) == ("a", 1, 5)
    assert "," in other_languages["cmn"]["rows"][1]["tokens"]["symbols"]      # ，


def test_fingerprint_is_versioned():
    assert frontend_fingerprint() != "e0837f47d19e118aa04859251f7e8297ebcc83655fa479ef7ccc07b63be36aa6"   # version 1 (symbols only)


def phone_features(p, name):
    """Per word occurrence, the set of values feature ``name`` takes on its phones."""
    values = {}
    for w, value in zip(p.word, getattr(p, name)):
        if w >= 0:
            values.setdefault(w, set()).add(value)
    return [values[w] for w in sorted(values)]


def test_part_of_speech_per_spoken_word():
    p = english((["I", "do", "n't"], "aɪ # d ˈoʊ n t", ["PRON", "AUX", "PART"]), (["know"], "n ˈoʊ", ["VERB"]),
                (["Tom"], "t ˈɑː m", ["PROPN"]), (["hmm"], "h m", ["WEIRD"]))
    assert phone_features(p, "pos") == [{POS_ID["PRON"]}, {POS_ID["AUX"]}, {POS_ID["VERB"]}, {POS_ID["PROPN"]}, {POS_OTHER}]
    assert all(pos == 0 for s, pos in zip(p.symbols, p.pos) if s in SPECIALS)   # only phones have one
    # One group for two words keeps the first's; groups that cannot be matched to words get none.
    assert phone_features(english((["can", "not"], "k æ n ˈɑː t", ["AUX", "PART"])), "pos") == [{POS_ID["AUX"]}]
    assert phone_features(english((["she", "can", "not"], "ʃ iː # k æ n ˈɑː t", ["PRON", "AUX", "PART"])), "pos") == [{0}, {0}]


def test_rarity_covers_the_whole_gram():
    p = english((["the", "cat"], "ð ə # k ˈæ t", ["DET", "NOUN"], 174), (["Tom"], "t ˈɑː m", ["PROPN"], 0), (["."], "", ["PUNCT"], 0))
    assert phone_features(p, "rarity") == [{rarity_bucket(174)}, {rarity_bucket(174)}, {1}]
    assert rarity_bucket(0) == 1 and rarity_bucket(1) == 2 and rarity_bucket(174) == 1 + 7 and rarity_bucket(10 ** 9) == RARITY_BUCKETS
    assert p.gram[1:4] == (0, 0, -1) and max(p.gram) == 1


def test_boundary_type_tells_chunks_apart():
    p = english((["look", ",", "I", "'m"], "l ˈʊ k # aɪ m"), (["so", "very"], "s oʊ # v ɛ ɹ i"), (["tired"], "t ˈaɪ ɚ d"), (["."], ""))
    breaks = [(s, b) for s, b in zip(p.symbols, p.boundary) if b]
    assert breaks == [(",", WITHIN_GRAM), (SP, BETWEEN_GRAMS), (SP, WITHIN_GRAM), (SP, BETWEEN_GRAMS), (".", BETWEEN_GRAMS)]
    assert all(b for s, b in zip(p.symbols, p.boundary) if s == SP)


def test_words_are_runs_of_phones():
    """``voca.model.word_layout`` finds tokenize's word occurrences from the tokens alone, and phrases end at punctuation."""
    import torch
    from voca.model import word_layout
    for line in open(ROOT / "data/eng_script.jsonl", encoding="utf-8"):
        p = tokenize(json.loads(line)["grams"])
        assert word_layout(torch.tensor([p.ids]), torch.ones(1, len(p), dtype=torch.bool)).index[0].tolist() == list(p.word)
    p = english((["yes"], "j ɛ s"), ([","], ""), (["I", "do"], "aɪ # d uː"), (["?"], ""))
    words = word_layout(torch.tensor([p.ids]), torch.ones(1, len(p), dtype=torch.bool))
    assert words.mask[0].tolist()[:4] == [True, True, True, False]
    assert words.phrase[0, :3].tolist() == [0, 1, 1] and words.final[0, :3].tolist() == [True, False, True]
    assert words.same[0, 1, 2] and not words.same[0, 0, 1]
    z = words.standardize(torch.tensor([[5.0, 1.0, 3.0] + [0.0] * (len(p) - 3)]))[0, :3]
    assert z.tolist() == pytest.approx([0.0, -1.0, 1.0], abs=1e-3)


PUNCTUATION_OR_SPECIAL = set(SPECIALS) | set(PUNCTUATION)
