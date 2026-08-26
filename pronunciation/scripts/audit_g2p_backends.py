#!/usr/bin/env python3
"""Generate reproducible text-to-pronunciation backend samples for review.

This is deliberately upstream of `phonemes.jsonl`: it preserves each tool's
native output and suprasegmental metadata so a backend can be evaluated before
we write a lossy IPA/token mapping or treat it as training truth.

Examples:
  python scripts/audit_g2p_backends.py --lang hin --provider epitran-hin
  python scripts/audit_g2p_backends.py --lang zho-hans --provider pypinyin-ipa
  python scripts/audit_g2p_backends.py --lang zho-hans --provider g2pw-ipa
  python scripts/audit_g2p_backends.py --lang jpn --provider pyopenjtalk
  python scripts/audit_g2p_backends.py --lang tha --provider thaig2p-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "audio"


def _epitran_hin(text: str) -> dict[str, Any]:
    import epitran

    # Constructing Epitran is relatively expensive; cache it on the function.
    if not hasattr(_epitran_hin, "engine"):
        _epitran_hin.engine = epitran.Epitran("hin-Deva")
    return {"ipa": _epitran_hin.engine.transliterate(text)}


# Words whose non-nuqta फ is a genuine native aspirated stop /pʰ/. Everything
# else defaults to /f/: the 2026-08-24 listening audit (36 clips + 82
# word-level forced-choice judgments, scripts/audit_hin_aspiration.py and
# .work/hin_ph_word_verify.jsonl) found Perso-Arabic and English loans —
# the overwhelming majority of फ tokens (काफी, फिल्म, सिर्फ, फोन…) — are
# categorically [f] in every source, while espeak-lineage labels claimed
# [pʰ]. Hindi writes loan /f/ as फ़ (nuqta) in careful orthography, but real
# transcripts mostly drop the dot, so spelling alone can't decide.
#
# The native set below is closed-ish (Sanskrit phala/sphuṭ derivatives and
# onomatopoeia), matched by prefix to cover inflection. Native speakers
# variably fricativize even these ([pʰ]~[f], the ongoing merger — the audit
# heard both across clips of the same lemma); we keep the canonical /pʰ/
# for them and leave per-clip realization to a future acoustic narrowing
# rule (frication vs stop burst is a robust cue, same class as flapping).
# फव्वारा stays native: Persian origin but audited [pʰ] (nativized).
_HINDI_NATIVE_PH_PREFIXES = (
    "फिर", "फल", "फूल", "फैल", "फेंक", "फंस", "फँस", "फूट", "फट", "फोड़",
    "फाड़", "फाडऩ", "फिसल", "फुसफुस", "फुँफ", "फफ", "फांसी", "फाटक",
    "फीका", "फीकी", "फीत", "फेर", "फुफेर", "दुफेर", "फगवाड़", "फुलझड़",
    "फूँक", "फुस", "फूस", "फुहार", "फव्वार", "फलांग", "फलद",
)
_HINDI_NATIVE_PH_SUBSTRINGS = (
    # Non-initial native फ: Sanskrit compounds and reduplication.
    "सफल", "विफल", "क्षेत्रफल", "स्फीति", "स्फोट", "हेरफेर", "हेराफेर",
    "फटाफट", "दोफहर",
)


def _hindi_word_ph_is_native(word: str) -> bool:
    return word.startswith(_HINDI_NATIVE_PH_PREFIXES) or any(
        s in word for s in _HINDI_NATIVE_PH_SUBSTRINGS
    )


_HINDI_GOOGLE_TO_IPA = {
    "a": "ə", "aa": "aː", "i": "ɪ", "ii": "iː", "u": "ʊ", "uu": "uː",
    "e": "eː", "E": "ɛː", "o": "oː", "O": "ɔː",
    "k": "k", "kh": "kʰ", "g": "ɡ", "gh": "ɡʱ", "ng": "ŋ",
    "c": "t͡ʃ", "ch": "t͡ʃʰ", "j": "d͡ʒ", "jh": "d͡ʒʱ",
    "tt": "ʈ", "tth": "ʈʰ", "dd": "ɖ", "ddh": "ɖʱ",
    "t": "t̪", "th": "t̪ʰ", "d": "d̪", "dh": "d̪ʱ", "n": "n",
    "p": "p", "ph": "pʰ", "b": "b", "bh": "bʱ", "m": "m",
    "y": "j", "r": "ɾ", "l": "l", "v": "ʋ", "sh": "ʃ", "s": "s",
    "h": "ɦ", "z": "z", "f": "f", "rr": "ɽ", "rrh": "ɽʱ",
    "q": "q", "x": "x", "Gh": "ɣ", "Zh": "ʒ",
    "?": "ɲ",
}


def _schwa_hin(text: str) -> dict[str, Any]:
    """Apply the ACL 2020 Hindi schwa classifier, then map to broad IPA."""
    import numpy as np
    from joblib import load

    root = Path(os.environ.get(
        "HINDI_SCHWA_MODEL_ROOT",
        str(REPO_ROOT / ".work" / "schwa-deletion" / "hindi"),
    ))
    if not root.exists():
        # A /tmp checkout evaporates on reboot; keep the clone under .work
        # and (re)fetch it on demand so the Hindi chain is self-contained.
        import subprocess
        print(f"cloning aryamanarora/schwa-deletion (MIT) → {root.parent} ...")
        root.parent.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/aryamanarora/schwa-deletion",
             str(root.parent)],
            check=True,
        )
    if not hasattr(_schwa_hin, "engine"):
        # The published pickle used an old private sklearn module path.
        import sklearn.linear_model._logistic as logistic
        sys.modules.setdefault("sklearn.linear_model.logistic", logistic)
        sys.path.insert(0, str(root))
        import transliterate

        model = load(root / "models/logistic/logistic.joblib")
        phons = load(root / "models/logistic/logistic_phons.joblib")
        _schwa_hin.engine = (model, phons, transliterate)
    model, phons, transliterate = _schwa_hin.engine

    words = re.findall(r"[\u0900-\u097f]+", text)
    ipa_words = []
    deletion_masks = []
    for word in words:
        units = transliterate.transliterate(word)
        # Loanword फ → /f/ (see _HINDI_NATIVE_PH_PREFIXES above). The
        # transliterator emits "ph" for फ and "f" for फ़; only the
        # non-native words get the swap, at the unit level so the schwa
        # classifier's phonological features see the corrected consonant.
        if "ph" in units and not _hindi_word_ph_is_native(word):
            units = ["f" if unit == "ph" else unit for unit in units]
        rows = []
        for index, unit in enumerate(units):
            if unit != "a":
                continue
            features = []
            for pos in list(range(index - 5, index)) + list(range(index + 1, index + 6)):
                for feature in phons:
                    unit_features = transliterate.phonological_features.get(
                        units[pos], transliterate.phonological_features[transliterate.UNK_CHAR],
                    ) if 0 <= pos < len(units) else ()
                    features.append(int(
                        0 <= pos < len(units)
                        and feature in unit_features
                    ))
            rows.append(features)
        keep = model.predict(np.asarray(rows)) if rows else []
        schwa_index = 0
        kept_units = []
        mask = []
        for unit in units:
            if unit == "a":
                retained = bool(keep[schwa_index])
                mask.append(retained)
                schwa_index += 1
                if not retained:
                    continue
            kept_units.append(unit)
        ipa = []
        for unit in kept_units:
            if unit == "~":
                # Repeated/misordered bindu in a few source transcripts can
                # otherwise create double-nasalized vowels or impossible
                # nasalized consonant tokens. Preserve one vowel nasalization;
                # the malformed extra mark carries no additional contrast.
                if (ipa and any(vowel in ipa[-1] for vowel in _HINDI_GOOGLE_TO_IPA.values()
                                if vowel[:1] in "əɪiʊueɛaoɔ")
                        and "̃" not in ipa[-1]):
                    ipa[-1] += "̃"
            elif unit in _HINDI_GOOGLE_TO_IPA:
                ipa.append(_HINDI_GOOGLE_TO_IPA[unit])
        ipa_words.append(ipa)
        deletion_masks.append(mask)
    return {"words": ipa_words, "schwa_retained": deletion_masks}


_HINDI_VOWELS = {"ə", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː", "oː", "ɔː", "aː"}
_HINDI_LONG_VOWELS = {phone for phone in _HINDI_VOWELS if "ː" in phone}
_HINDI_SEMIVOWELS = {"j", "ɾ", "l", "ʋ"}


def _hindi_surface_stress(phones: list[str]) -> dict[str, Any]:
    """Apply Roy's 2017 surface-syllable weight rules to one phone word.

    This deliberately runs *after* the independent schwa classifier. It does
    not reuse the legacy A2S converter's segment stream, whose schwa decisions
    and loan-vowel handling disagree materially with the primary backend.
    """
    def oral(phone: str) -> str:
        return phone.replace("̃", "")

    nuclei = [index for index, phone in enumerate(phones) if oral(phone) in _HINDI_VOWELS]
    if not nuclei:
        return {"stress": [0] * len(phones), "syllables": []}

    starts = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        cluster = phones[left + 1:right]
        if len(cluster) <= 1:
            onset_length = len(cluster)
        elif len(cluster) == 2 and cluster[-1] in _HINDI_SEMIVOWELS:
            # Roy's c1+c2 case: stop+semivowel is a complex onset.
            onset_length = 2
        else:
            # Split the first consonant into the preceding coda; maximize the
            # remaining onset (also covers the paper's three-consonant rule).
            onset_length = len(cluster) - 1
        starts.append(right - onset_length)
    ends = starts[1:] + [len(phones)]

    syllables = []
    aligned = [0] * len(phones)
    for index, (start, end, nucleus) in enumerate(zip(starts, ends, nuclei)):
        coda_count = max(0, end - nucleus - 1)
        weight = (2 if oral(phones[nucleus]) in _HINDI_LONG_VOWELS else 1) + coda_count
        final = index == len(nuclei) - 1
        stressed = (
            weight >= 3
            or (weight == 2 and not final)
            or (weight == 1 and len(nuclei) == 2 and index == 0)
        )
        if stressed:
            aligned[nucleus] = 1
        syllables.append({
            "start": start, "end": end, "nucleus": nucleus,
            "moras": weight, "stress": int(stressed),
        })
    return {"stress": aligned, "syllables": syllables}


def _schwa_stress_hin(text: str) -> dict[str, Any]:
    segmental = _schwa_hin(text)
    stress_words = []
    for phones in segmental["words"]:
        stress_words.append({"phonemes": phones, **_hindi_surface_stress(phones)})
    return {
        "words": stress_words,
        "schwa_retained": segmental["schwa_retained"],
        "stress_source": "roy-2017-surface-weight-rules",
    }


def _pinyin_syllables_to_ipa(syllables: list[str | None]) -> list[dict | None]:
    from pinyin_to_ipa import pinyin_to_ipa

    result: list[dict | None] = []
    for syllable in syllables:
        if not syllable:
            result.append(None)
            continue
        variants = [list(v) for v in pinyin_to_ipa(syllable)]
        result.append({"pinyin": syllable, "ipa_variants": variants})
    return result


def _pypinyin_ipa(text: str) -> dict[str, Any]:
    from pypinyin import Style, lazy_pinyin

    syllables = lazy_pinyin(
        text, style=Style.TONE3, neutral_tone_with_five=True,
        errors=lambda chars: [None] * len(chars),
    )
    return {"syllables": _pinyin_syllables_to_ipa(syllables)}


def _g2pw_ipa(text: str) -> dict[str, Any]:
    from g2pw import G2PWConverter

    if not hasattr(_g2pw_ipa, "engine"):
        _g2pw_ipa.engine = G2PWConverter(
            style="pinyin", enable_non_tradional_chinese=True,
        )
    syllables = _g2pw_ipa.engine(text)[0]
    return {"syllables": _pinyin_syllables_to_ipa(syllables)}


def _g2pm_ipa(text: str) -> dict[str, Any]:
    """Mainland-Mandarin contextual polyphone baseline."""
    from g2pM import G2pM

    if not hasattr(_g2pm_ipa, "engine"):
        _g2pm_ipa.engine = G2pM()
    raw = _g2pm_ipa.engine(text, tone=True, char_split=True)
    syllables = []
    for item in raw:
        item = item.replace("u:", "v")
        # g2pM tags erhua 儿 two different ways depending on the word: 这儿 and
        # 一点儿 come back as a full "er2" syllable, but 哪儿 comes back as the
        # bare suffix "r5", which pinyin_to_ipa rejects outright ("Final
        # couldn't be detected"). Same phenomenon, same sound — only the
        # spelling differs — so normalize the suffix form to the syllable form
        # and let both take the path that already works. "er5" yields ɚ, which
        # is exactly the segment 这儿 already contributes, so this makes the
        # corpus internally consistent rather than inventing a label.
        # Without it, every 哪儿 fails the audit and the fail-closed sidecar
        # builder drops the whole language — 114 rows here, all of them 哪儿,
        # a word Pimsleur drills constantly.
        if item == "r5":
            item = "er5"
        syllables.append(item if re.fullmatch(r"[a-z]+[1-5]", item) else None)
    return {"syllables": _pinyin_syllables_to_ipa(syllables)}


def _pyopenjtalk(text: str) -> dict[str, Any]:
    import pyopenjtalk

    return {
        "phones": pyopenjtalk.g2p(text, kana=False).split(),
        # Preserve the frontend representation before JPCommon rewrites a
        # flat accent (acc=0) to acc=mora_size in HTS labels. Without this,
        # heiban and final-mora accent are indistinguishable downstream.
        "njd_features": pyopenjtalk.run_frontend(text),
        # Retain the labels verbatim. They encode mora/accent and phrase
        # context that cannot be recovered from the flat phone string.
        "fullcontext": pyopenjtalk.extract_fullcontext(text),
    }


def _thaig2p_v2(text: str) -> dict[str, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not hasattr(_thaig2p_v2, "engine"):
        tokenizer = AutoTokenizer.from_pretrained("pythainlp/thaig2p-v2.0")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "pythainlp/thaig2p-v2.0",
        )
        model.eval()
        _thaig2p_v2.engine = (tokenizer, model)
    tokenizer, model = _thaig2p_v2.engine
    inputs = tokenizer(text, return_tensors="pt")
    # The checkpoint's legacy generation config has a 512-token ceiling. On
    # newer Transformers, overriding that with max_new_tokens can prevent its
    # forced terminal token from firing and make every short sentence decode
    # all 512 positions. Thai IPA is normally only a few times longer than the
    # input, so retain the checkpoint's beam search but set a bounded total
    # length proportional to the encoded sentence.
    input_length = int(inputs["input_ids"].shape[1])
    output_ids = model.generate(
        **inputs,
        max_length=min(512, max(24, input_length * 5)),
        # Beam search in the legacy checkpoint becomes pathologically slow
        # through the Transformers 5 cache compatibility layer. Greedy decode
        # is suitable for backend screening and makes a corpus audit feasible.
        num_beams=1,
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"ipa_with_tone": generated}


def _vachana_thai(text: str) -> dict[str, Any]:
    """Fast rule/lexicon Thai baseline derived from TLTK."""
    from vachana_g2p import th2ipa

    return {"ipa_with_tone": th2ipa(text)}


PROVIDERS: dict[str, tuple[str, Callable[[str], dict[str, Any]]]] = {
    "epitran-hin": ("hin", _epitran_hin),
    "schwa-hin": ("hin", _schwa_hin),
    "schwa-stress-hin": ("hin", _schwa_stress_hin),
    "pypinyin-ipa": ("zho-hans", _pypinyin_ipa),
    "g2pw-ipa": ("zho-hans", _g2pw_ipa),
    "g2pm-ipa": ("zho-hans", _g2pm_ipa),
    "pyopenjtalk": ("jpn", _pyopenjtalk),
    "thaig2p-v2": ("tha", _thaig2p_v2),
    "vachana-thai": ("tha", _vachana_thai),
}

PROVIDER_SCHEMA = {name: 1 for name in PROVIDERS}
PROVIDER_SCHEMA["pyopenjtalk"] = 2
PROVIDER_SCHEMA["schwa-hin"] = 2
# 5: loanword फ → /f/ via _HINDI_NATIVE_PH_PREFIXES (2026-08-24 listen audit)
PROVIDER_SCHEMA["schwa-stress-hin"] = 5


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def select_length_stratified(records: list[dict], limit: int, seed: int) -> list[dict]:
    """Sample across the sentence-length distribution, reproducibly."""
    if limit <= 0 or len(records) <= limit:
        return records
    ordered = sorted(records, key=lambda r: len(r["sentence"]))
    rng = random.Random(seed)
    # One random item from each equal-sized length stratum avoids an audit set
    # made almost entirely of the corpus's common medium-length sentences.
    selected = []
    for i in range(limit):
        lo = i * len(ordered) // limit
        hi = (i + 1) * len(ordered) // limit
        selected.append(ordered[rng.randrange(lo, max(lo + 1, hi))])
    return selected


def run_audit(lang: str, provider: str, *, manifest: Path, output: Path,
              limit: int = 0, seed: int = 42) -> Path:
    """Bring the canonical audit file up to date with the manifest.

    Incremental: rows whose (file, sentence hash) already have a successful,
    schema-current entry are reused without touching the G2P tool, so a
    fully-cached refresh needs none of the backend dependencies installed.
    A missing dependency aborts immediately (rather than recording thousands
    of error rows) and names the module to install.
    """
    expected_lang, transcribe = PROVIDERS[provider]
    if lang != expected_lang:
        raise ValueError(f"{provider} is for {expected_lang}, not {lang}")

    records = select_length_stratified(load_manifest(manifest), limit, seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed: dict[tuple[str, str], dict] = {}
    if output.exists():
        with output.open() as f:
            for line in f:
                if line.strip():
                    old = json.loads(line)
                    # Retry failures on resume: dependency/model fixes should
                    # not require deleting the whole successful audit cache.
                    if (old.get("error") is None and
                            old.get("provider_schema", 1) == PROVIDER_SCHEMA[provider]):
                        completed[(old["file"], old["sentence_sha256"])] = old

    missing = [
        rec for rec in records
        if (rec["file"], hashlib.sha256(rec["sentence"].encode()).hexdigest())
        not in completed
    ]
    if missing:
        # Probe the backend on one row BEFORE rewriting the cache file, so a
        # missing dependency aborts with the existing audit fully intact.
        try:
            transcribe(missing[0]["sentence"])
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"{provider} needs {exc.name!r} installed to audit "
                f"{len(missing)} new/changed {lang} row(s) "
                f"(pip install {exc.name}); audit cache left untouched."
            ) from exc
        except Exception:
            pass  # per-row failures are recorded as error rows below

    computed = 0
    with output.open("w") as out:
        for index, rec in enumerate(records, 1):
            sentence = rec["sentence"]
            digest = hashlib.sha256(sentence.encode()).hexdigest()
            key = (rec["file"], digest)
            result = completed.get(key)
            if result is None:
                try:
                    backend_output = transcribe(sentence)
                    error = None
                except ModuleNotFoundError as exc:
                    raise SystemExit(
                        f"{provider} needs {exc.name!r} installed to audit "
                        f"{len(records) - index + 1} new/changed {lang} row(s) "
                        f"(pip install {exc.name})."
                    ) from exc
                except Exception as exc:
                    backend_output = None
                    error = f"{type(exc).__name__}: {exc}"
                computed += 1
                result = {
                    "file": rec["file"],
                    "lang": lang,
                    "source": rec.get("source"),
                    "sentence": sentence,
                    "sentence_sha256": digest,
                    "provider": provider,
                    "provider_schema": PROVIDER_SCHEMA[provider],
                    "output": backend_output,
                    "error": error,
                }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if computed and (index % 25 == 0 or index == len(records)):
                print(f"{provider}: {index}/{len(records)} ({computed} computed)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    # The canonical audit file is also the cache used to build label sidecars.
    # Default to the complete manifest so a later invocation cannot silently
    # truncate a full audit back to a 500-row screening sample.
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if PROVIDERS[args.provider][0] != args.lang:
        parser.error(f"{args.provider} is for {PROVIDERS[args.provider][0]}, not {args.lang}")
    manifest = args.manifest or args.data_root / args.lang / "manifest.jsonl"
    output = args.output or args.data_root / args.lang / f"g2p_audit_{args.provider}.jsonl"
    run_audit(args.lang, args.provider, manifest=manifest, output=output,
              limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
