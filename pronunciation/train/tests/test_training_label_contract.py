"""Shared frozen pre-change label oracle, offline tokenizer and staging gates."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "pronunciation/train/scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess

DEFINITION = ROOT / "tagging/lexide/data/training_labels.json"
FIXTURE = json.loads((DEFINITION.parent / "training_labels_conformance.json").read_text())


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_frozen_prechange_conformance(case, monkeypatch):
    base = preprocess._tokenizer_vocab() | set(case["extra_vocab"])
    monkeypatch.setattr(preprocess, "_tokenizer_vocab", lambda: base)
    phones, stress, unknown = preprocess.validate_phonemes(
        case["phones"], case["stress"], case["lang"],
    )
    assert wire([phones, stress, sorted(unknown)]) == wire(case["expected"])


def test_fixture_generator_is_a_byte_identical_noop():
    script = SCRIPTS / "generate_training_label_fixture.py"
    fixture = DEFINITION.parent / "training_labels_conformance.json"
    before = fixture.read_bytes()
    result = subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    assert result.stdout == before
    subprocess.run([sys.executable, str(script), "--check"], check=True, capture_output=True)
    assert fixture.read_bytes() == before
    assert FIXTURE["generated_by"] == (
        "python pronunciation/train/scripts/generate_training_label_fixture.py --write"
    )


def test_frozen_oracle_digest_and_inventory_coverage():
    # Captured before extracting the tables, using the original validator and
    # the actual cached tokenizer (provenance in the fixture). Never derive
    # expected outputs from the new contract or auto-refresh this oracle.
    assert hashlib.sha256(wire([c["expected"] for c in FIXTURE["cases"]])).hexdigest() == (
        "475cb3ef22332802623e8a301142c73b4e0fa74fbcd70bdc425b915b2e44133b"
    )
    accepted = {p for c in FIXTURE["cases"] if c["name"].startswith("accepted-")
                for p in c["expected"][0]}
    base = preprocess._tokenizer_vocab()
    extensions = preprocess.VOCAB_EXTENSIONS
    assert len(base) == 393 and len(extensions) == 77
    assert not base & extensions
    assert accepted == base | extensions
    assert {"<unk>", "<s>", "</s>", "<pad>", "|"} <= accepted
    assert len(preprocess.TOKEN_BLACKLIST) == 8
    assert not accepted & preprocess.TOKEN_BLACKLIST
    cases = {c["name"]: c for c in FIXTURE["cases"]}
    assert set(cases["blacklist"]["phones"]) == preprocess.TOKEN_BLACKLIST
    assert set(cases["global-remap"]["phones"]) == set(preprocess.TOKEN_REMAP)
    for lang, table in preprocess.LANG_PHONEME_REMAP.items():
        for phone, expected in table.items():
            case = cases[f"{lang}-exact-{phone}"]
            assert case["phones"][0] == phone
            assert case["expected"][0][0] == expected


def test_language_remap_precedes_global_and_vocab_precedes_blacklist(monkeypatch):
    monkeypatch.setattr(preprocess, "LANG_PHONEME_REMAP", {"synthetic": {"raw": "ε"}})
    monkeypatch.setattr(preprocess, "TOKEN_BLACKLIST", {"ɛ"})
    assert preprocess.validate_phonemes(["raw"], [2], "synthetic") == (["ɛ"], [2], set())


@pytest.mark.parametrize("phones,stress,expected", [
    (["a", "unknown"], [1], (["a"], [1], set())),
    (["a"], [1, 2], (["a"], [1], set())),
    (["unknown"], [], ([], [], set())),
])
def test_python_keeps_legacy_zip_truncation(phones, stress, expected):
    assert preprocess.validate_phonemes(phones, stress, "eng") == expected


def test_cached_tokenizer_matches_contract():
    from transformers import Wav2Vec2CTCTokenizer
    from huggingface_hub import hf_hub_download

    provenance = FIXTURE["provenance"]
    try:
        raw = Path(hf_hub_download(
            provenance["tokenizer_name"], "vocab.json",
            revision=provenance["tokenizer_revision"], local_files_only=True,
        ))
        pinned = Wav2Vec2CTCTokenizer.from_pretrained(
            provenance["tokenizer_name"], revision=provenance["tokenizer_revision"],
            local_files_only=True,
        )
        current = Wav2Vec2CTCTokenizer.from_pretrained(
            provenance["tokenizer_name"], local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"offline tokenizer snapshot unavailable: {exc}")
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == provenance["raw_vocab_sha256"]
    assert set(pinned.get_vocab()) - set(json.loads(raw.read_text())) == {"|"}
    assert set(pinned.get_vocab()) == preprocess._tokenizer_vocab()
    # Check the default cached source as well, not just the provenance revision.
    # Training performs this same guard on the tokenizer it actually loads.
    preprocess.check_training_label_vocab(preprocess.TOKENIZER_NAME, set(current.get_vocab()))


def test_default_tokenizer_guard_and_custom_source():
    base = preprocess._tokenizer_vocab()
    preprocess.check_training_label_vocab(preprocess.TOKENIZER_NAME, base)
    for changed, token in [(base - {"|"}, "|"), (base | {"unexpected"}, "unexpected")]:
        with pytest.raises(ValueError, match="Training-label vocabulary drift") as error:
            preprocess.check_training_label_vocab(preprocess.TOKENIZER_NAME, changed)
        assert token in str(error.value)
    preprocess.check_training_label_vocab("custom/processor", {"anything"})


def test_trainer_checks_before_adding_extensions(monkeypatch):
    from src import train_unified as training

    added = []
    base = preprocess._tokenizer_vocab()
    tokenizer = SimpleNamespace(get_vocab=lambda: dict.fromkeys(base),
                                add_tokens=lambda tokens: added.extend(tokens))
    monkeypatch.setattr(training.Wav2Vec2FeatureExtractor, "from_pretrained", lambda _: "features")
    monkeypatch.setattr(training.Wav2Vec2CTCTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(training, "Wav2Vec2Processor", lambda **kwargs: kwargs)
    result = training.load_processor(preprocess.TOKENIZER_NAME)
    assert result["tokenizer"] is tokenizer
    assert added == sorted(preprocess.VOCAB_EXTENSIONS)
    base = base - {"|"}
    added.clear()
    with pytest.raises(ValueError, match="Training-label vocabulary drift"):
        training.load_processor(preprocess.TOKENIZER_NAME)
    assert not added
    training.load_processor("custom/processor")
    assert added == sorted(preprocess.VOCAB_EXTENSIONS)


def import_staged(script):
    spec = importlib.util.spec_from_file_location("staged_preprocess", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recipe", ["sky_train.yaml", "sky_train_merged.yaml", "sky_smoke.yaml"])
def test_sky_mount_and_pronunciation_only_import(recipe, tmp_path):
    config = yaml.safe_load((ROOT / "pronunciation/train" / recipe).read_text())
    assert config["workdir"] == "."
    destination = "~/sky_workdir/train/scripts/training_labels.json"
    source = (ROOT / "pronunciation" / config["file_mounts"][destination]).resolve()
    assert source == DEFINITION
    stage = tmp_path / "sky_workdir/train/scripts"
    stage.mkdir(parents=True)
    script = stage / "preprocess.py"
    shutil.copyfile(SCRIPTS / "preprocess.py", script)
    shutil.copyfile(source, stage / "training_labels.json")
    # No tagging tree exists in this layout, exactly as in the Sky workdir.
    module = import_staged(script)
    assert module._TRAINING_LABELS == preprocess._TRAINING_LABELS
    assert module.validate_phonemes(["ɐ", ".", "ɪ"], [1, 0, 2], "eng") == (
        ["ə", "ɪ"], [1, 2], set(),
    )


def test_full_checkout_prefers_canonical_over_stale_adjacent_copy(tmp_path):
    script = tmp_path / "pronunciation/train/scripts/preprocess.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPTS / "preprocess.py", script)
    canonical = tmp_path / "tagging/lexide/data/training_labels.json"
    canonical.parent.mkdir(parents=True)
    shutil.copyfile(DEFINITION, canonical)
    script.with_name("training_labels.json").write_text("{}")
    assert import_staged(script)._TRAINING_LABELS == preprocess._TRAINING_LABELS
