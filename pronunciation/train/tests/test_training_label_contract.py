"""Model vocabulary, offline tokenizer and staging gates."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "pronunciation/train/scripts"
sys.path.insert(0, str(SCRIPTS))
import training_vocabulary as preprocess

DEFINITION = ROOT / "tagging/lexide/data/training_labels.json"
def test_cached_tokenizer_matches_contract():
    from transformers import Wav2Vec2CTCTokenizer
    from huggingface_hub import hf_hub_download

    provenance = json.loads(DEFINITION.read_text())["provenance"]
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
    script = stage / "training_vocabulary.py"
    shutil.copyfile(SCRIPTS / "training_vocabulary.py", script)
    shutil.copyfile(source, stage / "training_labels.json")
    # No tagging tree exists in this layout, exactly as in the Sky workdir.
    module = import_staged(script)
    assert module._TRAINING_LABELS == preprocess._TRAINING_LABELS
    assert module.unknown_phonemes(["ə", ".", "hʲ", "ɪ"]) == {".", "hʲ"}


def test_full_checkout_prefers_canonical_over_stale_adjacent_copy(tmp_path):
    script = tmp_path / "pronunciation/train/scripts/training_vocabulary.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPTS / "training_vocabulary.py", script)
    canonical = tmp_path / "tagging/lexide/data/training_labels.json"
    canonical.parent.mkdir(parents=True)
    shutil.copyfile(DEFINITION, canonical)
    script.with_name("training_labels.json").write_text("{}")
    assert import_staged(script)._TRAINING_LABELS == preprocess._TRAINING_LABELS
