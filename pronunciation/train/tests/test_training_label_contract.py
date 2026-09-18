"""Model vocabulary, offline tokenizer and staging gates."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "pronunciation/train/scripts"
sys.path.insert(0, str(SCRIPTS))
import training_vocabulary as preprocess

DEFINITION = ROOT / "tagging/lexide/data/training_labels.json"
def test_fresh_vocab_and_saved_processor_resume(tmp_path, monkeypatch):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    from src import train_unified as training

    with monkeypatch.context() as patch:
        patch.setattr(training.Wav2Vec2FeatureExtractor, "from_pretrained",
                      lambda _: Wav2Vec2FeatureExtractor())
        processor = training.load_processor("feature-extractor")
    vocab = processor.tokenizer.get_vocab()
    assert set(vocab) == preprocess.PHONEMES | {"<pad>", "<unk>"}
    assert sorted(vocab.values()) == list(range(len(vocab)))
    assert processor.tokenizer.pad_token_id == 0
    nasal_phones = ["ã", "ẽ", "ẽː", "ĩ", "õ", "õɪ̃", "ũ", "ũɪ̃"]
    assert not preprocess.unknown_phonemes(nasal_phones)
    assert all(processor.tokenizer.convert_tokens_to_ids(phone) != processor.tokenizer.unk_token_id
               for phone in nasal_phones)
    assert preprocess.unknown_phonemes(["a", "tʃ", "??", "d[", "a1", "aɜ", "ɜ", "ɜː", "ʲ", "<pad>", "|"]) == {
        "??", "d[", "a1", "aɜ", "ʲ", "<pad>", "|"}

    # A checkpoint's mapping wins even if current fresh IDs differ at the same size.
    processor.save_pretrained(tmp_path)
    assert training.load_processor("unused", tmp_path).tokenizer.get_vocab() == vocab
    saved = tmp_path / "vocab.json"
    mapping = json.loads(saved.read_text())
    mapping["a"], mapping["b"] = mapping["b"], mapping["a"]
    saved.write_text(json.dumps(mapping))
    expected = Wav2Vec2Processor.from_pretrained(tmp_path, local_files_only=True)
    resumed = training.load_processor("unused", tmp_path)
    assert resumed.tokenizer.get_vocab() == expected.tokenizer.get_vocab()
    assert resumed.tokenizer.convert_tokens_to_ids("a") == vocab["b"]


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
