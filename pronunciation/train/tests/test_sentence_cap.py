"""Sentence repetition caps run on surviving labels before source balancing."""

import json
import random
from collections import Counter

import pytest
from torch.utils.data import ConcatDataset, Subset

from src.dataset import StressDataset, get_audio_lengths
from src.train_unified import (
    cap_clips_per_sentence,
    cap_sources_second,
    normalize_sentence,
)


def make_dataset(tmp_path, lang, rows):
    path = tmp_path / lang / "phonemes_narrowed.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(file=f"{i}.wav", lang=lang, source=source, sentence=text)
               for i, (source, text) in enumerate(rows)]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    # Match StressDataset's retained metadata; no audio/model downloads needed.
    ds = StressDataset.__new__(StressDataset)
    ds.samples = [dict(wav_path=str(path.parent / row["file"]), lang=lang,
                       source=row["source"], n_audio_samples=16000 + i)
                  for i, row in enumerate(records)]
    return ds, path


@pytest.mark.parametrize("text, expected", [
    ("  HELLO\t world!  ", "hello world"),
    ("Bonjour \n le monde…?! »", "bonjour le monde"),
    ("你好。！？", "你好"),
    ("don't re-enter, please.", "don't re-enter, please"),
    ("¿Qué?", "¿qué"),
    ("  !… ", ""),
])
def test_normalization(text, expected):
    assert normalize_sentence(text) == expected


def test_default_cap_all_sources_deterministic_and_ordered(tmp_path, capsys):
    sources = ["pimsleur", "tts", "fleurs", "tatoeba", "film", None]
    rows = [(source, ["Hello world.", " HELLO\tWORLD! "][i % 2])
            for source in sources for i in range(30)]
    ds, path = make_dataset(tmp_path, "eng", rows)
    original = list(ds.samples)
    state = random.getstate()
    cap_clips_per_sentence(ds, path)
    assert random.getstate() == state
    assert Counter(s["source"] for s in ds.samples) == dict.fromkeys(sources, 20)
    assert [original.index(s) for s in ds.samples] == sorted(original.index(s) for s in ds.samples)
    assert ds.samples[:20] != original[:20]  # seeded sample, not the first clips
    again, _ = make_dataset(tmp_path, "eng", rows)
    random.seed(999)
    try:
        cap_clips_per_sentence(again, path)
    finally:
        random.setstate(state)
    assert again.samples == ds.samples
    assert "eng: sentence cap removed 60 / 180 clips" in capsys.readouterr().out


def test_small_groups_and_distinct_sentences_survive(tmp_path):
    rows = [("tts", "A.")] * 3 + [("tts", "B!")] * 2 + [("tts", "C?")]
    ds, path = make_dataset(tmp_path, "fra", rows)
    original = list(ds.samples)
    cap_clips_per_sentence(ds, path, 2)
    assert len(ds.samples) == 5
    assert ds.samples[-3:] == original[-3:]


def test_zero_is_noop_without_reading_labels_and_negative_rejected(tmp_path):
    ds, path = make_dataset(tmp_path, "eng", [("tts", "a")] * 25)
    original = ds.samples
    path.unlink()
    cap_clips_per_sentence(ds, path, 0)
    assert ds.samples is original
    with pytest.raises(ValueError, match="nonnegative"):
        cap_clips_per_sentence(ds, path, -1)


def test_source_cap_counts_sentence_survivors_and_preserves_lengths(tmp_path, capsys):
    eng, eng_path = make_dataset(tmp_path, "eng", [("tts", "repeated")] * 100
                                + [("tts", f"unique {i}") for i in range(10)])
    fra, fra_path = make_dataset(tmp_path, "fra", [("tts", f"unique {i}") for i in range(40)])
    cap_clips_per_sentence(eng, eng_path)
    cap_clips_per_sentence(fra, fra_path)
    assert (len(eng), len(fra)) == (30, 40)
    balanced = cap_sources_second([eng, fra])
    assert all(isinstance(ds, Subset) for ds in balanced)
    assert [len(ds) for ds in balanced] == [30, 30]
    expected_lengths = [ds.dataset.samples[i]["n_audio_samples"]
                        for ds in balanced for i in ds.indices]
    assert get_audio_lengths(ConcatDataset(balanced)) == expected_lengths
    output = capsys.readouterr().out
    assert "fra: sentence cap removed 0 / 40 clips" in output
    assert "{'tts': 30}" in output


def test_only_loaded_samples_count_and_languages_are_independent(tmp_path):
    for lang in ["eng", "fra"]:
        ds, path = make_dataset(tmp_path, lang, [("tts", "same")] * 25)
        ds.samples = ds.samples[-3:]  # earlier audio/quality filtering
        cap_clips_per_sentence(ds, path, 2)
        assert len(ds.samples) == 2
        assert all(s["lang"] == lang for s in ds.samples)


def test_cli_rejects_negative_cap_before_model_loading(monkeypatch, capsys):
    from src.train_unified import main

    monkeypatch.setattr("sys.argv", ["train_unified", "--max-clips-per-sentence", "-1"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--max-clips-per-sentence must be nonnegative" in capsys.readouterr().err


def test_cli_help_documents_sentence_cap(monkeypatch, capsys):
    from src.train_unified import main

    monkeypatch.setattr("sys.argv", ["train_unified", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "--max-clips-per-sentence" in capsys.readouterr().out
