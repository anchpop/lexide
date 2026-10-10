"""Training packs omit only current strict rejects, before any audio loading."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import preprocess_support as preprocess
from src.train_unified import load_training_datasets


def fixture(tmp_path):
    data, train = tmp_path / 'audio', tmp_path / 'train'
    lang = data / 'eng'
    lang.mkdir(parents=True)
    train.mkdir()
    records = [dict(file=f'{name}.wav', sentence=name, source='tts_word') for name in ('good', 'bad')]
    labels = [dict(r, lang='eng', phonemes=['a'], stress=[1],
                   g2p_language='eng', g2p_identity='test-build') for r in records]
    audits = [dict(r, lang='eng', ok=True, expected_sha256=hashlib.sha256(r['sentence'].encode()).hexdigest(),
                   phone_match=r['file']=='good.wav', phone_match_version=1,
                   g2p_language='eng', g2p_identity='test-build',
                   g2p_selection={'g2p_language':None,'variety':None,'espeak_voice':None}) for r in records]
    for name, rows in [('manifest.jsonl', records), ('phonemes.jsonl', labels)]:
        (lang / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    audit = train / 'tts_word_asr_exclusions.jsonl'
    audit.write_text(''.join(json.dumps(r) + '\n' for r in audits))
    for r in records:
        sf.write(lang / r['file'], np.full(16000, .1, dtype=np.float32), 16000)
    for directory in (data / '.cache', lang / '.cache'):
        directory.mkdir()
        (directory / 'ignored').write_text('cache')
    (tmp_path / 'quarantine').mkdir()
    (tmp_path / 'quarantine/removed.wav').write_bytes(b'not training audio')
    return data, train, audit, audits


def test_pack_omits_reject_audio_but_keeps_metadata_and_trainer_skips_it(tmp_path, monkeypatch, capsys):
    data, train, audit, _ = fixture(tmp_path)
    original = {name:(data / 'eng' / name).read_bytes() for name in ('manifest.jsonl','phonemes.jsonl')}
    output = tmp_path / 'data.tar'
    preprocess.build_dataset_tar(data, output, train_dir=train)
    extracted = tmp_path / 'extracted'
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert './eng/good.wav' in names and './eng/bad.wav' not in names
        assert not any('.cache' in name or 'quarantine' in name for name in names)
        archive.extractall(extracted, filter='data')
    for name, content in original.items():
        assert (extracted / 'eng' / name).read_bytes() == content
        assert (data / 'eng' / name).read_bytes() == content
    assert (data / 'eng/bad.wav').exists(), 'Packing must not delete local rejected audio'
    exists = Path.exists

    def checked_exists(path):
        assert path.name != 'bad.wav', 'Trainer inspected rejected audio before excluding it'
        return exists(path)

    monkeypatch.setattr(Path, 'exists', checked_exists)
    tokenizer = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda phone: 1 if phone == 'a' else -1)
    args = SimpleNamespace(data_dir=extracted, audit_path=[audit], fleurs_audit_path=None,
                           fleurs_audit_min_per=None, fleurs_audit_min_cer=None, fleurs_audit_min_wer=None,
                           audit_min_per=1e-12, audit_min_cer=1e-12, audit_min_wer=1e-12,
                           langs=None, use_narrowed=False, max_audio_sec=16, min_rms=.005,
                           min_whisper_logprob=-.7, max_clips_per_sentence=20, source_cap_second=False)
    datasets = load_training_datasets(args, SimpleNamespace(tokenizer=tokenizer))
    assert sum(map(len, datasets)) == 1
    assert "'asr_audit': 1" in capsys.readouterr().out


@pytest.mark.parametrize('state', ['missing', 'stale'])
def test_pack_refuses_missing_or_stale_strict_audit(tmp_path, state):
    data, train, audit, audits = fixture(tmp_path)
    if state == 'missing':
        audit.unlink()
    else:
        audits[1]['expected_sha256'] = 'stale'
        audit.write_text(''.join(json.dumps(r) + '\n' for r in audits))
    output = tmp_path / 'data.tar'
    with pytest.raises(ValueError, match='missing, stale or failed'):
        preprocess.build_dataset_tar(data, output, train_dir=train)
    assert not output.exists()


def test_gate_policy_change_invalidates_pack(tmp_path, monkeypatch):
    data, train, _, _ = fixture(tmp_path)
    root = tmp_path / 'policy'
    gate = root / 'train/src/audit_gate.py'
    gate.parent.mkdir(parents=True)
    gate.write_text('# packing admission policy\n')
    monkeypatch.setattr(preprocess, 'REPO_ROOT', root)
    run = preprocess.subprocess.run
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return run(*args, **kwargs)

    monkeypatch.setattr(preprocess.subprocess, 'run', counted)
    output = tmp_path / 'data.tar'
    preprocess.build_dataset_tar(data, output, train_dir=train)
    preprocess.build_dataset_tar(data, output, train_dir=train)
    assert len(calls) == 1
    newer = output.stat().st_mtime + 2
    os.utime(gate, (newer, newer))
    preprocess.build_dataset_tar(data, output, train_dir=train)
    assert len(calls) == 2


def test_changed_audit_invalidates_pack_without_changing_data(tmp_path):
    data, train, audit, audits = fixture(tmp_path)
    output = tmp_path / 'data.tar'
    preprocess.build_dataset_tar(data, output, train_dir=train)
    audits[1]['phone_match'] = True
    audit.write_text(''.join(json.dumps(r) + '\n' for r in audits))
    newer = output.stat().st_mtime + 2
    os.utime(audit, (newer, newer))
    preprocess.build_dataset_tar(data, output, train_dir=train)
    with tarfile.open(output) as archive:
        assert './eng/bad.wav' in archive.getnames()
