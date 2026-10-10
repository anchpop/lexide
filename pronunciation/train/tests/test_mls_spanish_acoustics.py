"""Dialect application preserves unrelated rows and quarantines abstentions."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import mls_spanish_acoustics as dialect


@pytest.mark.parametrize('stale', [None, 'text', 'build', 'coverage'])
def test_apply_preserves_existing_rows_and_labels(tmp_path, stale):
    audio, work, quarantine = (tmp_path / name for name in ('audio', 'work', 'quarantine'))
    work.mkdir()
    decisions = {'yes': {'clips': 1, 'variety': 'european'},
                 'no': {'clips': 2, 'variety': None}}
    (work / 'variety-invariant.jsonl').write_text(''.join(json.dumps({
        'file': file, 'g2p_identity': 'test-build', 'variety_invariant': invariant,
        'expected_sha256': hashlib.sha256(sentence.encode()).hexdigest(),
    }) + '\n' for file, sentence, invariant in [
        ('drop.wav', 'zapato', False), ('safe.wav', 'hola', True)]))
    (work / 'speaker-results.json').write_text(json.dumps(
        {'identity': dialect.IDENTITY.as_dict(), 'speakers': decisions}))
    old = b'{ "file": "old.wav", "source": "tts", "opaque": [1, 2] }\n'
    for lang in ('por', 'spa'):
        directory = audio / lang
        directory.mkdir(parents=True)
        rows = [{'file': 'keep.wav', 'source': 'mls', 'voice': 'yes'},
                {'file': 'drop.wav', 'source': 'mls', 'voice': 'no', 'sentence': 'zapato'}]
        if lang == 'spa':
            rows.append({'file': 'safe.wav', 'source': 'mls', 'voice': 'no',
                         'sentence': 'hola', 'duration_sec': 3.6})
        (directory / 'manifest.jsonl').write_bytes(old + b''.join(
            (json.dumps(row) + '\n').encode() for row in rows))
        (directory / 'phonemes.jsonl').write_bytes(old)
        (directory / 'vad.jsonl').write_text(
            ''.join(json.dumps({'file': file, 'vad_probs': []}) + '\n'
                    for file in ('old.wav', 'keep.wav', 'drop.wav')))
        for file in ('old.wav', 'keep.wav', 'drop.wav', 'safe.wav'):
            (directory / file).write_bytes(b'unchanged audio')
    if stale:
        path = work / 'variety-invariant.jsonl'
        evidence = [json.loads(line) for line in path.read_text().splitlines()]
        if stale == 'coverage':
            evidence.pop()
        else:
            evidence[0]['expected_sha256' if stale == 'text' else 'g2p_identity'] = 'stale'
        path.write_text(''.join(json.dumps(row) + '\n' for row in evidence))
        with pytest.raises(AssertionError):
            dialect.apply_metadata(audio, work, quarantine, expected_por_count=2, g2p_identity='test-build')
        assert not quarantine.exists()
        assert (audio / 'por/drop.wav').exists()
        return
    dialect.apply_metadata(audio, work, quarantine, expected_por_count=2, g2p_identity='test-build')
    assert (audio / 'por/manifest.jsonl').read_bytes() == old
    spanish = (audio / 'spa/manifest.jsonl').read_bytes()
    assert spanish.startswith(old)
    retained = json.loads(spanish.splitlines()[1])
    assert retained['file'] == 'keep.wav' and retained['variety'] == 'european'
    assert retained['dialect_evidence_sha256']
    recovered = json.loads(spanish.splitlines()[2])
    assert recovered['file'] == 'safe.wav'
    assert recovered['variety'] == 'european' and recovered['variety_invariant'] is True
    assert recovered['variety_invariant_g2p_identity'] == 'test-build'
    summary = json.loads((quarantine / 'summary.json').read_text())
    assert summary['variety_invariant_recovered'] == {'clips': 1, 'hours': .001}
    for lang in ('por', 'spa'):
        assert (audio / lang / 'phonemes.jsonl').read_bytes() == old
        assert (audio / lang / 'old.wav').read_bytes() == b'unchanged audio'
        assert not (audio / lang / 'drop.wav').exists()
        assert (quarantine / lang / 'audio/drop.wav').read_bytes() == b'unchanged audio'
        assert 'drop.wav' not in (audio / lang / 'vad.jsonl').read_text()
    assert (audio / 'spa/keep.wav').exists()
    assert not (audio / 'por/keep.wav').exists()
    with pytest.raises(AssertionError):
        dialect.apply_metadata(audio, work, quarantine, expected_por_count=2, g2p_identity='test-build')
