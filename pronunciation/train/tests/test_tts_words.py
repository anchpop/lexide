"""Safety properties of word-plan selection and bounded synthesis."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data'))
import generate_tts
import tts_words


@pytest.mark.parametrize('text,blocked', [
    ('L’été bleu-gris', {'été', 'bleu', 'gris', "l'été", 'bleu-gris'}),
    ('CAFÉ', {'café'}), ('café', {'café'}),
    ('aujourd’hui — dit‑elle', {'aujourd', 'hui', 'dit', 'elle'}),
])
def test_global_eval_constituents(text, blocked):
    assert blocked <= tts_words.constituents(text)
    for word in blocked:
        assert not tts_words.eligible(word, tts_words.constituents(text), [])


def test_no_nonspace_eval_subword_leakage():
    assert not tts_words.eligible('東京', set(), ['東京都に行く'])
    assert tts_words.eligible('大阪', set(), ['東京都に行く'])


def test_cloud_one_attempt_disables_sdk_retry(tmp_path, monkeypatch):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, 'google.cloud', SimpleNamespace(texttospeech=sdk))
    client = MagicMock()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 160)
    client.synthesize_speech.return_value.audio_content = buffer.getvalue()
    record = generate_tts.synthesize_one(client, 'bonjour', 'fr-FR-Chirp3-HD-Kore', 'fr-FR',
                                         None, tmp_path, 'fra', file='tts_word_voice.wav',
                                         source='tts_word', attempts=1)
    assert record['source'] == 'tts_word' and record['file'] == 'tts_word_voice.wav'
    assert client.synthesize_speech.call_args.kwargs['retry'] is None
    client.synthesize_speech.side_effect = RuntimeError('429')
    assert generate_tts.synthesize_one(client, 'bonjour', 'voice', 'fr-FR', None, tmp_path,
                                       'fra', attempts=1) is None
    assert client.synthesize_speech.call_count == 2


def test_gemini_usage_settles_reservations_and_resume_skips_attempts(tmp_path, monkeypatch):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, 'google.cloud', SimpleNamespace(texttospeech=sdk))
    monkeypatch.setattr(generate_tts, 'make_client', MagicMock())
    monkeypatch.setattr(tts_words, 'LANGS', ['fas'])
    monkeypatch.setattr(tts_words, 'eval_inventory', lambda: (set(), [], []))
    usage = {'promptTokenCount': 2, 'candidatesTokenCount': 23}
    cost = 2 * 0.5e-6 + 23 * 6e-6
    work = MagicMock(side_effect=lambda req, _: (
        {'file': req['file'], 'source': 'tts_word', 'sentence': req['word']}, cost, usage))
    monkeypatch.setattr(tts_words, 'gemini_word', work)
    requests = [{'lang': 'fas', 'word': 'allowed', 'file': f'{i}.wav', 'backend': 'gemini',
                 'model': 'test', 'frequency': 1, 'cost_usd': cost,
                 'reserve_usd': 0.1024} for i in range(8)]
    plan = {'version': 1, 'retries': 0, 'estimate_usd': 8 * cost, 'cap_usd': 0.11,
            'provenance': {'eval_manifests': []}, 'requests': requests}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    # The cap permits just one reservation at a time, but all actual receipts.
    tts_words.execute(path, tmp_path / 'audio', workers=1, rps=100000)
    assert work.call_count == 8
    events = [json.loads(line) for line in path.with_suffix('.ledger.jsonl').read_text().splitlines()]
    latest = {event['file']: event for event in events}
    assert sum(event['debit_usd'] for event in latest.values()) == pytest.approx(8 * cost)
    assert all(event['state'] == 'received' for event in latest.values())
    tts_words.execute(path, tmp_path / 'audio', workers=1, rps=100000)
    assert work.call_count == 8


def test_retry_keeps_uncertain_spend_and_never_repeats_success(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setitem(sys.modules, 'google.cloud', SimpleNamespace(texttospeech=MagicMock()))
    monkeypatch.setattr(generate_tts, 'make_client', MagicMock())
    monkeypatch.setattr(tts_words, 'LANGS', ['fas'])
    monkeypatch.setattr(tts_words, 'eval_inventory', lambda: (set(), [], []))
    req = dict(lang='fas', word='allowed', file='word.wav', backend='gemini',
               model='test', frequency=1, cost_usd=0.001, reserve_usd=0.1)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(dict(version=1, retries=0, estimate_usd=0.001,
                                   cap_usd=0.21, provenance={'eval_manifests': []},
                                   requests=[req])))
    work = MagicMock(side_effect=[urllib.error.URLError('connection lost'),
        ({'file': req['file'], 'sentence': req['word']}, 0.001, {'receipt': True})])
    monkeypatch.setattr(tts_words, 'gemini_word', work)
    for _ in range(3):
        tts_words.execute(path, tmp_path / 'audio', workers=1, rps=100000)
    assert work.call_count == 2
    events = [json.loads(line) for line in path.with_suffix('.ledger.jsonl').read_text().splitlines()]
    assert [e['debit_usd'] for e in events] == pytest.approx([0.1, 0.1, 0.2, 0.101])
    assert events[-1]['actual_cost_usd'] == 0.001
    assert len((tmp_path / 'audio/fas/manifest.jsonl').read_text().splitlines()) == 1


def test_install_is_append_only_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_words, 'assert_no_running_audit', lambda: None)
    monkeypatch.setattr(tts_words, 'eval_inventory', lambda: (set(), [], []))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'provenance': {'eval_manifests': []},
                               'requests': [{'file': 'new.wav', 'lang': 'eng', 'word': 'allowed'}]}))
    plan_hash = tts_words.fingerprint(plan)['sha256']
    plan.with_suffix('.ledger.jsonl').write_text(json.dumps(
        {'file': 'new.wav', 'plan_sha256': plan_hash, 'state': 'received'}) + '\n')
    staging, output = tmp_path / 'staging', tmp_path / 'audio'
    for root in (staging, output):
        (root / 'eng').mkdir(parents=True)
    row = {'file': 'new.wav', 'lang': 'eng', 'sentence': 'allowed',
           'source': 'tts_word', 'word_plan_sha256': plan_hash}
    (staging / 'eng/manifest.jsonl').write_text(json.dumps(row) + '\n')
    (staging / 'eng/new.wav').write_bytes(b'unchanged audio')
    old = b'{ "file": "old.wav", "opaque": 3 }\n'
    target = output / 'eng/manifest.jsonl'
    target.write_bytes(old)
    tts_words.install(plan, staging, output)
    first = target.read_bytes()
    assert first.startswith(old) and len(first.splitlines()) == 2
    assert (output / 'eng/new.wav').read_bytes() == b'unchanged audio'
    tts_words.install(plan, staging, output)
    assert target.read_bytes() == first
    (output / 'eng/new.wav').write_bytes(b'different audio')
    with pytest.raises(AssertionError):
        tts_words.install(plan, staging, output)
    assert target.read_bytes() == first


def test_budget_drains_successes_and_resume_never_retries(tmp_path, monkeypatch):
    sdk = MagicMock()
    monkeypatch.setitem(sys.modules, 'google.cloud', SimpleNamespace(texttospeech=sdk))
    monkeypatch.setattr(generate_tts, 'make_client', MagicMock())
    monkeypatch.setattr(tts_words, 'LANGS', ['eng'])
    monkeypatch.setattr(tts_words, 'eval_inventory', lambda: (set(), [], []))
    original_glob = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda p, pat: [] if str(p) == '/proc' else original_glob(p, pat))
    work = MagicMock(side_effect=lambda *a, **kw: {'file': kw['file'], 'source': 'tts_word', 'sentence': a[1]})
    monkeypatch.setattr(generate_tts, 'synthesize_one', work)
    requests = [{'lang': 'eng', 'word': 'allowed', 'file': f'{i}.wav', 'backend': 'chirp3',
                 'model': 'Chirp3-HD', 'voice': 'voice', 'language_code': 'en-US',
                 'frequency': 1, 'cost_usd': 1, 'reserve_usd': 1} for i in range(8)]
    plan = {'version': 1, 'retries': 0, 'estimate_usd': 5, 'cap_usd': 5,
            'provenance': {'eval_manifests': []}, 'requests': requests}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    tts_words.execute(path, tmp_path / 'audio', workers=3, rps=100000)
    assert work.call_count == 5
    manifest = tmp_path / 'audio/eng/manifest.jsonl'
    assert len(manifest.read_text().splitlines()) == 5
    tts_words.execute(path, tmp_path / 'audio', workers=3, rps=100000)
    assert work.call_count == 5
    assert len(manifest.read_text().splitlines()) == 5
