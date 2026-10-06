"""Strict sources cannot bypass transcript coverage via PER=0 or missing audits."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from src.audit_gate import STRICT_SOURCES, load_coverage, strict_verdict
import preprocess_support as preprocess
from src.train_unified import load_asr_audit_exclusions


@pytest.mark.parametrize('source', sorted(STRICT_SOURCES))
@pytest.mark.parametrize('state', ['missing','stale','failed','legacy','malformed','pass','reject'])
def test_strict_coverage(tmp_path, source, state):
    record = {'file':'a.wav','sentence':'their sentence café Straße','source':source}
    audit = {'file':'a.wav','lang':'eng','source':source,'ok':True,'per':0,
             'expected_sha256':hashlib.sha256(record['sentence'].encode()).hexdigest(),
             'phone_match':state != 'reject', 'phone_match_version':1,
             'g2p_selection':{'g2p_language':None,'variety':None,'espeak_voice':None},
             'word_match':False, 'word_match_version':1}
    if state == 'stale': audit['expected_sha256'] = 'old'
    if state == 'failed': audit['ok'] = False
    if state == 'legacy': audit.pop('phone_match_version')
    if state == 'malformed': audit['phone_match'] = 'true'
    path = tmp_path / f'{source}_asr_exclusions.jsonl'
    if state != 'missing': path.write_text(json.dumps(audit)+'\n')
    coverage = load_coverage([path])
    if state in {'pass','reject'}:
        assert strict_verdict(record,'eng',coverage) is (state == 'pass')
    else:
        with pytest.raises(ValueError, match='missing, stale or failed'):
            strict_verdict(record,'eng',coverage)


@pytest.mark.parametrize('source', sorted(STRICT_SOURCES))
def test_legacy_loaders_leave_strict_sources_to_strict_verdict(tmp_path, source):
    # A failing PER must not exclude here: strict rows are gated only by phone_match.
    row = {'lang':'eng','file':'a.wav','source':source,'ok':True,'expected':'their',
           'phone_match':True,'phone_match_version':1,'per':1,'cer':1,'wer':1}
    path = tmp_path / f'{source}_asr_exclusions.jsonl'
    path.write_text(json.dumps(row)+'\n')
    assert preprocess.load_training_exclusions(tmp_path) == {}
    assert load_asr_audit_exclusions(path,min_per=.5,min_cer=.5,min_wer=.5) == {}


@pytest.mark.parametrize('field,value', [('variety','latin_american'),('espeak_voice','es-419'),('g2p_language','spa-419')])
def test_changed_raw_language_selection_is_stale(field, value):
    record = {'file':'a.wav','sentence':'llama','source':'mls'}
    audit = {'ok':True,'expected_sha256':hashlib.sha256(b'llama').hexdigest(),
             'phone_match':True,'phone_match_version':1,'g2p_language':'spa',
             'g2p_selection':{'g2p_language':None,'variety':None,'espeak_voice':None}}
    coverage = {('mls','spa','a.wav'):audit}
    assert strict_verdict(record,'spa',coverage) is True
    record[field] = value
    with pytest.raises(ValueError,match='stale'):
        strict_verdict(record,'spa',coverage)


@pytest.mark.parametrize('field,value', [('g2p_language','spa-419'),('g2p_identity','new-build')])
def test_changed_finalized_language_or_build_is_stale(field, value):
    record = {'file':'a.wav','sentence':'llama','source':'mls','phonemes':['ʎ'],
              'g2p_language':'spa','g2p_identity':'old-build'}
    audit = {'ok':True,'expected_sha256':hashlib.sha256(b'llama').hexdigest(),
             'phone_match':True,'phone_match_version':1,
             'g2p_language':'spa','g2p_identity':'old-build'}
    coverage = {('mls','spa','a.wav'):audit}
    assert strict_verdict(record,'spa',coverage) is True
    record[field] = value
    with pytest.raises(ValueError,match='stale'):
        strict_verdict(record,'spa',coverage)


def test_word_only_audit_no_longer_satisfies_coverage(tmp_path):
    record = {'file':'a.wav','sentence':'hello','source':'mls'}
    row = {'file':'a.wav','lang':'eng','source':'mls','ok':True,
           'expected_sha256':hashlib.sha256(b'hello').hexdigest(),
           'word_match':True,'word_match_version':1,'per':0}
    path = tmp_path/'mls_asr_exclusions.jsonl'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='phone-match audit'):
        strict_verdict(record,'eng',load_coverage([path]))


def test_prepare_checks_coverage_before_audio(tmp_path):
    root = tmp_path/'audio'/'eng'
    root.mkdir(parents=True)
    record = {'file':'absent.wav','sentence':'their','source':'mls'}
    (root/'manifest.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='phone-match audit'):
        preprocess.prepare(root.parent,'eng',tmp_path/'prepared',False,tmp_path)
    assert not (tmp_path/'prepared').exists()


def test_incremental_selection_and_byte_preserving_merge(tmp_path):
    for lang in ('eng','deu','spa'):
        root=tmp_path/lang
        root.mkdir()
        old=b'{ "file": "old.wav", "sentence": "old", "opaque": [1, 2] }\n'
        (root/'phonemes.jsonl').write_bytes(old)
        (root/'manifest.jsonl').write_text(json.dumps({'file':'old.wav','sentence':'old','source':'mls'})+'\n')
        # Existing strict-source rows are not reprocessed in incremental mode.
        prepared=tmp_path/(lang+'.jsonl')
        preprocess.prepare(tmp_path,lang,prepared,False,tmp_path,new_only=True)
        assert prepared.read_bytes()==b''
        preprocess.write_jsonl(root/'phonemes.jsonl',[{'file':'new.wav'}],True)
        assert (root/'phonemes.jsonl').read_bytes().startswith(old)
        with pytest.raises(ValueError,match='refusing to replace'):
            preprocess.write_jsonl(root/'phonemes.jsonl',[{'file':'old.wav'}],True)
        assert (root/'phonemes.jsonl').read_bytes().startswith(old)
