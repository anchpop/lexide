"""Private pack publication verifies pinned remote bytes and resumes immutably."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import publish_pack


def test_publish_private_snapshot_and_resume_without_reupload(tmp_path, monkeypatch):
    root = tmp_path / 'pronunciation'
    (root / '.work/run2').mkdir(parents=True)
    (root / 'train').mkdir()
    (tmp_path / '.env').write_text('HF_TOKEN=test-only-not-a-real-token\n')
    tar = root / '.work/pron_audio.tar'
    tar.write_bytes(b'test packed corpus')
    (root / 'train/tts_word_asr_exclusions.jsonl').write_text('{}\n')
    for name in ('admitted-after.json', 'audit-summary.json', 'tts-plan-final.json',
                 'word-candidates.provenance.json', 'tts-costs.json',
                 'spanish-acoustics/speaker-results.json',
                 'quarantine/read-speech-unsupported/unsupported-phones.json',
                 'quarantine/word-audit-failures/tts-audit-failures-for-quarantine.json'):
        path = root / '.work/run2' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    monkeypatch.setattr(publish_pack, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['publish_pack', '--tar', str(tar), '--snapshot', 'fixed-test'])
    api = MagicMock()
    revision = 'a' * 40
    tags = []
    api.list_repo_refs.side_effect = lambda *a, **kw: SimpleNamespace(tags=tags)
    api.upload_folder.return_value = SimpleNamespace(oid=revision)
    api.create_tag.side_effect = lambda *a, **kw: tags.append(SimpleNamespace(name=kw['tag'], target_commit=kw['revision']))

    def info(*args, **kwargs):
        stage = root / '.work/fixed-test'
        siblings = []
        if stage.exists() and ('revision' in kwargs or tags):
            for path in stage.rglob('*'):
                if path.is_file():
                    siblings.append(SimpleNamespace(rfilename='snapshots/fixed-test/' + str(path.relative_to(stage)),
                                                     size=path.stat().st_size,
                                                     lfs=SimpleNamespace(sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        return SimpleNamespace(private=True, sha=revision, siblings=siblings)

    api.dataset_info.side_effect = info
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda **kw: api, hf_hub_download=MagicMock()))
    publish_pack.main()
    result = json.loads((root / '.work/run2/hf-upload.json').read_text())
    assert result['private'] and result['revision'] == revision
    assert result['verified_files'] == 11  # tar, sidecar, eight certificates, snapshot.json
    publish_pack.main()
    assert api.upload_folder.call_count == 1
    assert api.create_tag.call_count == 1
    (root / '.work/fixed-test/pron_audio.tar.000.part').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        publish_pack.main()
    assert api.upload_folder.call_count == 1
