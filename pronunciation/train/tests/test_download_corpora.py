"""Offline acquisition fixtures; never download or process the real corpus."""
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'data'))
import download_aishell as aishell
import download_cv as cv
from corpus_import import extract_archive


@pytest.fixture(autouse=True)
def ample_fixture_disk(monkeypatch):
    monkeypatch.setattr(aishell.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024**4))


def wav(rate=16000):
    data=io.BytesIO()
    sf.write(data,np.full(rate//10,.1,dtype=np.float32),rate,format='WAV',subtype='PCM_16')
    return data.getvalue()


def archive(path, files):
    with tarfile.open(path,'w:gz') as out:
        for name,data in files.items():
            if isinstance(data,str): data=data.encode()
            info=tarfile.TarInfo(name); info.size=len(data)
            out.addfile(info,io.BytesIO(data))


@pytest.mark.parametrize('source',['aishell1','aishell3'])
def test_complete_append_resume_and_provenance(tmp_path,source):
    packed=tmp_path/'input.tgz'
    if source=='aishell1':
        files={'data_aishell/transcript/aishell_transcript_v0.8.txt':'BAC001S0001W0001 你 好\n',
               'data_aishell/wav/train/S0001/BAC001S0001W0001.wav':wav()}
    else:
        files={'train/content.txt':'SSB00010001.wav\t你 ni3 好 hao3\n',
               'train/wav/SSB0001/SSB00010001.wav':wav(44100)}
    archive(packed,files)
    root=tmp_path/'audio'/'zho-hans'; root.mkdir(parents=True)
    old=b'{ "file": "old.wav", "sentence": "old", "source": "tts" }\n'
    (root/'manifest.jsonl').write_bytes(old)
    args=SimpleNamespace(source=source,archive=packed,data_dir=root.parent,cache_dir=tmp_path/'cache')
    aishell.acquire(args)
    before=(root/'manifest.jsonl').read_bytes()
    assert before.startswith(old)
    rows=[json.loads(s) for s in before.splitlines()]
    added=rows[1]
    assert added['sentence']=='你好'
    assert added['voice']==source+':'+('S0001' if source=='aishell1' else 'SSB0001')
    if source=='aishell3': assert added['annotated_pinyin']=='ni3 hao3'
    info=sf.info(root/added['file'])
    assert (info.samplerate,info.channels,info.subtype)==(16000,1,'PCM_16')
    aishell.acquire(args)
    assert (root/'manifest.jsonl').read_bytes()==before
    assert not (root/'phonemes.jsonl').exists()
    assert not (root/'vad.jsonl').exists()


@pytest.mark.parametrize('accent,status',[
    ('','blank'),('United States English','native_region'),
    ('non-native American English','nonnative'),('Russian accent','unknown'),
    ('French-accented English','unknown'),('learner','nonnative'),
])
def test_accent_gate(accent,status):
    assert cv.accent_status('eng',accent)==status


@pytest.mark.parametrize('lang,accent,variety',[
    ('spa','Mexican','latin_american'),('spa','España','european'),('spa','Spanish',None),('spa','',None),
    ('por','Brasil','brazilian'),('por','Portugal','european'),('por','Portuguese',None),
])
def test_variety_comes_only_from_a_named_region(lang,accent,variety):
    assert cv.variety(lang,accent)==variety


def test_cv_authorized_archive_and_votes(tmp_path):
    packed=tmp_path/'cv.tgz'
    metadata='client_id\tpath\tsentence\tup_votes\tdown_votes\taccents\tlocale\n'
    metadata+='alice\ta.wav\thello\t2\t0\tUnited States English\ten\n'
    metadata+='bob\tb.wav\thello\t1\t0\t\ten\n'
    metadata+='carol\tc.wav\thello\t9\t1\t\ten\n'
    metadata+='dave\td.wav\thello\t9\t0\tnon-native\ten\n'
    archive(packed,{'en/validated.tsv':metadata,'en/clips/a.wav':wav(44100)})
    args=SimpleNamespace(archive=packed,dataset_url='https://mozilladatacollective.com/datasets/test',
                         license='CC0-1.0',lang='eng',hours=15,cache_dir=tmp_path/'cache',data_dir=tmp_path/'audio')
    cv.acquire(args)
    manifest=args.data_dir/'eng'/'manifest.jsonl'
    before=manifest.read_bytes()
    row=json.loads(before)
    assert row['voice']=='cv:alice'
    assert row['source']=='cv'
    report=json.loads((args.cache_dir/'cv-eng-summary.json').read_text())
    assert report['metadata_coverage']['validated_rows']==4
    assert report['metadata_coverage']['rejected_votes']==2
    cv.acquire(args)
    assert manifest.read_bytes()==before


def test_archive_traversal_rejected(tmp_path):
    packed=tmp_path/'bad.tgz'
    archive(packed,{'../escape':'bad'})
    with pytest.raises(ValueError,match='Unsafe archive path'):
        extract_archive(packed,tmp_path/'extract')
