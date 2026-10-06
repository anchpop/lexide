"""Append-only imports shared by authorized Common Voice and public AISHELL."""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import soundfile as sf
from download_mls import decode_audio, fingerprint, manifest_lock, atomic_json


def extract_archive(archive: Path, destination: Path):
    """Only regular archive members, never links or paths outside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.extracted.json'
    identity = {'archive': str(archive.resolve()), 'size': archive.stat().st_size}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f'{destination}: different archive already extracted')
        return
    with tarfile.open(archive, 'r|*') as tar:
        for member in tar:
            path = destination / member.name
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe archive path: {member.name}')
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open('wb') as out:
                    shutil.copyfileobj(src, out)
            else:
                raise ValueError(f'Unsupported archive member: {member.name}')
    atomic_json(marker, identity)


def read_rows(path):
    if not path.exists():
        return []
    result = []
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError(f'{path}: existing row lacks newline')
            if line.strip():
                result.append(json.loads(line))
    if len({r['file'] for r in result}) != len(result):
        raise ValueError(f'{path}: duplicate filenames')
    return result


@contextmanager
def importer(data_dir, lang, source, state_dir):
    root = data_dir / lang
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.jsonl'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f'{source}-{lang}-prefix.json'
    with manifest_lock(manifest):
        if not state_path.exists():
            atomic_json(state_path, fingerprint(manifest))
        prefix = json.loads(state_path.read_text())
        if fingerprint(manifest, prefix['bytes']) != prefix:
            raise ValueError(f'{manifest}: original prefix changed')
        rows = read_rows(manifest)
        with manifest.open('ab') as stream:
            yield root, rows, stream
            stream.flush()
            os.fsync(stream.fileno())
        if fingerprint(manifest, prefix['bytes']) != prefix:
            raise ValueError(f'{manifest}: original prefix changed during import')


def convert_audio(root, row, path, max_seconds=None):
    audio = decode_audio(path.read_bytes())
    duration = len(audio) / 16000
    if not duration or (max_seconds is not None and duration > max_seconds):
        return False
    row['duration_sec'] = duration
    # libsndfile syncs a disk fd on close; doing that for every tiny clip stalls
    # bcachefs. Encode in memory, then atomically rename a buffered file, as for
    # MLS's process-resumable (not power-loss-durable) audio writes.
    encoded = io.BytesIO()
    sf.write(encoded, audio, 16000, format='WAV', subtype='PCM_16')
    with tempfile.NamedTemporaryFile(dir=root, suffix='.part.wav', delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(encoded.getvalue())
            out.close()
            temporary.replace(root / row['file'])
        finally:
            temporary.unlink(missing_ok=True)
    return True


def append_audio(root, stream, row, path, max_seconds=None):
    if not convert_audio(root, row, path, max_seconds):
        return False
    stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
    stream.flush()
    return True


def summarize(data_dir, lang, source):
    rows = [r for r in read_rows(data_dir / lang / 'manifest.jsonl') if r.get('source') == source]
    checks = []
    for row in rows[::max(1, len(rows)//10)][:10]:
        info = sf.info(data_dir / lang / row['file'])
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16'
        assert abs(info.duration - row['duration_sec']) < 1/16000
        checks.append({'file':row['file'], 'duration':info.duration, 'sentence':row['sentence'], 'voice':row['voice'], 'annotated_pinyin':row.get('annotated_pinyin')})
    return {'source':source, 'lang':lang, 'clips':len(rows), 'hours':sum(r['duration_sec'] for r in rows)/3600,
            'speakers':len({r['voice'] for r in rows}), 'spotchecks':checks}
