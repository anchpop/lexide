#!/usr/bin/env python3
"""Publish an immutable private packed-corpus snapshot, with exclusion sidecars."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

# Classic LFS avoids a second local Xet cache for this already-packed payload.
os.environ['HF_HUB_DISABLE_XET'] = '1'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--tar', type=Path, default=ROOT / '.work/pron_audio.tar')
    parser.add_argument('--repo', default='anchpop/lexide-pronunciation-pack')
    parser.add_argument('--snapshot', default='run2-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ'))
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    secrets = {}
    for line in (ROOT.parent / '.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            name, value = line.split('=', 1)
            secrets[name.strip()] = value.strip().strip('\"\'')
    api = HfApi(token=secrets['HF_TOKEN'])
    api.create_repo(args.repo, repo_type='dataset', private=True, exist_ok=True)
    assert api.dataset_info(args.repo).private, 'Refusing to upload private corpus to a public repo'
    stage = ROOT / '.work' / args.snapshot
    stage.mkdir(exist_ok=True)
    metadata_path = stage / 'snapshot.json'
    if not metadata_path.exists():
        parts, total = [], hashlib.sha256()
        with args.tar.open('rb') as source:
            index = 0
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                path = stage / f'pron_audio.tar.{index:03d}.part'
                part_hash, size = hashlib.sha256(), 0
                with path.open('wb') as dest:
                    while block:
                        dest.write(block)
                        part_hash.update(block)
                        total.update(block)
                        size += len(block)
                        if size >= 8 * 1024 ** 3:
                            break
                        block = source.read(8 * 1024 * 1024)
                parts.append({'path': path.name, 'size': size, 'sha256': part_hash.hexdigest()})
                print(f'Prepared {path.name}: {size} bytes', flush=True)
                index += 1
        sidecars = stage / 'sidecars'
        sidecars.mkdir(exist_ok=True)
        for path in sorted((ROOT / 'train').glob('*_exclusions.jsonl')):
            dest = sidecars / path.name
            shutil.copyfile(path, dest)
            parts.append({'path': str(dest.relative_to(stage)), 'size': dest.stat().st_size, 'sha256': sha256(dest)})
        # Keep the admission/audit certification with the exact uploaded pack.
        for name in ('admitted-after.json', 'audit-summary.json', 'tts-plan-final.json',
                     'word-candidates.provenance.json', 'tts-costs.json',
                     'spanish-acoustics/speaker-results.json',
                     'quarantine/read-speech-unsupported/unsupported-phones.json',
                     'quarantine/word-audit-failures/tts-audit-failures-for-quarantine.json'):
            path = ROOT / '.work/run2' / name
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            parts.append({'path': name, 'size': dest.stat().st_size, 'sha256': sha256(dest)})
        metadata = {'snapshot': args.snapshot, 'tar': {'size': args.tar.stat().st_size, 'sha256': total.hexdigest()},
                    'files': parts, 'restore': 'Concatenate pron_audio.tar.*.part in numeric order; extract into data/audio (language directories are tar roots). Copy sidecars into train/.'}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    metadata = json.loads(metadata_path.read_text())
    assert args.tar.stat().st_size == metadata['tar']['size'] and sha256(args.tar) == metadata['tar']['sha256']
    for item in metadata['files']:
        path = stage / item['path']
        assert path.stat().st_size == item['size'] and sha256(path) == item['sha256'], item['path']
    prefix = f"snapshots/{args.snapshot}"
    tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(args.repo, repo_type='dataset').tags}
    head = api.dataset_info(args.repo, files_metadata=True)
    occupied = any(s.rfilename.startswith(prefix + '/') for s in head.siblings)
    if args.snapshot in tags:
        revision = tags[args.snapshot]  # retry verification; never overwrite a named snapshot
    elif occupied:
        # A previous upload may have committed before tag creation. Verify that
        # exact commit below before attaching the still-unused immutable name.
        revision = head.sha
    else:
        commit = api.upload_folder(folder_path=str(stage), path_in_repo=prefix, repo_id=args.repo,
                                   repo_type='dataset', commit_message=f'Private pronunciation pack {args.snapshot}')
        revision = commit.oid
    info = api.dataset_info(args.repo, revision=revision, files_metadata=True)
    assert info.private and info.sha == revision
    remote = {s.rfilename: s for s in info.siblings}
    expected = metadata['files'] + [{'path': 'snapshot.json', 'size': metadata_path.stat().st_size,
                                      'sha256': sha256(metadata_path)}]
    for item in expected:
        name = f"{prefix}/{item['path']}"
        sibling = remote[name]
        assert sibling.size == item['size'], name
        if sibling.lfs:
            remote_hash = sibling.lfs.sha256 if hasattr(sibling.lfs, 'sha256') else sibling.lfs['sha256']
        else:
            downloaded = Path(hf_hub_download(args.repo, name, repo_type='dataset', revision=revision, token=secrets['HF_TOKEN']))
            remote_hash = sha256(downloaded)
        assert remote_hash == item['sha256'], name
    if args.snapshot not in tags:
        api.create_tag(args.repo, tag=args.snapshot, revision=revision, repo_type='dataset')
    assert api.dataset_info(args.repo, revision=args.snapshot).sha == revision
    result = {'repo': args.repo, 'revision': revision, 'tag': args.snapshot, 'private': info.private,
              'tar': metadata['tar'], 'verified_files': len(expected)}
    (ROOT / '.work/run2/hf-upload.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
