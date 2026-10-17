#!/usr/bin/env python3
"""Download Mozilla Data Collective datasets whose terms you've accepted.

Terms can only be accepted in the website; after that, the API key in
MOZILLA_DATA_COLLECTIVE_API_KEY (profile credentials) fetches presigned URLs.
Downloads resume, are sha256-checked, and wait out the 30/day 429 limit, so a
long list finishes over several days unattended.

    scripts/py-linux.sh data/download_mdc.py --out .work/cv-archives ID...
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

API = 'https://mozilladatacollective.com/api/datasets'


def presign(dataset_id):
    request = urllib.request.Request(f'{API}/{dataset_id}/download', method='POST', headers={
        'Authorization': f'Bearer {os.environ["MOZILLA_DATA_COLLECTIVE_API_KEY"]}',
        # Cloudflare rejects urllib's default agent (error 1010).
        'User-Agent': 'curl/8'})
    while True:
        try:
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise RuntimeError(f'{dataset_id}: {error.code} {error.read().decode()}') from error
            wait = int(error.headers.get('Retry-After', 3600))
            print(f'{dataset_id}: rate limited, sleeping {wait}s', flush=True)
            time.sleep(wait)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset_id, out):
    done = out / f'{dataset_id}.json'
    if done.exists():
        return
    meta = presign(dataset_id)
    path = out / meta['filename']
    # A failed transfer is usually an expired URL; presign again and resume.
    while subprocess.run(['curl', '-fsSL', '--retry', '5', '-C', '-', '-o', path, meta['downloadUrl']]).returncode:
        meta = presign(dataset_id)
    if path.stat().st_size != int(meta['sizeBytes']) or sha256(path) != meta['checksum']:
        path.unlink()
        raise RuntimeError(f'{dataset_id}: {path.name} failed size/checksum; deleted')
    done.write_text(json.dumps({k: meta[k] for k in ('filename', 'sizeBytes', 'checksum')}, indent=1))
    print(f'{dataset_id}: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('ids', nargs='+')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset_id in args.ids:
        download(dataset_id, args.out)


if __name__ == '__main__':
    main()
