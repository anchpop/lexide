#!/usr/bin/env python3
"""Download/import the complete public OpenSLR AISHELL-1 and AISHELL-3 corpora.

No ASR or labels run here. These sources remain quarantined by the strict audit
coverage gate. Pass --archive to import an already downloaded official archive.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from corpus_import import extract_archive, importer, convert_audio, summarize, atomic_json

URLS = {
    'aishell1': 'https://openslr.trmal.net/resources/33/data_aishell.tgz',
    'aishell3': 'https://openslr.trmal.net/resources/93/data_aishell3.tgz',
}


def transcripts(source, root):
    result = {}
    if source == 'aishell1':
        path, = root.rglob('aishell_transcript_v0.8.txt')
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            utterance, text = line.split(maxsplit=1)
            result[utterance] = {'sentence': ''.join(text.split()), 'original_transcript': text,
                                 'transcript_path':str(path.relative_to(root))}
    else:
        for path in sorted(root.rglob('content.txt')):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                name, annotation = line.split(maxsplit=1)
                fields = annotation.split()
                if len(fields) % 2:
                    raise ValueError(f'{path}: unpaired Hanzi/pinyin: {line}')
                utterance = Path(name).stem
                if utterance in result:
                    raise ValueError(f'Duplicate AISHELL-3 transcript: {utterance}')
                result[utterance] = {'sentence': ''.join(fields[::2]),
                    'annotated_pinyin':' '.join(fields[1::2]), 'original_annotation':annotation,
                    'transcript_path':str(path.relative_to(root))}
    if not result:
        raise ValueError(f'{source}: no transcripts found')
    return result


def acquire(args):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Main archives + extraction + PCM conversion require well below 150 GiB.
    if shutil.disk_usage(args.cache_dir).free < 150 * 1024**3:
        raise RuntimeError('Need at least 150 GiB free before AISHELL acquisition')
    archive = args.archive or args.cache_dir / Path(URLS[args.source]).name
    if not args.archive:
        partial = archive.with_suffix(archive.suffix + '.part')
        if not archive.exists():
            subprocess.run(['curl','--fail','--location','--retry','5','--continue-at','-',
                            '--output',str(partial), URLS[args.source]], check=True)
            partial.replace(archive)
    extracted = args.cache_dir / args.source
    extract_archive(archive, extracted)
    if args.source == 'aishell1':
        for nested in sorted(extracted.rglob('*.tar.gz')):
            extract_archive(nested, nested.parent / nested.name.removesuffix('.tar.gz'))
    text = transcripts(args.source, extracted)
    missing = []
    mapped = set()
    with importer(args.data_dir, 'zho-hans', args.source, args.cache_dir) as (root, rows, stream):
        existing = {r['file']:r for r in rows}
        pending = []
        for path in sorted(extracted.rglob('*.wav')):
            utterance = path.stem
            if utterance not in text:
                missing.append(str(path.relative_to(extracted)))
                continue
            mapped.add(utterance)
            speaker = path.parent.name
            row = {'file':f'{args.source}_{utterance}.wav', 'source':args.source,
                   'voice':f'{args.source}:{speaker}', 'speaker_id':speaker, 'utterance_id':utterance,
                   'license':'Apache-2.0', 'archive_url':URLS[args.source],
                   'original_audio':str(path.relative_to(extracted)), **text[utterance]}
            if row['file'] in existing:
                old = existing[row['file']]
                if old['sentence'] != row['sentence'] or old['voice'] != row['voice']:
                    raise ValueError(f'Existing provenance differs: {row["file"]}')
                continue
            pending.append((row, path))
            existing[row['file']] = row
        def convert(item):
            row, path = item
            if not convert_audio(root, row, path):
                raise ValueError(f'Empty audio: {path}')
            return row
        print(f'{args.source}: converting {len(pending)} new clips', flush=True)
        # Bounded work queue; manifest writes remain on the one owning thread.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(pending), 128):
                for row in pool.map(convert, pending[start:start+128]):
                    stream.write((json.dumps(row, ensure_ascii=False)+'\n').encode())
                    stream.flush()
                if start % 1280 == 0:
                    print(f'{args.source}: {min(start+128,len(pending))}/{len(pending)} converted', flush=True)
    report = summarize(args.data_dir, 'zho-hans', args.source)
    report.update({'untranscribed_audio':missing, 'transcripts_without_audio':sorted(set(text)-mapped)})
    atomic_json(args.cache_dir / f'{args.source}-summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=URLS, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent/'audio')
    parser.add_argument('--cache-dir', type=Path, required=True)
    acquire(parser.parse_args())


if __name__ == '__main__':
    main()
