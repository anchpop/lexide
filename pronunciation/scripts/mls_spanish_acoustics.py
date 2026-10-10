#!/usr/bin/env python3
"""Conservative same-speaker Spanish MLS sibilance investigation.

The pinned model supplies boundaries ONLY. Decisions use waveform spectra and
intensity, never model readings, scores or posterior-derived dialect labels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'espeak_audit'))
from production_alignment import ModelIdentity, align_batch

IDENTITY = ModelIdentity('anchpop/lexide-pronunciation-merged', '95f4b185676627ffe566e8760349ebb42cc55dde')
VOWELS = set('aeiouɛɔɐ')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(work, clips_per_speaker=24):
    speakers = defaultdict(list)
    for row in read(ROOT / 'data/audio/spa/manifest.jsonl'):
        if row.get('source') == 'mls':
            speakers[row['voice']].append(row)
    work.mkdir(parents=True, exist_ok=True)
    selected = []
    for speaker, rows in sorted(speakers.items()):
        candidates = [r for r in rows if re.search(r'c[eéií]|z[aeiouáéíóú]', r['sentence'].lower())
                      and re.search(r's[aeiouáéíóú]', r['sentence'].lower())]
        # Hash order avoids taking just the first contiguous chapter. It does
        # not assume that book metadata provides an acoustic recording session.
        candidates.sort(key=lambda r: hashlib.sha256(r['file'].encode()).hexdigest())
        chosen = candidates[:clips_per_speaker]
        selected.extend({**r, 'lang': 'spa', 'word': r['sentence']} for r in chosen)
    (work / 'targets-input.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    (work / 'speakers.json').write_text(json.dumps({s: {'clips': len(r), 'hours': sum(x['duration_sec'] for x in r) / 3600}
                                                for s, r in speakers.items()}, indent=2))
    print(f'{len(speakers)} speakers; {len(selected)} clips selected for boundaries')


def spectrum(audio, sr, start, end):
    segment = audio[max(0, int(start * sr)):int(end * sr)]
    if len(segment) < sr * .025:
        return None
    rms = np.sqrt(np.mean(segment ** 2))
    if rms < 1e-5:
        return None
    power = abs(np.fft.rfft(segment * np.hanning(len(segment)), n=2048)) ** 2
    freq = np.fft.rfftfreq(2048, 1 / sr)
    band = (freq >= 1000) & (freq <= min(7500, sr / 2))
    energy = power[band].sum()
    if energy <= 1e-12:
        return None
    centroid = np.sum(power[band] * freq[band]) / energy
    high_fraction = power[(freq >= 3500) & (freq <= 7500)].sum() / energy
    flatness = np.exp(np.log(power[band] + 1e-20).mean()) / (power[band].mean() + 1e-20)
    return float(centroid), float(20 * np.log10(rms)), float(high_fraction), float(flatness)


def features(row, aligned):
    if aligned['align_error'] or aligned['keep'] != list(range(len(row['phonemes']))):
        return []
    audio, sr = sf.read(ROOT / 'data/audio/spa' / row['file'], dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    phones, spans = row['phonemes'], aligned['spans']
    result = []
    for i, phone in enumerate(phones[:-1]):
        if phone not in {'s', 'θ'} or phones[i + 1] not in VOWELS:
            continue
        start, end = spans[i][:2]
        center = (start + end) / 2
        vowel_start, vowel_end = spans[i + 1][:2]
        vowel_center = (vowel_start + vowel_end) / 2
        # Don't expand into the neighboring vowel. A CTC emission alone may
        # be only 20 ms: allow a 40 ms DSP window around its center, not a
        # claimed full phone duration. Shift checks below reject unstable cues.
        if vowel_center - center < .05 or (i and center - sum(spans[i - 1][:2]) / 2 < .04):
            continue
        vowel = spectrum(audio, sr, vowel_center - .02, vowel_center + .02)
        measurements = [spectrum(audio, sr, center + shift - .02, center + shift + .02)
                        for shift in (-.01, 0, .01)]
        if vowel is None or any(m is None for m in measurements):
            continue
        values = np.array(measurements)
        if np.ptp(values[:, 0]) > 1500 or np.ptp(values[:, 1]) > 6:
            continue
        centroid, intensity, high, flat = np.median(values, axis=0)
        if flat < .03:  # require aperiodic energy, not a vowel/harmonic sliver
            continue
        result.append({'file': row['file'], 'speaker': row['voice'], 'phone_index': i,
                       'class': 'target' if phone == 'θ' else 'control',
                       'centroid_hz': float(centroid), 'relative_db': float(intensity - vowel[1]),
                       'high_fraction': float(high), 'flatness': float(flat), 'center_sec': center})
    return result


def measure(work):
    rows = [r for r in read(work / 'targets.jsonl') if r.get('phonemes')]
    cache = work / 'boundaries'
    cache.mkdir(exist_ok=True)
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset + 64]
        todo = [r for r in batch if not (cache / (r['file'] + '.json')).exists()]
        if todo:
            aligned = align_batch([(ROOT / 'data/audio/spa' / r['file'], 'spa', r['phonemes']) for r in todo], IDENTITY)
            for row, result in zip(todo, aligned):
                # Explicitly discard readings and per-span likelihoods.
                keep = {k: result[k] for k in ('keep', 'align_error')}
                keep['spans'] = [s[:2] for s in result['spans']]
                keep['identity'] = IDENTITY.as_dict()
                keep['phones'] = row['phonemes']
                keep['audio_sha256'] = hashlib.sha256((ROOT / 'data/audio/spa' / row['file']).read_bytes()).hexdigest()
                (cache / (row['file'] + '.json')).write_text(json.dumps(keep))
        print(f'Boundaries {min(offset + 64, len(rows))}/{len(rows)}', flush=True)
    observations = []
    for row in rows:
        alignment = json.loads((cache / (row['file'] + '.json')).read_text())
        assert alignment['identity'] == IDENTITY.as_dict() and alignment['phones'] == row['phonemes']
        assert alignment['audio_sha256'] == hashlib.sha256((ROOT / 'data/audio/spa' / row['file']).read_bytes()).hexdigest()
        observations.extend(features(row, alignment))
    (work / 'tokens.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in observations))
    summarize(work)


def summarize(work):
    observations = read(work / 'tokens.jsonl')
    speakers = json.loads((work / 'speakers.json').read_text())
    rng = np.random.default_rng(42)
    for speaker, output in speakers.items():
        output.update(variety=None, reason='insufficient_evidence')
        classes = {}
        for category in ('target', 'control'):
            group = [r for r in observations if r['speaker'] == speaker and r['class'] == category]
            clips = defaultdict(list)
            for row in group:
                clips[row['file']].append([row['centroid_hz'], row['relative_db'], row['high_fraction']])
            values = np.array([np.median(v, axis=0) for v in clips.values()])
            output[category] = {'tokens': len(group), 'clips': len(clips),
                                'median': np.median(values, axis=0).tolist() if len(values) else None}
            if len(group) >= 20 and len(clips) >= 8:
                classes[category] = values
        if len(classes) != 2:
            continue
        # Resample CLIPS, not correlated neighboring spectral frames.
        bootstrap = []
        for _ in range(1000):
            sampled = {k: np.median(v[rng.integers(0, len(v), len(v))], axis=0) for k, v in classes.items()}
            bootstrap.append(sampled['target'] - sampled['control'])
        low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        output['difference_95ci'] = {'low': low.tolist(), 'high': high.tolist()}
        output['reason'] = 'ambiguous_acoustics'
        # Require a sibilant control baseline before any dialect decision.
        control = output['control']['median']
        if control[0] < 3500 or control[2] < .45:
            output['reason'] = 'weak_control_sibilance'
        elif high[0] < -1000 and high[1] < -6 and high[2] < -.15:
            output.update(variety='european', reason='separated_nonsibilant_targets')
        elif low[0] > -600 and high[0] < 600 and low[1] > -3 and high[1] < 3 and low[2] > -.1 and high[2] < .1:
            output.update(variety='latin_american', reason='equivalent_sibilant_targets')
    result = {'identity': IDENTITY.as_dict(), 'method': 'spectral-centroid-intensity-clip-bootstrap-v1', 'speakers': speakers}
    (work / 'speaker-results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({v: sum(r['variety'] == v for r in speakers.values()) for v in ('european', 'latin_american', None)}))


def apply_metadata(data_root, work, quarantine, expected_por_count, g2p_identity):
    """Keep decided speakers or dialect-invariant clips; quarantine other MLS."""
    import shutil

    result_path = work / 'speaker-results.json'
    evidence = json.loads(result_path.read_text())
    assert evidence['identity'] == IDENTITY.as_dict()
    evidence_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    decisions = evidence['speakers']
    invariant_rows = read(work / 'variety-invariant.jsonl')
    invariants = {row['file']: row for row in invariant_rows}
    assert len(invariants) == len(invariant_rows), 'Duplicate variety-invariance results'
    assert all(row['g2p_identity'] == g2p_identity
               and type(row['variety_invariant']) is bool for row in invariant_rows)
    recovered = {'clips': 0, 'hours': 0.}
    plans = {}
    for lang in ('por', 'spa'):
        directory = data_root / lang
        raw = (directory / 'manifest.jsonl').read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in raw]
        mls = [r for r in records if r.get('source') == 'mls']
        if lang == 'por':
            assert len(mls) == expected_por_count, 'Portuguese MLS census changed'
        else:
            assert set(r['voice'] for r in mls) == set(decisions), 'Spanish speaker census changed'
            assert all(sum(r['voice'] == s for r in mls) == d['clips'] for s, d in decisions.items())
            undecided = [r for r in mls if decisions[r['voice']]['variety'] is None]
            assert set(invariants) == {r['file'] for r in undecided}, 'Incomplete variety-invariance census'
            assert all(invariants[r['file']]['expected_sha256'] == hashlib.sha256(
                r['sentence'].encode()).hexdigest() for r in undecided), 'Stale variety-invariance text'
        # Existing MLS phone labels would become stale after a dialect change.
        # Run2 has none; do not silently rewrite or preserve mislabelled rows.
        labels = directory / 'phonemes.jsonl'
        assert not labels.exists() or not any(r.get('source') == 'mls' for r in read(labels))
        kept, removed = [], []
        for line, row in zip(raw, records):
            if row.get('source') != 'mls':
                kept.append(line)
            elif lang == 'por':
                removed.append((line, row))
            else:
                variety = decisions[row['voice']]['variety']
                if variety is None:
                    if not invariants[row['file']]['variety_invariant']:
                        removed.append((line, row))
                        continue
                    # This is a label-routing choice, NOT a rescued speaker
                    # decision. Both varieties produce identical phones/stress.
                    variety = 'european'
                    row['variety_invariant'] = True
                    row['variety_invariant_g2p_identity'] = g2p_identity
                    recovered['clips'] += 1
                    recovered['hours'] += row['duration_sec'] / 3600
                row['variety'] = variety
                row['dialect_evidence_sha256'] = evidence_hash
                kept.append((json.dumps(row, ensure_ascii=False) + '\n').encode())
        plans[lang] = (kept, removed)
    assert not quarantine.exists(), 'Quarantine already exists; inspect before retrying'
    summary = {}
    for lang, (kept, removed) in plans.items():
        directory = data_root / lang
        dest = quarantine / lang
        dest.mkdir(parents=True)
        shutil.copyfile(directory / 'manifest.jsonl', dest / 'manifest.original.jsonl')
        (dest / 'manifest.jsonl').write_bytes(b''.join(line for line, _ in removed))
        removed_files = {r['file'] for _, r in removed}
        for _, row in removed:
            target = dest / 'audio' / row['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(directory / row['file'], target)
        for name in ('manifest.jsonl', 'vad.jsonl'):
            path = directory / name
            if not path.exists():
                continue
            if name == 'manifest.jsonl':
                content = b''.join(kept)
            else:
                original = path.read_bytes().splitlines(keepends=True)
                selected = [line for line in original if json.loads(line)['file'] in removed_files]
                if not selected:
                    continue
                (dest / name).write_bytes(b''.join(selected))
                content = b''.join(line for line in original if json.loads(line)['file'] not in removed_files)
            temp = path.with_suffix('.run2.tmp')
            temp.write_bytes(content)
            temp.replace(path)
        summary[lang] = {'quarantined': len(removed), 'retained_manifest_rows': len(kept)}
    summary['variety_invariant_recovered'] = recovered
    (quarantine / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


def apply(work):
    sys.path.insert(0, str(ROOT / 'data'))
    from tts_words import assert_no_running_audit
    assert_no_running_audit()
    with (work.parent / 'word-phones.jsonl').open() as source:
        identity = json.loads(next(source))['g2p_identity']
    apply_metadata(ROOT / 'data/audio', work, ROOT / '.work/run2/quarantine', 9797, identity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'measure', 'summarize', 'apply'])
    parser.add_argument('--work', type=Path, default=ROOT / '.work/run2/spanish-acoustics')
    parser.add_argument('--clips-per-speaker', type=int, default=24)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.work, args.clips_per_speaker)
    else:
        globals()[args.stage](args.work)
