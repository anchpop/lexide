"""Cache clean Chirp3 clips, exact-grid CTC timing, and Praat pitch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import parselmouth
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'espeak_audit'))
from production_alignment import viterbi_spans

OUT = Path(__file__).parent / 'out'
AUDIO = ROOT / 'data/audio/eng'
MODEL = ROOT / '.work/onnx-small'


def make_session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(MODEL / 'model.onnx'), options,
                               providers=['CPUExecutionProvider'])


def prepare(selected, destination):
    destination.mkdir(parents=True, exist_ok=True)
    meta = json.loads((MODEL / 'frame_matrix.json').read_text())['heads']['phone']
    vocabulary, blank = meta['labels'], meta['blank_id']
    identity = hashlib.sha256((MODEL / 'model.onnx').read_bytes()).hexdigest()
    session = make_session()
    accepted, rejected = [], []
    for row in selected:
        row = row.copy()
        cache = destination / (Path(row['file']).stem + '.npz')
        key_row = {k: v for k, v in row.items() if k not in ('cache_key', 'split')}
        key = hashlib.sha256(json.dumps([identity, key_row, 'praat-65-500-grid-v1'], sort_keys=True).encode()
                             + (AUDIO / row['file']).read_bytes()).hexdigest()
        row['cache_key'] = key
        if cache.exists() and str(np.load(cache)['cache_key']) == key:
            accepted.append(row)
            continue
        audio, sr = sf.read(AUDIO / row['file'], dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(1)
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= .999 or np.sqrt(np.mean(audio**2)) < .005:
            rejected.append(dict(file=row['file'], reason='nonfinite/clipped/quiet'))
            continue
        heads = dict(zip([o.name for o in session.get_outputs()], session.run(None, {
            'waveform': audio[None], 'lengths': np.array([len(audio)], dtype=np.int64)})))
        count = (len(audio)-400)//320+1
        scores = heads['phone'][0, :count]
        targets = [vocabulary.index(p) for p in row['phonemes']]
        spans, error = viterbi_spans(scores, targets, blank, count)
        if error:
            rejected.append(dict(file=row['file'], reason=str(error)))
            continue
        centers = np.array([(start+end-1)/2 for start, end, _ in spans])*.02 + .0125
        boundaries = np.r_[0., (centers[:-1]+centers[1:])/2, len(audio)/16000]
        assert (np.diff(boundaries) > 0).all()
        times = np.arange(len(audio)//160 + 1)*.01
        index = np.searchsorted(boundaries[1:-1], times, side='right')
        relative = (times-boundaries[index])/(boundaries[index+1]-boundaries[index])
        pitch = parselmouth.Sound(audio, 16000).to_pitch(time_step=.01, pitch_floor=65,
                                                        pitch_ceiling=500)
        measured = pitch.selected_array['frequency']
        voiced = np.interp(times, pitch.xs(), (measured > 0).astype(float), left=0, right=0)
        good = measured > 0
        if not good.any():
            rejected.append(dict(file=row['file'], reason='unvoiced'))
            continue
        f0 = np.interp(times, pitch.xs()[good], measured[good])
        np.savez(cache, audio=audio, f0=f0.astype(np.float32), voiced=voiced.astype(np.float32),
                 phone=np.array(targets)[index], relative=relative.astype(np.float32),
                 boundaries=boundaries, phonemes=np.array(row['phonemes']), cache_key=key)
        sf.write(destination / (Path(row['file']).stem+'-original.wav'), audio, 16000, subtype='FLOAT')
        accepted.append(row)
        if len(accepted) % 50 == 0:
            print('prepared', len(accepted), 'rejected', len(rejected), flush=True)
    (destination / 'clips.json').write_text(json.dumps(accepted, indent=2))
    (destination / 'provenance.json').write_text(json.dumps(dict(model_sha256=identity,
        rejected=rejected, grid_hz=100), indent=2))
    return accepted


def select(count):
    labels = {r['file']: r for r in map(json.loads, (AUDIO / 'phonemes.jsonl').open())}
    selected = []
    for row in map(json.loads, (AUDIO / 'manifest.jsonl').open()):
        if (row['source'] != 'tts' or 'Chirp3-HD' not in (row.get('voice') or '')
                or row['file'] not in labels):
            continue
        info = sf.info(AUDIO / row['file'])
        phones = labels[row['file']]['phonemes']
        if 1.2 <= info.duration <= 4 and 8 <= len(phones) <= 45:
            selected.append(row | {'phonemes': phones})
        if len(selected) == count:
            break
    assert len(selected) == count, len(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['ablation', 'corpus'])
    args = parser.parse_args()
    if args.stage == 'ablation':
        rows = prepare(select(10), OUT / 'ablation-data')
        assert len(rows) == 10
    else:
        rows = prepare(select(2050), OUT / 'corpus')
        assert len(rows) >= 2000, len(rows)
        # This corpus has one recording per sentence: a seen-sentence heldout
        # category would be fabricated. Every heldout sentence is genuinely new.
        ablation_sentences = {r['sentence'] for r in select(10)}
        eligible = sorted({r['sentence'] for r in rows} - ablation_sentences,
                          key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        heldout = set(eligible[:20])
        for row in rows:
            row['split'] = 'heldout' if row['sentence'] in heldout else 'train'
        train_sentences = {r['sentence'] for r in rows if r['split'] == 'train'}
        assert not train_sentences & heldout
        assert len({r['voice'] for r in rows if r['split'] == 'train'}) > 1
        (OUT / 'corpus/clips.json').write_text(json.dumps(rows, indent=2))
        print('split', len(rows)-20, 'train, 20 heldout', flush=True)


if __name__ == '__main__':
    main()
