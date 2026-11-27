"""Compare the Rust runtime against onnxruntime on dumps from `cargo run --release --example trace`.

Per stage: feed onnxruntime the exact inputs ours saw and compare outputs. Pipeline: run the whole chain in
onnxruntime from the same tokens and controls, delivery draw z, source noise and pitch offsets, and compare the
raw waves. With ORT_TIMES=n, also time that chain
n times per utterance (ORT_THREADS sets intra-op threads, 0 = onnxruntime's default).
    python parity.py <bundle.voca> <trace dir> [wav dir]
"""
import json, os, struct, sys, time, wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

bundle, root = Path(sys.argv[1]), Path(sys.argv[2])
wavs = Path(sys.argv[3]) if len(sys.argv) > 3 else None
threads = int(os.environ.get('ORT_THREADS', 1))
b = bundle.read_bytes()
n = struct.unpack('<I', b[:4])[0]
manifest = json.loads(b[4:4 + n])
sessions, offset = {}, 4 + n
for name, size in manifest['graphs'].items():
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    sessions[name] = ort.InferenceSession(b[offset:offset + size], o, providers=['CPUExecutionProvider'])
    offset += size


def run(stage, values):
    s = sessions[stage]
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, {i.name: values[i.name] for i in s.get_inputs()})))


def load(d):
    return {p.stem.replace('~', '/'): np.load(p) for p in d.glob('*.npy')}


def compare(ours, theirs):
    if ours.dtype.kind in 'biu':
        return ('exact' if ours.shape == theirs.shape and (ours == theirs).all()
                else f'MISMATCH {int((ours != theirs).sum()) if ours.shape == theirs.shape else "shape"}')
    err = np.abs(ours.astype(np.float64) - theirs)
    return f'{err.max():.2e} abs, {err.max() / max(np.abs(theirs).max(), 1e-30):.2e} rel'


def chain(common, z, render, variety, fix_ending):
    """The engine's stage chain in onnxruntime; returns the final reading and the wave (None if frame counts differ)."""
    common = {**common, **run('encode', common)}

    def predict(draw):
        p = run('predict', {**common, 'draw': draw})
        d = p['durations'][0]
        r = {**common, **p, 'owner': np.repeat(np.arange(len(d)), d)[None]}
        return {**r, **run('melody', r)}
    spread = (np.float32(variety) * common['scale']).astype(np.float32)
    reading = predict((common['mean'] + z.reshape(common['mean'].shape) * spread).astype(np.float32))
    if fix_ending and variety != 0 and reading['rising']:
        typical = predict(common['mean'])
        reading['prosody'] = run('blend', {**reading, 'typical': typical['prosody'], 'typical_durations': typical['durations']})['blended']
    if render['noise'].shape[2] != reading['owner'].size * manifest['hop']:
        return reading, None
    return reading, run('render', {**reading, 'noise': render['noise'], 'pitch_shift': render['pitch_shift']})['wave'][0]


rows = []
for u in sorted(root.glob('u*')):
    calls = sorted(p for p in u.iterdir() if p.is_dir())
    for call in calls:
        theirs = run(call.name.split('_', 1)[1], load(call / 'in'))
        ours = load(call / 'out')
        for name in sorted(theirs):
            rows.append((u.name, call.name, name, compare(ours[name], theirs[name])))
    # Only the request scalars and tokens are shared with ours, plus the draw z and source noise.
    keys = ('tokens', 'lexical', 'speaker', 'speed', 'comma_pause', 'stop_pause', 'pitch_range', 'voiced_vowels')
    common = {k: v for call in calls for k, v in load(call / 'in').items() if k in keys}
    z, ours, options = np.load(u / 'z.npy'), np.load(u / 'wave.npy'), json.loads((u / 'options.json').read_text())
    render = load([c for c in calls if c.name.endswith('render')][0] / 'in')
    args = (common, z, render, options['variety'], options['fixEnding'])
    reading, theirs = chain(*args)
    if theirs is None:
        rows.append((u.name, 'pipeline', 'wave', 'MISMATCH frame count'))
        continue
    rows.append((u.name, 'pipeline', 'durations', compare(np.load(u / '01_predict' / 'out' / 'durations.npy'), reading['durations'])))
    rows.append((u.name, 'pipeline', 'wave', compare(ours, theirs)))
    snr = 10 * np.log10((theirs ** 2).sum() / max(((ours - theirs) ** 2).sum(), 1e-30))
    rows.append((u.name, 'pipeline', 'wave SNR', f'{snr:.1f} dB over {ours.size} samples'))
    times = []
    for _ in range(int(os.environ.get('ORT_TIMES', 0))):
        start = time.perf_counter()
        chain(*args)
        times.append(time.perf_counter() - start)
    if times:
        rtf = ' '.join(f'{theirs.size / manifest["sampleRate"] / t:.1f}' for t in times)
        rows.append((u.name, 'speed', f'ort native, {threads or "default"} threads', f'realtime x{rtf}'))
    if wavs:
        wavs.mkdir(parents=True, exist_ok=True)
        for who, w in (('ours', ours), ('ort', theirs)):
            with wave.open(str(wavs / f'{bundle.stem[:8]}-{u.name}-{who}.wav'), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(manifest['sampleRate'])
                f.writeframes((np.clip(w, -1, 1) * 32767).astype('<i2').tobytes())
for r in rows:
    print(' | '.join(r))
