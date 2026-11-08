"""Direct MRI contour supervision of phone-conditioned controls; no fitted labels.

Run with py-linux.sh -I mri_train.py {prepare,train,evaluate}. Annot16 outputs
are local research artifacts in ignored out/network-mri, never redistribute.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import parselmouth
import torch
import maeda
import mri_fit as fit
import usc
from network import ControlNet, NasalLoss
from objective import AudioLoss
from synth import Synth, effort, effort_weights, smoothness
import train as audio_train

OUT = Path(__file__).resolve().parent / 'out/network-mri'
OLD = OUT.parent / 'network-anatomy/epoch-19.pt'
HOLDOUT = ('sub013', 'sub022')
SURFACES = ('tongue', 'hard_palate', 'pharyngeal_wall')
# Broad ARPAbet categories, not inferred allophones; retain whole diphthongs.
ARPABET = dict(AA='ɑː', AE='æ', AH='ʌ', AO='ɔː', AW='aʊ', AY='aɪ',
    B='b', CH='tʃ', D='d', DH='ð', EH='ɛ', ER='ɚ', EY='eɪ', F='f', G='ɡ',
    HH='h', H='h', IH='ɪ', IY='iː', JH='dʒ', K='k', L='l', M='m', N='n',
    NG='ŋ', OW='oʊ', OY='ɔɪ', P='p', R='ɹ', S='s', SH='ʃ', T='t', TH='θ',
    UH='ʊ', UW='uː', V='v', W='w', Y='j', Z='z', ZH='ʒ')
SILENCE = {'', 'SIL', 'SP', 'SPN'}


def phone_label(phone):
    phone = phone.strip().upper()
    if phone in SILENCE:
        return '<pad>'
    if phone == 'AH0':
        return 'ə'
    return ARPABET.get(phone.rstrip('012'))


def timeline(intervals, times, vocabulary):
    phone = np.zeros(len(times), np.int64)
    relative = np.zeros(len(times), np.float32)
    valid = np.zeros(len(times), bool)
    labels = np.full(len(times), '', dtype='<U32')
    for a, b, label in intervals:
        mask = (times >= a) & (times < b)
        mapped = phone_label(label)
        if mapped is not None:
            phone[mask] = vocabulary.index(mapped)
            valid[mask] = mapped != '<pad>'
        relative[mask] = (times[mask]-a)/max(b-a, 1e-8)
        labels[mask] = label
    return dict(phone=phone, relative=relative, valid=valid, labels=labels)


def nearest(times, observed, tolerance):
    right = np.searchsorted(observed, times).clip(0, len(observed)-1)
    left = (right-1).clip(0)
    index = np.where(abs(observed[left]-times) < abs(observed[right]-times), left, right)
    return index, abs(observed[index]-times) <= tolerance


def checkpoint(path=OLD, extra=()):
    state = torch.load(path, map_location='cpu', weights_only=False)
    speakers = state['speakers'] + list(extra)
    net = ControlNet(len(state['vocabulary']), len(speakers))
    weights = state['state_dict'].copy()
    if extra:
        original = weights['speaker.weight']
        weights['speaker.weight'] = torch.cat([original, original.mean(0).repeat(len(extra), 1)])
    net.load_state_dict(weights)
    return net.cuda(), state | {'speakers': speakers}


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def prepare():
    OUT.mkdir(exist_ok=True, parents=True)
    cache = OUT / 'data'; cache.mkdir(exist_ok=True)
    _, state = checkpoint()
    vocabulary = state['vocabulary']
    speakers = sorted(p.name for p in usc.ANNOT.glob('sub*'))
    assert len(speakers) == 16
    registrations = json.loads((OUT/'registration.json').read_text()) if (OUT/'registration.json').exists() else {}
    previous = {r['stem']:r for r in json.loads((OUT/'data.json').read_text())} if (OUT/'data.json').exists() else {}
    rows = []; inventory = Counter(); unknown = Counter(); excluded = []
    for speaker in speakers:
        stems = []
        for path in sorted((usc.ANNOT / speaker / 'track').glob('*_track.mat')):
            stem = path.stem.removesuffix('_track')
            alignment = usc.alignment_path(stem)
            if not alignment.exists():
                excluded.append(dict(stem=stem, reason='no TextGrid for tracked utterance'))
                continue
            try:
                usc.intervals(alignment)
            except IndexError:
                excluded.append(dict(stem=stem, reason='TextGrid has no phones tier'))
                continue
            stems.append(stem)
        if speaker not in registrations:
            old = fit.OUT / f'{speaker}-fit.npz'
            registration_path = cache / f'{speaker}-registration.npz'
            if old.exists():
                global_ = fit.tensor(np.load(old)['global_'])
                origin = 'step1 r1 registration'
            else:
                targets, _, _, _, test = fit.dataset(speaker, 5, stems)
                targets = {k: v[~test] for k, v in targets.items()}
                _, global_, _ = fit.fit(targets, 200, starts=('rest', 'aa'))
                origin = 'bounded r1 registration: 5 frames/utterance plus rare phone centers'
            # Independent bounded audit; only global parameters survive registration.
            targets, _, _, _, test = fit.dataset(speaker, 3, stems)
            targets = {k: v[~test] for k, v in targets.items()}
            raw, _, _ = fit.fit(targets, 100, global_, starts=('rest',))
            with torch.no_grad():
                error = float(fit.errors(raw, global_, targets).mean())*10
            np.savez(registration_path, global_=global_.cpu().numpy())
            registrations[speaker] = dict(global_parameters=global_.cpu().tolist(),
                anatomy=fit.anatomy(global_).cpu().tolist(), registration_error_mm=error,
                origin=origin, demographics=usc.demographics(speaker), split='heldout' if speaker in HOLDOUT else 'train')
            dump('registration.json', registrations)
        for stem in stems:
            dest = cache / f'{stem}.npz'
            intervals = usc.intervals(usc.alignment_path(stem))
            inventory.update(p for a, b, p in intervals)
            unknown.update(p for a, b, p in intervals if phone_label(p) is None)
            if dest.exists() and stem in previous:
                rows.append(previous[stem])
                continue
            u = usc.load(stem)
            if not dest.exists():
                audio, sr = u.audio
                audio = audio.mean(1)
                times = np.arange(int(u.times[-1]*100)+1)*.01
                data = timeline(u.phone_intervals, times, vocabulary)
                pitch = parselmouth.Sound(audio, sr).to_pitch(time_step=.01, pitch_floor=65, pitch_ceiling=500)
                measured = pitch.selected_array['frequency']; good = measured > 0
                voiced = np.interp(times, pitch.xs(), good.astype(float), left=0, right=0)
                f0 = np.interp(times, pitch.xs()[good], measured[good]) if good.any() else np.full(len(times), 150.)
                index, present = nearest(times, u.times, .55/u.fps)
                # Missing frames and unknown/silence labels have no contour loss.
                selected = np.flatnonzero(present & data['valid'])
                if not len(selected):
                    excluded.append(dict(stem=stem, reason='no labelled tracked frames')); continue
                curves = usc.curves(u, index[selected])
                np.savez_compressed(dest, **data, f0=f0.astype(np.float32), voiced=voiced.astype(np.float32),
                    grid=selected, frame=u.frames[index[selected]], time=times[selected],
                    **{f'target_{k}': v.astype(np.float32) for k, v in curves.items()})
            with np.load(dest) as data:
                voiced_f0 = data['f0'][data['voiced'] > .5]
                rows.append(dict(stem=stem, voice=speaker, split=registrations[speaker]['split'],
                    frames=len(data['grid']), seconds=len(data['phone'])/100,
                    discarded_track_frames=u.discarded_track_frames,
                    voiced_fraction=float((data['voiced'] > .5).mean()),
                    f0_quantiles=np.quantile(voiced_f0, [.1,.5,.9]).tolist() if len(voiced_f0) else []))
        print('PREPARED', speaker, len(rows), flush=True)
        dump('data.json', rows)
        dump('mapping.json', dict(mapping=ARPABET, special={'AH0':'ə', 'silence/spn':'<pad>, no contour loss'},
             normalization='trim whitespace, uppercase, remove trailing stress digits 0/1/2; preserve diphthongs as single phones',
             inventory=inventory, unmapped=unknown, excluded_utterances=excluded,
             unknown_policy='unknown intervals use pad context and receive no contour loss',
             f0_policy='Praat 65–500 Hz, 10 ms; interpolate voiced F0; 150 Hz only if entirely unvoiced'))


def load(row):
    with np.load(OUT / 'data' / (row['stem']+'.npz')) as d:
        return {k: d[k] for k in d.files}


def geometry_errors(art, global_, targets):
    """All distances in mm; lips are aperture supervision, never lip skin fit."""
    public = fit.public_regions(art[:, :7], global_, targets)
    errors = {k: 10*fit.distance(public[k], targets[k]) for k in SURFACES}
    outer, inner = maeda.contours(art, fit.anatomy(global_))
    gap = (outer[:, -1]-inner[:, -1]).norm(dim=-1)*global_[0].exp()
    errors['lip_gap'] = 10*(gap-fit.minimum_gap(targets['upper_lip'], targets['lower_lip'])).abs()
    return errors


def inference(net, data, row, speakers):
    return net(**audio_train.inputs(data, row, speakers))


def mri_terms(net, data, row, speakers, global_, nasal, weights):
    # A two-second target window with 26 frames of true context on each side.
    grid = data['grid']
    center = random.choice(grid)
    lo, hi = max(0, center-100), min(len(data['phone']), center+100)
    start, end = max(0, lo-26), min(len(data['phone']), hi+26)
    window = {k: data[k][start:end] for k in ('phone','relative','f0','voiced')}
    art, _, _ = inference(net, window, row, speakers)
    selection = np.flatnonzero((grid >= lo) & (grid < hi))[::2]
    targets = {k: fit.tensor(data['target_'+k][selection]) for k in usc.REGIONS}
    errors = geometry_errors(art[grid[selection]-start], global_, targets)
    terms = {k: weights[k]*value.mean() for k, value in errors.items()}
    terms['smoothness'] = .002*smoothness(art)
    terms['saturation'] = .1*(art.abs()-3).relu().square().mean()
    terms['nasal'] = .1*nasal(art, fit.anatomy(global_),
        torch.as_tensor(window['phone'], device='cuda'), fit.tensor(window['relative']))
    return terms


def audit_pitch():
    """Reject gross per-recording octave/scanner locks, not normal intonation.

    Pitch enters the source only, never the articulation network. Keep Praat's
    voicing, but use the speaker's robust median for suspect F0 recordings.
    """
    rows = json.loads((OUT / 'data.json').read_text())
    audit = {}
    for speaker in sorted({r['voice'] for r in rows}):
        recordings = [r for r in rows if r['voice'] == speaker]
        center = float(np.median([r['f0_quantiles'][1] for r in recordings if r['f0_quantiles']]))
        replacements = []
        for row in recordings:
            q = row['f0_quantiles']
            if q and not .55*center <= q[1] <= 1.8*center:
                data = load(row)
                data['f0'] = np.full_like(data['f0'], center)
                np.savez_compressed(OUT / 'data' / (row['stem']+'.npz'), **data)
                replacements.append(dict(stem=row['stem'], original_quantiles=q, constant_hz=center))
        audit[speaker] = dict(speaker_median_hz=center, replacements=replacements,
            voiced_fraction_median=float(np.median([r['voiced_fraction'] for r in recordings])))
    dump('pitch-audit.json', dict(rule='replace F0 if utterance voiced median outside [0.55,1.8] times speaker median; keep Praat voicing',
                                  speakers=audit))
    return audit


def run_training(minutes):
    audit_pitch()
    random.seed(43); torch.manual_seed(43)
    rows = json.loads((OUT / 'data.json').read_text())
    mri_rows = [r for r in rows if r['split'] == 'train']
    mri_speakers = sorted({r['voice'] for r in mri_rows})
    assert len(mri_speakers) == 14 and not set(mri_speakers) & set(HOLDOUT)
    net, state = checkpoint(extra=mri_speakers)
    speakers = state['speakers']; nasal = NasalLoss(state['vocabulary']).cuda()
    registrations = json.loads((OUT / 'registration.json').read_text())
    globals_ = {s: fit.tensor(r['global_parameters']) for s, r in registrations.items()}
    cached = {r['stem']: load(r) for r in mri_rows}
    tts_rows = [r for r in json.loads((audio_train.DATA/'clips.json').read_text()) if r['split']=='train']
    tts_cache = {r['file']: audio_train.load(r) for r in tts_rows}
    synth = Synth(True).cuda()
    # Calibrate once, training speakers only: each surface contributes ~0.4 and
    # lips ~0.4 initially, comparable to the order-one TTS reconstruction loss.
    probes = defaultdict(list)
    unit = {k: 1. for k in (*SURFACES, 'lip_gap')}
    with torch.no_grad():
        for row in random.sample(mri_rows, min(24, len(mri_rows))):
            for k, v in mri_terms(net, cached[row['stem']], row, speakers, globals_[row['voice']], nasal, unit).items():
                probes[k].append(float(v))
    weights = {k: .4/max(np.mean(probes[k]), .1) for k in unit}
    config = dict(seed=43, minutes=minutes, learning_rate=.0001, alternation='one TTS then one MRI',
        contour_weights=weights, initial_unweighted_terms={k:float(np.mean(v)) for k,v in probes.items()},
        final_checkpoint='final.pt at declared wall-clock budget; no test selection',
        holdout=list(HOLDOUT), MRI_audio_loss=False, fixed_anatomy=True, initialization=str(OLD),
        mri_window_frames=200, mri_context_frames=26, mri_target_stride=2,
        smoothness=.002, saturation=.1, nasal=.1, tts_effort=1.,
        mri_speakers=mri_speakers, mri_training_utterances=len(mri_rows), tts_training_utterances=len(tts_rows))
    dump('config.json', config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    start = time.monotonic(); step=0; totals=defaultdict(list); curve=[]
    print('TRAIN_START', config, flush=True)
    while time.monotonic()-start < minutes*60:
        row = random.choice(tts_rows); data = tts_cache[row['file']]
        args = audio_train.inputs(data, row, speakers)
        art, source, scale = net(**args); target = fit.tensor(data['audio'])
        output = synth(art, source, args['f0'], args['voiced'], len(target), scale)
        terms = dict(audio=AudioLoss(target, 'linear')(output),
            effort=effort(art, effort_weights(target)), smoothness=.002*smoothness(art),
            saturation=.1*(art.abs()-3).relu().square().mean(),
            nasal=.1*nasal(art, scale, args['phone'], args['relative']))
        for kind in ('tts', 'mri'):
            if kind == 'mri':
                row = random.choice(mri_rows)
                terms = mri_terms(net, cached[row['stem']], row, speakers, globals_[row['voice']], nasal, weights)
            loss = sum(terms.values()); assert torch.isfinite(loss), (kind, row)
            optimizer.zero_grad(); loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step(); step += 1
            for k,v in terms.items(): totals[kind+'/'+k].append(float(v.detach()))
        if step % 100 == 0:
            entry = dict(step=step, seconds=time.monotonic()-start,
                         weighted_terms={k:float(np.mean(v)) for k,v in totals.items()})
            curve.append(entry); totals.clear(); dump('progress.json', curve)
            print('TRAIN', entry, flush=True)
        if step % 1000 == 0:
            torch.save(dict(state_dict=net.state_dict(), speakers=speakers, vocabulary=state['vocabulary'],
                config=config, steps=step), OUT / 'latest.pt')
    torch.save(dict(state_dict=net.state_dict(), speakers=speakers, vocabulary=state['vocabulary'],
        config=config, steps=step, seconds=time.monotonic()-start), OUT / 'final.pt')
    print('TRAIN_DONE', step, time.monotonic()-start, flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare','train','evaluate'])
    p.add_argument('--minutes',type=float,default=65)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare': prepare()
    elif args.stage == 'train': run_training(args.minutes)
    else:
        from mri_evaluate import run
        run()


if __name__ == '__main__': main()
