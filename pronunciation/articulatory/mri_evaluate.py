"""Predeclared whole-speaker MRI comparison and independent heldout TTS check."""
import json
import subprocess
import tempfile

import numpy as np
import torch
import maeda
import mri_fit as fit
import usc
import mri_train as mt
from synth import REST, Synth, geometry, levels

CLASSES = dict(vowels=set('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split()),
               bilabial={'M','B','P'}, alveolar=set('T D N S Z L'.split()), velar={'K','G','NG'})


@torch.no_grad()
def predict(net, data, speaker, speakers, global_):
    art, source, _ = mt.inference(net, data, {'voice':speaker}, speakers)
    # The network's speaker-dependent scale NEVER enters the MRI prediction.
    return art, source, fit.anatomy(global_)


def measures(art, global_, targets):
    result = mt.geometry_errors(art, global_, targets)
    public = fit.public_regions(art[:, :7], global_, targets)
    for k in ('upper_lip', 'lower_lip'):
        result[k] = 10*fit.distance(public[k], targets[k])
    result['palate_wall'] = (result['hard_palate']+result['pharyngeal_wall'])/2
    result['lips'] = (result['upper_lip']+result['lower_lip'])/2
    return {k:v.detach().cpu().numpy() for k,v in result.items()}


def statistics(errors, phones):
    base = np.char.rstrip(np.char.upper(np.char.strip(phones)), '012')
    return dict(all={k:fit.summary(v) for k,v in errors.items()},
        by_class={name:{k:fit.summary(v[np.isin(base, list(group))]) for k,v in errors.items()}
                  for name,group in CLASSES.items() if np.isin(base, list(group)).any()})


def frozen_mouth(targets, global_):
    """Oracle constant: one vector shared by ALL r2 frames, not a neural fit."""
    count = len(targets['tongue'])
    best = None
    runs = []
    with torch.enable_grad():
        for start in ('rest','iy','aa','uw'):
            initial = torch.zeros(7,device='cuda') if start == 'rest' else maeda.preset(start,'cuda')[:7]
            raw = initial.clone().requires_grad_()
            optimizer = torch.optim.Adam([raw],lr=.035)
            for step in range(400):
                art = torch.cat([raw,raw.new_zeros(1)]).expand(count,-1)
                loss = sum(v.mean() for v in mt.geometry_errors(art,global_,targets).values())/4
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step == 299: optimizer.param_groups[0]['lr'] = .008
            with torch.no_grad():
                art = torch.cat([raw,raw.new_zeros(1)]).expand(count,-1)
                loss = float(sum(v.mean() for v in mt.geometry_errors(art,global_,targets).values())/4)
                runs.append(dict(start=start,mean_objective_mm=loss))
                if best is None or loss < best[0]: best = (loss,art.clone())
    return best[1], dict(objective='unweighted mean of tongue, hard palate, pharyngeal wall and lip-gap errors in mm',
                        shared_controls=best[1][0].cpu().tolist(),starts=runs)


@torch.no_grad()
def evaluate_speaker(speaker, nets, states, global_, out):
    # Reuse precisely step1 r2 frames and raw fits. No neural control optimization.
    with np.load(fit.OUT / f'{speaker}-fit.npz') as d:
        test = d['test'] & np.array([mt.phone_label(p) is not None for p in d['phone']])
        excluded = dict(zip(*np.unique(d['phone'][d['test'] & ~test], return_counts=True)))
        mt.dump(out, f'{speaker}-evaluation-exclusions.json', {str(k):int(v) for k,v in excluded.items()})
        ids, frames, phones = [d[k][test] for k in ('utterance','frame','phone')]
        targets = {k:fit.tensor(d['target_'+k][test]) for k in usc.REGIONS}
        raw = fit.tensor(d['raw'][test])
        torch.testing.assert_close(global_, fit.tensor(d['global_']))
    predictions = {name:torch.empty(len(frames), 8, device='cuda') for name in nets}
    for stem in np.unique(ids):
        row = dict(stem=stem, voice=speaker)
        data = mt.load(row, out)
        u = usc.load(stem)
        indices = np.flatnonzero(ids == stem)
        grid = np.rint(frames[indices]/u.fps*100).astype(int).clip(0, len(data['phone'])-1)
        for name, net in nets.items():
            art, source, scale = predict(net, data, speaker, states[name]['speakers'], global_)
            predictions[name][indices] = art[grid]
    predictions['geometry_fit_floor'] = torch.cat([raw, torch.zeros_like(raw[:, :1])], -1)
    predictions['rest'] = REST.to(device='cuda').logit().expand(len(frames),-1)
    constant_path = out / f'{speaker}-frozen-mouth.json'
    if constant_path.exists():
        frozen = json.loads(constant_path.read_text())
        predictions['frozen_mouth'] = fit.tensor(frozen['shared_controls']).expand(len(frames),-1)
    else:
        predictions['frozen_mouth'], frozen = frozen_mouth(targets,global_)
        mt.dump(out, constant_path.name,frozen)
    results = {}
    arrays = dict(utterance=ids, frame=frames, phone=phones, global_=global_.cpu().numpy())
    for name,art in predictions.items():
        values = measures(art, global_, targets)
        results[name] = statistics(values, phones)
        arrays[name+'_articulation'] = art.cpu().numpy()
        arrays.update({name+'_'+k+'_mm':v for k,v in values.items()})
    arrays.update({'target_'+k:v.cpu().numpy() for k,v in targets.items()})
    np.savez_compressed(out / f'{speaker}-comparison.npz', **arrays)
    return results, (ids, frames, phones, targets, predictions)


def contact(speaker, global_, comparison, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    ids, frames, phones, targets, predictions = comparison
    chosen = []
    for phone in ['IY','AA','UW','K','T','S','M','NG']:
        indices = np.flatnonzero(np.char.rstrip(np.char.upper(phones),'012') == phone)
        assert len(indices), (speaker, phone)
        chosen.append(indices[len(indices)//2])
    names = ('old', 'step2', 'new')
    fig, axes = plt.subplots(2*len(names), 4, figsize=(16, 28), constrained_layout=True)
    cache = {}
    with torch.no_grad():
        curves = {name:[v.cpu().numpy() for v in fit.register(maeda.contours(art,fit.anatomy(global_)),global_)]
                  for name, art in predictions.items() if name in names}
    for j, idx in enumerate(chosen):
        if ids[idx] not in cache:
            cache[ids[idx]] = usc.video_frames(usc.load(ids[idx]).video)
        for side, name in enumerate(names):
            ax = axes[len(names)*(j//4)+side, j%4]
            target = {k:v[idx].cpu().numpy() for k,v in targets.items()}
            fit.draw(ax, cache[ids[idx]][frames[idx]], target, curves[name][0][idx], curves[name][1][idx],
                     f'{speaker} /{mt.phone_label(phones[idx])}/ {name} | f{frames[idx]}')
    fig.savefig(out / f'{speaker}-contact.png', dpi=120)
    plt.close(fig)


@torch.no_grad()
def video(speaker, global_, nets, states, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    stem = speaker+'_2drt_01_vcv1_r2'
    u = usc.load(stem)
    data = mt.load(dict(stem=stem), out)
    frames = usc.video_frames(u.video)
    first = int(2*u.fps)
    video_indices = np.arange(first,int(8*u.fps),4)
    selected = np.searchsorted(u.frames,video_indices)
    assert np.array_equal(u.frames[selected], video_indices)
    targets = usc.curves(u,selected)
    grid = np.rint(u.times[selected]*100).astype(int)
    curves = {}
    for name, net in nets.items():
        art, source, scale = predict(net,data,speaker,states[name]['speakers'],global_)
        curves[name] = [v.cpu().numpy() for v in fit.register(maeda.contours(art[grid],scale),global_)]
        # Preserve the normal inference NPZ schema; registration is extra metadata.
        intervals = u.phone_intervals
        areas, _, coupling, _ = geometry(art, scale, True)
        np.savez_compressed(out / f'{speaker}-{name}-fit.npz', articulation=art.cpu().numpy(),
            source=levels(areas, coupling, source).cpu().numpy(), lung_glottis=source.cpu().numpy(), scale=scale.cpu().numpy(), f0=data['f0'], voiced=data['voiced'],
            phone=data['phone'], relative=data['relative'], phonemes=np.array([mt.phone_label(p) or '<unk>' for a,b,p in intervals]),
            boundaries=np.array([a for a,b,p in intervals]+[intervals[-1][1]]),
            samples=len(data['phone'])*160, sample_rate=16000,control_hz=100,closures=True,
            language='eng',speaker=speaker,sentence=stem,global_=global_.cpu().numpy())
    fig, axes = plt.subplots(len(nets), 2, figsize=(10, 5.5*len(nets)), constrained_layout=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for j, idx in enumerate(selected):
            target = {k:v[j] for k,v in targets.items()}
            for row, name in enumerate(nets):
                for ax in axes[row]:
                    ax.clear()
                outer,inner = curves[name][0][j],curves[name][1][j]
                fit.draw(axes[row,0],frames[u.frames[idx]],target,outer,inner,
                         f'{speaker} {name} | {u.times[idx]:.2f}s {u.phones[idx]}')
                fit.draw(axes[row,1],None,{},outer,inner,f'{name} sagittal | fixed MRI anatomy')
            fig.savefig(f'{tmp}/{j:05d}.png',dpi=80)
        subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(u.fps/4),'-i',f'{tmp}/%05d.png',
            '-ss',str(first/u.fps),'-i',str(u.video),'-map','0:v','-map','1:a','-c:v','libx264',
            '-pix_fmt','yuv420p','-c:a','aac','-shortest',str(out/f'{speaker}-comparison.mp4')],check=True)
    plt.close(fig)


def run(out):
    old, old_state = mt.checkpoint(extra=mt.HOLDOUT)
    step2, step2_state = mt.checkpoint(out.parent/'network-mri/final.pt', extra=mt.HOLDOUT)
    new, new_state = mt.checkpoint(out/'final.pt', extra=mt.HOLDOUT)
    # Holdout source identity is the mean of original TTS embeddings, not MRI.
    with torch.no_grad():
        tts_count = len(old_state['speakers'])-len(mt.HOLDOUT)
        for net in (step2, new):
            net.speaker.weight[-len(mt.HOLDOUT):] = net.speaker.weight[:tts_count].mean(0)
    nets = dict(old=old.eval(), step2=step2.eval(), new=new.eval())
    states = dict(old=old_state, step2=step2_state, new=new_state)
    registrations = json.loads((out/'registration.json').read_text())
    result = dict(protocol='Step1 r2 frames; complete heldout speakers; fixed r1 registration; no neural test optimization',
        floor='Step1 independent native-geometry fits, scored using the same public contours as the networks',
        constant='Oracle one-vector fit to all r2 targets, four starts, fixed registration; rest uses zero raw Maeda factors',
        exclusions='Unmapped phone frames excluded identically from all conditions; see speaker evaluation-exclusions.json',
        lips='Aperture-only training; mucosal lip-curve distances are secondary diagnostics',speakers={})
    for speaker in mt.HOLDOUT:
        global_ = fit.tensor(registrations[speaker]['global_parameters'])
        result['speakers'][speaker], comparison = evaluate_speaker(speaker,nets,states,global_,out)
        mt.dump(out, 'mri-metrics.json',result)
        print('MRI',speaker,json.dumps(result['speakers'][speaker]),flush=True)
        contact(speaker,global_,comparison,out)
        video(speaker,global_,nets,states,out)
        print('ARTIFACTS',speaker,flush=True)
    rows = [r for r in json.loads((mt.audio_train.DATA/'clips.json').read_text()) if r['split']=='heldout']
    assert len(rows)==20
    tts = {}
    for name,net in nets.items():
        tts[name] = mt.audio_train.evaluate(net,Synth(True).cuda(),rows,states[name]['speakers'],name,out)
    mt.dump(out, 'tts-metrics.json',tts)
    print('EVALUATION_DONE',flush=True)
