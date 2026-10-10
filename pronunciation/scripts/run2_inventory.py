"""Run2 coverage certification and actual trainer-admitted duration census."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
from src.audit_gate import STRICT_SOURCES, strict_verdict


def audit(output, sources):
    identity_path = ROOT / '.work/run2/word-phones.jsonl'
    with identity_path.open() as f:
        identity = json.loads(next(f))['g2p_identity']
    fields = ('file', 'source', 'lang', 'ok', 'phone_match', 'phone_match_version',
              'expected_sha256', 'g2p_selection', 'g2p_identity', 'g2p_language',
              'expected', 'whisper_text')
    coverage = {}
    for source in sources:
        path = ROOT / 'train' / f'{source}_asr_exclusions.jsonl'
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                assert row['g2p_identity'] == identity, f'{path}: stale g2p identity'
                coverage[(source, row['lang'], row['file'])] = {k: row.get(k) for k in fields}
    summary = defaultdict(lambda: {'clips': 0, 'pass': 0, 'reject': 0, 'reject_examples': []})
    for manifest in sorted((ROOT / 'data/audio').glob('*/manifest.jsonl')):
        lang = manifest.parent.name
        with manifest.open() as f:
            for line in f:
                row = json.loads(line)
                source = row.get('source')
                if source not in sources:
                    continue
                verdict = strict_verdict(row, lang, coverage)
                item = summary[f'{lang}/{source}']
                item['clips'] += 1
                item['pass' if verdict else 'reject'] += 1
                if not verdict and len(item['reject_examples']) < 3:
                    record = coverage[(source, lang, row['file'])]
                    item['reject_examples'].append({k: record[k] for k in ('file', 'expected', 'whisper_text')})
    result = {'g2p_identity': identity, 'coverage': dict(summary)}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for key, value in summary.items():
        print(key, value['pass'], '/', value['clips'])


def batching(lengths):
    from src.dataset import augmented_audio_lengths, PlannedPaddingBatchSampler, TokenBudgetBatchSampler

    options = dict(max_batch_size=128, bucket_size=3200, seed=42)
    legacy = TokenBudgetBatchSampler(lengths, token_budget=170 * 16000, **options)
    legacy_bounds = augmented_audio_lengths(lengths, speed_min=.85)
    planned = PlannedPaddingBatchSampler(lengths, token_budget=200 * 16000,
                                         speed_min=.85, padding_prob=.3, **options)
    return {
        'run1_batching': {
            'stats': legacy.stats(),
            'padded_hours': sum(max(legacy_bounds[i] for i in batch) * len(batch)
                                for batch in legacy) / 16000 / 3600,
        },
        'run2': {
            'stats': planned.stats(),
            'planned_padding_hours': sum(sum(p) for p in planned.padding_frames) * 256 / 16000 / 3600,
            'padded_hours': planned.stats()['padded_audio_samples'] / 16000 / 3600,
        },
    }


def admitted(output):
    from torch.utils.data import ConcatDataset, Subset
    from src.train_unified import load_processor, load_training_datasets
    from src.validation_metrics import sentence_split

    sidecars = ['fleurs_asr_exclusions', 'tatoeba_asr_exclusions', 'tts_asr_exclusions',
                'tts_word_asr_exclusions', 'mls_asr_exclusions', 'cv_asr_exclusions',
                'aishell1_asr_exclusions', 'aishell3_asr_exclusions', 'lang_exclusions',
                'mixed_script_exclusions', 'boilerplate_exclusions']
    paths = [ROOT / 'train' / (s + '.jsonl') for s in sidecars]
    args = SimpleNamespace(data_dir=ROOT / 'data/audio', audit_path=[p for p in paths if p.exists()],
                           fleurs_audit_path=None, fleurs_audit_min_per=None, fleurs_audit_min_cer=None,
                           fleurs_audit_min_wer=None, audit_min_per=1e-12, audit_min_cer=1e-12,
                           audit_min_wer=1e-12, langs=None, use_narrowed=False,
                           max_audio_sec=16., min_rms=.005, min_whisper_logprob=-.7,
                           max_clips_per_sentence=20, source_cap_second=True)
    datasets = load_training_datasets(args, load_processor('scratch'))
    full = ConcatDataset(datasets)
    train, val = sentence_split(full, .05)

    def metadata(dataset):
        if isinstance(dataset, Subset):
            parent = metadata(dataset.dataset)
            return [parent[i] for i in dataset.indices]
        if isinstance(dataset, ConcatDataset):
            return [row for child in dataset.datasets for row in metadata(child)]
        return dataset.samples

    with (ROOT / 'data/audio/spa/manifest.jsonl').open() as source:
        invariant_files = {r['file'] for r in map(json.loads, source) if r.get('variety_invariant')}
    invariant_admission = {}
    summary = defaultdict(lambda: {'admitted_clips': 0, 'admitted_hours': 0.,
                                   'train_clips': 0, 'train_hours': 0., 'val_clips': 0, 'val_hours': 0.})
    for stage, rows in [('admitted', metadata(full)), ('train', metadata(train)), ('val', metadata(val))]:
        invariant_admission[stage] = {'clips': 0, 'hours': 0.}
        for row in rows:
            if row.get('source') in STRICT_SOURCES:
                assert row['vad_probs'], f"Missing strict-row VAD: {row['wav_path']}"
            if row['lang'] == 'spa' and Path(row['wav_path']).name in invariant_files:
                invariant_admission[stage]['clips'] += 1
                invariant_admission[stage]['hours'] += row['n_audio_samples'] / 16000 / 3600
            item = summary[f"{row['lang']}/{row.get('source') or 'unknown'}"]
            item[stage + '_clips'] += 1
            item[stage + '_hours'] += row['n_audio_samples'] / 16000 / 3600
    lengths = [r['n_audio_samples'] for r in metadata(train)]
    # Small metadata cache for revisiting batching without re-decoding the corpus.
    output.with_suffix('.lengths.json').write_text(json.dumps(lengths) + '\n')
    sampler_work = batching(lengths)
    result = {'table': dict(sorted(summary.items())), 'sampler': sampler_work,
              'variety_invariant': invariant_admission,
              'admitted_hours': sum(v['admitted_hours'] for v in summary.values()),
              'train_hours': sum(v['train_hours'] for v in summary.values())}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'table'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['audit', 'admitted'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--sources', nargs='+', default=['mls', 'aishell1', 'aishell3', 'tts_word'])
    args = parser.parse_args()
    if args.stage == 'audit':
        audit(args.output, args.sources)
    else:
        admitted(args.output)
