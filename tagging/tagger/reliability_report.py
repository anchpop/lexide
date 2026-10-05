"""Score blind reliability judgments, including optional Claude chunk verdicts.

/home/andrep/.venv-lexide-tests/bin/python tagging/tagger/reliability_report.py
Rates count independent error findings (not unique erroneous tokens). Bootstrap units are
sentences; judgments for the same sentence always travel together. Overall bootstrap is
stratified by language. Pooled results use only sentences judged by EVERY available family.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from reliability_eval import OUT, LANGS, SCHEMA, SEED, TEST, PRED, cost, dump, lines, load, usage, validate

CATEGORIES = ['tokenization', 'pos', 'lemma']
SYSTEMS = ['gemma', 'parsley']
BOOTSTRAPS = 4000


def unblind(row, verdict):
    counts = {s: {severity: Counter() for severity in ('wrong', 'convention')} for s in SYSTEMS}
    shared = {'wrong': Counter(), 'convention': Counter()}
    for error in verdict['errors']:
        targets = SYSTEMS if error['in'] == 'both' else [row['mapping'][error['in']]]
        for system in targets:
            counts[system][error['severity']][error['category']] += 1
        if error['in'] == 'both':
            shared[error['severity']][error['category']] += 1
    winner = 'tie' if verdict['verdict'] == 'tie' else row['mapping'][verdict['verdict']]
    return counts, shared, winner


def token_counts(row):
    return {system: len(row['analyses'][label]) for label, system in row['mapping'].items()}


def ci(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def summarize(rows, judgments):
    """judgments contains one or more equally weighted judges on exactly these rows."""
    n = len(rows)
    if not n:
        return None
    wrong = np.zeros((n, 2, 3))
    conventional = np.zeros_like(wrong)
    shared = np.zeros((n, 3))
    shared_convention = np.zeros_like(shared)
    denominators = np.array([[token_counts(r)[s] for s in SYSTEMS] for r in rows], dtype=float)
    wins = Counter()
    for i, row in enumerate(rows):
        for judge in judgments:
            counts, common, winner = unblind(row, judge[row['id']])
            wins[winner] += 1
            for s, system in enumerate(SYSTEMS):
                for c, cat in enumerate(CATEGORIES):
                    wrong[i, s, c] += counts[system]['wrong'][cat] / len(judgments)
                    conventional[i, s, c] += counts[system]['convention'][cat] / len(judgments)
            for c, cat in enumerate(CATEGORIES):
                shared[i, c] += common['wrong'][cat] / len(judgments)
                shared_convention[i, c] += common['convention'][cat] / len(judgments)
    wrong = np.concatenate([wrong, wrong.sum(axis=2, keepdims=True)], axis=2)
    conventional = np.concatenate([conventional, conventional.sum(axis=2, keepdims=True)], axis=2)
    shared = np.concatenate([shared, shared.sum(axis=1, keepdims=True)], axis=1)
    shared_convention = np.concatenate([shared_convention, shared_convention.sum(axis=1, keepdims=True)], axis=1)
    rates = wrong.sum(axis=0) / denominators.sum(axis=0)[:, None] * 100
    rng = np.random.default_rng(SEED)
    groups = [[i for i, r in enumerate(rows) if r['lang'] == lang] for lang in LANGS]
    groups = [g for g in groups if g]
    boots = []
    # Chunked to avoid allocating a huge B x N x systems x categories tensor.
    for start in range(0, BOOTSTRAPS, 100):
        size = min(100, BOOTSTRAPS - start)
        indices = np.concatenate([rng.choice(g, (size, len(g)), replace=True) for g in groups], axis=1)
        boots.append(wrong[indices].sum(axis=1) / denominators[indices].sum(axis=1)[:, :, None] * 100)
    boots = np.concatenate(boots)
    result = {'sentences': n, 'judge_sentence_pairs': n * len(judgments),
              'disagreement_sentences': sum(r['disagreement'] for r in rows),
              'tokens': dict(zip(SYSTEMS, denominators.sum(axis=0).astype(int).tolist())),
              'verdicts': dict(wins), 'categories': {}}
    for c, cat in enumerate(CATEGORIES + ['all']):
        result['categories'][cat] = {
            **{s: {'wrong_per_100_tokens': float(rates[k, c]), 'ci95': ci(boots[:, k, c]),
                    'wrong_findings': float(wrong[:, k, c].sum()),
                    'convention_findings': float(conventional[:, k, c].sum())}
               for k, s in enumerate(SYSTEMS)},
            'parsley_minus_gemma': {'rate_difference': float(rates[1, c] - rates[0, c]),
                                   'ci95': ci(boots[:, 1, c] - boots[:, 0, c])},
            'shared_wrong_findings': float(shared[:, c].sum()),
            'shared_wrong_per_100_mean_tokens': float(shared[:, c].sum() / denominators.mean(axis=1).sum() * 100),
            'shared_convention_findings': float(shared_convention[:, c].sum())}
    return result


def agreement(rows, left, right):
    common = [r for r in rows if r['id'] in left and r['id'] in right]
    verdicts = []
    counts = [[], []]
    fewer = []
    for r in common:
        a, b = [unblind(r, j[r['id']]) for j in (left, right)]
        verdicts.append((a[2], b[2]))
        totals = [[sum(v[0][s]['wrong'].values()) for s in SYSTEMS] for v in (a, b)]
        for i in range(2):
            counts[i].append(totals[i])
        fewer.append(np.sign(totals[0][1] - totals[0][0]) == np.sign(totals[1][1] - totals[1][0]))
    n = len(common)
    if not n:
        return {'n': 0}
    observed = sum(a == b for a, b in verdicts) / n
    marginals = [Counter(v[i] for v in verdicts) for i in range(2)]
    chance = sum(marginals[0][v] * marginals[1][v] for v in SYSTEMS + ['tie']) / n ** 2
    correlations = {}
    exact = {}
    for i, s in enumerate(SYSTEMS):
        x, y = [np.array(c)[:, i] for c in counts]
        correlations[s] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        exact[s] = float(np.mean(x == y))
    return {'n': n, 'verdict_agreement': observed,
            'cohen_kappa': (observed - chance) / (1 - chance) if chance < 1 else None,
            'verdict_confusion': dict(Counter(a + '/' + b for a, b in verdicts)),
            'wrong_count_pearson': correlations, 'exact_wrong_count_agreement': exact,
            'fewer_wrong_errors_agreement_including_ties': float(np.mean(fewer))}


def choices(records, lang, string):
    segmentation = Counter()
    lemmas = Counter()
    occurrences = embedded = 0
    for row in records:
        if row['lang'] != lang:
            continue
        for match in re.finditer(re.escape(string), row['text']):
            start, end = match.span()
            tokens = [t for t in row['tokens'] if t['start'] < end and t['end'] > start]
            occurrences += 1
            inside = sorted({t['end'] - start for t in tokens if start < t['end'] < end})
            boundaries = [0] + inside + [len(string)]
            segmentation['|'.join(string[a:b] for a, b in zip(boundaries, boundaries[1:]))] += 1
            aligned = tokens[0]['start'] == start and tokens[-1]['end'] == end
            if aligned:
                lemmas['|'.join(t['lemma'] for t in tokens)] += 1
            else:
                embedded += 1
    return {'occurrences': occurrences, 'segmentation_choices': dict(segmentation),
            'segmentation_choice_rates': {k: v / occurrences for k, v in segmentation.items()},
            'majority_segmentation_rate': max(segmentation.values()) / occurrences if occurrences else None,
            'outer_boundary_embedded_occurrences': embedded,
            'lemma_sequences_outer_aligned_only': dict(lemmas)}


def nonjudge():
    teacher, student = lines(TEST), lines(PRED)
    strings = [('zho-hans', '不用'), ('zho-hans', '有人'), ('jpn', 'ので'), ('jpn', 'どうか')]
    # Supplement with observed frequent alternatives, not a hand-picked success list.
    for lang in ('zho-hans', 'jpn'):
        candidate = Counter(r['text'][t['start']:t['end']] for r in teacher if r['lang'] == lang
                            for t in r['tokens'] if 2 <= t['end'] - t['start'] <= 4)
        ranked = []
        for text, count in candidate.items():
            if count < 8 or (lang, text) in strings:
                continue
            c = choices(teacher, lang, text)
            counts = c['segmentation_choices'].values()
            minority = c['occurrences'] - max(counts)
            if minority >= 3:
                ranked.append((minority, text))
        strings.extend((lang, text) for _, text in sorted(ranked, reverse=True)[:3])
    consistency = {lang + ':' + text: {system: choices(records, lang, text)
                  for system, records in [('gemma', teacher), ('parsley', student)]}
                  for lang, text in strings}
    coverage = {}
    for system, records in [('gemma', teacher), ('parsley', student)]:
        invalid = 0
        for r in records:
            covered = set()
            previous = 0
            valid = True
            for t in r['tokens']:
                valid &= previous <= t['start'] < t['end'] <= len(r['text'])
                covered.update(range(t['start'], t['end']))
                previous = t['end']
            valid &= all(i in covered or char.isspace() for i, char in enumerate(r['text']))
            invalid += not valid
        coverage[system] = {'rows': len(records), 'invalid_nonwhitespace_coverage_or_spans': invalid}
    rejects = {}
    log_path = OUT / 'export.log'
    if log_path.exists():
        for lang, path, count in re.findall(r'tokenization\[([^]]+)\] (.*): rejected (\d+) invalid rows', log_path.read_text()):
            p = Path(path)
            with p.open('rb') as f:
                total = sum(1 for _ in f)
            rejects[path] = {'lang': lang, 'invalid': int(count), 'rows': total,
                             'fraction': int(count) / total, 'gold': p.name.startswith('cleaned_')}
        # Include valid stores in denominator, not just stores with rejection messages.
        stores = [p for lang in LANGS for p in (Path('/data/coding/yap/out') / lang).glob('*tokenization*.jsonl')
                  if p.name in ('target_language_sentences_tokenization.jsonl', 'restricted_sentences_tokenization.jsonl',
                                'target_language_multiword_terms_tokenization.jsonl', 'target_language_sentences_tokenization_augmented.jsonl')]
        total = 0
        by_lang = {}
        for lang in LANGS:
            n = 0
            for p in stores:
                if p.parent.name == lang:
                    with p.open('rb') as f:
                        n += sum(1 for _ in f)
            bad = sum(v['invalid'] for v in rejects.values() if not v['gold'] and v['lang'] == lang)
            total += n
            by_lang[lang] = {'rows': n, 'invalid': bad, 'fraction': bad / n if n else None}
        bad = sum(v['invalid'] for v in rejects.values() if not v['gold'] and v['lang'] in LANGS)
        failure = {'silver_rows': total, 'silver_invalid': bad, 'silver_invalid_fraction': bad / total,
                   'per_language': by_lang, 'rejected_files': rejects,
                   'caveat': 'Surviving stored-row rejection rate, not original generation failure probability; successful retries and discarded responses are unobserved.',
                   'retry_policy': {'model_attempts_per_run': 3, 'failed_runs_before_giving_up': 3, 'endpoint_attempts': 8},
                   'source': '/data/coding/yap/generate-data/src/nlp.rs:73-95,155-193'}
    else:
        failure = {'unavailable': 'Export log absent'}
    return {'consistency': consistency, 'test_span_coverage': coverage, 'hard_failures': failure,
            'consistency_caveat': 'Choice rates across contexts containing the same surface string, not repeated inference determinism or proof of inconsistency. Internal boundaries counted even when substring is embedded; lemma sequences only for outer-aligned occurrences.'}


def read_judges(rows):
    judges = {}
    ids = {r['id'] for r in rows}
    for provider in ('gemini', 'openai'):
        path = OUT / f'{provider}_verdicts.json'
        if path.exists():
            values = load(path)
            judges[provider] = {v['id']: v for v in values}
            assert len(judges[provider]) == len(values)
    values = []
    for path in sorted((OUT / 'claude_subset').glob('verdicts_[0-9][0-9].json')):
        chunk = load(path)
        expected = {r['id'] for r in load(path.with_name(path.name.replace('verdicts_', 'chunk_')))['requests']}
        assert {v['id'] for v in chunk} == expected
        values.extend(chunk)
    if values:
        judges['claude-opus-5.5'] = {v['id']: v for v in values}
        assert len(judges['claude-opus-5.5']) == len(values)
    for judge in judges.values():
        assert judge.keys() <= ids
        for v in judge.values():
            validate({k: v[k] for k in ('errors', 'verdict')})
    return judges


def main():
    rows = load(OUT / 'sample.json')
    judges = read_judges(rows)
    assert judges, 'No completed verdicts'
    results = {}
    for name, judge in judges.items():
        selected = [r for r in rows if r['id'] in judge]
        results[name] = {lang: summarize([r for r in selected if lang == 'ALL' or r['lang'] == lang], [judge])
                         for lang in list(LANGS) + ['ALL']}
    common = [r for r in rows if all(r['id'] in j for j in judges.values())]
    results['pooled_common_subset'] = {lang: summarize([r for r in common if lang == 'ALL' or r['lang'] == lang], list(judges.values()))
                                      for lang in list(LANGS) + ['ALL']}
    agreement_results = {a + ' vs ' + b: agreement(rows, judges[a], judges[b]) for a, b in combinations(judges, 2)}
    # Same support gives a fair view of family bias, unlike comparing 1200 vs 240.
    common_results = {name: summarize(common, [judge]) for name, judge in judges.items()}
    examples = {}
    for lang in LANGS:
        examples[lang] = {}
        for kind in ('parsley_right_gemma_wrong', 'gemma_right_parsley_wrong', 'shared'):
            candidates = []
            for r in rows:
                if r['lang'] != lang:
                    continue
                evidence = {}
                for name, judge in judges.items():
                    if r['id'] not in judge:
                        continue
                    counts, shared, winner = unblind(r, judge[r['id']])
                    g, p = [sum(counts[s]['wrong'].values()) for s in SYSTEMS]
                    qualifies = (p == 0 and g > 0 if kind == 'parsley_right_gemma_wrong'
                                 else g == 0 and p > 0 if kind == 'gemma_right_parsley_wrong'
                                 else sum(shared['wrong'].values()) > 0)
                    if qualifies:
                        evidence[name] = judge[r['id']]
                if evidence:
                    candidates.append((len(evidence), r, evidence))
            if candidates:
                _, row, evidence = max(candidates, key=lambda x: x[0])
                examples[lang][kind] = {'id': row['id'], 'text': row['text'], 'mapping': row['mapping'],
                                        'analyses': row['analyses'], 'judgments': evidence,
                                        'caveat': 'Right means zero wrong-severity findings from the cited judge, not independently verified correctness.'}
            else:
                examples[lang][kind] = None
    costs = {}
    for provider in ('gemini', 'openai'):
        pilot = sum(cost(provider, usage(provider, load(p)), batch=False) for p in (OUT / 'pilot').glob(provider + '*.json'))
        raw = (load(OUT / 'gemini_outputs.json') if provider == 'gemini' and (OUT / 'gemini_outputs.json').exists()
               else lines(OUT / 'openai_outputs.jsonl') if provider == 'openai' and (OUT / 'openai_outputs.jsonl').exists() else [])
        usages = [usage(provider, r['response'] if provider == 'gemini' else r['response']['body']) for r in raw]
        batch = sum(cost(provider, u) for u in usages)
        rejected = native_pilot = 0
        if provider == 'gemini':
            rejected_path = OUT / 'gemini_rejected_json_schema/gemini_outputs.json'
            if rejected_path.exists():
                rejected = sum(cost(provider, usage(provider, r['response'])) for r in load(rejected_path))
            native_path = OUT / 'gemini_native_pilot_status.json'
            if native_path.exists():
                native_pilot = sum(cost(provider, usage(provider, r['response']))
                                   for r in load(native_path)['response']['inlinedResponses']['inlinedResponses'])
        costs[provider] = {'synchronous_pilot_usd': pilot, 'final_batch_usd': batch,
                           'discarded_batch_usd': rejected, 'native_schema_batch_pilot_usd': native_pilot,
                           'total_usd': pilot + batch + rejected + native_pilot,
                           'final_batch_usage': dict(sum((Counter(u) for u in usages), Counter()))}
    facts_path = OUT / 'nonjudge.json'
    if not facts_path.exists():
        dump(facts_path, nonjudge())
    report = {'manifest': load(OUT / 'manifest.json'), 'bootstrap_resamples': BOOTSTRAPS,
              'results': results, 'same_subset_results': common_results, 'judge_agreement': agreement_results,
              'nonjudge': load(facts_path), 'costs': costs, 'api_total_usd': sum(c['total_usd'] for c in costs.values()),
              'budget_estimate_and_incident': load(OUT / 'estimate.json'),
              'discarded_gemini_run': 'All 1200 results discarded: responseJsonSchema batch parsing corrupted nested errors; native responseSchema passed a batch pilot before full rerun.',
              'claude_cost': 'Driver-managed subagents; no API usage billing available here.', 'examples': examples,
              'caveats': ['Judge findings are not independently verified gold. Severity and linguistic-policy judgments vary across families.',
                          'Only tokenization/POS/lemma are judged; dependencies explicitly removed from scope.',
                          'Gemma means exported non-gold silver after project normalization/corrections, not unprocessed model text.',
                          'Rates count independent findings, not unique erroneous tokens; multiple categories may affect one token.',
                          'Punctuation is included in token denominators. Each system uses its own token count; shared rate uses mean token count.',
                          'Pooled view is the intersection of judge coverage, not a naive mixture of the full and small samples.',
                          'Bootstrap is paired, sentence-clustered, stratified by language overall; it quantifies sampling, not judge systematic error.',
                          'Actual cost is computed from returned usage and verified posted rates, not a provider invoice.']}
    dump(OUT / 'report.json', report)
    print('Primary judge: Gemini; wrong findings / 100 tokens (95% sentence bootstrap CI)')
    print('lang       Gemma                 parsley               difference')
    for lang, result in results.get('gemini', {}).items():
        c = result['categories']['all']
        cells = [f'{c[s]["wrong_per_100_tokens"]:.2f} [{c[s]["ci95"][0]:.2f}, {c[s]["ci95"][1]:.2f}]' for s in SYSTEMS]
        d = c['parsley_minus_gemma']
        print(f'{lang:8}   {cells[0]:22} {cells[1]:22} {d["rate_difference"]:.2f} [{d["ci95"][0]:.2f}, {d["ci95"][1]:.2f}]')
    print(json.dumps({'agreement': agreement_results, 'costs': costs}, indent=2))


if __name__ == '__main__':
    main()
