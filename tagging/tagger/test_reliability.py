"""Offline reliability-evaluation regression tests."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reliability_eval as evaluator
import reliability_report as reporter


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'eng-000', 'lang': 'eng', 'text': 'a b', 'subset': True,
                    'disagreement': True, 'mapping': {'A': 'parsley', 'B': 'gemma'},
                    'analyses': {'A': [{'index': 1, 'text': 'a', 'pos': 'DET', 'lemma': 'a'}],
                                 'B': [{'index': 1, 'text': 'a', 'pos': 'DET', 'lemma': 'a'},
                                       {'index': 2, 'text': 'b', 'pos': 'NOUN', 'lemma': 'b'}]}}
        self.verdict = {'verdict': 'B', 'errors': [
            {'in': 'both', 'category': 'lemma', 'tokens': ['a'], 'severity': 'wrong', 'reason': 'shared'},
            {'in': 'A', 'category': 'pos', 'tokens': ['a'], 'severity': 'convention', 'reason': 'policy'},
            {'in': 'A', 'category': 'tokenization', 'tokens': ['b'], 'severity': 'wrong', 'reason': 'missing'}]}

    def test_validation(self):
        evaluator.validate(self.verdict)
        bad = copy.deepcopy(self.verdict)
        bad['errors'][0]['category'] = 'head'
        with self.assertRaises(AssertionError):
            evaluator.validate(bad)

    def test_unblind_shared_once_each(self):
        counts, shared, winner = reporter.unblind(self.row, self.verdict)
        self.assertEqual(winner, 'gemma')
        self.assertEqual(sum(counts['gemma']['wrong'].values()), 1)
        self.assertEqual(sum(counts['parsley']['wrong'].values()), 2)
        self.assertEqual(sum(shared['wrong'].values()), 1)
        self.assertEqual(sum(counts['parsley']['convention'].values()), 1)

    def test_rates_and_paired_bootstrap(self):
        result = reporter.summarize([self.row], [{'eng-000': self.verdict}])
        all_rates = result['categories']['all']
        self.assertEqual(all_rates['gemma']['wrong_per_100_tokens'], 50)
        self.assertEqual(all_rates['parsley']['wrong_per_100_tokens'], 200)
        self.assertEqual(all_rates['parsley_minus_gemma']['ci95'], [150, 150])
        # Duplicating identical judges must not double the error rate.
        pooled = reporter.summarize([self.row], [{'eng-000': self.verdict}] * 2)
        self.assertEqual(pooled['categories'], result['categories'])

    def test_agreement_unblinded(self):
        result = reporter.agreement([self.row], {'eng-000': self.verdict}, {'eng-000': self.verdict})
        self.assertEqual(result['verdict_agreement'], 1)
        self.assertEqual(result['fewer_wrong_errors_agreement_including_ties'], 1)
        self.assertIsNone(result['wrong_count_pearson']['gemma'])

    def test_no_dependencies_or_identity_in_prompt(self):
        text = evaluator.prompt(self.row, {'eng': 'English rules'})
        self.assertTrue(text.startswith('Project language tips (eng):\nEnglish rules'))
        self.assertNotIn('parsley', text)
        self.assertNotIn('gemma', text)
        self.assertIn('index, text, POS, lemma', text)
        self.assertNotIn('head index', text)

    def test_surface_choices(self):
        rows = [{'lang': 'jpn', 'text': 'ので', 'tokens': [{'start': 0, 'end': 1, 'lemma': 'の'},
                                                        {'start': 1, 'end': 2, 'lemma': 'で'}]},
                {'lang': 'jpn', 'text': 'ので', 'tokens': [{'start': 0, 'end': 2, 'lemma': 'ので'}]}]
        result = reporter.choices(rows, 'jpn', 'ので')
        self.assertEqual(result['segmentation_choices'], {'の|で': 1, 'ので': 1})
        self.assertEqual(result['majority_segmentation_rate'], .5)

    def test_tips_all_languages(self):
        tips = evaluator.extract_tips()
        self.assertEqual(set(tips), set(evaluator.LANGS))
        self.assertEqual(tips['eng'], '')
        self.assertTrue(all(tips[lang] for lang in tips if lang != 'eng'))

    def test_budget_guard_bound_to_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, 'OUT', Path(directory)):
            hashes = {}
            for provider in evaluator.MODELS:
                data = provider.encode()
                (Path(directory) / f'{provider}_inputs.jsonl').write_bytes(data)
                hashes[provider] = hashlib.sha256(data).hexdigest()
            estimate = {'approved_total_usd': 29, 'pilot_reviewed': True, 'input_sha256': hashes}
            evaluator.check_estimate(estimate)
            with self.assertRaises(RuntimeError):
                evaluator.check_estimate({**estimate, 'approved_total_usd': 31})
            with self.assertRaises(RuntimeError):
                evaluator.check_estimate({**estimate, 'submissions_closed': True})
            (Path(directory) / 'gemini_inputs.jsonl').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                evaluator.check_estimate(estimate)

    def test_gemini_native_schema(self):
        body = evaluator.request('gemini', self.row, {'eng': ''})
        config = body['generationConfig']
        self.assertNotIn('responseJsonSchema', config)
        schema = config['responseSchema']
        self.assertEqual(schema['properties']['errors']['items']['properties']['in']['enum'], ['A', 'B', 'both'])
        self.assertNotIn('additionalProperties', schema)

    def test_null_errors_not_repaired(self):
        bad = {'errors': [None], 'verdict': 'tie'}
        with self.assertRaises(TypeError):
            evaluator.validate(bad)

    def test_reasoning_token_billing(self):
        result = evaluator.usage('gemini', {'usageMetadata': {'promptTokenCount': 100,
            'candidatesTokenCount': 20, 'thoughtsTokenCount': 80, 'cachedContentTokenCount': 50}})
        self.assertEqual(result, {'input': 100, 'cached': 50, 'output': 100})
        self.assertAlmostEqual(evaluator.cost('gemini', result), .000655)


if __name__ == '__main__':
    unittest.main()
