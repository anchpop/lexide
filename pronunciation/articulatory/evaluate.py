"""Post-fit diagnostics only: free transcription and vowel formant agreement.

The pronunciation model is NOT a loss and these diagnostics are NOT a listening
substitute. Compare originals too: errors of the diagnostic model are visible.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import parselmouth
import soundfile as sf

from prepare import MODEL, OUT

VOWELS = set('aeiouyæɑɒɔəɚɛɜɞɤɨɪɯɵœøʊʌɐ')


def decode(session, audio, labels, blank):
    outputs = session.run(None, {'waveform': audio[None].astype(np.float32),
                                 'lengths': np.array([len(audio)], np.int64)})
    heads = dict(zip([o.name for o in session.get_outputs()], outputs))
    scores = heads['phone'][0].copy()
    scores[:, blank] = -np.inf
    ids = np.where(heads['nonblank'][0, :, 0] > .5, scores.argmax(-1), blank)
    return [labels[i] for j, i in enumerate(ids) if i != blank and (j == 0 or i != ids[j-1])]


def edit_distance(reference, hypothesis):
    """Token Levenshtein distance; IPA strings are indivisible vocabulary tokens."""
    previous = list(range(len(hypothesis)+1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1,
                               previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def summarize(results):
    for row in results:
        row['edits'] = edit_distance(row['original_reading'], row['fitted_reading'])
        row['reference_count'] = len(row['original_reading'])
        assert row['reference_count'] > 0
        row['per'] = row['edits']/row['reference_count']
    edits = sum(r['edits'] for r in results)
    count = sum(r['reference_count'] for r in results)
    return dict(clips=results, edits=edits, reference_count=count, per=edits/count)


def score_files(rows, data_dir, output_dir):
    from prepare import make_session
    session = make_session()
    meta = json.loads((MODEL / 'frame_matrix.json').read_text())['heads']['phone']
    results = []
    for row in rows:
        stem = Path(row['file']).stem
        original = np.load(data_dir / (stem+'.npz'))['audio']
        fitted, sr = sf.read(output_dir / (stem+'-fit.wav'), dtype='float32')
        assert sr == 16000 and len(original) == len(fitted)
        results.append(dict(file=stem, sentence=row['sentence'],
            original_reading=decode(session, original, meta['labels'], meta['blank_id']),
            fitted_reading=decode(session, fitted, meta['labels'], meta['blank_id'])))
    result = summarize(results)
    (output_dir / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    meta = json.loads((MODEL / 'frame_matrix.json').read_text())['heads']['phone']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(MODEL / 'model.onnx'), options,
                                   providers=['CPUExecutionProvider'])
    results = []
    for row in json.loads((OUT / 'clips.json').read_text()):
        stem = Path(row['file']).stem
        cache = np.load(OUT / (stem+'.npz'))
        original, sr = sf.read(OUT / f'{stem}-original.wav', dtype='float32')
        fitted, _ = sf.read(OUT / f'{stem}-fit.wav', dtype='float32')
        formants = [parselmouth.Sound(y, sr).to_formant_burg(time_step=.01,
                    max_number_of_formants=5, maximum_formant=5500) for y in (original, fitted)]
        vowels = []
        for i, phone in enumerate(row['phonemes']):
            if phone[0] not in VOWELS:
                continue
            a, b = cache['boundaries'][i:i+2]
            times = np.linspace(a+.3*(b-a), a+.7*(b-a), 5)
            measured = [[float(np.nanmedian([f.get_value_at_time(k, t) for t in times]))
                         for k in (1, 2)] for f in formants]
            vowels.append(dict(phone=phone, midpoint=(a+b)/2, original_hz=measured[0], fit_hz=measured[1]))
        result = dict(file=stem, expected=row['phonemes'],
                      original_reading=decode(session, original, meta['labels'], meta['blank_id']),
                      fitted_reading=decode(session, fitted, meta['labels'], meta['blank_id']),
                      vowel_formants=vowels)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / 'diagnostics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (OUT / 'metrics.json').write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
