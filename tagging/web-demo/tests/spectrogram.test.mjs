import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSpectrogram } from '../www/spectrogram.mjs';

test('silence has no spectral energy and retains its time axis', () => {
  const result = buildSpectrogram(new Float32Array(16000));
  assert.equal(result.columns, 100);
  assert.equal(result.hop, 160);
  assert.equal(result.bins, 257);
  assert.ok(result.values.every(value => value === 0));
});

test('a 1 kHz sine peaks at 1 kHz at the expected amplitude', () => {
  const samples = Float32Array.from({length: 16000}, (_, i) => .5 * Math.sin(2 * Math.PI * 1000 * i / 16000));
  const result = buildSpectrogram(samples);
  const middle = result.values.slice(50 * result.bins, 51 * result.bins);
  const peak = middle.indexOf(Math.max(...middle));
  assert.equal(peak * result.sampleRate / result.size, 1000);
  const db = middle[peak] / 255 * 80 - 80;
  assert.ok(Math.abs(db - 20 * Math.log10(.5)) < .4);
  assert.equal(middle[100], 0);
});

test('edge windows and non-hop-aligned clips are retained', () => {
  const result = buildSpectrogram(Float32Array.from({length: 163}, () => .2));
  assert.equal(result.columns, 2);
  assert.equal(result.values.length, 2 * result.bins);
  assert.ok(result.values[0] > 0);
});
