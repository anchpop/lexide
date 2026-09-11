// Centered 512-sample Hann-window STFT, 10 ms hop, amplitude in dBFS.
// Zero-padding at clip edges keeps column time = column * hop / sampleRate.
export function buildSpectrogram(samples, sampleRate = 16000) {
  const size = 512, hop = Math.round(sampleRate * 0.01), bins = size / 2 + 1;
  const columns = Math.ceil(samples.length / hop);
  const values = new Uint8Array(columns * bins);
  const window = Float64Array.from({length: size}, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / size));
  const gain = window.reduce((sum, value) => sum + value, 0);
  const real = new Float64Array(size), imag = new Float64Array(size);
  for (let column = 0; column < columns; column++) {
    for (let i = 0; i < size; i++) {
      const sample = column * hop + i - size / 2;
      real[i] = (samples[sample] || 0) * window[i];
      imag[i] = 0;
    }
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) [real[i], real[j]] = [real[j], real[i]];
    }
    for (let length = 2; length <= size; length *= 2) {
      const angle = -2 * Math.PI / length;
      const wr = Math.cos(angle), wi = Math.sin(angle);
      for (let offset = 0; offset < size; offset += length) {
        let ur = 1, ui = 0;
        for (let j = 0; j < length / 2; j++) {
          const a = offset + j, b = a + length / 2;
          const tr = ur * real[b] - ui * imag[b], ti = ur * imag[b] + ui * real[b];
          real[b] = real[a] - tr; imag[b] = imag[a] - ti;
          real[a] += tr; imag[a] += ti;
          [ur, ui] = [ur * wr - ui * wi, ur * wi + ui * wr];
        }
      }
    }
    for (let bin = 0; bin < bins; bin++) {
      const scale = bin === 0 || bin === bins - 1 ? 1 : 2;
      const amplitude = Math.hypot(real[bin], imag[bin]) * scale / gain;
      const db = 20 * Math.log10(Math.max(amplitude, 1e-4));
      values[column * bins + bin] = Math.round(Math.max(0, Math.min(1, (db + 80) / 80)) * 255);
    }
  }
  return { values, columns, bins, sampleRate, hop, size };
}
