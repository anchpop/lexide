import { buildSpectrogram } from "./spectrogram.mjs";
self.onmessage = ({ data }) => {
  try {
    const result = buildSpectrogram(data.samples, data.sampleRate);
    self.postMessage(result, [result.values.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
