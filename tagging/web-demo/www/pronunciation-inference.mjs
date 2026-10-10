// A canceled/failed run discards the worker, including its WASM session.
export class LocalTranscriber {
  constructor(createWorker, assets) {
    this.createWorker = createWorker;
    this.assets = assets;
    this.worker = null;
    this.reject = null;
  }

  run(samples, onStatus) {
    if (this.reject) throw new Error("Transcription already running.");
    const worker = this.worker ??= this.createWorker();
    return new Promise((resolve, reject) => {
      this.reject = reject;
      worker.onmessage = ({ data }) => {
        if (this.worker !== worker) return;
        if (data.status) { onStatus(data.status); return; }
        if (data.error) { this.cancel(new Error(data.error)); return; }
        this.reject = null;
        resolve(data);
      };
      worker.onerror = event => {
        if (this.worker === worker) this.cancel(new Error(event.message || "Local model failed to load."));
      };
      // Keep the original audio for playback, the spectrogram and Retry.
      const waveform = Float32Array.from(samples);
      worker.postMessage({ waveform, ...this.assets }, [waveform.buffer]);
    });
  }

  cancel(error = new DOMException("Aborted", "AbortError")) {
    this.worker?.terminate();
    this.worker = null;
    this.reject?.(error);
    this.reject = null;
  }
}
