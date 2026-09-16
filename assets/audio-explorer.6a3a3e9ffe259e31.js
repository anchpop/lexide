// One shared time axis for phoneme emissions, STFT and playback.
export class AudioExplorer {
  constructor({ audio, onPhone, onFrame }) {
    this.audio = audio;
    this.onPhone = onPhone;
    this.onFrame = onFrame;
    this.$ = id => document.getElementById(id);
    this.duration = 0;
    this.phones = [];
    this.frameSeconds = null;
    this.animation = 0;
    this.$("explorer-play").onclick = () => {
      if (!audio.paused) audio.pause();
      else audio.play().catch(() => { this.$("spectrum-status").textContent = "Use the audio player above to start playback."; });
    };
    this.$("audio-zoom").onchange = () => { this.layout(); this.update(true); };
    const spectrum = this.$("spectrogram");
    const seekPointer = event => {
      const bounds = spectrum.getBoundingClientRect();
      this.seek((event.clientX - bounds.left) / bounds.width * this.duration);
    };
    spectrum.onpointerdown = event => {
      if (event.button !== 0) return;
      spectrum.setPointerCapture(event.pointerId);
      spectrum.focus({ preventScroll: true });
      seekPointer(event);
    };
    spectrum.onpointermove = event => {
      if (spectrum.hasPointerCapture(event.pointerId)) seekPointer(event);
    };
    spectrum.onpointerup = event => {
      if (spectrum.hasPointerCapture(event.pointerId)) spectrum.releasePointerCapture(event.pointerId);
    };
    spectrum.onkeydown = event => {
      // UI-only navigation step before inference, not a matrix timebase fallback.
      const step = event.shiftKey ? .1 : (this.frameSeconds ?? .02);
      const time = { ArrowLeft: this.audio.currentTime - step, ArrowDown: this.audio.currentTime - step,
        ArrowRight: this.audio.currentTime + step, ArrowUp: this.audio.currentTime + step,
        Home: 0, End: this.duration }[event.key];
      if (time === undefined) return;
      event.preventDefault();
      this.seek(time);
    };
    for (const event of ["timeupdate", "seeked", "pause", "ended"]) audio.addEventListener(event, () => this.update());
    audio.addEventListener("play", () => {
      cancelAnimationFrame(this.animation);
      const tick = () => {
        this.update(true);
        if (!audio.paused && !audio.ended) this.animation = requestAnimationFrame(tick);
      };
      tick();
    });
    this.resizeObserver = new ResizeObserver(() => this.layout());
    this.resizeObserver.observe(this.$("audio-scroll"));
    window.addEventListener("pagehide", () => { cancelAnimationFrame(this.animation); this.worker?.terminate(); });
  }

  clear() {
    this.worker?.terminate();
    this.worker = null;
    cancelAnimationFrame(this.animation);
    this.duration = 0;
    this.phones = [];
    this.frameSeconds = null;
    this.image = null;
    this.active = -1;
    this.selected = -1;
    this.$("audio-explorer").hidden = true;
    this.$("phoneme-track").replaceChildren();
    this.$("selected-region").hidden = true;
    this.$("audio-scroll").scrollLeft = 0;
    this.$("audio-zoom").value = "1";
  }

  load(samples, sampleRate) {
    this.clear();
    this.duration = samples.length / sampleRate;
    this.$("spectrum-status").textContent = "Drawing the spectrogram…";
    this.$("spectrogram").setAttribute("aria-valuemax", this.duration);
    // Off the UI thread: a long clip must not block recording or playback controls.
    try {
      this.worker = new Worker(new URL("./spectrogram-worker.0ea77c477369445f.js", import.meta.url), { type: "module" });
      const worker = this.worker;
      this.worker.onmessage = ({ data }) => {
        if (this.worker !== worker) return;
        if (data.error) { this.$("spectrum-status").textContent = "Spectrogram unavailable for this clip."; return; }
        this.image = document.createElement("canvas");
        this.image.width = data.columns;
        this.image.height = data.bins;
        const context = this.image.getContext("2d");
        const pixels = context.createImageData(data.columns, data.bins);
        const stops = [[10, 13, 30], [49, 27, 91], [125, 38, 112], [206, 75, 76], [248, 148, 65], [252, 239, 148]];
        for (let x = 0; x < data.columns; x++) for (let bin = 0; bin < data.bins; bin++) {
          const level = data.values[x * data.bins + bin] / 255 * (stops.length - 1);
          const low = Math.min(stops.length - 2, Math.floor(level)), mix = level - low;
          const i = ((data.bins - 1 - bin) * data.columns + x) * 4;
          for (let c = 0; c < 3; c++) pixels.data[i + c] = stops[low][c] * (1 - mix) + stops[low + 1][c] * mix;
          pixels.data[i + 3] = 255;
        }
        context.putImageData(pixels, 0, 0);
        this.$("spectrum-status").textContent = "Brighter = louder · 0–8 kHz";
        this.layout();
        this.worker?.terminate();
        this.worker = null;
      };
      this.worker.onerror = () => { this.$("spectrum-status").textContent = "Spectrogram unavailable. Audio and phoneme timing still work."; };
      const copy = Float32Array.from(samples);
      this.worker.postMessage({ samples: copy, sampleRate }, [copy.buffer]);
    } catch {
      this.$("spectrum-status").textContent = "Spectrogram unavailable in this browser.";
    }
  }

  show(phones, frameSeconds) {
    if (!Number.isFinite(frameSeconds) || frameSeconds <= 0) throw new Error("Invalid frame interval.");
    this.frameSeconds = frameSeconds;
    this.phones = phones;
    this.active = -1;
    const track = this.$("phoneme-track");
    track.replaceChildren();
    for (const [index, phone] of phones.entries()) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = phone.phoneme;
      button.title = `${phone.phoneme} · ${(phone.startFrame * this.frameSeconds).toFixed(2)}–${(phone.endFrame * this.frameSeconds).toFixed(2)}s (approximate)`;
      button.setAttribute("aria-label", button.title);
      button.tabIndex = -1; // The full IPA row below provides keyboard sound selection.
      button.onclick = () => this.onPhone(index, true);
      track.append(button);
    }
    this.$("audio-explorer").hidden = false;
    this.layout();
    this.update();
  }

  layout() {
    if (!this.duration || this.$("audio-explorer").hidden) return;
    const viewport = this.$("audio-scroll").clientWidth;
    this.width = Math.max(viewport, this.duration * 100) * +this.$("audio-zoom").value;
    this.$("audio-plot").style.width = `${this.width}px`;
    const canvas = this.$("spectrogram");
    canvas.width = Math.ceil(this.width);
    canvas.height = 200;
    const context = canvas.getContext("2d");
    context.fillStyle = "#0a0d1e";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (this.image) context.drawImage(this.image, 0, 0, canvas.width, canvas.height);
    for (const [index, button] of Array.from(this.$("phoneme-track").children).entries()) {
      const phone = this.phones[index];
      const width = (phone.endFrame - phone.startFrame) * this.frameSeconds / this.duration * this.width;
      button.style.left = `${phone.startFrame * this.frameSeconds / this.duration * this.width}px`;
      button.style.width = `${Math.max(2, width)}px`;
      button.classList.toggle("narrow", width < 15);
    }
    const axis = this.$("audio-axis");
    axis.replaceChildren();
    const step = this.width / this.duration > 180 ? .25 : .5;
    for (let time = 0; time < this.duration; time += step) {
      const tick = document.createElement("span");
      tick.textContent = `${time.toFixed(2)}s`;
      tick.style.left = `${time / this.duration * this.width}px`;
      axis.append(tick);
    }
    this.markSelection();
    this.update();
  }

  select(index, follow = false) {
    this.selected = index;
    this.markSelection();
    if (follow) this.ensureVisible(this.phones[index].startFrame * this.frameSeconds);
  }

  markSelection() {
    const phone = this.phones[this.selected];
    const region = this.$("selected-region");
    region.hidden = !phone;
    if (!phone || !this.width) return;
    region.style.left = `${phone.startFrame * this.frameSeconds / this.duration * this.width}px`;
    region.style.width = `${Math.max(2, (phone.endFrame - phone.startFrame) * this.frameSeconds / this.duration * this.width)}px`;
    for (const [i, button] of Array.from(this.$("phoneme-track").children).entries()) button.classList.toggle("selected", i === this.selected);
  }

  ensureVisible(time) {
    const scroller = this.$("audio-scroll"), x = time / this.duration * this.width;
    if (x < scroller.scrollLeft || x > scroller.scrollLeft + scroller.clientWidth - 24) {
      scroller.scrollLeft = Math.max(0, x - scroller.clientWidth * .25);
    }
  }

  seek(time) {
    if (!this.duration) return;
    this.audio.currentTime = Math.max(0, Math.min(this.duration, time));
    this.update(true);
  }

  update(follow = false) {
    if (!this.duration || this.$("audio-explorer").hidden) return;
    const time = Math.min(this.duration, Math.max(0, this.audio.currentTime));
    this.$("audio-playhead").style.left = `${time / this.duration * this.width}px`;
    this.$("audio-clock").textContent = `${time.toFixed(2)} / ${this.duration.toFixed(2)}s`;
    this.$("spectrogram").setAttribute("aria-valuenow", time.toFixed(2));
    this.$("spectrogram").setAttribute("aria-valuetext", `${time.toFixed(2)} seconds of ${this.duration.toFixed(2)}`);
    this.$("explorer-play").textContent = this.audio.paused ? "Play" : "Pause";
    const frame = this.frameSeconds ? Math.floor((time + 1e-6) / this.frameSeconds) : -1;
    const index = this.phones.findIndex(phone => frame >= phone.startFrame && frame < phone.endFrame);
    if (index !== this.active) {
      this.active = index;
      this.$("now-playing").textContent = index < 0 ? "—" : this.phones[index].phoneme;
      for (const [i, button] of Array.from(this.$("phoneme-track").children).entries()) button.classList.toggle("playing", i === index);
      if (index >= 0) this.onPhone(index, false);
    }
    this.onFrame(time);
    if (follow) this.ensureVisible(time);
  }
}
