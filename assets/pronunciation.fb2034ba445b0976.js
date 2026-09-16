import { AudioExplorer } from "./audio-explorer.733cd109b8ff4594.js";
import { unpackMatrix, decodePath, matrixMetadata } from "./pronunciation-decoder.de966b137c8e07fa.mjs";

// Existing production Lexide model, also used by Yap. No credentials in the page.
const ENDPOINT = "https://anchpop--wav2vec2-phoneme-wav2vec2phoneme-predict.modal.run";
const SAMPLE_RATE = 16000;
const MAX_SECONDS = 20;
const $ = (id) => document.getElementById(id);
let samples = null;
let playbackURL = null;
let recorder = null;
let stream = null;
let recordingTimer = null;
let recordingClock = null;
let request = null;
let canRetry = false;
let preparing = false;
let modelOutput = null;
let decoded = null;
let copyTimer = null;
let frameSeconds = null;
let blankId = null;
const explorer = new AudioExplorer({
  audio: $("playback"),
  onPhone: (index, seek) => selectPhone(index, seek),
  onFrame: time => {
    if (decoded) inspectFrame(Math.min(decoded.path.length - 1, Math.floor((time + 1e-6) / frameSeconds)));
  },
});


function controls() {
  const recording = recorder !== null;
  $("audio-file").disabled = preparing || recording || !!request;
  $("upload").disabled = preparing || recording || !!request;
  $("record").disabled = preparing || !!request;
  $("record").setAttribute("aria-pressed", String(recording));
  $("drop-zone").setAttribute("aria-busy", String(preparing || !!request));
  $("record").textContent = recording ? "Stop recording" : "Record microphone";
  $("retry").hidden = !canRetry || preparing || recording || !!request;
  $("cancel").hidden = !request;
}

function clearResult() {
  $("drop-zone").classList.remove("has-result");
  $("audio-explorer").hidden = true;
  modelOutput = null;
  decoded = null;
  $("model-version").textContent = "";
  $("result").hidden = true;
  $("ipa").textContent = "";
  $("phones").hidden = true;
  $("model-details").open = false;
  $("copy-status").textContent = "";
  clearTimeout(copyTimer);
  $("timing").textContent = "";
}

function clearAudio() {
  canRetry = false;
  explorer.clear();
  samples = null;
  $("clip-info").hidden = true;
  $("drop-zone").classList.remove("has-audio");
  $("playback").pause();
  $("playback").removeAttribute("src");
  $("playback").hidden = true;
  if (playbackURL) URL.revokeObjectURL(playbackURL);
  playbackURL = null;
  clearResult();
}

async function prepareAudio(blob, label) {
  clearAudio();
  preparing = true;
  controls();
  $("status").textContent = "Preparing audio…";
  let context;
  try {
    if (!blob.size) throw new Error("The recording is empty. Please try again.");
    if (blob.size > 25 * 1024 * 1024) throw new Error("Choose a file smaller than 25 MB.");
    context = new AudioContext();
    let decoded;
    try { decoded = await context.decodeAudioData(await blob.arrayBuffer()); }
    catch { throw new Error("Could not decode this audio. Try a WAV or MP3 file."); }
    if (decoded.duration > MAX_SECONDS) throw new Error("Choose a clip of 20 seconds or less.");
    if (decoded.duration < 0.1) throw new Error("The clip is too short. Record at least a moment of speech.");
    // Browser resampling also mixes stereo down to mono; keep the raw waveform.
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const mono = await offline.startRendering();
    samples = Array.from(mono.getChannelData(0));
    explorer.load(samples, SAMPLE_RATE);
    playbackURL = URL.createObjectURL(blob);
    $("playback").src = playbackURL;
    $("playback").hidden = false;
    $("clip-name").textContent = label;
    $("clip-name").title = label;
    $("clip-duration").textContent = `${decoded.duration.toFixed(1)} sec`;
    $("clip-info").hidden = false;
    $("drop-zone").classList.add("has-audio");
    $("status").textContent = "Ready to transcribe.";
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    if (context) await context.close().catch(() => {});
    preparing = false;
    controls();
  }
  if (samples) void transcribe();
}

$("audio-file").onchange = () => {
  const file = $("audio-file").files[0];
  if (file) void prepareAudio(file, file.name);
  $("audio-file").value = ""; // Selecting the same file again should also work.
};
$("upload").onclick = () => $("audio-file").click();

const dropZone = $("drop-zone");
let dragDepth = 0;
const inputBusy = () => preparing || recorder !== null || request !== null;
// Prevent a dropped file from navigating away, including drops outside the card.
window.addEventListener("dragover", (event) => {
  if (Array.from(event.dataTransfer.types).includes("Files")) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (Array.from(event.dataTransfer.types).includes("Files")) event.preventDefault();
  dragDepth = 0;
  dropZone.classList.remove("dragging");
});
dropZone.addEventListener("dragenter", (event) => {
  if (!Array.from(event.dataTransfer.types).includes("Files")) return;
  event.preventDefault();
  dragDepth++;
  if (!inputBusy()) dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.dataTransfer.dropEffect = inputBusy() ? "none" : "copy";
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (inputBusy()) return;
  const files = Array.from(event.dataTransfer.files);
  if (files.length !== 1) {
    $("status").textContent = "Drop one audio file at a time.";
    return;
  }
  void prepareAudio(files[0], files[0].name);
});

function stopRecording() {
  clearTimeout(recordingTimer);
  clearInterval(recordingClock);
  if (recorder?.state === "recording") recorder.stop();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
}

$("record").onclick = async () => {
  if (recorder) { stopRecording(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    $("status").textContent = "Microphone recording is unavailable here. Upload an audio file instead (recording requires HTTPS or localhost).";
    return;
  }
  preparing = true;
  controls();
  $("status").textContent = "Waiting for microphone permission…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    clearAudio();
    $("audio-file").value = "";
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearTimeout(recordingTimer);
      clearInterval(recordingClock);
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      const type = recorder.mimeType;
      recorder = null;
      void prepareAudio(new Blob(chunks, { type }), "Microphone recording");
    };
    recorder.onerror = () => {
      recorder.onstop = null;
      stopRecording();
      recorder = null;
      $("status").textContent = "Recording failed. Try again or upload an audio file.";
      controls();
    };
    recorder.start();
    const started = performance.now();
    const tick = () => {
      const elapsed = Math.floor((performance.now() - started) / 1000);
      $("status").textContent = `Recording · 0:${String(elapsed).padStart(2, "0")} / 0:19`;
    };
    tick();
    recordingClock = setInterval(tick, 250);
    // Leave headroom for the recorder's final encoded audio frames.
    recordingTimer = setTimeout(stopRecording, (MAX_SECONDS - 1) * 1000);
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    recorder = null;
    $("status").textContent = error.name === "NotAllowedError"
      ? "Microphone access was denied. Allow access in your browser or upload a file."
      : "Could not start the microphone. Try uploading an audio file.";
  } finally {
    preparing = false;
    controls();
  }
};

function renderResult(result, output) {
  ({ frameSeconds, blankId } = matrixMetadata(output.frame_matrix));
  decoded = result;
  modelOutput = output;
  $("ipa").append("[");
  for (const [index, phone] of result.phones.entries()) {
    const symbol = document.createElement("button");
    symbol.type = "button";
    symbol.className = "sound-choice";
    symbol.textContent = phone.phoneme;
    symbol.dataset.start = phone.startFrame;
    symbol.dataset.end = phone.endFrame;
    symbol.setAttribute("aria-label", `Sound ${index + 1}: ${phone.phoneme}`);
    symbol.setAttribute("aria-pressed", "false");
    symbol.onclick = () => selectPhone(index, true);
    symbol.onkeydown = (event) => {
      const next = { ArrowLeft: Math.max(0, index - 1), ArrowRight: Math.min(result.phones.length - 1, index + 1),
        Home: 0, End: result.phones.length - 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      selectPhone(next, true);
      $("ipa").querySelectorAll("button")[next].focus();
    };
    $("ipa").append(symbol);
  }
  $("ipa").append("]");
  $("phone-count").textContent = `${result.phones.length} sounds`;
  $("copy-ipa").disabled = !result.phones.length;
  $("frame-cursor").max = result.path.length - 1;
  $("frame-cursor").value = 0;
  $("model-version").textContent = `Production model · ${output.deploy_marker || "version not reported"} · ${result.path.length} frames`;
  $("result").hidden = false;
  $("drop-zone").classList.add("has-result");
  paintFrames();
  explorer.show(result.phones, frameSeconds);
  inspectFrame(0);
  if (result.phones.length) selectPhone(0);
}

function selectPhone(index, seek = false) {
  if (!decoded) return;
  const phone = decoded.phones[index];
  for (const [i, symbol] of $("ipa").querySelectorAll("button").entries()) {
    symbol.setAttribute("aria-pressed", String(i === index));
    symbol.tabIndex = i === index ? 0 : -1;
  }
  $("phones").hidden = false;
  $("selected-symbol").textContent = phone.phoneme;
  $("selected-position").textContent = `${index + 1} / ${decoded.phones.length}`;
  $("selected-time").textContent = `${(phone.startFrame * frameSeconds).toFixed(2)}–${(phone.endFrame * frameSeconds).toFixed(2)} sec · approximate`;
  $("selected-confidence").textContent = `${Math.round(phone.confidence * 100)}% confidence`;
  $("alternatives").replaceChildren();
  for (const alternative of phone.top_k) {
    const chip = document.createElement("span");
    chip.className = "alternative";
    chip.append(alternative.id === blankId ? "Blank" : alternative.phoneme);
    const percent = document.createElement("small");
    percent.textContent = `${(alternative.probability * 100).toFixed(1)}%`;
    chip.append(percent);
    $("alternatives").append(chip);
  }
  explorer.select(index, seek);
  if (seek) explorer.seek(phone.startFrame * frameSeconds);
}

$("copy-ipa").onclick = async () => {
  if (!decoded) return;
  clearTimeout(copyTimer);
  try {
    await navigator.clipboard.writeText(`[${decoded.phones.map(phone => phone.phoneme).join("")}]`);
    $("copy-status").textContent = "Copied";
  } catch {
    $("copy-status").textContent = "Copy unavailable. Select the IPA text to copy it.";
  }
  copyTimer = setTimeout(() => { $("copy-status").textContent = ""; }, 3000);
};
$("model-details").addEventListener("toggle", () => {
  if ($("model-details").open) paintFrames();
});

function paintFrames() {
  if (!decoded) return;
  const canvas = $("frame-strip");
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
  canvas.height = Math.round(40 * devicePixelRatio);
  const context = canvas.getContext("2d");
  const width = canvas.width / decoded.path.length;
  for (const [i, frame] of decoded.path.entries()) {
    context.fillStyle = frame.blank ? "#858585" : `hsl(${(frame.id * 137.5) % 360} 45% 65%)`;
    context.fillRect(i * width, 0, Math.ceil(width), canvas.height);
  }
}

function inspectFrame(index) {
  if (!decoded) return;
  const frame = decoded.path[index];
  $("frame-cursor").value = index;
  const text = `Frame ${index} · ${(index * frameSeconds).toFixed(2)}s · ${frame.phoneme} · ${(frame.probability * 100).toFixed(1)}%`;
  $("frame-detail").textContent = text;
  $("frame-cursor").setAttribute("aria-valuetext", text);
  for (const symbol of $("ipa").querySelectorAll("button")) {
    symbol.classList.toggle("current-phone", index >= +symbol.dataset.start && index < +symbol.dataset.end);
  }
}

$("frame-cursor").oninput = () => {
  const index = +$("frame-cursor").value;
  inspectFrame(index);
  $("playback").currentTime = index * frameSeconds;
};
$("playback").addEventListener("timeupdate", () => {
  if (decoded) inspectFrame(Math.min(decoded.path.length - 1, Math.floor(($("playback").currentTime + 1e-6) / frameSeconds)));
});
window.addEventListener("resize", paintFrames);
$("download-frames").onclick = () => {
  if (!modelOutput) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify({
    ...modelOutput,
    demo_metadata: { sample_rate: SAMPLE_RATE, frame_stride_seconds: frameSeconds,
      decoder: "nonblank-first CTC over float16 phoneme matrix", phones: decoded.phones },
  })], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "lexide-model-output.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

async function transcribe() {
  if (!samples || request) return;
  canRetry = false;
  clearResult();
  request = new AbortController();
  controls();
  const start = performance.now();
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; request?.abort(); }, 180000);
  $("status").textContent = "Transcribing… The model may need a minute to wake up.";
  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio: samples, sample_rate: SAMPLE_RATE, top_k: 3, return_frames: true, return_frame_matrix: true }),
      signal: request.signal,
    });
    if (!response.ok) throw new Error(`The model returned an error (${response.status}). Please try again shortly.`);
    const data = await response.json();
    const matrix = await unpackMatrix(data.frame_matrix);
    let result;
    try { result = decodePath(matrix, data.frames); }
    finally { matrix.free(); }
    if (request.signal.aborted) throw new DOMException("Aborted", "AbortError");
    renderResult(result, data);
    $("status").textContent = result.phones.length
      ? `${result.phones.length} phonemes recognized.`
      : "No phonemes detected. Try a clip with clearer speech.";
    $("timing").textContent = `${((performance.now() - start) / 1000).toFixed(1)}s`;
  } catch (error) {
    canRetry = true;
    $("status").textContent = error.name === "AbortError"
      ? (timedOut ? "The model took too long to respond. Please try again." : "Transcription canceled.")
      : error instanceof TypeError
        ? "Could not reach the model. Check your connection and try again."
        : error.message;
  } finally {
    clearTimeout(timeout);
    request = null;
    controls();
  }
}
$("retry").onclick = () => void transcribe();
$("cancel").onclick = () => request?.abort();
window.addEventListener("pagehide", () => {
  if (recorder) recorder.onstop = null;
  stopRecording();
  recorder = null;
  request?.abort();
  controls();
});
