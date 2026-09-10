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
let request = null;
let preparing = false;

function controls() {
  const recording = recorder !== null;
  $("audio-file").disabled = preparing || recording || !!request;
  $("record").disabled = preparing || !!request;
  $("record").textContent = recording ? "Stop recording" : "Record microphone";
  $("transcribe").disabled = !samples || preparing || recording || !!request;
  $("cancel").hidden = !request;
}

function clearResult() {
  $("result").hidden = true;
  $("ipa").textContent = "";
  $("phones").replaceChildren();
  $("timing").textContent = "";
}

function clearAudio() {
  samples = null;
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
    playbackURL = URL.createObjectURL(blob);
    $("playback").src = playbackURL;
    $("playback").hidden = false;
    $("status").textContent = `${label} · ${decoded.duration.toFixed(1)} seconds · ready to transcribe.`;
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    if (context) await context.close().catch(() => {});
    preparing = false;
    controls();
  }
}

$("audio-file").onchange = () => {
  const file = $("audio-file").files[0];
  if (file) void prepareAudio(file, file.name);
};

function stopRecording() {
  clearTimeout(recordingTimer);
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
    $("status").textContent = "Recording… Select Stop recording when finished (stops automatically before 20 seconds).";
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

function renderResult(phonemes) {
  $("ipa").textContent = phonemes.map((phone) => phone.phoneme).join("");
  for (const phone of phonemes) {
    const detail = document.createElement("details");
    detail.className = "phone";
    const summary = document.createElement("summary");
    summary.append(phone.phoneme + " ");
    const confidence = document.createElement("small");
    confidence.textContent = `${Math.round(phone.confidence * 100)}%`;
    summary.append(confidence);
    detail.append(summary);
    for (const alternative of phone.top_k || []) {
      const row = document.createElement("p");
      row.textContent = `${alternative.phoneme} · ${(alternative.probability * 100).toFixed(1)}%`;
      detail.append(row);
    }
    $("phones").append(detail);
  }
  $("result").hidden = false;
}

$("transcribe").onclick = async () => {
  if (!samples || request) return;
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
      body: JSON.stringify({ audio: samples, sample_rate: SAMPLE_RATE, top_k: 3 }),
      signal: request.signal,
    });
    if (!response.ok) throw new Error(`The model returned an error (${response.status}). Please try again shortly.`);
    const data = await response.json();
    if (!Array.isArray(data.phonemes) || data.phonemes.some((phone) =>
      typeof phone.phoneme !== "string" || !Number.isFinite(phone.confidence))) {
      throw new Error("The model returned an unexpected response. Please try again.");
    }
    if (data.phonemes.length) {
      renderResult(data.phonemes);
      $("status").textContent = `${data.phonemes.length} phonemes recognized.`;
    } else {
      $("status").textContent = "No phonemes detected. Try a clip with clearer speech.";
    }
    $("timing").textContent = `${((performance.now() - start) / 1000).toFixed(1)}s`;
  } catch (error) {
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
};
$("cancel").onclick = () => request?.abort();
window.addEventListener("pagehide", () => {
  if (recorder) recorder.onstop = null;
  stopRecording();
  recorder = null;
  request?.abort();
  controls();
});
