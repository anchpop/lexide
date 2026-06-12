"""Forced alignment of espeak's broad/citation target to real audio, on Modal.

Extends the proven `yap/modal-envs/wav2vec2_phoneme.py` pattern for the SAME
model (anchpop/lexide-pronunciation-unified-vad-clean). The model is used ONLY
to place time boundaries — it inherits espeak's broadness (trained on espeak
labels) so it gets no vote on correctness. The acoustics (local parselmouth)
arbitrate.

Method `force_align_batch` takes pre-phonemized targets (ids computed locally
from the espeak fork + vocab.json) plus raw f32 samples, runs the model's CTC
log-probs, and returns per-target-token time spans via torchaudio.forced_align.
Per-token alignment score + duration are returned too: an espeak phone that
isn't in the audio (e.g. a deleted French schwa) gets ~0 frames / low score,
an independent detector of over-specification.
"""

import modal

app = modal.App("espeak-audit-aligner")

MODEL_ID = "anchpop/lexide-pronunciation-unified-vad-clean"
# PINNED commit, not "main": alignment depends on the exact weights, so a moving
# ref would silently change spans/measurements (and invalidate caches without the
# cache key knowing). Bumping vad-clean = bump this SHA = a fresh cache namespace.
MODEL_REVISION = "2926e06f8092935f597e0018beb5d579b95b889a"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchaudio", "transformers", "huggingface_hub", "numpy",
        # measure-on-Modal: parselmouth (Praat DSP) + scipy (welch) for the
        # acoustic arbiter, so per-segment measurement runs where the audio
        # already is (parallel across containers) instead of a local bottleneck.
        "praat-parselmouth", "scipy",
    )
    .run_commands(
        "echo 'WEIGHTS=vad_clean_align_v1' && "
        "python -c \"from transformers import Wav2Vec2Model, Wav2Vec2Processor; "
        "from huggingface_hub import hf_hub_download; "
        f"Wav2Vec2Processor.from_pretrained('{MODEL_ID}', revision='{MODEL_REVISION}'); "
        f"Wav2Vec2Model.from_pretrained('{MODEL_ID}', revision='{MODEL_REVISION}'); "
        f"hf_hub_download('{MODEL_ID}', 'factorized_heads.pt', revision='{MODEL_REVISION}')\""
    )
    # The measurement code (phonetics.py) ships into the container so it runs
    # identically here and locally.
    .add_local_python_source("phonetics")
)


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=300,
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VadCleanAligner:
    @modal.enter(snap=True)
    def load_model(self):
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.backbone = Wav2Vec2Model.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16
        ).to("cuda")
        self.backbone.eval()
        H = self.backbone.config.hidden_size

        ckpt = torch.load(
            hf_hub_download(MODEL_ID, "factorized_heads.pt", revision=MODEL_REVISION),
            map_location="cpu", weights_only=False,
        )
        assert not ckpt.get("regularized_heads", False), (
            "this aligner targets the simple (mode=off) vad-clean variant"
        )
        self.blank_id = int(ckpt["blank_id"])
        self.masked_slots = list(ckpt.get("masked_slots") or [self.blank_id])
        V = ckpt["vocab_size"]

        self.nonblank_head = nn.Linear(H, 1)
        self.phoneme_head = nn.Linear(H, V)
        self.nonblank_head.load_state_dict(ckpt["nonblank_head"])
        self.phoneme_head.load_state_dict(ckpt["phoneme_head"])
        self.nonblank_head = self.nonblank_head.to("cuda").eval()
        self.phoneme_head = self.phoneme_head.to("cuda").eval()

        self.inv_vocab = {i: t for t, i in self.processor.tokenizer.get_vocab().items()}

        # Warmup (captured in snapshot).
        dummy = self.processor([0.0] * 16000, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            self._log_probs(dummy.input_values.to("cuda").to(torch.float16))

    def _log_probs(self, input_values):
        """CTC log-probs (1, T, V) in fp32 — same factorization as training."""
        import torch
        import torch.nn.functional as F

        out = self.backbone(input_values)
        h = out.last_hidden_state.float()
        l_nb = self.nonblank_head(h).squeeze(-1)            # (1, T)
        l_ph = self.phoneme_head(h).clone()                 # (1, T, V)
        for slot in self.masked_slots:
            l_ph[..., slot] = float("-inf")
        log_p_blank = F.logsigmoid(-l_nb)
        log_p_nonblank = F.logsigmoid(l_nb)
        log_probs = (log_p_nonblank.unsqueeze(-1) + F.log_softmax(l_ph, dim=-1)).clone()
        log_probs[..., self.blank_id] = log_p_blank
        return log_probs                                    # (1, T, V)

    def _reading(self, log_probs):
        """Greedy CTC decode -> phoneme symbols (the model's 'second opinion')."""
        ids = log_probs[0].argmax(-1).tolist()
        seq, prev = [], None
        for i in ids:
            if i != prev and i != self.blank_id:
                seq.append(self.inv_vocab.get(i, "?"))
            prev = i
        return seq

    def _align_samples(self, samples, sr: int, target_ids: list):
        """Core CTC forced-alignment for one clip. Returns
        (n_frames, audio_sec, spans[[start_s,end_s,score]], reading, align_error)."""
        import torch
        import torchaudio.functional as AF

        audio_sec = len(samples) / sr
        inputs = self.processor(samples, sampling_rate=sr, return_tensors="pt", padding=True)
        iv = inputs.input_values.to("cuda").to(torch.float16)
        with torch.no_grad():
            log_probs = self._log_probs(iv)                 # (1, T, V) fp32
        T = log_probs.shape[1]
        sec_per_frame = audio_sec / T

        target_ids = [int(x) for x in target_ids]
        spans_out = []
        align_error = None
        # CTC forced alignment requires T >= len(target) + #adjacent-repeats.
        # Long espeak targets on short/fast clips can violate this; skip those.
        n_repeats = sum(1 for a, b in zip(target_ids, target_ids[1:]) if a == b)
        if target_ids and (len(target_ids) + n_repeats) <= T:
            try:
                targets = torch.tensor([target_ids], dtype=torch.long)
                lp = log_probs.cpu().contiguous()
                aligned, scores = AF.forced_align(lp, targets, blank=self.blank_id)
                spans = AF.merge_tokens(aligned[0], scores[0], blank=self.blank_id)
                if len(spans) != len(target_ids):
                    raise RuntimeError(f"span count {len(spans)} != target {len(target_ids)}")
                for sp in spans:
                    spans_out.append([float(sp.start * sec_per_frame),
                                      float(sp.end * sec_per_frame), float(sp.score)])
            except Exception as e:  # noqa: BLE001
                align_error = f"{type(e).__name__}: {e}"
        elif target_ids:
            align_error = f"target_too_long: L={len(target_ids)}+rep{n_repeats} > T={T}"
        return int(T), float(audio_sec), spans_out, self._reading(log_probs), align_error

    @modal.method()
    def force_align_batch(self, items: list) -> list:
        """items: [{key, audio: list[float], sample_rate, target_ids: [int]}].
        Returns [{key, n_frames, audio_sec, spans, reading, align_error}]."""
        out = []
        for it in items:
            sr = int(it.get("sample_rate", 16000))
            T, audio_sec, spans, reading, err = self._align_samples(
                it["audio"], sr, it["target_ids"])
            out.append({"key": it["key"], "n_frames": T, "audio_sec": audio_sec,
                        "spans": spans, "reading": reading, "align_error": err})
        return out

    @modal.method()
    def align_batch_b64(self, items: list) -> list:
        """ALIGN-ONLY at scale (no parselmouth). Audio arrives as compact base64
        float32; only the tiny alignment spans come back. Measurement is done
        LOCALLY by the caller — the GPU container does the one thing it must (CTC
        forced-align) and never burns GPU time on CPU DSP. This is the cheap path:
        boundaries cached once, then any measurement/param change re-runs locally
        with zero Modal involvement.

        items: [{key, audio_b64, sample_rate, target_ids}].
        Returns [{key, n_frames, audio_sec, spans, reading, align_error}].
        """
        import base64
        import numpy as np

        out = []
        for it in items:
            sr = int(it.get("sample_rate", 16000))
            samples = np.frombuffer(base64.b64decode(it["audio_b64"]), dtype="<f4").astype(np.float64)
            T, audio_sec, spans, reading, err = self._align_samples(
                samples.tolist(), sr, it["target_ids"])
            out.append({"key": it["key"], "n_frames": T, "audio_sec": audio_sec,
                        "spans": spans, "reading": reading, "align_error": err,
                        "model_revision": MODEL_REVISION})  # caller's stale-deploy guard
        return out

    @modal.method()
    def align_and_measure_batch(self, items: list) -> list:
        """Align AND parselmouth-measure each clip on the container — the
        at-scale path. Audio arrives as base64 float32 bytes (compact); only the
        tiny per-segment measurements come back.

        items: [{key, audio_b64, sample_rate, target_ids, phonemes, stress,
                 word_of, ok, keep}].
        Returns [{key, n_frames, audio_sec, reading, align_error, segments}].
        """
        import base64
        import numpy as np
        import phonetics

        out = []
        for it in items:
            sr = int(it.get("sample_rate", 16000))
            samples = np.frombuffer(base64.b64decode(it["audio_b64"]), dtype="<f4").astype(np.float64)
            T, audio_sec, spans, reading, err = self._align_samples(
                samples.tolist(), sr, it["target_ids"])
            segments = phonetics.measure_segments(
                samples, sr, it["phonemes"], it["stress"], it["word_of"],
                it["ok"], it["keep"], spans)
            out.append({"key": it["key"], "n_frames": T, "audio_sec": audio_sec,
                        "reading": reading, "align_error": err, "segments": segments,
                        "model_revision": MODEL_REVISION})  # for the caller's stale-deploy guard
        return out
