"""Drop-in wav2vec2-style backbone built on Cohere Transcribe's Conformer encoder.

Pipeline:  raw 16k waveform
  → Cohere's exact log-mel (FilterbankFeatures, on-device)   [128 mel, 10 ms hop]
  → 48-layer Fast-Conformer encoder                           [8× subsample → 80 ms / 12.5 fps, d=1280]
  → per-coarse-frame MLP upsampler (4×, NON-overlapping)      [→ 20 ms / 50 fps, so CTC has enough frames
                                                               AND the trainer's 50 fps frame logic is unchanged]
  → last_hidden_state (B, T≈audio/320, 1280)

Why the upsampler: CTC needs frames ≥ phonemes. At 80 ms (12.5 fps) that's *below* the
phoneme rate (~13/s), so plain CTC can't fit the labels. 4× → 50 fps fixes it and exactly
matches wav2vec2's frame rate, so factorized_ctc/train_unified work without frame-rate edits.

Presents the minimal interface factorized_ctc.py expects of a Wav2Vec2Model:
  .config.hidden_size (=1280), .config.num_hidden_layers (=48)
  forward(input_values, attention_mask, output_hidden_states=False) -> obj with .last_hidden_state
  _get_feat_extract_output_lengths(lengths) -> upsampled frame counts
  save_pretrained(dir) / load via from_pretrained(dir)
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

COHERE_ID = "CohereLabs/cohere-transcribe-03-2026"
UPSAMPLE = 4  # 80 ms -> 20 ms (match wav2vec2's 50 fps)


class CohereBackbone(nn.Module):
    def __init__(self, model_name: str = COHERE_ID, upsample: int = UPSAMPLE, token: str | None = None):
        super().__init__()
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        token = token or os.environ.get("HF_TOKEN")
        full = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float32,
            low_cpu_mem_usage=True, token=token)
        self.encoder = full.get_encoder()                 # ConformerEncoder (~1.9B), weights loaded
        del full                                          # drop the decoder
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=token)
        # The exact mel module the official extractor uses (correct params + buffers).
        self.featurizer = proc.feature_extractor.filterbank
        self.featurizer.dither = 0.0                      # deterministic mel for training

        self.d_model = int(self.encoder.d_model)          # 1280
        self.upsample = int(upsample)
        # Bigger, strictly NON-OVERLAPPING upscaler: a per-coarse-frame MLP that
        # fans each 80 ms encoder frame into `upsample` fine (20 ms) frames. Each
        # fine block is a function of ONLY its parent coarse frame — there is no
        # mixing across coarse frames — so all cross-frame (global) reasoning
        # stays in the encoder; this module only adds *local* expressiveness.
        # (The previous single linear ConvTranspose1d couldn't fan out well —
        # nonblank_prob stalled at ~0.14. Nonlinear + wider gives it room without
        # an overlapping kernel's cross-frame smoothing.)
        hidden = self.upsample * self.d_model
        self.upsampler = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.upsample * self.d_model),
        )

        self.config = SimpleNamespace(
            hidden_size=self.d_model,
            num_hidden_layers=len(self.encoder.layers),
            final_dropout=0.0,
            model_type="cohere_conformer_ctc",
        )
        self._cohere_model_name = model_name
        self._grad_ckpt = False

    # -- activation checkpointing ---------------------------------------------
    # The Conformer encoder's own forward loops over layers with no checkpoint
    # hook (supports_gradient_checkpointing = False), so we wrap each layer's
    # forward with torch.utils.checkpoint to cut activation memory for the 1.9B
    # encoder fine-tune. Trainer calls this via factorized_ctc.gradient_checkpointing_enable.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kw):
        if self._grad_ckpt:
            return
        self._grad_ckpt = True
        import torch.utils.checkpoint as _ckpt
        for layer in self.encoder.layers:
            if getattr(layer, "_orig_forward", None) is None:
                layer._orig_forward = layer.forward

            def _make(_layer):
                def _ckpt_forward(*a, **k):
                    if not (self.training and torch.is_grad_enabled()):
                        return _layer._orig_forward(*a, **k)
                    return _ckpt.checkpoint(_layer._orig_forward, *a, use_reentrant=False, **k)
                return _ckpt_forward
            layer.forward = _make(layer)

    def gradient_checkpointing_disable(self, **kw):
        if not self._grad_ckpt:
            return
        self._grad_ckpt = False
        for layer in self.encoder.layers:
            if getattr(layer, "_orig_forward", None) is not None:
                layer.forward = layer._orig_forward

    # -- length bookkeeping (kept consistent with forward) --------------------
    def _mel_len(self, wav_len: torch.Tensor) -> torch.Tensor:
        return self.featurizer.get_seq_len(wav_len.float())

    def _subsample_len(self, mel_len: torch.Tensor) -> torch.Tensor:
        # ConvSubsampling: 3× Conv2d stride 2, kernel 3, pad 1 → out = (in - 1)//2 + 1
        n = mel_len.long()
        for _ in range(3):
            n = (n - 1) // 2 + 1
        return n

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # waveform samples -> mel frames -> 8× subsample -> 4× upsample
        return (self._subsample_len(self._mel_len(input_lengths)) * self.upsample).to(torch.long)

    # -- forward --------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, output_hidden_states: bool = False, **kw):
        if attention_mask is None:
            wav_len = torch.full((input_values.shape[0],), input_values.shape[-1],
                                 device=input_values.device, dtype=torch.long)
        else:
            wav_len = attention_mask.sum(-1)

        # waveform -> exact Cohere mel (B, 128, T_mel)
        mel, mel_len = self.featurizer(input_values.float(), wav_len.float())
        # Fast-Conformer encoder (B, T_sub≈T_mel/8, 1280)
        enc, _sub_len = self.encoder(input_features=mel.to(next(self.encoder.parameters()).dtype),
                                     length=mel_len)
        # Per-coarse-frame fan-out → ~50 fps. enc (B,T,D) → (B,T,upsample*D) →
        # reshape to (B, T*upsample, D), interleaving the `upsample` fine frames
        # of coarse frame t before those of t+1. Strictly non-overlapping: fine
        # frame (t*upsample + j) depends only on enc[:, t].
        B, T, D = enc.shape
        up = self.upsampler(enc)                                  # (B, T, upsample*D)
        up = up.view(B, T, self.upsample, D).reshape(B, T * self.upsample, D)

        out = SimpleNamespace(last_hidden_state=up)
        if output_hidden_states:
            out.hidden_states = (up,)   # standard mode only uses last_hidden_state
        return out

    # -- checkpointing (mirror Wav2Vec2Model.save_pretrained semantics) -------
    def save_pretrained(self, save_dir, **kw):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "model_name": self._cohere_model_name,
            "upsample": self.upsample,
        }, os.path.join(save_dir, "cohere_backbone.pt"))

    @classmethod
    def from_pretrained(cls, load_dir, token=None, **kw):
        ckpt_path = os.path.join(str(load_dir), "cohere_backbone.pt")
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            obj = cls(model_name=ck["model_name"], upsample=ck["upsample"], token=token)
            obj.load_state_dict(ck["state_dict"])
            return obj
        return cls(model_name=str(load_dir), token=token, **kw)
