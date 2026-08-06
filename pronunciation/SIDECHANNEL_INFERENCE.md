# Inference for the `mel_sidechannel` + `mlp_heads` model

**Model:** `anchpop/lexide-pronunciation-vad-clean-sidechannel`
(xls-r-2b backbone, narrowed labels, 12 epochs, `--mel-sidechannel --mlp-heads`).

## TL;DR — there is NO new inference path

The mel side-channel and MLP heads are **entirely internal to the model's forward
pass**. Inference is the *standard* path used for every other checkpoint:

```python
from src.factorized_ctc import FactorizedCTCModel   # train/src on sys.path
model = FactorizedCTCModel.load_from_dir(snapshot_dir)   # rebuilds arch from saved flags
model.eval().to(device)
out = model(input_values)                                # input_values: (B, T) raw-ish waveform
# phone-only default: language_head_logits is an empty dict (no extra MLP work)
# pass active_language_heads=["jpn_pitch_accent"] for prosody-aware inference
```

`load_from_dir` reads `mel_sidechannel`/`mlp_heads` from the saved `factorized_heads.pt`
and reconstructs the modules itself — you do **not** pass any flags. Existing callers
can continue reading `log_probs`, `nonblank_logit`, and `stress_logits`; the additional
`language_head_logits` mapping is empty unless heads are explicitly activated.

### Two corrections to what you were told
- **No new Modal inference path is needed.** Same `load_from_dir` + `model(input_values)`.
- **There is NO feature-emission in this model.** Feature-emission belongs to the
  *articulatory-aux* variant (`--use-aux-features`). This run used neither
  `--use-features` nor `--use-aux-features`; `feature_head is None`,
  `feature_emission_weight == 0`. Whoever mentioned feature-emission conflated it with a
  different experiment.

## The only real requirements for the image

1. Ship the **current** `train/src/factorized_ctc.py` (with `mel_sidechannel`/`mlp_heads`
   support) — not an old pinned copy.
2. `pip install torchaudio` — the internal `mel_spec` imports it lazily. The plain
   vad-clean model never needed torchaudio, so the stock image may lack it. This is the
   ONLY genuine gap.

## What the code actually does

### 1. Construction (`FactorizedCTCModel.__init__`)
```python
elif mel_sidechannel:
    self.layer_weights = None
    self.shared_base = None
    import torchaudio                                  # <-- the only extra dep
    self.mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400,
        hop_length=320, n_mels=n_mels, center=False)
    self.mel_norm = nn.LayerNorm(n_mels)
    self.mel_proj = nn.Linear(n_mels, acoustic_dim)
    head_input_dim = hidden_size + acoustic_dim         # 1920 + 64 = 1984

def _mk_head(dout):                                     # MLP heads when mlp_heads=True
    if mlp_heads:
        return nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(head_input_dim, dout))
    return nn.Linear(head_input_dim, dout)
# self.nonblank_head = _mk_head(1);  self.phoneme_head = _mk_head(vocab_size)
```

### 2. Forward (`_compute_head_input`) — the entire "side-channel path", self-contained
```python
if self.mel_sidechannel:
    out = self.backbone(input_values=input_values, attention_mask=attention_mask)
    hidden = self.dropout(out.last_hidden_state)
    with torch.autocast(device_type=input_values.device.type, enabled=False):
        mel = self.mel_spec(input_values.float())            # mel straight from the waveform
    mel = mel.transpose(1, 2)
    T_target = hidden.shape[1]
    if mel.shape[1] > T_target:   mel = mel[:, :T_target, :]
    elif mel.shape[1] < T_target: mel = F.pad(mel, (0, 0, 0, T_target - mel.shape[1]))
    mel = torch.log(mel + 1e-6).to(hidden.dtype)
    mel = self.mel_norm(mel)                                  # per-frame LayerNorm over mel bins
    mel_proj = self.mel_proj(mel)
    return torch.cat([hidden, mel_proj], dim=-1)             # heads consume [hidden ; mel_proj]
```
The heads then run on this exactly as in the base model.

### 3. Round-trip (`load_from_dir`) — rebuilds from the checkpoint, no manual flags
```python
mel_sidechannel = heads.get("mel_sidechannel", False)
mlp_heads       = heads.get("mlp_heads", False)
if mel_sidechannel:
    reg_kwargs["acoustic_dim"] = heads["acoustic_dim"]; reg_kwargs["n_mels"] = heads["n_mels"]
model = cls(..., mel_sidechannel=mel_sidechannel, mlp_heads=mlp_heads, **reg_kwargs)
if mel_sidechannel:
    model.mel_norm.load_state_dict(heads["mel_norm"])
    model.mel_proj.load_state_dict(heads["mel_proj"])
```

## Audio normalization (a subtlety, not a blocker)

The model computes mel from `input_values`, and **training fed the raw waveform**
(dataset.py applies no normalization). `infer.py`'s wav2vec2 path normalizes
(`do_normalize=True`), which is per-utterance `(x-mean)/std` ≈ a uniform scale → a
*constant* log-mel offset across all bins → and the per-frame `mel_norm` LayerNorm
subtracts exactly that. So the side-channel is robust to the normalization; the existing
normalized path works.

`inference/infer.py` now feeds **raw audio** automatically for any mel-computing model
(`mel_sidechannel`, `regularized_heads`, or the Cohere front-end) — so it matches training
exactly with no caller action needed:
```python
_needs_raw = (
    getattr(model.backbone.config, "model_type", "") == "cohere_conformer_ctc"
    or getattr(model, "mel_sidechannel", False)
    or getattr(model, "regularized_heads", False)
)
input_values = (torch.from_numpy(audio).float().unsqueeze(0).to(device) if _needs_raw
                else processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device))
```
If you have your own inference path, do the same — feed raw for these models. (mel_norm
largely absorbs the per-utterance scale either way, which is why normalized input didn't
corrupt earlier evals, but raw removes the dependence.)
