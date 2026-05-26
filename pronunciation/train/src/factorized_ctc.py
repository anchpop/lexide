"""Factorized-CTC wav2vec2 with optional articulatory-feature phoneme head.

Two factorization layers:

1. Blank vs phoneme (always on):
       P(blank   | t)         = σ(-l_nb_t)
       P(phoneme | t, ¬blank) = softmax(l_ph_t)
       P(phoneme | t)         = σ(l_nb_t) * softmax(l_ph_t)
   Decouples the blank decision from phoneme identity. See feedback_no_safety_nets
   and the original derivation: blank gets a clean binary gradient that isn't
   coupled to phoneme-identity noise through the softmax denominator.

2. Articulatory features (optional, opt-in via `feature_table`):
   Instead of a single Linear(H, V) projecting directly to phoneme logits,
   project to F feature heads (F=24, panphon's Hayes schema), each predicting
   ternary {-, 0, +} for one feature (nasal, voiced, high, back, …). Phoneme
   log-prob is derived by *summing* feature log-probs at the indices given by
   a lookup table (per-phoneme feature signature):

       log P(phoneme=v | t) = sum_i log P(feature_i = table[v, i] | t)

   This forces the model to learn each articulatory dimension explicitly,
   which should help rare phonemes (nasal vowels in our case) because every
   nasal-bearing language in training contributes to the shared `nas` head.

   An `invalid_mass_weight` regularizer penalizes feature distributions that
   put probability on combinations not in the vocab — discouraging the model
   from leaving the phoneme manifold.

Combined log-probs are passed to standard `F.ctc_loss`.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


NUM_FEATURE_VALUES = 3  # panphon ternary {-1, 0, +1} → encoded as {0, 1, 2}


class FactorizedCTCModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-2b",
        vocab_size: int = 392,
        blank_id: int = 0,
        num_stress_labels: int = 3,
        feature_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_stress_labels = num_stress_labels

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(self.backbone.config.final_dropout)
        self.nonblank_head = nn.Linear(hidden_size, 1)
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_stress_labels),
        )

        if feature_table is not None:
            # Articulatory factorization: predict each feature, derive phoneme log-prob.
            assert feature_table.shape[0] == vocab_size, (
                f"feature_table rows ({feature_table.shape[0]}) must match vocab_size ({vocab_size})"
            )
            self.num_features = int(feature_table.shape[1])
            # One Linear that outputs F*K logits, reshaped to (B, T, F, K).
            self.feature_head = nn.Linear(hidden_size, self.num_features * NUM_FEATURE_VALUES)
            self.phoneme_head = None
            self.register_buffer("feature_table", feature_table.long())
            nn.init.zeros_(self.feature_head.bias)
        else:
            # Direct phoneme head (original).
            self.num_features = 0
            self.phoneme_head = nn.Linear(hidden_size, vocab_size)
            self.feature_head = None
            self.feature_table = None  # not a buffer, just None
            nn.init.zeros_(self.phoneme_head.bias)

        nn.init.zeros_(self.nonblank_head.bias)

    @property
    def use_features(self) -> bool:
        return self.feature_head is not None

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def _phoneme_log_probs_from_features(self, hidden: torch.Tensor):
        """Return (l_ph_norm, invalid_mass) using feature factorization.

        l_ph_norm: (B, T, V) — normalized log P(phoneme | ¬blank). The blank
                   slot is set to -inf.
        invalid_mass: (B, T) — probability mass that fell outside the vocab
                   under the *unnormalized* feature factorization. Used as a
                   regularization signal.
        """
        B, T, _ = hidden.shape
        F_dim = self.num_features
        V = self.vocab_size

        # (B, T, F*K) → (B, T, F, K), then log_softmax over K.
        feat = self.feature_head(hidden).view(B, T, F_dim, NUM_FEATURE_VALUES)
        feat_log_probs = F.log_softmax(feat, dim=-1)  # (B, T, F, K)

        # For each phoneme v, look up its feature indices and sum the log-probs.
        # feature_table: (V, F)  →  indices into K for each (v, i).
        # Gather feat_log_probs[..., table[v, i]] then sum over i.
        # Expand table to (1, 1, F, V), feat to (B, T, F, K), gather along K dim.
        idx = self.feature_table.t().contiguous()              # (F, V)
        idx = idx.view(1, 1, F_dim, V).expand(B, T, F_dim, V)  # (B, T, F, V)
        gathered = torch.gather(
            feat_log_probs.unsqueeze(-1).expand(B, T, F_dim, NUM_FEATURE_VALUES, V).gather(
                3, idx.unsqueeze(3)
            ).squeeze(3),
            # equivalent simpler form below — keep code readable
            dim=-1, index=torch.zeros_like(idx)
        )  # placeholder, replaced below

        # Cleaner equivalent: stack feat_log_probs gather per-feature.
        # feat_log_probs: (B, T, F, K). For each v: pick feat_log_probs[..., i, table[v, i]] then sum over i.
        # Easiest: use gather along the K dim.
        # idx shape (F, V), we want (B, T, F, V) where last dim picks K via table.
        idx2 = self.feature_table.t().view(1, 1, F_dim, V).expand(B, T, F_dim, V)  # (B, T, F, V)
        # feat_log_probs: (B, T, F, K). Gather along K: out[b, t, f, v] = feat_log_probs[b, t, f, idx2[b, t, f, v]]
        l_ph_unnorm = torch.gather(feat_log_probs, dim=3, index=idx2).sum(dim=2)  # (B, T, V)

        # Mask blank slot — blank prob comes from nonblank head, not features.
        neg_inf = torch.finfo(l_ph_unnorm.dtype).min
        l_ph_unnorm = l_ph_unnorm.clone()
        l_ph_unnorm[..., self.blank_id] = neg_inf

        # Invalid-mass: probability mass that the unnormalized feature distribution
        # placed outside the vocab. Stable form using logsumexp.
        log_valid_mass = torch.logsumexp(l_ph_unnorm, dim=-1)             # (B, T)
        invalid_mass = (1.0 - log_valid_mass.exp()).clamp(min=0.0)        # (B, T)

        # Normalize so log-probs are a proper distribution over V (sum to 1
        # excluding the masked blank slot).
        l_ph_norm = l_ph_unnorm - log_valid_mass.unsqueeze(-1)            # (B, T, V)

        return l_ph_norm, invalid_mass

    def forward(self, input_values, attention_mask=None, labels=None, label_lengths=None):
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)

        l_nb = self.nonblank_head(hidden).squeeze(-1)              # (B, T)

        invalid_mass = None
        if self.use_features:
            l_ph, invalid_mass = self._phoneme_log_probs_from_features(hidden)
        else:
            l_ph = self.phoneme_head(hidden)
            neg_inf = torch.finfo(l_ph.dtype).min
            l_ph = l_ph.clone()
            l_ph[..., self.blank_id] = neg_inf
            l_ph = F.log_softmax(l_ph, dim=-1)

        log_p_blank = F.logsigmoid(-l_nb).unsqueeze(-1)            # (B, T, 1)
        log_p_nonblank = F.logsigmoid(l_nb).unsqueeze(-1)          # (B, T, 1)
        log_p_phonemes = log_p_nonblank + l_ph                     # (B, T, V), blank=-inf

        log_probs = log_p_phonemes.clone()
        log_probs[..., self.blank_id] = log_p_blank.squeeze(-1)

        stress_logits = self.stress_head(hidden)

        result = {
            "log_probs": log_probs,
            "nonblank_logit": l_nb,
            "stress_logits": stress_logits,
        }
        if invalid_mass is not None:
            result["invalid_mass"] = invalid_mass  # (B, T)

        if labels is not None:
            if attention_mask is None:
                input_lengths = torch.full(
                    (input_values.shape[0],), log_probs.shape[1],
                    dtype=torch.long, device=log_probs.device,
                )
            else:
                input_lengths = self.backbone._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).to(torch.long)

            log_probs_tbv = log_probs.transpose(0, 1).float()
            if label_lengths is None:
                label_lengths = (labels != -100).sum(-1).to(torch.long)
            flat_labels = labels.masked_select(labels != -100).to(torch.long)

            loss = F.ctc_loss(
                log_probs_tbv,
                flat_labels,
                input_lengths,
                label_lengths,
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
            result["loss"] = loss

        return result

    def save_to_dir(self, save_dir: Path):
        """Save backbone via HF + factorized heads as a separate state dict."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        payload = {
            "nonblank_head": self.nonblank_head.state_dict(),
            "stress_head": self.stress_head.state_dict(),
            "vocab_size": self.vocab_size,
            "blank_id": self.blank_id,
            "num_stress_labels": self.num_stress_labels,
            "uses_features": self.use_features,
        }
        if self.use_features:
            payload["feature_head"] = self.feature_head.state_dict()
            payload["feature_table"] = self.feature_table.cpu()
            payload["num_features"] = self.num_features
        else:
            payload["phoneme_head"] = self.phoneme_head.state_dict()
        torch.save(payload, save_dir / "factorized_heads.pt")

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        heads = torch.load(load_dir / "factorized_heads.pt", map_location="cpu", weights_only=False)
        ft = heads.get("feature_table") if heads.get("uses_features") else None
        model = cls(
            model_name=str(load_dir),
            vocab_size=heads["vocab_size"],
            blank_id=heads["blank_id"],
            num_stress_labels=heads.get("num_stress_labels", 3),
            feature_table=ft,
        )
        model.nonblank_head.load_state_dict(heads["nonblank_head"])
        model.stress_head.load_state_dict(heads["stress_head"])
        if heads.get("uses_features"):
            model.feature_head.load_state_dict(heads["feature_head"])
        else:
            model.phoneme_head.load_state_dict(heads["phoneme_head"])
        return model

    def head_parameters(self):
        yield from self.nonblank_head.parameters()
        yield from self.stress_head.parameters()
        if self.use_features:
            yield from self.feature_head.parameters()
        else:
            yield from self.phoneme_head.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()
