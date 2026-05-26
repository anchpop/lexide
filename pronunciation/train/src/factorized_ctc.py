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
        aux_feature_table: torch.Tensor | None = None,
        special_token_ids: list[int] | None = None,
    ):
        super().__init__()
        if feature_table is not None and aux_feature_table is not None:
            raise ValueError(
                "feature_table (factorized) and aux_feature_table (auxiliary) are mutually exclusive. "
                "Pick one feature mode."
            )

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

        if feature_table is not None or aux_feature_table is not None:
            table = feature_table if feature_table is not None else aux_feature_table
            assert table.shape[0] == vocab_size, (
                f"feature_table rows ({table.shape[0]}) must match vocab_size ({vocab_size})"
            )
            # Precompute a "first occurrence" mask per signature. The off-manifold
            # regularizer sums probability mass *per unique signature* — otherwise
            # tokens that share a feature signature get double-counted, valid_mass
            # can exceed 1, and -log(valid_mass) goes negative (rewards collisions).
            seen = {}
            first_occurrence = torch.zeros(vocab_size, dtype=torch.bool)
            for v in range(vocab_size):
                sig = tuple(table[v].tolist())
                if sig not in seen:
                    seen[sig] = v
                    first_occurrence[v] = True
            self.register_buffer("_first_occurrence_mask", first_occurrence)

        if feature_table is not None:
            # Mode: factorized. No direct phoneme head; phoneme log-probs derived
            # from feature heads via gather+sum against the lookup table.
            self.num_features = int(feature_table.shape[1])
            self.feature_head = nn.Linear(hidden_size, self.num_features * NUM_FEATURE_VALUES)
            self.phoneme_head = None
            self.register_buffer("feature_table", feature_table.long())
            # Special token ids (e.g. <s>, </s>, <unk>) must be masked to -inf
            # in the derived phoneme logits — they're not real phonemes and
            # shouldn't be emittable. blank_id is handled separately by
            # nonblank_head; include it here too for consistency.
            mask_ids = sorted(set((special_token_ids or []) + [blank_id]))
            self.register_buffer(
                "_masked_slots",
                torch.tensor(mask_ids, dtype=torch.long),
            )
            nn.init.zeros_(self.feature_head.bias)
        elif aux_feature_table is not None:
            # Mode: auxiliary. Direct phoneme head handles MAIN CTC; feature
            # head runs a custom factorial/vector-label CTC against
            # feature-encoded targets. Encoder gets one shared alignment over
            # whole feature vectors, while comparison stays in feature space.
            assert aux_feature_table.shape[0] == vocab_size
            self.num_features = int(aux_feature_table.shape[1])
            self.phoneme_head = nn.Linear(hidden_size, vocab_size)
            # The feature head predicts only feature values. Blank/nonblank is
            # shared with the main CTC route via nonblank_head.
            self.feature_head = nn.Linear(hidden_size, self.num_features * NUM_FEATURE_VALUES)
            self.register_buffer("feature_table", aux_feature_table.long())
            mask_ids = sorted(set((special_token_ids or []) + [blank_id]))
            self.register_buffer(
                "_masked_slots",
                torch.tensor(mask_ids, dtype=torch.long),
            )
            nn.init.zeros_(self.feature_head.bias)
            nn.init.zeros_(self.phoneme_head.bias)
        else:
            # Mode: off. Direct phoneme head only (original baseline-VAD behavior).
            self.num_features = 0
            self.phoneme_head = nn.Linear(hidden_size, vocab_size)
            self.feature_head = None
            self.feature_table = None
            self._masked_slots = None
            nn.init.zeros_(self.phoneme_head.bias)

        nn.init.zeros_(self.nonblank_head.bias)

    @property
    def use_features(self) -> bool:
        """True if phoneme logits are *derived* from features (factorized mode)."""
        return self.phoneme_head is None and self.feature_head is not None

    @property
    def use_aux_features(self) -> bool:
        """True if features are an *auxiliary* supervision signal (direct head + aux head)."""
        return self.phoneme_head is not None and self.feature_head is not None

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def _phoneme_log_probs_from_features(self, hidden: torch.Tensor):
        """Return (l_ph_norm, neg_log_valid_mass) using feature factorization.

        l_ph_norm: (B, T, V) — normalized log P(phoneme | ¬blank). Special-token
                   and blank slots are masked to -inf.
        neg_log_valid_mass: (B, T) — `-log(sum_v exp(l_ph_unnorm[v]))` over
                   real phoneme slots. Equivalent to the negative log probability
                   that the independent feature distributions assigned to the
                   on-manifold (in-vocab) region. Use as a regularization signal:
                   minimize this to push feature distributions toward producing
                   real phoneme combinations. Stronger gradient signal than
                   `1 - exp(log_valid_mass)` (which saturates near 1 when off-manifold).
        """
        B, T, _ = hidden.shape
        F_dim = self.num_features
        V = self.vocab_size

        # (B, T, F*K) → (B, T, F, K), then log_softmax over K.
        feat = self.feature_head(hidden).view(B, T, F_dim, NUM_FEATURE_VALUES)
        feat_log_probs = F.log_softmax(feat, dim=-1)  # (B, T, F, K)

        # For each phoneme v, sum log-probs of its feature values across F.
        # feature_table: (V, F) — each row is the target K-index for each feature.
        # Gather feat_log_probs[..., i, table[v, i]] along K, then sum over F.
        idx = self.feature_table.t().view(1, 1, F_dim, V).expand(B, T, F_dim, V)
        l_ph_unnorm = torch.gather(feat_log_probs, dim=3, index=idx).sum(dim=2)  # (B, T, V)

        # Mask non-emittable slots (blank + special tokens) to -inf so they
        # don't contribute to the valid-mass sum and can't be greedy-emitted.
        neg_inf = torch.finfo(l_ph_unnorm.dtype).min
        l_ph_unnorm = l_ph_unnorm.clone()
        l_ph_unnorm[..., self._masked_slots] = neg_inf

        # Two different "valid mass" quantities — they look similar but mean
        # different things and are used for different purposes:
        #
        # 1. V-level log_valid_mass: sum over all V token slots (with duplicate
        #    signatures double-counted). This is what's needed to NORMALIZE
        #    the V-slot distribution to a proper probability distribution
        #    (sums to 1 over V), which CTC requires.
        #
        # 2. U-level log_valid_mass: sum over UNIQUE signatures (using
        #    _first_occurrence_mask). This is the actual probability the
        #    feature distributions assigned to the in-vocab manifold. Used
        #    for the off_manifold penalty: without de-dup, duplicate
        #    signatures inflate the apparent "valid mass" and -log goes
        #    negative, rewarding signature collisions.
        log_valid_mass_v = torch.logsumexp(l_ph_unnorm, dim=-1)              # (B, T)
        l_unique = l_ph_unnorm.clone()
        l_unique[..., ~self._first_occurrence_mask] = neg_inf
        log_valid_mass_u = torch.logsumexp(l_unique, dim=-1)                 # (B, T)
        neg_log_valid_mass_u = -log_valid_mass_u                             # ≥ 0

        # Normalize V-slot distribution using V-level mass so probs sum to 1.
        l_ph_norm = l_ph_unnorm - log_valid_mass_v.unsqueeze(-1)

        return l_ph_norm, neg_log_valid_mass_u

    def forward(self, input_values, attention_mask=None, labels=None, label_lengths=None):
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)

        l_nb = self.nonblank_head(hidden).squeeze(-1)              # (B, T)

        off_manifold = None
        if self.use_features:
            # Factorized: derive phoneme log-probs from features. Includes the
            # off_manifold regularizer for the V→U signature dedup.
            l_ph, off_manifold = self._phoneme_log_probs_from_features(hidden)
        else:
            l_ph = self.phoneme_head(hidden)
            neg_inf = torch.finfo(l_ph.dtype).min
            l_ph = l_ph.clone()
            # Mask blank + special tokens (<s>, </s>, <unk>) from the direct
            # phoneme distribution. In aux mode the feature table treats these
            # as non-phonemes, and we don't want the direct head emitting them
            # either — main CTC's blank lives on the separate nonblank_head.
            if self._masked_slots is not None:
                l_ph[..., self._masked_slots] = neg_inf
            else:
                l_ph[..., self.blank_id] = neg_inf
            l_ph = F.log_softmax(l_ph, dim=-1)

        log_p_blank = F.logsigmoid(-l_nb).unsqueeze(-1)            # (B, T, 1)
        log_p_nonblank = F.logsigmoid(l_nb).unsqueeze(-1)          # (B, T, 1)
        log_p_phonemes = log_p_nonblank + l_ph                     # (B, T, V), blank=-inf

        log_probs = log_p_phonemes.clone()
        log_probs[..., self.blank_id] = log_p_blank.squeeze(-1)

        # Feature log-probs (aux mode). The training loop runs a custom
        # factorial/vector-label CTC: for each target phoneme position, it sums
        # the log-probs of that position's feature vector, then applies the CTC
        # forward recursion with one shared blank/nonblank head.
        feature_log_probs = None
        if self.use_aux_features:
            B, T = hidden.shape[0], hidden.shape[1]
            feat = self.feature_head(hidden).view(B, T, self.num_features, NUM_FEATURE_VALUES)
            feature_log_probs = F.log_softmax(feat, dim=-1)                    # (B, T, F, 3)

        stress_logits = self.stress_head(hidden)

        result = {
            "log_probs": log_probs,
            "nonblank_logit": l_nb,
            "stress_logits": stress_logits,
        }
        if feature_log_probs is not None:
            result["feature_log_probs"] = feature_log_probs  # (B, T, F, 3)
        if off_manifold is not None:
            result["off_manifold"] = off_manifold  # (B, T), ≥ 0

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
            # The factorial feature CTC loss (aux mode) is computed by the
            # training loop, not here — it needs phoneme_ids to encode targets
            # as feature sequences via the feature_table.

        return result

    def save_to_dir(self, save_dir: Path):
        """Save backbone via HF + factorized heads as a separate state dict."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        # Determine the feature mode for round-trip.
        if self.use_features:
            mode = "factorized"
        elif self.use_aux_features:
            mode = "aux"
        else:
            mode = "off"
        payload = {
            "nonblank_head": self.nonblank_head.state_dict(),
            "stress_head": self.stress_head.state_dict(),
            "vocab_size": self.vocab_size,
            "blank_id": self.blank_id,
            "num_stress_labels": self.num_stress_labels,
            "feature_mode": mode,
            # Keep `uses_features` for backward compatibility with older ckpts.
            "uses_features": (mode == "factorized"),
        }
        if mode == "factorized":
            payload["feature_head"] = self.feature_head.state_dict()
            payload["feature_table"] = self.feature_table.cpu()
            payload["num_features"] = self.num_features
            payload["masked_slots"] = self._masked_slots.cpu().tolist()
        elif mode == "aux":
            payload["phoneme_head"] = self.phoneme_head.state_dict()
            payload["feature_head"] = self.feature_head.state_dict()
            payload["feature_table"] = self.feature_table.cpu()
            payload["num_features"] = self.num_features
            # The aux phoneme distribution also masks special tokens; without
            # this, a reloaded model would let <s>, </s>, <unk> through the
            # aux logits and silently change loss / off_manifold behavior.
            payload["masked_slots"] = self._masked_slots.cpu().tolist()
        else:
            payload["phoneme_head"] = self.phoneme_head.state_dict()
        torch.save(payload, save_dir / "factorized_heads.pt")

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        heads = torch.load(load_dir / "factorized_heads.pt", map_location="cpu", weights_only=False)
        # Backward-compat: older ckpts only have `uses_features` (bool); newer
        # ones have `feature_mode` ∈ {"off", "factorized", "aux"}.
        mode = heads.get("feature_mode")
        if mode is None:
            mode = "factorized" if heads.get("uses_features", False) else "off"

        factorized_table = heads.get("feature_table") if mode == "factorized" else None
        aux_table = heads.get("feature_table") if mode == "aux" else None
        # masked_slots was saved for both factorized and aux from this code
        # version onward; strip blank since the ctor always re-adds it.
        special_ids = None
        if mode in ("factorized", "aux"):
            ms = heads.get("masked_slots", [])
            special_ids = [i for i in ms if i != heads["blank_id"]]
        model = cls(
            model_name=str(load_dir),
            vocab_size=heads["vocab_size"],
            blank_id=heads["blank_id"],
            num_stress_labels=heads.get("num_stress_labels", 3),
            feature_table=factorized_table,
            aux_feature_table=aux_table,
            special_token_ids=special_ids,
        )
        model.nonblank_head.load_state_dict(heads["nonblank_head"])
        model.stress_head.load_state_dict(heads["stress_head"])
        if mode == "factorized":
            model.feature_head.load_state_dict(heads["feature_head"])
        elif mode == "aux":
            model.phoneme_head.load_state_dict(heads["phoneme_head"])
            model.feature_head.load_state_dict(heads["feature_head"])
        else:
            model.phoneme_head.load_state_dict(heads["phoneme_head"])
        return model

    def head_parameters(self):
        yield from self.nonblank_head.parameters()
        yield from self.stress_head.parameters()
        # Yield whichever heads actually exist. In aux mode BOTH exist; the
        # optimizer must step the feature head too (otherwise aux CTC's
        # gradients accumulate but never apply, and the encoder is shaped by
        # a fixed random feature head).
        if self.phoneme_head is not None:
            yield from self.phoneme_head.parameters()
        if self.feature_head is not None:
            yield from self.feature_head.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()
