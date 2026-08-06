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
from transformers import AutoModel

# torchaudio is imported lazily inside the regularized_heads branch so that
# inference environments that don't use that feature can skip the dep.


NUM_FEATURE_VALUES = 3  # panphon ternary {-1, 0, +1} → encoded as {0, 1, 2}
NUM_LAYER_MIXTURES = 5  # learned soft selections over encoder hidden states (regularized-heads mode)
CTC_MASK_VALUE = -1e4  # finite "impossible" log-prob; avoids NaNs in CTC backward

# These heads are factors of the joint CTC symbol, not independent CTC tasks.
# `target` names the aligned tensor supplied by the dataset.  The extra class
# for tone/accent means "this phone is not a bearer"; examples in other
# languages do not use the head at all (and therefore produce no gradient).
DEFAULT_LANGUAGE_HEAD_SPECS = {
    "tha_tone": {
        "lang": "tha", "target": "tone", "num_labels": 6, "weight": 0.3,
    },
    "zho_hans_tone": {
        "lang": "zho-hans", "target": "tone", "num_labels": 6, "weight": 0.3,
    },
    "jpn_pitch_accent": {
        "lang": "jpn", "target": "pitch_accent", "num_labels": 3, "weight": 0.3,
    },
}


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
        regularized_heads: bool = False,
        head_base_dim: int = 768,
        acoustic_dim: int = 64,
        n_mels: int = 80,
        feature_emission_weight: float = 0.0,
        mel_sidechannel: bool = True,
        mlp_heads: bool = True,
        language_head_specs: dict[str, dict] | None = None,
    ):
        super().__init__()
        if feature_table is not None and aux_feature_table is not None:
            raise ValueError(
                "feature_table (factorized) and aux_feature_table (auxiliary) are mutually exclusive. "
                "Pick one feature mode."
            )
        if regularized_heads and mel_sidechannel:
            raise ValueError(
                "regularized_heads and mel_sidechannel are alternative head-base modes "
                "(both build the mel side-channel) and are mutually exclusive."
            )

        # Backbone swap: Cohere Transcribe's Conformer encoder (via CohereBackbone,
        # which presents a wav2vec2-shaped interface at 50 fps) when the model name
        # points at it OR a saved cohere checkpoint is present; else AutoModel.
        # AutoModel resolves the right encoder class from the config — Wav2Vec2Model
        # for wav2vec2 (xls-r), HubertModel for hubert/distilhubert — so a tiny
        # DistilHuBERT student drops in unchanged. All share the 320-stride conv
        # feature extractor (50 fps), the .config.{hidden_size,final_dropout} fields,
        # _get_feat_extract_output_lengths, and last_hidden_state the heads consume.
        # Works for HF repo ids AND saved checkpoint dirs (config.json carries
        # model_type), so load_from_dir round-trips regardless of backbone family.
        import os as _os
        _is_cohere = ("cohere" in str(model_name).lower()
                      or _os.path.exists(_os.path.join(str(model_name), "cohere_backbone.pt")))
        if _is_cohere:
            try:
                from .cohere_backbone import CohereBackbone
            except ImportError:
                from cohere_backbone import CohereBackbone
            self.backbone = CohereBackbone.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_stress_labels = num_stress_labels
        self.regularized_heads = regularized_heads
        self.head_base_dim = head_base_dim
        self.acoustic_dim = acoustic_dim
        self.n_mels = n_mels
        self.feature_emission_weight = feature_emission_weight
        self.mel_sidechannel = mel_sidechannel
        self.mlp_heads = mlp_heads
        self.language_head_specs = {
            name: dict(spec)
            for name, spec in (
                DEFAULT_LANGUAGE_HEAD_SPECS
                if language_head_specs is None else language_head_specs
            ).items()
        }

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(self.backbone.config.final_dropout)

        # When regularized_heads is on, all four heads project from a shared
        # learned base instead of directly from the encoder hidden state. The
        # base input itself is enriched in two ways:
        #
        #   1. Layer-weighted hidden state: instead of just the final encoder
        #      layer (which has already abstracted away low-level acoustic
        #      detail), take a learned softmax-weighted sum over ALL encoder
        #      hidden states. The articulatory feature head benefits most —
        #      nasality, voicing, place are physically lower-level than the
        #      phoneme identity the final layer is optimized for.
        #
        #   2. Acoustic side-channel: a log-mel projection computed directly
        #      from the input waveform, concatenated with the layer-weighted
        #      hidden. Gives heads direct access to raw signal in case the
        #      encoder discarded something the heads could use.
        #
        # All four heads (nonblank, phoneme, feature, stress) consume the same
        # 768-dim base. The encoder is already the de facto shared base; this
        # just adds a learned shared projection between encoder and heads,
        # which empirically helps multi-task setups regularize.
        if regularized_heads:
            # 49 hidden states for XLS-R 2B (embedding + 48 transformer layers).
            num_hidden_states = self.backbone.config.num_hidden_layers + 1
            # K=5 independent learned softmax mixtures, each over the 49 layers.
            # Each row of layer_weights is one mixture's logits; row k's softmax
            # produces a (B,T,1920) weighted sum over the encoder's hidden
            # states. The K mixtures are concatenated to (B,T,K*1920), so the
            # downstream projection can route per-dim per-mixture nonlinearly
            # — strictly more expressive than a single shared mixture.
            #
            # Diverse init: each mixture peaked at a different layer evenly
            # spaced across depth (with softmax temp such that the peak gets
            # ~75% mass at init, leaving room to broaden or shift). If all 5
            # collapse to the same distribution during training, that's an
            # interesting result — but the diverse init prevents collapse from
            # being the default behavior.
            init_logits = torch.zeros(NUM_LAYER_MIXTURES, num_hidden_states)
            peak_layers = torch.linspace(0, num_hidden_states - 1, NUM_LAYER_MIXTURES).round().long()
            for k in range(NUM_LAYER_MIXTURES):
                init_logits[k, peak_layers[k]] = 5.0  # softmax(5.0) ≈ 75% on peak
            self.layer_weights = nn.Parameter(init_logits)
            # MelSpectrogram with hop=320 gives the same 50fps frame rate as
            # wav2vec2's conv feature extractor (16000 / 320 = 50). Any
            # remaining ±1-frame mismatch from edge effects is reconciled in
            # forward via truncate-or-pad.
            import torchaudio  # lazy: only train environments need this
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=400,
                win_length=400,
                hop_length=320,
                n_mels=n_mels,
                center=False,
            )
            # Normalize log-mel before projecting: utterance-level loudness
            # and channel/recording variation introduce huge offsets that
            # would otherwise dominate the projection's early gradient.
            # LayerNorm over the n_mels axis is the standard choice (per-frame,
            # per-batch) and avoids needing pre-computed dataset statistics.
            self.mel_norm = nn.LayerNorm(n_mels)
            self.mel_proj = nn.Linear(n_mels, acoustic_dim)
            # Input dim: K mixtures concat'd along feature axis + mel projection.
            self.shared_base = nn.Sequential(
                nn.Linear(NUM_LAYER_MIXTURES * hidden_size + acoustic_dim, head_base_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            head_input_dim = head_base_dim
        elif mel_sidechannel:
            # Lightweight variant of the regularized base: keep the simple
            # final-layer hidden (the layer-mixture experiment showed the model
            # picks L48 + raw L0 anyway), but concat a raw log-mel side-channel
            # so the heads get the low-level acoustic the final layer abstracts
            # away — the signal the learned mixtures kept ~5-45% mass on (L0).
            self.layer_weights = None
            self.shared_base = None
            import torchaudio  # lazy: only train environments need this
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, win_length=400,
                hop_length=320, n_mels=n_mels, center=False,
            )
            self.mel_norm = nn.LayerNorm(n_mels)
            self.mel_proj = nn.Linear(n_mels, acoustic_dim)
            head_input_dim = hidden_size + acoustic_dim
        else:
            self.layer_weights = None
            self.mel_spec = None
            self.mel_norm = None
            self.mel_proj = None
            self.shared_base = None
            head_input_dim = hidden_size

        def _mk_head(dout):
            # "Bigger head" option: nonlinear MLP readout (din→din→dout) instead
            # of a single linear projection. Cheap (a few M params) on top of the
            # 2B encoder; tests whether a thicker readout helps.
            if mlp_heads:
                return nn.Sequential(
                    nn.Linear(head_input_dim, head_input_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(head_input_dim, dout),
                )
            return nn.Linear(head_input_dim, dout)

        def _zero_head_bias(head):
            # Final Linear's bias — works for both plain Linear and the MLP Sequential.
            (head[-1] if isinstance(head, nn.Sequential) else head).bias.data.zero_()

        self.nonblank_head = _mk_head(1)
        if regularized_heads:
            # Shared base already provides the bottleneck; one Linear suffices.
            self.stress_head = nn.Linear(head_input_dim, num_stress_labels)
        else:
            self.stress_head = nn.Sequential(
                nn.Linear(head_input_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_stress_labels),
            )
        self.language_heads = nn.ModuleDict({
            name: _mk_head(int(spec["num_labels"]))
            for name, spec in self.language_head_specs.items()
        })
        for head in self.language_heads.values():
            _zero_head_bias(head)

        if feature_table is not None or aux_feature_table is not None:
            table = feature_table if feature_table is not None else aux_feature_table
            assert table.shape[0] == vocab_size, (
                f"feature_table rows ({table.shape[0]}) must match vocab_size ({vocab_size})"
            )
            # Precompute a "first occurrence" mask per signature. The off-manifold
            # regularizer sums probability mass *per unique signature* — otherwise
            # tokens that share a feature signature get double-counted, valid_mass
            # can exceed 1, and -log(valid_mass) goes negative (rewards collisions).
            #
            # Skip special tokens (blank + sentinels) when picking representatives.
            # They share an all-unknown signature with each other; if one of them
            # got picked as the representative for that signature, any real
            # phoneme panphon couldn't cover (also all-unknown) would be excluded
            # from the unique-mass sum.
            masked = set((special_token_ids or []) + [blank_id])
            seen = {}
            first_occurrence = torch.zeros(vocab_size, dtype=torch.bool)
            for v in range(vocab_size):
                if v in masked:
                    continue
                sig = tuple(table[v].tolist())
                if sig not in seen:
                    seen[sig] = v
                    first_occurrence[v] = True
            self.register_buffer("_first_occurrence_mask", first_occurrence)

        if feature_table is not None:
            # Mode: factorized. No direct phoneme head; phoneme log-probs derived
            # from feature heads via gather+sum against the lookup table.
            self.num_features = int(feature_table.shape[1])
            self.feature_head = nn.Linear(head_input_dim, self.num_features * NUM_FEATURE_VALUES)
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
            # Mode: auxiliary factors. Direct phone identity and articulatory
            # feature scores are combined inside the main CTC emission model.
            assert aux_feature_table.shape[0] == vocab_size
            self.num_features = int(aux_feature_table.shape[1])
            self.phoneme_head = nn.Linear(head_input_dim, vocab_size)
            # Blank/nonblank remains shared through nonblank_head.
            self.feature_head = nn.Linear(head_input_dim, self.num_features * NUM_FEATURE_VALUES)
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
            self.phoneme_head = _mk_head(vocab_size)
            self.feature_head = None
            self.feature_table = None
            self._masked_slots = None
            _zero_head_bias(self.phoneme_head)

        _zero_head_bias(self.nonblank_head)

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

    def _feature_scores_for_vocab(
        self,
        feature_log_probs: torch.Tensor,
        *,
        reduce: str,
    ) -> torch.Tensor:
        """Score every vocab token by the feature heads' log-probs.

        Returns (B, T, V). In aux mode this is one factor in the same CTC
        emission score as the direct phoneme head:

            score(v) = log P_phoneme(v) + w * feature_score(v)

        The mean reduction keeps the feature factor on roughly one-head scale
        rather than growing with the number of articulatory dimensions.
        """
        B, T, F_dim, K = feature_log_probs.shape
        V = self.vocab_size
        idx = self.feature_table.t().view(1, 1, F_dim, V).expand(B, T, F_dim, V)
        scores = torch.gather(feature_log_probs, dim=3, index=idx)  # (B, T, F, V)
        if reduce == "sum":
            return scores.sum(dim=2)
        if reduce == "mean":
            return scores.mean(dim=2)
        raise ValueError(f"Unsupported feature score reduction: {reduce}")

    def _compute_head_input(self, input_values, attention_mask):
        """Return (B, T, head_input_dim) tensor feeding all heads.

        Flat mode: just the final encoder hidden state. Regularized mode:
        softmax-weighted sum over all encoder layers, concatenated with a
        log-mel acoustic side-channel, passed through a shared MLP.
        """
        if self.regularized_heads:
            out = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # K independent softmax mixtures over the 49 hidden states.
            # layer_weights: (K, L). Softmax along L, then for each mixture k
            # compute its weighted sum (B, T, H) via streaming accumulation
            # (avoids materializing a (L, B, T, H) intermediate).
            weights = F.softmax(self.layer_weights, dim=1)        # (K, L)
            mixtures = []
            for k in range(NUM_LAYER_MIXTURES):
                w_k = weights[k]
                mix_k = sum(w * h for w, h in zip(w_k, out.hidden_states))  # (B, T, H)
                mixtures.append(mix_k)
            hidden = self.dropout(torch.cat(mixtures, dim=-1))    # (B, T, K*H)

            # Log-mel side-channel computed from the same input waveform.
            # MelSpectrogram is BF16-incompatible inside autocast, so cast to
            # float32 around it then back to hidden's dtype.
            with torch.autocast(device_type=input_values.device.type, enabled=False):
                mel = self.mel_spec(input_values.float())             # (B, n_mels, T_mel)
            mel = mel.transpose(1, 2)                                 # (B, T_mel, n_mels)
            # Align mel's time axis to wav2vec2's. They share hop=320 so
            # they match to within ±1 frame from edge handling — truncate
            # or pad the difference.
            T_target = hidden.shape[1]
            if mel.shape[1] > T_target:
                mel = mel[:, :T_target, :]
            elif mel.shape[1] < T_target:
                mel = F.pad(mel, (0, 0, 0, T_target - mel.shape[1]))
            mel = torch.log(mel + 1e-6).to(hidden.dtype)
            mel = self.mel_norm(mel)                                  # per-frame LayerNorm
            mel_proj = self.mel_proj(mel)                             # (B, T, acoustic_dim)

            combined = torch.cat([hidden, mel_proj], dim=-1)
            return self.shared_base(combined)

        if self.mel_sidechannel:
            # Final-layer hidden + raw log-mel side-channel (no layer mixture,
            # no shared-base bottleneck — heads consume [hidden ; mel_proj]).
            out = self.backbone(input_values=input_values, attention_mask=attention_mask)
            hidden = self.dropout(out.last_hidden_state)
            with torch.autocast(device_type=input_values.device.type, enabled=False):
                mel = self.mel_spec(input_values.float())             # (B, n_mels, T_mel)
            mel = mel.transpose(1, 2)                                 # (B, T_mel, n_mels)
            T_target = hidden.shape[1]
            if mel.shape[1] > T_target:
                mel = mel[:, :T_target, :]
            elif mel.shape[1] < T_target:
                mel = F.pad(mel, (0, 0, 0, T_target - mel.shape[1]))
            mel = torch.log(mel + 1e-6).to(hidden.dtype)
            mel = self.mel_norm(mel)                                  # per-frame LayerNorm
            mel_proj = self.mel_proj(mel)                            # (B, T, acoustic_dim)
            return torch.cat([hidden, mel_proj], dim=-1)

        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state)

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        label_lengths=None,
        active_language_heads: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        hidden = self._compute_head_input(input_values, attention_mask)

        l_nb = self.nonblank_head(hidden).squeeze(-1)              # (B, T)

        off_manifold = None
        feature_log_probs = None
        if self.use_features:
            # Factorized: derive phoneme log-probs from features. Includes the
            # off_manifold regularizer for the V→U signature dedup.
            l_ph, off_manifold = self._phoneme_log_probs_from_features(hidden)
        else:
            l_ph = self.phoneme_head(hidden)
            mask_value = l_ph.new_tensor(CTC_MASK_VALUE)
            l_ph = l_ph.clone()
            # Mask blank + special tokens (<s>, </s>, <unk>) from the direct
            # phoneme distribution. In aux mode the feature table treats these
            # as non-phonemes, and we don't want the direct head emitting them
            # either — main CTC's blank lives on the separate nonblank_head.
            if self._masked_slots is not None:
                l_ph[..., self._masked_slots] = mask_value
            else:
                l_ph[..., self.blank_id] = mask_value
            l_ph = F.log_softmax(l_ph, dim=-1)

            if self.use_aux_features:
                B, T = hidden.shape[0], hidden.shape[1]
                feat = self.feature_head(hidden).view(B, T, self.num_features, NUM_FEATURE_VALUES)
                feature_log_probs = F.log_softmax(feat, dim=-1)  # (B, T, F, 3)

                if self.feature_emission_weight > 0:
                    # Treat phoneme identity as one CTC emission factor and
                    # articulatory features as additional factors for that same
                    # emitted phoneme. Phoneme factor weight is 1.0; feature
                    # factor gets the configured weight, then the combined
                    # distribution is renormalized over phoneme slots.
                    feature_scores = self._feature_scores_for_vocab(
                        feature_log_probs,
                        reduce="mean",
                    )
                    feature_scores = feature_scores.clone()
                    feature_scores[..., self._masked_slots] = feature_scores.new_tensor(CTC_MASK_VALUE)
                    l_ph = l_ph + self.feature_emission_weight * feature_scores
                    l_ph = l_ph - torch.logsumexp(l_ph, dim=-1, keepdim=True)
                    l_ph = l_ph.clone()
                    l_ph[..., self._masked_slots] = l_ph.new_tensor(CTC_MASK_VALUE)

        log_p_blank = F.logsigmoid(-l_nb).unsqueeze(-1)            # (B, T, 1)
        log_p_nonblank = F.logsigmoid(l_nb).unsqueeze(-1)          # (B, T, 1)
        log_p_phonemes = log_p_nonblank + l_ph                     # (B, T, V), blank=-inf

        log_probs = log_p_phonemes.clone()
        log_probs[..., self.blank_id] = log_p_blank.squeeze(-1)

        stress_logits = self.stress_head(hidden)
        # Phone-only inference remains the cheap/backward-compatible default.
        # Training and prosody-aware inference opt in to the applicable heads.
        active = set() if active_language_heads is None else set(active_language_heads)
        unknown_heads = active - set(self.language_heads)
        if unknown_heads:
            raise ValueError(f"unknown language heads: {sorted(unknown_heads)}")
        language_head_logits = {
            name: self.language_heads[name](hidden) for name in sorted(active)
        }

        result = {
            "log_probs": log_probs,
            "nonblank_logit": l_nb,
            "stress_logits": stress_logits,
            "language_head_logits": language_head_logits,
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
            # The trainer's prosody-aware joint CTC composes additional target
            # factors around this normalized phone distribution.

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
            "feature_emission_weight": self.feature_emission_weight,
            # Keep `uses_features` for backward compatibility with older ckpts.
            "uses_features": (mode == "factorized"),
            "regularized_heads": self.regularized_heads,
            "mel_sidechannel": self.mel_sidechannel,
            "mlp_heads": self.mlp_heads,
            "language_head_specs": self.language_head_specs,
            "language_heads": {
                name: head.state_dict() for name, head in self.language_heads.items()
            },
        }
        if self.regularized_heads:
            payload["head_base_dim"] = self.head_base_dim
            payload["acoustic_dim"] = self.acoustic_dim
            payload["n_mels"] = self.n_mels
            payload["layer_weights"] = self.layer_weights.detach().cpu()
            payload["mel_norm"] = self.mel_norm.state_dict()
            payload["mel_proj"] = self.mel_proj.state_dict()
            payload["shared_base"] = self.shared_base.state_dict()
        if self.mel_sidechannel:
            payload["acoustic_dim"] = self.acoustic_dim
            payload["n_mels"] = self.n_mels
            payload["mel_norm"] = self.mel_norm.state_dict()
            payload["mel_proj"] = self.mel_proj.state_dict()
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
        regularized = heads.get("regularized_heads", False)
        mel_sidechannel = heads.get("mel_sidechannel", False)
        mlp_heads = heads.get("mlp_heads", False)
        reg_kwargs = {}
        if regularized:
            reg_kwargs["head_base_dim"] = heads["head_base_dim"]
            reg_kwargs["acoustic_dim"] = heads["acoustic_dim"]
            reg_kwargs["n_mels"] = heads["n_mels"]
        if mel_sidechannel:
            reg_kwargs["acoustic_dim"] = heads["acoustic_dim"]
            reg_kwargs["n_mels"] = heads["n_mels"]
        model = cls(
            model_name=str(load_dir),
            vocab_size=heads["vocab_size"],
            blank_id=heads["blank_id"],
            num_stress_labels=heads.get("num_stress_labels", 3),
            feature_table=factorized_table,
            aux_feature_table=aux_table,
            special_token_ids=special_ids,
            regularized_heads=regularized,
            feature_emission_weight=heads.get("feature_emission_weight", 0.0),
            mel_sidechannel=mel_sidechannel,
            mlp_heads=mlp_heads,
            language_head_specs=heads.get("language_head_specs"),
            **reg_kwargs,
        )
        if regularized:
            model.layer_weights.data.copy_(heads["layer_weights"])
            model.mel_norm.load_state_dict(heads["mel_norm"])
            model.mel_proj.load_state_dict(heads["mel_proj"])
            model.shared_base.load_state_dict(heads["shared_base"])
        if mel_sidechannel:
            model.mel_norm.load_state_dict(heads["mel_norm"])
            model.mel_proj.load_state_dict(heads["mel_proj"])
        model.nonblank_head.load_state_dict(heads["nonblank_head"])
        model.stress_head.load_state_dict(heads["stress_head"])
        for name, state in heads.get("language_heads", {}).items():
            if name in model.language_heads:
                model.language_heads[name].load_state_dict(state)
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
        yield from self.language_heads.parameters()
        # Yield whichever heads actually exist. In aux mode BOTH exist; the
        # optimizer must step the feature head too (otherwise aux CTC's
        # gradients accumulate but never apply, and the encoder is shaped by
        # a fixed random feature head).
        if self.phoneme_head is not None:
            yield from self.phoneme_head.parameters()
        if self.feature_head is not None:
            yield from self.feature_head.parameters()
        # Regularized-heads sub-modules step at head LR — they sit between
        # encoder and heads, train from scratch, and the layer weights in
        # particular should adapt fast.
        if self.regularized_heads:
            yield self.layer_weights
            yield from self.mel_norm.parameters()
            yield from self.mel_proj.parameters()
            yield from self.shared_base.parameters()
        if self.mel_sidechannel:
            # The side-channel mel projection trains from scratch at head LR too.
            yield from self.mel_norm.parameters()
            yield from self.mel_proj.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()
