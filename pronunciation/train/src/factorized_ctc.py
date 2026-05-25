"""Factorized-CTC wav2vec2: splits the (V+1)-way classifier into two heads.

Standard CTC has one softmax over V phonemes + blank. When label noise spreads
the model's posterior across several "wrong-but-close" labels, no individual
phoneme accumulates enough mass to beat blank — even when the model is
collectively confident *some* phoneme is present. Blank wins by plurality and
the phoneme drops from the output.

Factorization decouples the two decisions:

    P(blank   | t)         = σ(-l_nb_t)
    P(phoneme | t, ¬blank) = softmax(l_ph_t)
    P(phoneme | t)         = σ(l_nb_t) * softmax(l_ph_t)

The model can be confidently non-blank (Head 1 sees vowel-like energy) while
Head 2's mass is spread thin — and still emit a (possibly wrong) phoneme
instead of dropping it. Substitution > drop for SLA transcription.

Combined log-probs are passed to standard F.ctc_loss; no custom CTC machinery
needed.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class FactorizedCTCModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-2b",
        vocab_size: int = 392,
        blank_id: int = 0,
        num_stress_labels: int = 3,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_stress_labels = num_stress_labels

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(self.backbone.config.final_dropout)
        self.nonblank_head = nn.Linear(hidden_size, 1)
        self.phoneme_head = nn.Linear(hidden_size, vocab_size)
        # Same MLP shape as the prior stress-only model — known to work.
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_stress_labels),
        )

        nn.init.zeros_(self.nonblank_head.bias)
        nn.init.zeros_(self.phoneme_head.bias)

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_values, attention_mask=None, labels=None, label_lengths=None):
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)  # (B, T, H)

        l_nb = self.nonblank_head(hidden).squeeze(-1)         # (B, T)
        l_ph = self.phoneme_head(hidden)                       # (B, T, V)

        # Mask the blank slot in the phoneme distribution so it stays purely
        # non-blank (sigmoid handles blank vs not-blank entirely).
        neg_inf = torch.finfo(l_ph.dtype).min
        l_ph = l_ph.clone()
        l_ph[..., self.blank_id] = neg_inf

        log_p_blank = F.logsigmoid(-l_nb).unsqueeze(-1)        # (B, T, 1)
        log_p_nonblank = F.logsigmoid(l_nb).unsqueeze(-1)      # (B, T, 1)
        log_p_phonemes = log_p_nonblank + F.log_softmax(l_ph, dim=-1)  # (B, T, V); blank slot = -inf

        log_probs = log_p_phonemes.clone()
        log_probs[..., self.blank_id] = log_p_blank.squeeze(-1)  # (B, T, V)

        stress_logits = self.stress_head(hidden)  # (B, T, num_stress_labels)

        result = {
            "log_probs": log_probs,
            "nonblank_logit": l_nb,
            "stress_logits": stress_logits,
        }

        if labels is not None:
            # Derive input lengths from attention_mask using the backbone's stride.
            if attention_mask is None:
                input_lengths = torch.full(
                    (input_values.shape[0],), log_probs.shape[1],
                    dtype=torch.long, device=log_probs.device,
                )
            else:
                input_lengths = self.backbone._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).to(torch.long)

            # CTC expects log-probs in (T, B, V) and labels as the flat valid sequence.
            log_probs_tbv = log_probs.transpose(0, 1).float()
            if label_lengths is None:
                # Labels padded with -100 (huggingface convention); derive lengths.
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
        torch.save(
            {
                "nonblank_head": self.nonblank_head.state_dict(),
                "phoneme_head": self.phoneme_head.state_dict(),
                "stress_head": self.stress_head.state_dict(),
                "vocab_size": self.vocab_size,
                "blank_id": self.blank_id,
                "num_stress_labels": self.num_stress_labels,
            },
            save_dir / "factorized_heads.pt",
        )

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        heads = torch.load(load_dir / "factorized_heads.pt", map_location="cpu", weights_only=False)
        model = cls(
            model_name=str(load_dir),
            vocab_size=heads["vocab_size"],
            blank_id=heads["blank_id"],
            num_stress_labels=heads.get("num_stress_labels", 3),
        )
        model.nonblank_head.load_state_dict(heads["nonblank_head"])
        model.phoneme_head.load_state_dict(heads["phoneme_head"])
        model.stress_head.load_state_dict(heads["stress_head"])
        return model

    def head_parameters(self):
        yield from self.nonblank_head.parameters()
        yield from self.phoneme_head.parameters()
        yield from self.stress_head.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()
