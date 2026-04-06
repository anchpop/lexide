"""Conformer + CTC model for multilingual IPA pronunciation grading."""

import torch
import torch.nn as nn
import torchaudio

from .audio import SpecAugment


class ConformerCTCModel(nn.Module):
    """Conformer encoder with Conv2d subsampling and CTC head.

    Default config (~110M params):
        model_dim=512, num_layers=24, ffn_dim=2048, num_heads=8
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        model_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 24,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Conv2d subsampling: reduces time dimension by 4x
        self.subsample = nn.Sequential(
            nn.Conv2d(1, model_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After subsampling: (B, model_dim, T/4, input_dim/4)
        # Flatten channel+freq dims and project to model_dim
        self.subsample_proj = nn.Linear(model_dim * (input_dim // 4), model_dim)
        self.subsample_norm = nn.LayerNorm(model_dim)
        self.subsample_dropout = nn.Dropout(dropout)

        # Conformer encoder
        self.conformer = torchaudio.models.Conformer(
            input_dim=model_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        # CTC projection
        self.ctc_head = nn.Linear(model_dim, vocab_size)

        # SpecAugment (applied during training only)
        self.spec_augment = SpecAugment()

    def _subsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after two stride-2 conv layers."""
        # Each conv with kernel=3, stride=2, padding=1: L_out = floor((L_in + 1) / 2)
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths

    def forward(
        self, mels: torch.Tensor, mel_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mels: (B, T, n_mels)
            mel_lengths: (B,)
        Returns:
            log_probs: (T', B, vocab_size) — CTC time-first convention
            output_lengths: (B,)
        """
        # SpecAugment during training
        if self.training:
            # SpecAugment expects (B, n_mels, T)
            x = mels.transpose(1, 2)
            x = self.spec_augment(x)
            x = x.transpose(1, 2)
        else:
            x = mels

        # Subsampling: (B, T, n_mels) -> (B, 1, T, n_mels) -> conv
        x = x.unsqueeze(1)
        x = self.subsample(x)  # (B, model_dim, T/4, n_mels/4)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T/4, model_dim * n_mels/4)
        x = self.subsample_proj(x)  # (B, T/4, model_dim)
        x = self.subsample_norm(x)
        x = self.subsample_dropout(x)

        output_lengths = self._subsample_lengths(mel_lengths)

        # Conformer
        x, output_lengths = self.conformer(x, output_lengths)

        # CTC head
        logits = self.ctc_head(x)  # (B, T', vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T', B, vocab_size)

        return log_probs, output_lengths
