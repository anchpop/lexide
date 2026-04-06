"""Audio preprocessing: mel spectrogram extraction and SpecAugment."""

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


class AudioProcessor:
    """Extracts normalized log-mel spectrograms from audio files."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __call__(self, audio_path: str) -> torch.Tensor:
        """Load audio and return normalized log-mel spectrogram (T, n_mels)."""
        # Use soundfile instead of torchaudio.load to avoid torchcodec dependency
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        mel = self.mel_transform(waveform)  # (1, n_mels, T)
        mel = torch.clamp(mel, min=1e-10).log()
        mel = mel.squeeze(0).transpose(0, 1)  # (T, n_mels)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking for training augmentation."""

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [torchaudio.transforms.TimeMasking(time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply masks to mel spectrogram. Input: (B, n_mels, T)."""
        for fm in self.freq_masks:
            mel = fm(mel)
        for tm in self.time_masks:
            mel = tm(mel)
        return mel
