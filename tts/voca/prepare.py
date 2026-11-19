"""Acoustic features of one recording: waveform, log-mel, pYIN pitch/voicing, energy."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .config import Config

MIN_FRAMES = 32


def records(source):
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        yield {"_input_error":str(error)}
    elif source.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            yield from batch.to_pylist()
    else:
        raise ValueError("input must be .jsonl or .parquet")


def extract(wave, config):
    import librosa
    c = config
    if wave.ndim != 1 or not np.isfinite(wave).all() or not wave.size:
        raise ValueError("expected finite mono audio")
    frames = math.ceil(len(wave)/c.hop)
    if frames < 1:
        raise ValueError("empty audio")
    if np.max(np.abs(wave)) > 1.01 or np.sqrt(np.mean(wave**2)) < 1e-5:
        raise ValueError("audio is clipped/out of range or effectively silent")
    wave = np.pad(wave, (0, frames*c.hop-len(wave))).astype(np.float32)
    spectrum = librosa.stft(wave, n_fft=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[:, :frames]
    filters = librosa.filters.mel(sr=c.sample_rate, n_fft=c.n_fft, n_mels=c.n_mels)
    mel = np.log(np.maximum(filters @ np.abs(spectrum)**2, 1e-5)).T
    f0, voiced, _ = librosa.pyin(wave, fmin=c.f0_min, fmax=c.f0_max, sr=c.sample_rate,
                                frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")
    f0, voiced = f0[:frames], voiced[:frames]
    pitch = np.where(voiced, np.log(np.maximum(np.nan_to_num(f0, nan=200.0), 1)/200.0), 0)
    rms = librosa.feature.rms(y=wave, frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[0, :frames]
    prosody = np.stack((pitch, voiced.astype(np.float32), np.log(np.maximum(rms, 1e-5))), axis=-1)
    return {"wave": torch.from_numpy(wave), "mel": torch.from_numpy(mel.astype(np.float32)),
            "prosody": torch.from_numpy(prosody.astype(np.float32))}
