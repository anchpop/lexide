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
    prosody = np.stack((pitch, voiced.astype(np.float32), log_energy(wave, c)), axis=-1)
    return {"wave": torch.from_numpy(wave), "mel": torch.from_numpy(mel.astype(np.float32)),
            "prosody": torch.from_numpy(prosody.astype(np.float32)),
            # A second tracker: where the two disagree, neither is trusted (voca.data).
            "pitch_check": torch.from_numpy(praat_pitch(wave, c).astype(np.float32))}


def log_energy(wave, config):
    """log RMS per frame [T] (``len(wave) // hop`` frames); training normalizes it
    to ``(log RMS - energy_mean) / energy_scale``."""
    import librosa
    c = config
    rms = librosa.feature.rms(y=wave, frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant")[0, :len(wave) // c.hop]
    return np.log(np.maximum(rms, 1e-5))


def praat_pitch(wave, config):
    """Praat's autocorrelation pitch on the frame grid: [T, 2] log(F0 / 200 Hz)
    (0 when unvoiced) and voiced. Fast, and independent of pYIN's HMM."""
    import parselmouth
    c, frames = config, len(wave) // config.hop
    track = parselmouth.Sound(wave.astype(np.float64), c.sample_rate).to_pitch_ac(
        time_step=c.hop / c.sample_rate, pitch_floor=c.f0_min, pitch_ceiling=c.f0_max)
    index = np.rint((np.arange(frames) * c.hop / c.sample_rate - track.x1) / track.dx).astype(int)
    hz = np.where((index >= 0) & (index < track.n_frames), track.selected_array["frequency"][np.clip(index, 0, track.n_frames - 1)], 0)
    return np.stack((np.where(hz > 0, np.log(np.maximum(hz, 1) / 200.0), 0), hz > 0), axis=-1)


def waveform_prosody(wave, config):
    """Pitch, voicing and energy measured on any waveform (a recording or a
    synthesis), in training units: [T, 3] log(F0 / 200 Hz), voiced, normalized energy."""
    c = config
    wave = np.pad(wave, (0, -len(wave) % c.hop)).astype(np.float32)
    return np.concatenate((praat_pitch(wave, c), ((log_energy(wave, c) - c.energy_mean) / c.energy_scale)[:, None]), axis=-1)
