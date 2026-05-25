"""Dataset: load audio + pre-computed espeak phonemes with stress labels."""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2
NUM_STRESS_LABELS = 3

VAD_FRAME_SAMPLES = 256  # earshot's native stride (16 ms @ 16 kHz)


class StressDataset(Dataset):
    """Yields (audio, phoneme_token_ids, stress_per_phoneme) triples.

    The phoneme token IDs are mapped to the base model's tokenizer vocabulary
    so that torchaudio's forced_align can align them against the frozen CTC
    logits at training time.
    """

    def __init__(
        self,
        phonemes_path: Path,
        tokenizer,
        max_audio_sec: float = 16.0,
        min_rms: float = 0.005,
        min_duration_sec: float = 0.3,
    ):
        self.tokenizer = tokenizer
        self.max_audio_samples = int(max_audio_sec * 16000)
        min_samples = int(min_duration_sec * 16000)

        audio_dir = phonemes_path.parent
        self.samples = []
        unk_id = tokenizer.unk_token_id

        skipped = {"missing_file": 0, "no_phonemes": 0, "silent": 0, "too_short": 0, "unreadable": 0}

        # Optional: per-clip VAD probabilities at 16ms stride from earshot.
        # See vad_compare/. Used as a soft regularizer on the nonblank head
        # (BCE between sigmoid(nonblank_logit) and vad_prob).
        vad_path = audio_dir / "vad.jsonl"
        vad_by_file: dict[str, list[float]] = {}
        if vad_path.exists():
            with open(vad_path) as f:
                for line in f:
                    rec = json.loads(line)
                    vad_by_file[rec["file"]] = rec["vad_probs"]

        with open(phonemes_path) as f:
            records = [json.loads(line) for line in f]

        for rec in tqdm(records, desc=f"Scanning {phonemes_path.parent.name}", leave=False):
            wav_path = audio_dir / rec["file"]
            if not wav_path.exists():
                skipped["missing_file"] += 1
                continue

            # Audio quality check — cheap (reads file header + samples once).
            # Catches the e12ee45c06f03906.wav case (valid WAV, no actual audio).
            try:
                info = sf.info(wav_path)
                if info.frames < min_samples:
                    skipped["too_short"] += 1
                    continue
                audio_sample, _ = sf.read(wav_path, dtype="float32")
                rms = float(np.sqrt(np.mean(audio_sample.astype(np.float64) ** 2)))
                if rms < min_rms:
                    skipped["silent"] += 1
                    continue
            except Exception:
                skipped["unreadable"] += 1
                continue

            # Map espeak phonemes to base model's vocabulary
            phoneme_ids = []
            stress_seq = []
            for phoneme, stress in zip(rec["phonemes"], rec["stress"]):
                tid = tokenizer.convert_tokens_to_ids(phoneme)
                if tid == unk_id or tid is None:
                    continue  # Skip unknown phonemes — forced align can't place them
                phoneme_ids.append(tid)
                stress_seq.append(stress)

            if not phoneme_ids:
                skipped["no_phonemes"] += 1
                continue

            self.samples.append({
                "wav_path": str(wav_path),
                "phoneme_ids": phoneme_ids,
                "stress_seq": stress_seq,
                "lang": rec["lang"],
                # May be empty list if vad.jsonl wasn't present — training loop
                # treats empty as "no VAD signal, skip VAD loss for this clip".
                "vad_probs": vad_by_file.get(rec["file"], []),
            })

        total_skipped = sum(skipped.values())
        if total_skipped:
            print(f"  Skipped {total_skipped}: {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample["wav_path"])
        assert sr == 16000, f"Expected 16kHz audio, got {sr}"

        if len(audio) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]

        # Truncate VAD to the audio's frame count (in case audio was clipped above).
        # vad_probs is at 16ms stride — len(audio) // 256 frames cover the same span.
        n_vad_frames = len(audio) // VAD_FRAME_SAMPLES
        vad_probs = sample["vad_probs"][:n_vad_frames]

        return {
            "audio": torch.from_numpy(audio).float(),
            "phoneme_ids": torch.tensor(sample["phoneme_ids"], dtype=torch.long),
            "stress_seq": torch.tensor(sample["stress_seq"], dtype=torch.long),
            "lang": sample["lang"],
            "vad_probs": torch.tensor(vad_probs, dtype=torch.float32),  # may be empty
        }


def collate_fn(batch):
    """Pad audio and phoneme sequences to batch max length."""
    return _collate(batch, augment=False)


def collate_fn_augment(batch):
    """Training-time collator: silence head/tail pad + low-amplitude noise overlay.

    Pad: 100-200ms head, 300-500ms tail of zeros — teaches the model that
    silence after a word means no trailing consonant (addresses liaison-style
    over-emission). Same trick on the head for symmetry.

    Noise: Gaussian, ~1e-4 RMS, applied to the full clip including pad
    regions. Cheap robustness — pure silence at exact zero is unrealistic.
    """
    return _collate(batch, augment=True)


def _collate(batch, *, augment: bool):
    sr = 16000
    augmented = []
    for item in batch:
        audio = item["audio"]
        vad = item["vad_probs"]
        head = tail = 0
        if augment:
            head = int(random.uniform(0.1, 0.2) * sr)
            tail = int(random.uniform(0.3, 0.5) * sr)
            audio = torch.cat([torch.zeros(head), audio, torch.zeros(tail)])
            noise = torch.randn_like(audio) * 1e-4
            audio = audio + noise
            # Pad VAD with zeros for the silence head/tail (silence = no speech).
            # Use VAD_FRAME_SAMPLES = 256 to convert sample counts to VAD frames.
            if vad.numel() > 0:
                vad = torch.cat([
                    torch.zeros(head // VAD_FRAME_SAMPLES),
                    vad,
                    torch.zeros(tail // VAD_FRAME_SAMPLES),
                ])
        augmented.append({**item, "audio": audio, "vad_probs": vad})

    max_audio = max(item["audio"].shape[0] for item in augmented)
    max_phonemes = max(item["phoneme_ids"].shape[0] for item in augmented)
    max_vad = max((item["vad_probs"].shape[0] for item in augmented), default=0)

    audio_batch = torch.zeros(len(augmented), max_audio)
    audio_mask = torch.zeros(len(augmented), max_audio, dtype=torch.long)
    phoneme_batch = torch.zeros(len(augmented), max_phonemes, dtype=torch.long)
    stress_batch = torch.zeros(len(augmented), max_phonemes, dtype=torch.long)
    audio_lens = torch.zeros(len(augmented), dtype=torch.long)
    phoneme_lens = torch.zeros(len(augmented), dtype=torch.long)
    vad_batch = torch.zeros(len(augmented), max_vad) if max_vad > 0 else None
    vad_lens = torch.zeros(len(augmented), dtype=torch.long)

    for i, item in enumerate(augmented):
        a = item["audio"].shape[0]
        p = item["phoneme_ids"].shape[0]
        audio_batch[i, :a] = item["audio"]
        audio_mask[i, :a] = 1
        phoneme_batch[i, :p] = item["phoneme_ids"]
        stress_batch[i, :p] = item["stress_seq"]
        audio_lens[i] = a
        phoneme_lens[i] = p
        if vad_batch is not None:
            v = item["vad_probs"].shape[0]
            vad_batch[i, :v] = item["vad_probs"]
            vad_lens[i] = v

    return {
        "audio": audio_batch,
        "audio_mask": audio_mask,
        "audio_lens": audio_lens,
        "phoneme_ids": phoneme_batch,
        "phoneme_lens": phoneme_lens,
        "stress_seq": stress_batch,
        "vad_probs": vad_batch,   # (B, T_vad) or None if no VAD data in batch
        "vad_lens": vad_lens,
        "langs": [item["lang"] for item in augmented],
    }
