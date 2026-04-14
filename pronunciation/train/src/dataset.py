"""Dataset: load audio + pre-computed espeak phonemes with stress labels."""

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2
NUM_STRESS_LABELS = 3


class StressDataset(Dataset):
    """Yields (audio, phoneme_token_ids, stress_per_phoneme) triples.

    The phoneme token IDs are mapped to the base model's tokenizer vocabulary
    so that torchaudio's forced_align can align them against the frozen CTC
    logits at training time.
    """

    def __init__(self, phonemes_path: Path, tokenizer, max_audio_sec: float = 16.0):
        self.tokenizer = tokenizer
        self.max_audio_samples = int(max_audio_sec * 16000)

        audio_dir = phonemes_path.parent
        self.samples = []
        unk_id = tokenizer.unk_token_id

        with open(phonemes_path) as f:
            for line in f:
                rec = json.loads(line)
                wav_path = audio_dir / rec["file"]
                if not wav_path.exists():
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
                    continue

                self.samples.append({
                    "wav_path": str(wav_path),
                    "phoneme_ids": phoneme_ids,
                    "stress_seq": stress_seq,
                    "lang": rec["lang"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample["wav_path"])
        assert sr == 16000, f"Expected 16kHz audio, got {sr}"

        if len(audio) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]

        return {
            "audio": torch.from_numpy(audio).float(),
            "phoneme_ids": torch.tensor(sample["phoneme_ids"], dtype=torch.long),
            "stress_seq": torch.tensor(sample["stress_seq"], dtype=torch.long),
            "lang": sample["lang"],
        }


def collate_fn(batch):
    """Pad audio and phoneme sequences to batch max length."""
    max_audio = max(item["audio"].shape[0] for item in batch)
    max_phonemes = max(item["phoneme_ids"].shape[0] for item in batch)

    audio_batch = torch.zeros(len(batch), max_audio)
    audio_mask = torch.zeros(len(batch), max_audio, dtype=torch.long)
    phoneme_batch = torch.zeros(len(batch), max_phonemes, dtype=torch.long)
    stress_batch = torch.zeros(len(batch), max_phonemes, dtype=torch.long)
    audio_lens = torch.zeros(len(batch), dtype=torch.long)
    phoneme_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        a = item["audio"].shape[0]
        p = item["phoneme_ids"].shape[0]
        audio_batch[i, :a] = item["audio"]
        audio_mask[i, :a] = 1
        phoneme_batch[i, :p] = item["phoneme_ids"]
        stress_batch[i, :p] = item["stress_seq"]
        audio_lens[i] = a
        phoneme_lens[i] = p

    return {
        "audio": audio_batch,
        "audio_mask": audio_mask,
        "audio_lens": audio_lens,
        "phoneme_ids": phoneme_batch,
        "phoneme_lens": phoneme_lens,
        "stress_seq": stress_batch,
        "langs": [item["lang"] for item in batch],
    }
