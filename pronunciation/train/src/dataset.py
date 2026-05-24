"""Dataset: load audio + pre-computed espeak phonemes (with separate stress field).

On disk the phonemes.jsonl stores phonemes and stress as parallel arrays — that
format is what train/relabel-french/ produces and edits. At load time we
*interleave* stress markers (ˈ for primary, ˌ for secondary) into the CTC
token sequence so a single CTC head can learn to emit them inline.
"""

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


# Per-phoneme stress codes used by preprocess.py and relabel-french.
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2


class PhonemeDataset(Dataset):
    """Yields (audio, phoneme_token_ids) pairs.

    Phoneme sequences include inline stress markers (ˈ, ˌ) — the tokenizer must
    have them in its vocab (see STRESS_TOKENS in train_phoneme.py).
    Unknown phonemes are skipped.
    """

    def __init__(self, phonemes_path: Path, tokenizer, max_audio_sec: float = 16.0):
        self.tokenizer = tokenizer
        self.max_audio_samples = int(max_audio_sec * 16000)

        audio_dir = phonemes_path.parent
        self.samples = []
        unk_id = tokenizer.unk_token_id
        primary_id = tokenizer.convert_tokens_to_ids("ˈ")
        secondary_id = tokenizer.convert_tokens_to_ids("ˌ")
        if primary_id == unk_id or secondary_id == unk_id:
            raise RuntimeError(
                "Tokenizer must have ˈ and ˌ in its vocab — call "
                "tokenizer.add_tokens(['ˈ', 'ˌ']) before constructing PhonemeDataset."
            )

        with open(phonemes_path) as f:
            for line in f:
                rec = json.loads(line)
                wav_path = audio_dir / rec["file"]
                if not wav_path.exists():
                    continue

                # Interleave stress markers into the phoneme sequence.
                phoneme_ids = []
                for phoneme, s in zip(rec["phonemes"], rec["stress"]):
                    if s == STRESS_PRIMARY:
                        phoneme_ids.append(primary_id)
                    elif s == STRESS_SECONDARY:
                        phoneme_ids.append(secondary_id)
                    tid = tokenizer.convert_tokens_to_ids(phoneme)
                    if tid == unk_id or tid is None:
                        continue
                    phoneme_ids.append(tid)

                if not phoneme_ids:
                    continue

                self.samples.append({
                    "wav_path": str(wav_path),
                    "phoneme_ids": phoneme_ids,
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
            "lang": sample["lang"],
        }


def collate_fn(batch):
    """Pad audio and phoneme sequences to batch max length."""
    max_audio = max(item["audio"].shape[0] for item in batch)
    max_phonemes = max(item["phoneme_ids"].shape[0] for item in batch)

    audio_batch = torch.zeros(len(batch), max_audio)
    audio_mask = torch.zeros(len(batch), max_audio, dtype=torch.long)
    phoneme_batch = torch.zeros(len(batch), max_phonemes, dtype=torch.long)
    audio_lens = torch.zeros(len(batch), dtype=torch.long)
    phoneme_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        a = item["audio"].shape[0]
        p = item["phoneme_ids"].shape[0]
        audio_batch[i, :a] = item["audio"]
        audio_mask[i, :a] = 1
        phoneme_batch[i, :p] = item["phoneme_ids"]
        audio_lens[i] = a
        phoneme_lens[i] = p

    return {
        "audio": audio_batch,
        "audio_mask": audio_mask,
        "audio_lens": audio_lens,
        "phoneme_ids": phoneme_batch,
        "phoneme_lens": phoneme_lens,
        "langs": [item["lang"] for item in batch],
    }
