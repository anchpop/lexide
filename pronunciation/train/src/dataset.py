"""Dataset: load audio + pre-computed espeak phonemes with stress labels."""

import hashlib
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset
from tqdm import tqdm

STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2
NUM_STRESS_LABELS = 3

VAD_FRAME_SAMPLES = 256  # earshot's native stride (16 ms @ 16 kHz)
VAD_TRAILING_SILENCE_RMS = 5e-4


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
        excluded_target_hashes: dict[str, str] | None = None,
        min_whisper_logprob: float | None = None,
    ):
        """
        min_whisper_logprob: if set, exclude rows whose `whisper_avg_logprob`
            (recorded by download_pimsleur.py) is below this threshold. A
            value of -0.7 catches the bulk of Pimsleur mistranscriptions
            (manual audit: ~17% wrong-rate in <-0.7, ~0% wrong-rate above).
            Rows lacking this field (FLEURS, Tatoeba) always pass — they
            went through their own audit pipeline.
        """
        self.tokenizer = tokenizer
        self.max_audio_samples = int(max_audio_sec * 16000)
        min_samples = int(min_duration_sec * 16000)
        excluded_target_hashes = excluded_target_hashes or {}

        audio_dir = phonemes_path.parent
        self.samples = []
        unk_id = tokenizer.unk_token_id

        skipped = {
            "missing_file": 0,
            "no_phonemes": 0,
            "silent": 0,
            "too_short": 0,
            "unreadable": 0,
            "asr_audit": 0,
            "stale_asr_audit": 0,
            "whisper_logprob": 0,
        }

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
            # Whisper confidence filter (Pimsleur clips only — FLEURS/Tatoeba
            # rows lack this field). Manual audit on eng Pimsleur showed
            # ~17% wrong-rate below logprob -0.7, ~0% above. Defaults to
            # None (no filter) at the constructor level; train_unified.py
            # plumbs through a CLI value.
            #
            # Pimsleur rows that lack the field are stale (preprocessed
            # before the field was wired through preprocess.py). Drop them
            # rather than silently bypassing the filter — re-run preprocess
            # to get them back. Pimsleur is detected by either the `source`
            # field (added in this PR) OR the `pimsleur_` filename prefix
            # (a stable contract from download_pimsleur.py), so very old
            # preprocessed rows that predate the `source` field are still
            # caught.
            if min_whisper_logprob is not None:
                lp = rec.get("whisper_avg_logprob")
                is_pimsleur = (rec.get("source") == "pimsleur"
                               or rec["file"].startswith("pimsleur_"))
                if lp is None:
                    if is_pimsleur:
                        skipped["whisper_logprob"] += 1
                        continue
                elif lp < min_whisper_logprob:
                    skipped["whisper_logprob"] += 1
                    continue

            audited_target_hash = excluded_target_hashes.get(rec["file"])
            if audited_target_hash is not None:
                current_hash = hashlib.sha256(rec.get("sentence", "").encode()).hexdigest()
                if audited_target_hash == current_hash:
                    skipped["asr_audit"] += 1
                    continue
                # The target changed since the audit, so don't exclude based on
                # stale CER/WER from a now-repaired label.
                skipped["stale_asr_audit"] += 1

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

            n_audio_samples = info.frames
            if n_audio_samples > self.max_audio_samples:
                n_audio_samples = self.max_audio_samples
            self.samples.append({
                "wav_path": str(wav_path),
                "phoneme_ids": phoneme_ids,
                "stress_seq": stress_seq,
                "lang": rec["lang"],
                # Source class (fleurs / tatoeba / pimsleur / tts). Used by
                # --source-cap-second in train_unified.py to balance per-lang
                # representation within each source so e.g. English (which has
                # 87k Pimsleur clips) doesn't drown the other langs.
                "source": rec.get("source"),
                # May be empty list if vad.jsonl wasn't present — training loop
                # treats empty as "no VAD signal, skip VAD loss for this clip".
                "vad_probs": vad_by_file.get(rec["file"], []),
                # Stored so the length-bucketed BatchSampler can query without
                # re-statting the file. Capped at max_audio_samples since
                # __getitem__ truncates above that.
                "n_audio_samples": n_audio_samples,
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


def make_train_collate(degrade_prob: float | None = None):
    """Training collate carrying the audio-degradation probability explicitly.

    Returns a picklable functools.partial (module-level _collate + simple args),
    so the degradation config reaches DataLoader workers under both fork AND
    spawn — unlike a module global, which spawn workers would re-import as off.
    degrade_prob=None → no degradation (the plain augment path).
    """
    return partial(_collate, augment=True, degrade_prob=degrade_prob)


def _match_vad_length(vad: torch.Tensor, target_len: int) -> torch.Tensor:
    """Trim/pad VAD to the number of complete 16 ms frames in the audio."""
    if vad.shape[0] > target_len:
        return vad[:target_len]
    if vad.shape[0] < target_len:
        return torch.cat([vad, vad.new_zeros(target_len - vad.shape[0])])
    return vad


def _zero_vad_after_trailing_silence(
    audio: torch.Tensor,
    vad: torch.Tensor,
    *,
    threshold: float = VAD_TRAILING_SILENCE_RMS,
) -> torch.Tensor:
    """Suppress VAD after the last frame with meaningful audio energy.

    This is trailing-only: internal low-energy pauses are preserved, but once
    the waveform has dropped below the threshold and never comes back up, VAD
    should not keep encouraging nonblank predictions.
    """
    n_frames = min(vad.shape[0], audio.shape[0] // VAD_FRAME_SAMPLES)
    if n_frames <= 0:
        return vad

    framed = audio[:n_frames * VAD_FRAME_SAMPLES].view(n_frames, VAD_FRAME_SAMPLES)
    rms = torch.sqrt(torch.mean(framed.float() ** 2, dim=1))
    active = torch.nonzero(rms > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        vad[:n_frames] = 0
        return vad

    first_trailing_silence = int(active[-1].item()) + 1
    if first_trailing_silence < n_frames:
        vad[first_trailing_silence:n_frames] = 0
    return vad


# --- Audio degradation augmentation (opt-in via --audio-degrade) -------------
# Realistic learner-mic corruptions so the mel side-channel learns to extract
# phonetics THROUGH noise instead of amplifying it (the momom2 failure: clean
# audio +8, noisy −10). Every op preserves phoneme identity (no pitch/time warp
# — those move formants and could flip the label) and preserves length (so the
# precomputed 16ms VAD grid stays aligned). The per-clip probability flows in
# as a _collate argument (see make_train_collate) — no module global, so it is
# carried explicitly to workers under both fork and spawn.


def _colored_noise(n: int, rng) -> torch.Tensor:
    """White / pink / brown noise — colored is closer to real ambient than white."""
    w = torch.randn(n)
    kind = rng.choice(["white", "pink", "brown"])
    if kind == "white":
        return w
    # 1/f-ish shaping in the frequency domain.
    spec = torch.fft.rfft(w)
    f = torch.arange(spec.shape[0], dtype=torch.float32).clamp_min(1.0)
    spec = spec / (f ** (0.5 if kind == "pink" else 1.0))
    x = torch.fft.irfft(spec, n=n)
    return x / x.std().clamp_min(1e-9)


def degrade_waveform(audio: torch.Tensor, sr: int = 16000, rng=random) -> torch.Tensor:
    """Apply a random subset of identity-preserving, length-preserving corruptions."""
    import torchaudio.functional as AF
    n = audio.shape[0]
    x = audio
    # random gain (mel_norm largely absorbs this, but it varies clipping/SNR interplay)
    if rng.random() < 0.5:
        x = x * (10.0 ** (rng.uniform(-8, 6) / 20.0))
    # reverb: convolve with a randomized exp-decay synthetic room impulse (FFT conv, trim to n)
    if rng.random() < 0.3:
        L = int(sr * rng.uniform(0.05, 0.35))
        t = torch.arange(L, dtype=torch.float32)
        rir = torch.randn(L) * torch.exp(-t / (sr * rng.uniform(0.02, 0.12)))
        rir[0] = 1.0                                  # direct path dominates
        rir = rir / rir.norm().clamp_min(1e-9)
        x = AF.fftconvolve(x, rir)[:n]
    # band-limit: laptop/phone mic + telephone band
    if rng.random() < 0.4:
        x = AF.lowpass_biquad(x, sr, rng.uniform(2500, 7000))
    if rng.random() < 0.25:
        x = AF.highpass_biquad(x, sr, rng.uniform(60, 300))
    # mic coloration: one random peaking-EQ band
    if rng.random() < 0.3:
        x = AF.equalizer_biquad(x, sr, rng.uniform(500, 4000),
                                gain=rng.uniform(-9, 9), Q=rng.uniform(0.5, 2.0))
    # additive colored noise at a random SNR
    if rng.random() < 0.8:
        snr_db = rng.uniform(3, 30)
        noise = _colored_noise(n, rng)
        sig = x.pow(2).mean().clamp_min(1e-12).sqrt()
        x = x + noise * (sig / (10.0 ** (snr_db / 20.0)))
    # clipping/saturation from a too-hot mic
    if rng.random() < 0.2:
        thr = x.abs().max().clamp_min(1e-6) * rng.uniform(0.3, 0.8)
        x = x.clamp(-thr, thr)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x[:n] if x.shape[0] >= n else torch.cat([x, x.new_zeros(n - x.shape[0])])


def _collate(batch, *, augment: bool, degrade_prob: float | None = None):
    sr = 16000
    augmented = []
    for item in batch:
        audio = item["audio"]
        vad = item["vad_probs"]
        if augment:
            # Quantize synthetic silence to VAD frames so the precomputed VAD
            # grid remains aligned after prepending silence.
            head_frames = random.randint(
                round(0.1 * sr / VAD_FRAME_SAMPLES),
                round(0.2 * sr / VAD_FRAME_SAMPLES),
            )
            tail_frames = random.randint(
                round(0.3 * sr / VAD_FRAME_SAMPLES),
                round(0.5 * sr / VAD_FRAME_SAMPLES),
            )
            head = head_frames * VAD_FRAME_SAMPLES
            tail = tail_frames * VAD_FRAME_SAMPLES
            audio = torch.cat([torch.zeros(head), audio, torch.zeros(tail)])
            # VAD target cleanup must run on the CLEAN waveform: the trailing-
            # silence detector is energy-based, so degradation noise/reverb in
            # the padded tail would make it see speech and stop suppressing
            # stale VAD positives. Finalize VAD here, THEN corrupt the audio.
            if vad.numel() > 0:
                vad = torch.cat([
                    vad.new_zeros(head_frames),
                    vad,
                    vad.new_zeros(tail_frames),
                ])
                vad = _match_vad_length(vad, audio.shape[0] // VAD_FRAME_SAMPLES)
                vad = _zero_vad_after_trailing_silence(audio, vad)
            if degrade_prob is not None and random.random() < degrade_prob:
                # Degrade the whole padded clip → reverb tail + noise floor cover
                # the silence too (realistic). Length-preserving, so VAD stays aligned.
                audio = degrade_waveform(audio, sr)
            else:
                # Clean-ish path (also the ~1-prob fraction that stays clean so
                # studio-quality performance like Jean-Cavard isn't lost).
                audio = audio + torch.randn_like(audio) * 1e-4
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


def get_audio_lengths(dataset) -> list[int]:
    """Walk a (possibly nested Subset of ConcatDataset of StressDataset) and
    return each sample's audio frame count without loading audio. Used by
    LengthBucketedBatchSampler to group same-length clips into batches.
    """
    def base(ds):
        if isinstance(ds, Subset):
            # Hoist the recursive call out of the list comprehension; otherwise
            # base(ds.dataset) is re-evaluated per index → O(n²) for the typical
            # Subset(ConcatDataset(StressDataset...)) shape from random_split.
            # That's the difference between seconds and not-completing-overnight
            # for a 285k-sample dataset.
            inner = base(ds.dataset)
            return [inner[i] for i in ds.indices]
        if isinstance(ds, ConcatDataset):
            out = []
            for d in ds.datasets:
                out.extend(base(d))
            return out
        if isinstance(ds, StressDataset):
            return [s["n_audio_samples"] for s in ds.samples]
        raise TypeError(f"get_audio_lengths doesn't know how to walk {type(ds)}")
    return base(dataset)


class LengthBucketedBatchSampler(Sampler):
    """Group dataset indices into batches by similar audio length.

    Why: the vectorized factorial-CTC DP allocates (B, T, P, F, K) tensors
    that scale with the longest sample in each batch. A single 6-minute clip
    in an otherwise-short batch can OOM a 96 GB GH200. Grouping similar
    lengths together bounds peak per-batch memory close to the average,
    and as a side benefit reduces padding waste.

    Procedure (standard speech-ML approach):
      1. Sort indices by length.
      2. Chunk into mega-buckets of `bucket_size_mul * batch_size` indices.
         Within each mega-bucket lengths are still similar but the
         within-bucket ordering is randomized to keep batches diverse.
      3. Chunk each mega-bucket into batch_size-sized batches.
      4. Shuffle the order of all batches across the epoch (kills any
         short→long curriculum effect from sorting).
      5. Drop the final incomplete batch (matches DataLoader drop_last=True).

    With B=16 and bucket_size_mul=100, lengths within a batch come from a
    1600-sample window of the sorted list — tight enough to bound memory,
    loose enough that within-batch correlation stays small.
    """

    def __init__(self, lengths: list[int], batch_size: int,
                 bucket_size_mul: int = 100, seed: int = 0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_mul
        self.seed = seed
        self.epoch = 0
        self._sorted = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the per-epoch shuffles. Callers should invoke before each epoch."""
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches = []
        for start in range(0, len(self._sorted), self.bucket_size):
            bucket = self._sorted[start:start + self.bucket_size]
            perm = torch.randperm(len(bucket), generator=g).tolist()
            shuffled = [bucket[i] for i in perm]
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        order = torch.randperm(len(batches), generator=g).tolist()
        for i in order:
            yield batches[i]

    def __len__(self):
        return len(self.lengths) // self.batch_size
