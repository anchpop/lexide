"""Full CREPE on the acoustic frame grid, with a continuous, sub-bin Viterbi estimate."""
import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchcrepe
import triton
import triton.language as tl


@triton.jit
def _decode(emissions, back, path, frames, bins: tl.constexpr, block: tl.constexpr):
    # One program owns the complete utterance; inference minibatches never break the path.
    i = tl.arange(0, block)
    score = tl.load(emissions + i, i < bins, -float("inf"))
    for t in range(1, frames):
        best = tl.full((block,), -float("inf"), tl.float32)
        predecessor = tl.full((block,), 0, tl.int32)
        for delta in tl.static_range(-11, 12):
            j = i + delta
            candidate = tl.gather(score, tl.maximum(0, tl.minimum(j, block - 1)), 0) + tl.log(12.0 - tl.abs(delta))
            better = (j >= 0) & (j < bins) & (candidate > best)
            best = tl.where(better, candidate, best)
            predecessor = tl.where(better, delta, predecessor)
        tl.store(back + t * bins + i, predecessor, i < bins)
        # Range lies in the interior of CREPE's 360 bins, so every allowed source
        # has the same transition normalization. Softmax normalization also cancels.
        score = best + tl.load(emissions + t * bins + i, i < bins, -float("inf"))
    state = tl.argmax(score, 0)
    for step in range(frames - 1):
        t = frames - 1 - step
        tl.store(path + t, state)
        state += tl.load(back + t * bins + state).to(tl.int32)
    tl.store(path, state)


@torch.inference_mode()
def track(wave, sample_rate=24000, hop=256):
    """Return Hz and periodicity at centers 0, hop, ... (wave is mono CUDA float)."""
    n = (wave.numel() + hop - 1) // hop
    audio = AF.resample(wave, sample_rate, 16000)
    centers = (torch.arange(n, device=wave.device, dtype=torch.float64) * (hop * 16000 / sample_rate)).round().long()
    frames = F.pad(audio, (512, 512))[centers[:, None] + torch.arange(1024, device=wave.device)]
    window = torch.hann_window(1024, device=wave.device)
    spectrum = torch.fft.rfft(frames * window).abs().clamp_min(1e-5).log10() * 20
    weights = torch.as_tensor(torchcrepe.loudness.perceptual_weights().ravel(), device=wave.device, dtype=torch.float32)
    loud = (spectrum.clamp_min(spectrum.max() - 80) + weights).clamp_min(-100).mean(1) >= -60
    frames = (frames - frames.mean(1, keepdim=True)) / frames.std(1, keepdim=True).clamp_min(1e-10)
    lo, hi = [round((1200 * math.log2(f / 10) - 1997.3794084376191) / 20) for f in (50, 1000)]
    with torch.autocast('cuda', dtype=torch.float16):
        scores = torch.cat([torchcrepe.infer(chunk, 'full', wave.device)[:, lo:hi+1] for chunk in frames.split(256)]).float().contiguous()
    bins = scores.shape[1]
    back = torch.empty_like(scores, dtype=torch.int8)
    path = torch.empty(n, device=wave.device, dtype=torch.int32)
    _decode[(1,)](scores, back, path, n, bins, triton.next_power_of_2(bins))
    neighbors = path[:, None] + torch.arange(-4, 5, device=wave.device)
    weights = scores.gather(1, neighbors.clamp(0, bins - 1)) * ((neighbors >= 0) & (neighbors < bins))
    center = (neighbors * weights).sum(1) / weights.sum(1).clamp_min(1e-10)
    hz = 10 * torch.exp2((1997.3794084376191 + 20 * (lo + center)) / 1200)
    confidence = scores.gather(1, path.long()[:, None]).flatten() * loud
    return hz, confidence
