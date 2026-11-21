"""Profile training steps of a voca.train configuration on real batches.

Reports wall time per step and how much of it the GPU was busy, then the top
operations by CPU and CUDA time. Run on the GPU you plan to train on:

    python scripts/profile_step.py --config configs/styletts.json \
        --train features/eng_gemini_10h/train.jsonl --gan
"""

import argparse
import json
import time
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile

from voca.config import Config
from voca.data import AnnotatedDataset, BucketSampler, collate
from voca.train import TRAINING_SYSTEMS, optimize_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--batch-frames", type=int, default=9000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--gan", action="store_true", help="Profile the adversarial phase")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace", type=Path, help="Write a Chrome trace here")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    c = Config.from_dict(json.loads(args.config.read_text()))
    data = AnnotatedDataset(args.train, c, voiced_vowels=True)
    pool = list(range(min(len(data), 2000)))
    data.preload(pool)
    system = TRAINING_SYSTEMS[c.architecture](c).cuda()
    critics = system.critics().cuda()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01, fused=True)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.0, fused=True)
    sampler = BucketSampler([data.lengths[i] for i in pool], args.batch_frames, args.max_batch, 0)
    step = 20000 if args.gan else 5000
    recipe = dict(weights=system.WEIGHTS, learning_rate=1e-4, warmup=1, lr_half_life=1e5, grad_clip=10.0, gan_start=8000,
                  gan_ratio=0.7, gan_ramp=3000, feature_matching=30.0, gan_scale_max=2000.0, crop_samples=8192, crops=2,
                  teacher_hold=3000, teacher_ramp=7000, teacher_final=0.0)
    system.train()

    def run(n):
        frames = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(n):
            batch = collate([data[pool[j]] for j in sampler.next()], "cuda")
            optimize_step(system, critics, optimizer, critic_optimizer, batch, step + i, recipe)
            frames += int(batch["host"]["frame_lengths"].sum())
        torch.cuda.synchronize()
        return time.perf_counter() - start, frames

    seconds, _ = run(args.warmup)
    print(f"warm-up ({args.warmup} steps, cuDNN autotuning): {seconds:.0f} s", flush=True)
    seconds, frames = run(args.steps)
    print(f"{args.steps / seconds:.2f} steps/s, {frames / args.steps:.0f} frames/step, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        seconds, _ = run(args.steps)
    events = prof.key_averages()
    # Busy time from the kernel timeline (overlapping streams counted once).
    kernels = sorted((e.time_range.start, e.time_range.end) for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    busy, end = 0, 0
    for a, b in kernels:
        if b > end:
            busy += b - max(a, end)
            end = b
    print(f"profiled: {seconds / args.steps * 1000:.0f} ms/step wall, GPU busy {busy / 1e3 / args.steps:.0f} ms/step "
          f"({busy / 1e6 / seconds:.0%}), {len(kernels) / args.steps:.0f} kernels/step")
    print(events.table(sort_by="self_cuda_time_total", row_limit=25))
    print(events.table(sort_by="self_cpu_time_total", row_limit=15))
    if args.trace:
        prof.export_chrome_trace(str(args.trace))


if __name__ == "__main__":
    main()
