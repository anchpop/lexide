"""Train stress prediction head on top of frozen wav2vec2."""

import argparse
import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

import wandb

from .dataset import StressDataset, collate_fn, NUM_STRESS_LABELS
from .model import Wav2Vec2StressModel, load_processor


def _walk_vectorized(alignments, stress_seq, blank_id):
    """Map a single-sample CTC alignment to per-frame stress labels via tensor ops.

    `alignments` is (T,) long, `stress_seq` is (L,) long. A "transition" at frame t
    is `alignments[t] != alignments[t-1] AND alignments[t] != blank`. CTC forced
    alignment separates same-token target positions with at least one blank, so
    comparing to the previous frame's token is enough — no need to track the
    previous *non-blank* token explicitly.
    """
    L = stress_seq.shape[0]
    is_non_blank = alignments != blank_id
    shifted = torch.cat(
        [alignments.new_full((1,), blank_id), alignments[:-1]], dim=0,
    )
    is_transition = (alignments != shifted) & is_non_blank
    phoneme_pos = is_transition.long().cumsum(0) - 1
    in_range = (phoneme_pos >= 0) & (phoneme_pos < L)
    phoneme_pos_safe = phoneme_pos.clamp(0, max(L - 1, 0))
    labels = stress_seq[phoneme_pos_safe]
    return torch.where(is_non_blank & in_range, labels, torch.zeros_like(labels))


def compute_frame_labels_batch(ctc_logits, phoneme_ids, stress_seq, phoneme_lens,
                               audio_lens, blank_id, conv_ratio):
    """Per-sample forced alignment, with a vectorized GPU walk to assign frame stress.

    The forced_align call is per-sample (variable-length batched forced_align in
    torchaudio was unreliable in this version — see the cancelled job 3 incident).
    The expensive part — the per-frame walk that previously did 800 `.item()` syncs
    per sample — is now a single cumsum + gather on GPU.

    `audio_lens` and `phoneme_lens` are pulled to CPU once up front so the Python
    loop over samples doesn't trigger per-iteration syncs.

    conv_ratio: audio samples per output frame (320 for wav2vec2 @ 16kHz).
    """
    batch_size, max_frames = ctc_logits.shape[:2]
    device = ctc_logits.device
    frame_labels = torch.zeros(batch_size, max_frames, dtype=torch.long, device=device)
    frame_masks = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=device)

    audio_lens_cpu = audio_lens.tolist()
    phoneme_lens_cpu = phoneme_lens.tolist()

    for i in range(batch_size):
        n_frames = min(audio_lens_cpu[i] // conv_ratio, max_frames)
        n_phonemes = phoneme_lens_cpu[i]
        if n_frames <= 0 or n_phonemes == 0 or n_frames < n_phonemes:
            continue

        try:
            log_probs = F.log_softmax(ctc_logits[i, :n_frames], dim=-1).unsqueeze(0)
            targets = phoneme_ids[i, :n_phonemes].unsqueeze(0)
            alignments, _ = AF.forced_align(log_probs, targets, blank=blank_id)
            labels = _walk_vectorized(
                alignments.squeeze(0), stress_seq[i, :n_phonemes], blank_id,
            )
            frame_labels[i, :n_frames] = labels
            frame_masks[i, :n_frames] = True
        except Exception:
            continue

    return frame_labels, frame_masks


def train_epoch(model, loader, optimizer, blank_id, conv_ratio, device, epoch, use_bf16=False):
    model.train()
    model.backbone.eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    total_loss = 0
    total_frames = 0
    correct = 0
    n_batches = 0
    total_samples = 0
    total_audio_sec = 0.0
    grad_norms = []
    timings = {"data": 0.0, "forward": 0.0, "forced_align": 0.0, "loss_backward": 0.0}

    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()
    last_end = time.perf_counter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Data loading = wallclock between previous batch end and now
        torch.cuda.synchronize()
        timings["data"] += time.perf_counter() - last_end

        audio = batch["audio"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        audio_lens = batch["audio_lens"].to(device, non_blocking=True)
        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lens = batch["phoneme_lens"].to(device, non_blocking=True)
        stress_seq = batch["stress_seq"].to(device, non_blocking=True)

        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with autocast_ctx:
            outputs = model(audio, attention_mask=audio_mask)
        torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - t_fwd

        stress_logits = outputs["stress_logits"]
        # forced_align needs fp32 log-probs for numerical stability
        ctc_logits = outputs["ctc_logits"].float()

        t_fa = time.perf_counter()
        frame_labels, frame_masks = compute_frame_labels_batch(
            ctc_logits, phoneme_ids, stress_seq, phoneme_lens,
            audio_lens, blank_id, conv_ratio,
        )
        torch.cuda.synchronize()
        timings["forced_align"] += time.perf_counter() - t_fa

        # Only compute loss on valid (masked) frames
        valid = frame_masks.view(-1)
        if valid.sum() == 0:
            last_end = time.perf_counter()
            continue

        flat_logits = stress_logits.view(-1, NUM_STRESS_LABELS)[valid].float()
        flat_labels = frame_labels.view(-1)[valid]

        t_lb = time.perf_counter()
        loss = F.cross_entropy(flat_logits, flat_labels)
        loss.backward()
        # Compute gradient norm without clipping (observation only)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.get_trainable_params()), max_norm=float("inf"),
        )
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timings["loss_backward"] += time.perf_counter() - t_lb

        preds = flat_logits.argmax(dim=-1)
        batch_correct = (preds == flat_labels).sum().item()
        batch_n = flat_labels.shape[0]

        total_loss += loss.item() * batch_n
        correct += batch_correct
        total_frames += batch_n
        n_batches += 1
        total_samples += audio.shape[0]
        total_audio_sec += audio_lens.sum().item() / 16000.0
        grad_norms.append(grad_norm.item())

        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_correct/batch_n:.3f}")
        last_end = time.perf_counter()

    epoch_sec = time.perf_counter() - epoch_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    nb = max(n_batches, 1)

    return {
        "loss": total_loss / max(total_frames, 1),
        "acc": correct / max(total_frames, 1),
        "wallclock_sec": epoch_sec,
        "samples_per_sec": total_samples / epoch_sec,
        "audio_realtime_factor": total_audio_sec / epoch_sec,
        "peak_mem_gb": peak_mem_gb,
        "ms_per_batch_data": timings["data"] / nb * 1000,
        "ms_per_batch_forward": timings["forward"] / nb * 1000,
        "ms_per_batch_forced_align": timings["forced_align"] / nb * 1000,
        "ms_per_batch_loss_backward": timings["loss_backward"] / nb * 1000,
        "grad_norm_mean": sum(grad_norms) / max(len(grad_norms), 1),
        "grad_norm_max": max(grad_norms, default=0.0),
    }


@torch.no_grad()
def eval_epoch(model, loader, blank_id, conv_ratio, device):
    model.eval()
    total_loss = 0
    total_frames = 0
    correct = 0
    per_class_correct = [0] * NUM_STRESS_LABELS
    per_class_total = [0] * NUM_STRESS_LABELS

    for batch in tqdm(loader, desc="Eval"):
        audio = batch["audio"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        audio_lens = batch["audio_lens"].to(device)
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_lens = batch["phoneme_lens"].to(device)
        stress_seq = batch["stress_seq"].to(device)

        outputs = model(audio, attention_mask=audio_mask)
        stress_logits = outputs["stress_logits"]
        ctc_logits = outputs["ctc_logits"]

        frame_labels, frame_masks = compute_frame_labels_batch(
            ctc_logits, phoneme_ids, stress_seq, phoneme_lens,
            audio_lens, blank_id, conv_ratio,
        )

        valid = frame_masks.view(-1)
        if valid.sum() == 0:
            continue

        flat_logits = stress_logits.view(-1, NUM_STRESS_LABELS)[valid]
        flat_labels = frame_labels.view(-1)[valid]

        loss = F.cross_entropy(flat_logits, flat_labels)
        total_loss += loss.item() * flat_labels.shape[0]

        preds = flat_logits.argmax(dim=-1)
        correct += (preds == flat_labels).sum().item()
        total_frames += flat_labels.shape[0]

        for c in range(NUM_STRESS_LABELS):
            mask_c = flat_labels == c
            per_class_correct[c] += (preds[mask_c] == c).sum().item()
            per_class_total[c] += mask_c.sum().item()

    per_class_acc = {
        f"acc_class_{c}": per_class_correct[c] / per_class_total[c]
        if per_class_total[c] > 0 else 0
        for c in range(NUM_STRESS_LABELS)
    }
    return total_loss / max(total_frames, 1), correct / max(total_frames, 1), per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../data/audio"))
    parser.add_argument("--model-name", type=str, default="anchpop/lexide-pronunciation-phoneme-xls-r-2b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", type=str, default="lexide-pronunciation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HF repo to push final model to (e.g. anchpop/lexide-pronunciation-stress)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 autocast on the backbone forward (recommended on Ampere+).")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — check torch install"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    processor = load_processor(args.model_name)
    model = Wav2Vec2StressModel(args.model_name).to(device)
    blank_id = processor.tokenizer.pad_token_id
    conv_ratio = 320  # wav2vec2-large downsamples 16kHz audio by 320x → 50fps frames

    trainable = sum(p.numel() for p in model.get_trainable_params())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # Load pre-computed phoneme datasets
    datasets = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        phonemes_file = lang_dir / "phonemes.jsonl"
        if phonemes_file.exists():
            ds = StressDataset(phonemes_file, processor.tokenizer, max_audio_sec=args.max_audio_sec)
            print(f"Loaded {lang_dir.name}: {len(ds)} samples")
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("No phonemes.jsonl files found — run preprocess.py first")

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # shuffle=True re-permutes every epoch via RandomSampler.
    # persistent_workers avoids respawning workers each epoch (matters at num_workers >> 0).
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    backbone_slug = args.model_name.split("/")[-1]
    wandb.init(project=args.wandb_project, name=f"stress-{backbone_slug}", config=vars(args))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model, train_loader, optimizer, blank_id, conv_ratio, device, epoch,
            use_bf16=args.bf16,
        )
        val_loss, val_acc, per_class_acc = eval_epoch(model, val_loader, blank_id, conv_ratio, device)
        scheduler.step()

        # Log learned layer weights
        weights = torch.softmax(model.layer_weights.detach(), dim=0).cpu().tolist()
        top_layer = max(range(len(weights)), key=lambda i: weights[i])

        print(
            f"Epoch {epoch}: train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} top_layer={top_layer} "
            f"wall={train_stats['wallclock_sec']:.1f}s "
            f"(data={train_stats['ms_per_batch_data']:.0f}ms "
            f"fwd={train_stats['ms_per_batch_forward']:.0f}ms "
            f"fa={train_stats['ms_per_batch_forced_align']:.0f}ms "
            f"bwd={train_stats['ms_per_batch_loss_backward']:.0f}ms) "
            f"mem={train_stats['peak_mem_gb']:.1f}GB"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "top_layer": top_layer,
            # Throughput / wallclock
            "perf/epoch_wallclock_sec": train_stats["wallclock_sec"],
            "perf/samples_per_sec": train_stats["samples_per_sec"],
            "perf/audio_realtime_factor": train_stats["audio_realtime_factor"],
            "perf/peak_mem_gb": train_stats["peak_mem_gb"],
            # Per-batch timing breakdown (lets us decide if forced_align is the bottleneck)
            "perf/ms_per_batch_data": train_stats["ms_per_batch_data"],
            "perf/ms_per_batch_forward": train_stats["ms_per_batch_forward"],
            "perf/ms_per_batch_forced_align": train_stats["ms_per_batch_forced_align"],
            "perf/ms_per_batch_loss_backward": train_stats["ms_per_batch_loss_backward"],
            # Gradient health (matters at lr=4e-3 / batch=64)
            "grad/norm_mean": train_stats["grad_norm_mean"],
            "grad/norm_max": train_stats["grad_norm_max"],
            **per_class_acc,
            **{f"layer_weight_{i}": w for i, w in enumerate(weights)},
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "stress_head": model.stress_head.state_dict(),
                "layer_weights": model.layer_weights.detach().cpu(),
                "args": vars(args),
            }, args.save_dir / "best.pt")

    wandb.finish()

    if args.hf_repo:
        from huggingface_hub import HfApi
        # Persist processor configs (preprocessor_config.json, tokenizer, vocab) alongside
        # best.pt so the HF repo is self-contained for downstream loading.
        processor.save_pretrained(args.save_dir)
        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.save_dir),
            repo_id=args.hf_repo,
        )
        print(f"Uploaded to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
