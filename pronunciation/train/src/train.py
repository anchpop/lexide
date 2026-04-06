"""Training loop for Conformer+CTC pronunciation model."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .audio import AudioProcessor
from .dataset import (
    PronunciationDataset,
    collate_fn,
    compute_language_weights,
)
from .evaluate import evaluate
from .model import ConformerCTCModel
from .vocab import IPAVocab


def noam_schedule(step: int, warmup_steps: int) -> float:
    """Noam-style LR: linear warmup then inverse sqrt decay."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    return (warmup_steps / step) ** 0.5


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    output_dir: str,
    best: bool = False,
) -> None:
    path = os.path.join(output_dir, "best_model.pt" if best else f"checkpoint_{step}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def cleanup_checkpoints(output_dir: str, keep: int = 3) -> None:
    ckpts = sorted(Path(output_dir).glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep]:
        p.unlink()


def get_manifest_paths(data_dir: str, split: str) -> list[str]:
    return sorted(str(p) for p in Path(data_dir).glob(f"manifest_*_{split}.jsonl"))


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    # Model
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--conv_kernel_size", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=25000)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--lang_temp", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # Output
    parser.add_argument("--output_dir", type=str, default="output/conformer")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=100)
    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="conformer-ctc-ipa-16lang")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Vocab
    vocab = IPAVocab.from_file(args.vocab_path)

    # Audio processor
    audio_proc = AudioProcessor()

    # Datasets
    train_manifests = get_manifest_paths(args.data_dir, "train")
    val_manifests = get_manifest_paths(args.data_dir, "val")
    assert train_manifests, f"No train manifests found in {args.data_dir}"
    assert val_manifests, f"No val manifests found in {args.data_dir}"

    train_dataset = PronunciationDataset(train_manifests, vocab, audio_proc)
    val_dataset = PronunciationDataset(val_manifests, vocab, audio_proc)

    # Language-balanced sampling
    weights = compute_language_weights(train_dataset.entries, temperature=args.lang_temp)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = ConformerCTCModel(
        vocab_size=vocab.size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        depthwise_conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"Vocab size: {vocab.size}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: noam_schedule(step + 1, args.warmup_steps)
    )

    # CTC loss
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, reduction="mean", zero_infinity=True)

    # bf16 mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # WandB
    if args.wandb:
        import wandb
        wandb.init(project="lexide-pronunciation", name=args.run_name, config=vars(args))

    # Training loop
    global_step = 0
    best_per = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            mels = batch["mels"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                log_probs, output_lengths = model(mels, mel_lengths)
                loss = ctc_loss(log_probs, labels, output_lengths, label_lengths)

            if args.grad_accum > 1:
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * (args.grad_accum if args.grad_accum > 1 else 1)
            epoch_steps += 1
            global_step += 1

            # Log
            if global_step % args.log_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | loss={avg_loss:.4f} | lr={lr:.2e}")
                if args.wandb:
                    wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

            # Eval
            if global_step % args.eval_steps == 0:
                results = evaluate(model, val_loader, vocab, device=device)
                print(f"Step {global_step} | PER={results['overall_per']:.4f}")
                for lang, per in sorted(results["per_by_lang"].items()):
                    print(f"  {lang}: {per:.4f}")

                if args.wandb:
                    log_dict = {"val/per": results["overall_per"]}
                    for lang, per in results["per_by_lang"].items():
                        log_dict[f"val/per_{lang}"] = per
                    wandb.log(log_dict, step=global_step)

                if results["overall_per"] < best_per:
                    best_per = results["overall_per"]
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, args.output_dir, best=True)

                model.train()

            # Save periodic checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, args.output_dir)
                cleanup_checkpoints(args.output_dir)

        print(f"Epoch {epoch} done | avg_loss={epoch_loss / max(epoch_steps, 1):.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
