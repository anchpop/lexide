"""Head-only CTC fine-tune of wav2vec2-xlsr-53-espeak-cv-ft on the multilingual data.

Freezes the 317M-param backbone, leaves only `lm_head` (~400K params) trainable, and
re-fits it on the existing FLEURS + TTS phoneme data across all 7 languages. The goal
is to move the close-call substitutions documented in the French quirks doc (e.g.
/ɛ/→/e/, /y/→/i/, /e/→/i/) where the correct phoneme is already in the head's top-10
but ranked below the chosen one.

Saves a full `Wav2Vec2ForCTC` via `save_pretrained()` so downstream callers (Modal
endpoint etc.) can load with vanilla `from_pretrained(repo)`.
"""

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

import wandb
from transformers import Wav2Vec2ForCTC

from .dataset import StressDataset, collate_fn
from .model import load_processor


def freeze_backbone(model: Wav2Vec2ForCTC):
    """Freeze everything except lm_head."""
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True


def unfreeze_all(model: Wav2Vec2ForCTC):
    """Train every parameter (full fine-tune mode)."""
    for p in model.parameters():
        p.requires_grad = True


def apply_lora(model: Wav2Vec2ForCTC, rank: int, target_modules: list[str]):
    """Wrap backbone with LoRA adapters on the given modules; keep lm_head fully trainable.

    Adapts the (self-supervised-pretrained) backbone for phoneme classification without
    full-model retraining. Param count scales with rank × number of target modules
    × number of layers (wav2vec2-xls-r-2b has 48 transformer layers).
    """
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        # Train lm_head normally (not as a LoRA adapter) since it's small and being
        # learned from random init.
        modules_to_save=["lm_head"],
    )
    return get_peft_model(model, lora_config)


def make_labels(phoneme_ids: torch.Tensor, phoneme_lens: torch.Tensor) -> torch.Tensor:
    """CTC loss expects padding marked with -100. Collator pads with 0; convert here."""
    labels = phoneme_ids.clone()
    L_max = labels.shape[1]
    valid = torch.arange(L_max, device=labels.device).unsqueeze(0) < phoneme_lens.unsqueeze(1)
    labels[~valid] = -100
    return labels


def train_epoch(model, loader, optimizer, device, epoch, use_bf16, backbone_eval_mode=True):
    model.train()
    if backbone_eval_mode:
        # Frozen-backbone case: keep wav2vec2 in eval mode so SpecAug and dropout
        # don't disturb the pretrained activations. For LoRA / full fine-tune we want
        # those regularizers on, so skip this.
        backbone = getattr(model, "wav2vec2", None) or getattr(
            getattr(model, "base_model", None), "model", model,
        ).wav2vec2
        backbone.eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    total_audio_sec = 0.0
    grad_norms = []
    timings = {"data": 0.0, "forward": 0.0, "loss_backward": 0.0}

    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()
    last_end = time.perf_counter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        torch.cuda.synchronize()
        timings["data"] += time.perf_counter() - last_end

        audio = batch["audio"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lens = batch["phoneme_lens"].to(device, non_blocking=True)
        labels = make_labels(phoneme_ids, phoneme_lens)

        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with autocast_ctx:
            outputs = model(audio, attention_mask=audio_mask, labels=labels)
            loss = outputs.loss
        torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - t_fwd

        t_lb = time.perf_counter()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=float("inf"),
        )
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timings["loss_backward"] += time.perf_counter() - t_lb

        n_batches += 1
        total_loss += loss.item()
        total_samples += audio.shape[0]
        total_audio_sec += audio_mask.sum().item() / 16000.0
        grad_norms.append(grad_norm.item())

        pbar.set_postfix(loss=f"{loss.item():.3f}")
        last_end = time.perf_counter()

    epoch_sec = time.perf_counter() - epoch_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    nb = max(n_batches, 1)

    return {
        "loss": total_loss / nb,
        "wallclock_sec": epoch_sec,
        "samples_per_sec": total_samples / epoch_sec,
        "audio_realtime_factor": total_audio_sec / epoch_sec,
        "peak_mem_gb": peak_mem_gb,
        "ms_per_batch_data": timings["data"] / nb * 1000,
        "ms_per_batch_forward": timings["forward"] / nb * 1000,
        "ms_per_batch_loss_backward": timings["loss_backward"] / nb * 1000,
        "grad_norm_mean": sum(grad_norms) / max(len(grad_norms), 1),
        "grad_norm_max": max(grad_norms, default=0.0),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, use_bf16):
    model.eval()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Eval"):
        audio = batch["audio"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lens = batch["phoneme_lens"].to(device, non_blocking=True)
        labels = make_labels(phoneme_ids, phoneme_lens)

        with autocast_ctx:
            outputs = model(audio, attention_mask=audio_mask, labels=labels)

        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../data/audio"))
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--processor-source", type=str, default=None,
                        help="Where to load the tokenizer + feature extractor from. "
                             "Defaults to --model-name. Set this when --model-name is a raw "
                             "backbone without a tokenizer (e.g. wav2vec2-xls-r-1b/2b); "
                             "use facebook/wav2vec2-xlsr-53-espeak-cv-ft for the espeak vocab.")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Override CTC head vocab size. Needed when loading a raw backbone "
                             "(default config has no vocab_size). Should match the processor "
                             "tokenizer's vocab (392 for espeak).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints-phoneme"))
    parser.add_argument("--wandb-project", type=str, default="lexide-pronunciation")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HF repo to push fine-tuned model to")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Restrict training to these language codes (e.g. fra eng). "
                             "Default: all phonemes.jsonl files found under data-dir.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora", action="store_true",
                        help="Wrap backbone in LoRA adapters; train adapters + lm_head.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", nargs="+",
                        default=["q_proj", "k_proj", "v_proj"],
                        help="Module names to apply LoRA to (e.g. q_proj k_proj v_proj out_proj "
                             "intermediate_dense output_dense). More modules = more capacity "
                             "but slower training.")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Unfreeze all parameters and train end-to-end. Use lower lr "
                             "(~1e-5) and gradient-checkpointing to fit in memory.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade compute for activation memory. Essential for full fine-tune.")
    args = parser.parse_args()
    if sum([args.lora, args.full_finetune]) > 1:
        raise SystemExit("Pick at most one of --lora / --full-finetune (default = head-only)")

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    processor = load_processor(args.processor_source or args.model_name)
    # When loading a raw backbone (no CTC head), override the head's vocab_size
    # so it matches the espeak tokenizer (392 tokens). `ignore_mismatched_sizes`
    # tolerates the absent / mis-shaped lm_head in the source checkpoint.
    model_kwargs = {}
    if args.vocab_size is not None:
        model_kwargs["vocab_size"] = args.vocab_size
        model_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
        model_kwargs["ignore_mismatched_sizes"] = True
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name, **model_kwargs).to(device)
    # Silently zero out CTC infinity losses (occurs when input_length is too short
    # for the target sequence — without this, a single bad sample turns the whole
    # batch's mean loss to inf and the next backward step NaNs the head weights).
    model.config.ctc_zero_infinity = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # When the backbone is frozen (or LoRA-adapted) and gradient checkpointing
        # is on, gradients need a path back from the embedding to the trainable
        # params. enable_input_require_grads forces the embedding output to
        # carry require_grad so the gradient graph isn't broken at the checkpoint
        # boundary.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.lora:
        model = apply_lora(model, args.lora_rank, args.lora_targets)
    elif args.full_finetune:
        unfreeze_all(model)
    else:
        freeze_backbone(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    datasets = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if args.langs is not None and lang_dir.name not in args.langs:
            continue
        phonemes_file = lang_dir / "phonemes.jsonl"
        if phonemes_file.exists():
            ds = StressDataset(phonemes_file, processor.tokenizer, max_audio_sec=args.max_audio_sec)
            print(f"Loaded {lang_dir.name}: {len(ds)} samples")
            datasets.append(ds)
    if not datasets:
        raise RuntimeError(
            f"No phonemes.jsonl files matched (langs={args.langs}). "
            "Run preprocess.py and/or check --langs values."
        )

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    backbone_slug = args.model_name.split("/")[-1]
    wandb.init(project=args.wandb_project, name=f"phoneme-{backbone_slug}", config=vars(args))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    hf_api = None
    if args.hf_repo:
        from huggingface_hub import HfApi
        hf_api = HfApi()
        hf_api.create_repo(args.hf_repo, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model, train_loader, optimizer, device, epoch, args.bf16,
            backbone_eval_mode=not (args.lora or args.full_finetune),
        )
        val_loss = eval_epoch(model, val_loader, device, args.bf16)
        scheduler.step()

        print(
            f"Epoch {epoch}: train_loss={train_stats['loss']:.4f} val_loss={val_loss:.4f} "
            f"wall={train_stats['wallclock_sec']:.1f}s "
            f"(data={train_stats['ms_per_batch_data']:.0f}ms "
            f"fwd={train_stats['ms_per_batch_forward']:.0f}ms "
            f"bwd={train_stats['ms_per_batch_loss_backward']:.0f}ms) "
            f"mem={train_stats['peak_mem_gb']:.1f}GB"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "perf/epoch_wallclock_sec": train_stats["wallclock_sec"],
            "perf/samples_per_sec": train_stats["samples_per_sec"],
            "perf/audio_realtime_factor": train_stats["audio_realtime_factor"],
            "perf/peak_mem_gb": train_stats["peak_mem_gb"],
            "perf/ms_per_batch_data": train_stats["ms_per_batch_data"],
            "perf/ms_per_batch_forward": train_stats["ms_per_batch_forward"],
            "perf/ms_per_batch_loss_backward": train_stats["ms_per_batch_loss_backward"],
            "grad/norm_mean": train_stats["grad_norm_mean"],
            "grad/norm_max": train_stats["grad_norm_max"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save full Wav2Vec2ForCTC + processor so HF repo is self-contained
            # and downstream `from_pretrained(repo)` just works.
            # For PEFT-wrapped models, deepcopy + merge_and_unload first so the saved
            # repo is a vanilla Wav2Vec2ForCTC (no peft dependency on the load side).
            # Costs ~4GB extra GPU memory during the copy; original training model
            # keeps its LoRA wrapping and continues training untouched.
            try:
                from peft import PeftModel
                _is_peft = isinstance(model, PeftModel)
            except ImportError:
                _is_peft = False
            if _is_peft:
                import copy
                save_model = copy.deepcopy(model).merge_and_unload()
                save_model.save_pretrained(args.save_dir)
                del save_model
                torch.cuda.empty_cache()
            else:
                model.save_pretrained(args.save_dir)
            processor.save_pretrained(args.save_dir)
            # Upload on every improvement so the HF repo always reflects the
            # current best — lets us stop training whenever without losing
            # progress. Synchronous upload; xls-r-2b is ~8GB so this can take
            # a couple minutes per epoch on a slow link.
            if hf_api is not None:
                print(f"Uploading epoch {epoch} (val_loss={val_loss:.4f}) to HF...")
                hf_api.upload_folder(
                    folder_path=str(args.save_dir),
                    repo_id=args.hf_repo,
                    commit_message=f"epoch {epoch} (val_loss={val_loss:.4f})",
                )
                print(f"Uploaded to https://huggingface.co/{args.hf_repo}")

    wandb.finish()


if __name__ == "__main__":
    main()
