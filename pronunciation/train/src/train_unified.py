"""Train the unified factorized-CTC + stress model end-to-end.

Three heads on top of a single Wav2Vec2 backbone:
  - nonblank   (binary)     — addresses CTC blank-dominance (factorized CTC)
  - phoneme    (V-way)      — joint with nonblank → standard CTC log-probs
  - stress     (3-way)      — per-frame stress prediction (none/primary/secondary)

Stress supervision uses online torchaudio.forced_align on the model's *own*
factorized log-probs to assign per-frame stress labels. To avoid garbage
supervision during the cold start, stress loss is disabled until --stress-warmup-steps
have elapsed (by which point phoneme alignment is reasonable).

Fresh from facebook/wav2vec2-xls-r-2b — no external model dependencies, no
warm-start from earlier phoneme model.
"""

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

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from .dataset import StressDataset, collate_fn, collate_fn_augment, NUM_STRESS_LABELS
from .factorized_ctc import FactorizedCTCModel


def load_processor(model_name: str):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


WAV2VEC2_CONV_RATIO = 320  # 16kHz / 320 = 50 fps frames


def make_labels(phoneme_ids: torch.Tensor, phoneme_lens: torch.Tensor) -> torch.Tensor:
    labels = phoneme_ids.clone()
    L_max = labels.shape[1]
    valid = torch.arange(L_max, device=labels.device).unsqueeze(0) < phoneme_lens.unsqueeze(1)
    labels[~valid] = -100
    return labels


def _walk_vectorized(alignments: torch.Tensor, stress_seq: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Map a single-sample CTC alignment to per-frame stress labels via tensor ops.

    Mirror of the approach in train.py. CTC forced_align separates same-token
    target positions with at least one blank, so a `current != previous` transition
    cleanly marks each new phoneme.
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


def compute_frame_stress_labels(log_probs, phoneme_ids, stress_seq, phoneme_lens,
                                audio_lens, blank_id):
    """Per-sample forced alignment → per-frame stress targets.

    Uses our factorized model's own log-probs (which sum to 1 over the vocab
    by construction). Per-sample because batched torchaudio.forced_align was
    unreliable in the version used by train.py.
    """
    batch_size, max_frames = log_probs.shape[:2]
    device = log_probs.device
    frame_labels = torch.zeros(batch_size, max_frames, dtype=torch.long, device=device)
    frame_masks = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=device)

    audio_lens_cpu = audio_lens.tolist()
    phoneme_lens_cpu = phoneme_lens.tolist()

    for i in range(batch_size):
        n_frames = min(audio_lens_cpu[i] // WAV2VEC2_CONV_RATIO, max_frames)
        n_phonemes = phoneme_lens_cpu[i]
        if n_frames <= 0 or n_phonemes == 0 or n_frames < n_phonemes:
            continue

        try:
            lp = log_probs[i, :n_frames].unsqueeze(0)
            targets = phoneme_ids[i, :n_phonemes].unsqueeze(0)
            alignments, _ = AF.forced_align(lp, targets, blank=blank_id)
            labels = _walk_vectorized(
                alignments.squeeze(0), stress_seq[i, :n_phonemes], blank_id,
            )
            frame_labels[i, :n_frames] = labels
            frame_masks[i, :n_frames] = True
        except Exception:
            continue

    return frame_labels, frame_masks


def vad_loss(nonblank_logit, vad_probs, vad_lens, n_frames):
    """BCE between sigmoid(nonblank_logit) and VAD probabilities.

    VAD is at 16 ms stride (62.5 fps); wav2vec2 nonblank_logit is at 20 ms
    stride (50 fps). Interpolate per-sample with F.interpolate(mode='linear')
    from the clip's valid VAD length to its valid wav2vec2 length, then
    BCE on the masked region only. Padded frames contribute nothing.
    """
    B = nonblank_logit.shape[0]
    losses = []
    for i in range(B):
        n_v = int(vad_lens[i].item())
        n_f = int(n_frames[i].item())
        if n_v < 2 or n_f < 1:
            continue
        # F.interpolate wants (B=1, C=1, L) → output (1, 1, n_f)
        src = vad_probs[i, :n_v].unsqueeze(0).unsqueeze(0).float()
        target = F.interpolate(src, size=n_f, mode="linear", align_corners=False).squeeze()
        target = target.clamp(0.0, 1.0)
        logit = nonblank_logit[i, :n_f].float()
        losses.append(F.binary_cross_entropy_with_logits(logit, target))
    if not losses:
        return torch.zeros((), device=nonblank_logit.device)
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, device, epoch, *, use_bf16,
                blank_id, stress_active: bool, stress_weight: float,
                vad_weight: float, invalid_mass_weight: float):
    model.train()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    total_ctc = 0.0
    total_stress = 0.0
    total_vad = 0.0
    total_invalid = 0.0
    n_stress_batches = 0
    n_vad_batches = 0
    n_invalid_batches = 0
    n_batches = 0
    total_samples = 0
    total_audio_sec = 0.0
    grad_norms = []
    nonblank_means = []
    stress_correct = 0
    stress_total = 0
    timings = {"data": 0.0, "forward": 0.0, "forced_align": 0.0, "loss_backward": 0.0}

    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()
    last_end = time.perf_counter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        torch.cuda.synchronize()
        timings["data"] += time.perf_counter() - last_end

        audio = batch["audio"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        audio_lens = batch["audio_lens"].to(device, non_blocking=True)
        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lens = batch["phoneme_lens"].to(device, non_blocking=True)
        stress_seq = batch["stress_seq"].to(device, non_blocking=True)
        labels = make_labels(phoneme_ids, phoneme_lens)

        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with autocast_ctx:
            outputs = model(audio, attention_mask=audio_mask, labels=labels,
                            label_lengths=phoneme_lens)
            ctc_loss = outputs["loss"]
        torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - t_fwd

        stress_loss = torch.zeros((), device=device)
        if stress_active:
            t_fa = time.perf_counter()
            # forced_align needs fp32 log-probs for numerical stability.
            log_probs_fp32 = outputs["log_probs"].detach().float()
            frame_labels, frame_masks = compute_frame_stress_labels(
                log_probs_fp32, phoneme_ids, stress_seq, phoneme_lens,
                audio_lens, blank_id,
            )
            torch.cuda.synchronize()
            timings["forced_align"] += time.perf_counter() - t_fa

            valid = frame_masks.view(-1)
            if valid.sum() > 0:
                stress_logits = outputs["stress_logits"]
                flat_logits = stress_logits.view(-1, NUM_STRESS_LABELS)[valid].float()
                flat_labels = frame_labels.view(-1)[valid]
                stress_loss = F.cross_entropy(flat_logits, flat_labels)
                with torch.no_grad():
                    preds = flat_logits.argmax(dim=-1)
                    stress_correct += (preds == flat_labels).sum().item()
                    stress_total += flat_labels.shape[0]
                n_stress_batches += 1
                total_stress += stress_loss.item()

        vl = torch.zeros((), device=device)
        if vad_weight > 0 and batch.get("vad_probs") is not None:
            vad_probs_b = batch["vad_probs"].to(device, non_blocking=True)
            vad_lens_b = batch["vad_lens"].to(device, non_blocking=True)
            # Number of wav2vec2 output frames per sample.
            backbone = model.backbone
            n_frames = backbone._get_feat_extract_output_lengths(audio_mask.sum(-1)).to(torch.long)
            vl = vad_loss(outputs["nonblank_logit"], vad_probs_b, vad_lens_b, n_frames)
            total_vad += vl.item()
            n_vad_batches += 1

        im_loss = torch.zeros((), device=device)
        if invalid_mass_weight > 0 and "invalid_mass" in outputs:
            # Per-frame invalid-feature-mass, masked to valid audio frames.
            im = outputs["invalid_mass"]                              # (B, T)
            backbone = model.backbone
            n_frames_im = backbone._get_feat_extract_output_lengths(audio_mask.sum(-1)).to(torch.long)
            mask_im = (torch.arange(im.shape[1], device=im.device)[None] < n_frames_im[:, None])
            im_loss = (im * mask_im).sum() / mask_im.sum().clamp(min=1)
            total_invalid += im_loss.item()
            n_invalid_batches += 1

        loss = (ctc_loss
                + stress_weight * stress_loss
                + vad_weight * vl
                + invalid_mass_weight * im_loss)

        t_lb = time.perf_counter()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf"),
        )
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timings["loss_backward"] += time.perf_counter() - t_lb

        n_batches += 1
        total_ctc += ctc_loss.item()
        total_samples += audio.shape[0]
        total_audio_sec += audio_mask.sum().item() / 16000.0
        grad_norms.append(grad_norm.item())
        nonblank_means.append(torch.sigmoid(outputs["nonblank_logit"]).mean().item())

        postfix = {"ctc": f"{ctc_loss.item():.3f}", "p_nb": f"{nonblank_means[-1]:.2f}"}
        if stress_active and n_stress_batches > 0:
            postfix["stress"] = f"{stress_loss.item():.3f}"
        pbar.set_postfix(**postfix)
        last_end = time.perf_counter()

    epoch_sec = time.perf_counter() - epoch_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    nb = max(n_batches, 1)

    return {
        "ctc_loss": total_ctc / nb,
        "stress_loss": total_stress / max(n_stress_batches, 1) if n_stress_batches else 0.0,
        "stress_acc": stress_correct / stress_total if stress_total else 0.0,
        "vad_loss": total_vad / max(n_vad_batches, 1) if n_vad_batches else 0.0,
        "invalid_mass": total_invalid / max(n_invalid_batches, 1) if n_invalid_batches else 0.0,
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
        "nonblank_prob_mean": sum(nonblank_means) / max(len(nonblank_means), 1),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, *, use_bf16, blank_id, stress_active: bool):
    model.eval()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )
    total_ctc = 0.0
    total_stress = 0.0
    n_batches = 0
    n_stress_batches = 0
    stress_correct = 0
    stress_total = 0
    per_class_correct = [0] * NUM_STRESS_LABELS
    per_class_total = [0] * NUM_STRESS_LABELS

    for batch in tqdm(loader, desc="Eval"):
        audio = batch["audio"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        audio_lens = batch["audio_lens"].to(device, non_blocking=True)
        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lens = batch["phoneme_lens"].to(device, non_blocking=True)
        stress_seq = batch["stress_seq"].to(device, non_blocking=True)
        labels = make_labels(phoneme_ids, phoneme_lens)

        with autocast_ctx:
            outputs = model(audio, attention_mask=audio_mask, labels=labels,
                            label_lengths=phoneme_lens)

        total_ctc += outputs["loss"].item()
        n_batches += 1

        if stress_active:
            log_probs_fp32 = outputs["log_probs"].float()
            frame_labels, frame_masks = compute_frame_stress_labels(
                log_probs_fp32, phoneme_ids, stress_seq, phoneme_lens,
                audio_lens, blank_id,
            )
            valid = frame_masks.view(-1)
            if valid.sum() > 0:
                stress_logits = outputs["stress_logits"]
                flat_logits = stress_logits.view(-1, NUM_STRESS_LABELS)[valid].float()
                flat_labels = frame_labels.view(-1)[valid]
                stress_loss = F.cross_entropy(flat_logits, flat_labels)
                total_stress += stress_loss.item()
                n_stress_batches += 1
                preds = flat_logits.argmax(dim=-1)
                stress_correct += (preds == flat_labels).sum().item()
                stress_total += flat_labels.shape[0]
                for c in range(NUM_STRESS_LABELS):
                    mask_c = flat_labels == c
                    per_class_correct[c] += (preds[mask_c] == c).sum().item()
                    per_class_total[c] += mask_c.sum().item()

    per_class_acc = {
        f"val_stress_acc_class_{c}": per_class_correct[c] / per_class_total[c]
        if per_class_total[c] > 0 else 0.0
        for c in range(NUM_STRESS_LABELS)
    }
    return {
        "ctc_loss": total_ctc / max(n_batches, 1),
        "stress_loss": total_stress / max(n_stress_batches, 1) if n_stress_batches else 0.0,
        "stress_acc": stress_correct / stress_total if stress_total else 0.0,
        "per_class": per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../data/audio"))
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-xls-r-2b")
    parser.add_argument("--processor-source", type=str,
                        default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--max-audio-sec", type=float, default=16.0)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--stress-weight", type=float, default=0.3,
                        help="Coefficient on stress CE loss (CTC is the primary task).")
    parser.add_argument("--vad-weight", type=float, default=0.0,
                        help="Coefficient on VAD-anchor BCE loss on the nonblank head. "
                             "Soft regularizer pulling the nonblank decision toward "
                             "earshot VAD output. 0 disables. ~0.1 is a reasonable start.")
    parser.add_argument("--use-features", action="store_true",
                        help="Replace the direct phoneme head with an articulatory-feature "
                             "factorization (panphon 24-feature schema). Each phoneme's "
                             "log-prob is derived by summing per-feature log-probs at the "
                             "indices given by a fixed lookup table.")
    parser.add_argument("--invalid-mass-weight", type=float, default=0.05,
                        help="Coefficient on the invalid-feature-combination penalty. "
                             "Only meaningful with --use-features. Penalizes probability "
                             "mass that the independent feature distributions place on "
                             "phoneme combinations not present in the vocab.")
    parser.add_argument("--stress-warmup-steps", type=int, default=400,
                        help="Disable stress loss for this many steps so phoneme "
                             "model can converge enough for forced alignment to be meaningful. "
                             "At ~440 steps/epoch this gives 1 epoch of CTC-only warmup.")
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints-unified"))
    parser.add_argument("--wandb-project", type=str, default="lexide-pronunciation")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--langs", nargs="*", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    processor = load_processor(args.processor_source)

    feature_table = None
    if args.use_features:
        from .articulatory import build_feature_table
        tokens = [t for t, _ in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]
        feature_table, fstats = build_feature_table(tokens)
        print(f"Articulatory features: {feature_table.shape[1]}-dim, "
              f"covered={fstats['covered']} multi={fstats['multi_segment']} "
              f"empty={fstats['empty_or_special']}")

    model = FactorizedCTCModel(
        model_name=args.model_name,
        vocab_size=len(processor.tokenizer),
        blank_id=processor.tokenizer.pad_token_id,
        feature_table=feature_table,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    print(f"Blank id: {model.blank_id}, vocab size: {model.vocab_size}")

    datasets = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if args.langs is not None and lang_dir.name not in args.langs:
            continue
        phonemes_file = lang_dir / "phonemes.jsonl"
        if phonemes_file.exists():
            ds = StressDataset(phonemes_file, processor.tokenizer,
                               max_audio_sec=args.max_audio_sec,
                               min_rms=args.min_rms)
            print(f"Loaded {lang_dir.name}: {len(ds)} samples")
            datasets.append(ds)
    if not datasets:
        raise RuntimeError(f"No phonemes.jsonl files matched (langs={args.langs}).")

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
        collate_fn=collate_fn_augment, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    steps_per_epoch = len(train_loader)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.backbone_lr},
            {"params": model.head_parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.wandb_project, name="unified-xls-r-2b", config=vars(args))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    hf_api = None
    if args.hf_repo:
        from huggingface_hub import HfApi
        hf_api = HfApi()
        hf_api.create_repo(args.hf_repo, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Stress loss enabled when (epoch-1)*steps_per_epoch >= warmup. The first
        # epoch has noisy alignment from random init — wait for CTC to find phonemes.
        steps_so_far = (epoch - 1) * steps_per_epoch
        stress_active = steps_so_far >= args.stress_warmup_steps
        print(f"\nEpoch {epoch}: stress_active={stress_active} "
              f"(steps_so_far={steps_so_far}, warmup={args.stress_warmup_steps})")

        train_stats = train_epoch(
            model, train_loader, optimizer, device, epoch,
            use_bf16=args.bf16, blank_id=model.blank_id,
            stress_active=stress_active, stress_weight=args.stress_weight,
            vad_weight=args.vad_weight,
            invalid_mass_weight=args.invalid_mass_weight if args.use_features else 0.0,
        )
        val_stats = eval_epoch(
            model, val_loader, device,
            use_bf16=args.bf16, blank_id=model.blank_id, stress_active=stress_active,
        )
        scheduler.step()

        val_loss = val_stats["ctc_loss"]
        print(
            f"Epoch {epoch}: ctc_train={train_stats['ctc_loss']:.4f} "
            f"ctc_val={val_loss:.4f} stress_acc={val_stats['stress_acc']:.3f} "
            f"p_nb={train_stats['nonblank_prob_mean']:.3f} "
            f"wall={train_stats['wallclock_sec']:.1f}s mem={train_stats['peak_mem_gb']:.1f}GB"
        )

        lrs = scheduler.get_last_lr()
        log_dict = {
            "epoch": epoch,
            "train/ctc_loss": train_stats["ctc_loss"],
            "train/stress_loss": train_stats["stress_loss"],
            "train/stress_acc": train_stats["stress_acc"],
            "train/vad_loss": train_stats["vad_loss"],
            "train/invalid_mass": train_stats["invalid_mass"],
            "train/nonblank_prob_mean": train_stats["nonblank_prob_mean"],
            "val/ctc_loss": val_stats["ctc_loss"],
            "val/stress_loss": val_stats["stress_loss"],
            "val/stress_acc": val_stats["stress_acc"],
            "lr/backbone": lrs[0],
            "lr/heads": lrs[1],
            "stress_active": int(stress_active),
            "perf/epoch_wallclock_sec": train_stats["wallclock_sec"],
            "perf/samples_per_sec": train_stats["samples_per_sec"],
            "perf/audio_realtime_factor": train_stats["audio_realtime_factor"],
            "perf/peak_mem_gb": train_stats["peak_mem_gb"],
            "perf/ms_per_batch_data": train_stats["ms_per_batch_data"],
            "perf/ms_per_batch_forward": train_stats["ms_per_batch_forward"],
            "perf/ms_per_batch_forced_align": train_stats["ms_per_batch_forced_align"],
            "perf/ms_per_batch_loss_backward": train_stats["ms_per_batch_loss_backward"],
            "grad/norm_mean": train_stats["grad_norm_mean"],
            "grad/norm_max": train_stats["grad_norm_max"],
            **val_stats["per_class"],
        }
        wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_to_dir(args.save_dir)
            processor.save_pretrained(args.save_dir)
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
