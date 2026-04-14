"""Train stress prediction head on top of frozen wav2vec2."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

import wandb

from .dataset import StressDataset, collate_fn, NUM_STRESS_LABELS
from .model import Wav2Vec2StressModel, load_processor


def forced_align_frame_labels(
    ctc_logits: torch.Tensor,   # (frames, vocab)
    phoneme_ids: torch.Tensor,  # (num_phonemes,)
    stress_seq: torch.Tensor,   # (num_phonemes,)
    num_frames: int,
    blank_id: int,
) -> torch.Tensor:
    """Use torchaudio's forced_align to map each frame to a phoneme,
    then use stress_seq to derive per-frame stress labels.

    Frames aligned to blank get label STRESS_NONE (0).
    """
    log_probs = F.log_softmax(ctc_logits[:num_frames], dim=-1).unsqueeze(0)  # (1, T, V)
    targets = phoneme_ids.unsqueeze(0)  # (1, L)

    # forced_align returns (frame_alignments, frame_scores)
    # frame_alignments[0, t] = token index (0..L-1) or blank
    # It uses the token IDs we give it, not indices — so we pass through.
    alignments, _ = AF.forced_align(log_probs, targets, blank=blank_id)
    alignments = alignments.squeeze(0)  # (T,)

    frame_stress = torch.zeros(num_frames, dtype=torch.long, device=ctc_logits.device)
    # Walk through alignments, mapping each emitted token to its position in the target
    phoneme_pos = -1
    prev_token = blank_id
    for t in range(num_frames):
        tok = alignments[t].item()
        if tok == blank_id:
            continue
        # When we see a new non-blank token, advance to next phoneme in target
        if tok != prev_token:
            phoneme_pos += 1
        prev_token = tok
        if 0 <= phoneme_pos < stress_seq.shape[0]:
            frame_stress[t] = stress_seq[phoneme_pos]

    return frame_stress


def compute_frame_labels_batch(ctc_logits, phoneme_ids, stress_seq, phoneme_lens,
                               audio_lens, blank_id, conv_ratio):
    """Compute per-frame stress labels for each sample in the batch via forced alignment.

    conv_ratio: how many audio samples per output frame (320 for wav2vec2 @ 16kHz).
    """
    batch_size, max_frames = ctc_logits.shape[:2]
    frame_labels = torch.zeros(batch_size, max_frames, dtype=torch.long, device=ctc_logits.device)
    frame_masks = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=ctc_logits.device)

    for i in range(batch_size):
        n_frames = min(audio_lens[i].item() // conv_ratio, max_frames)
        n_phonemes = phoneme_lens[i].item()
        if n_frames <= 0 or n_phonemes == 0 or n_frames < n_phonemes:
            continue

        try:
            labels = forced_align_frame_labels(
                ctc_logits[i],
                phoneme_ids[i, :n_phonemes],
                stress_seq[i, :n_phonemes],
                n_frames,
                blank_id,
            )
            frame_labels[i, :n_frames] = labels
            frame_masks[i, :n_frames] = True
        except Exception as e:
            # Forced alignment can fail if sequence is too long for the frames
            continue

    return frame_labels, frame_masks


def train_epoch(model, loader, optimizer, blank_id, conv_ratio, device, epoch):
    model.train()
    model.backbone.eval()

    total_loss = 0
    total_frames = 0
    correct = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
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

        # Only compute loss on valid (masked) frames
        valid = frame_masks.view(-1)
        if valid.sum() == 0:
            continue

        flat_logits = stress_logits.view(-1, NUM_STRESS_LABELS)[valid]
        flat_labels = frame_labels.view(-1)[valid]

        loss = F.cross_entropy(flat_logits, flat_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = flat_logits.argmax(dim=-1)
        batch_correct = (preds == flat_labels).sum().item()
        batch_n = flat_labels.shape[0]

        total_loss += loss.item() * batch_n
        correct += batch_correct
        total_frames += batch_n
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_correct/batch_n:.3f}")

    return total_loss / max(total_frames, 1), correct / max(total_frames, 1)


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
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-lv-60-espeak-cv-ft")
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb.init(project=args.wandb_project, config=vars(args))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, blank_id, conv_ratio, device, epoch)
        val_loss, val_acc, per_class_acc = eval_epoch(model, val_loader, blank_id, conv_ratio, device)
        scheduler.step()

        # Log learned layer weights
        weights = torch.softmax(model.layer_weights.detach(), dim=0).cpu().tolist()
        top_layer = max(range(len(weights)), key=lambda i: weights[i])

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} top_layer={top_layer}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "top_layer": top_layer,
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
        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.save_dir / "best.pt"),
            path_in_repo="best.pt",
            repo_id=args.hf_repo,
        )
        print(f"Uploaded to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
