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
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from tqdm import tqdm

import wandb

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from .dataset import (
    StressDataset, collate_fn, collate_fn_augment, NUM_STRESS_LABELS,
    LengthBucketedBatchSampler, get_audio_lengths,
)
from .factorized_ctc import FactorizedCTCModel


def load_processor(model_name: str):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    # Extend the borrowed vocab with phonemes the patched espeak emits that
    # the xlsr-53 vocab didn't have (e.g. German ʏ, Danish ɐ̯ / ʌː). The
    # xls-r-2b backbone has no pretrained phoneme embeddings, so the CTC
    # head's output dim is sized to len(tokenizer) AFTER this add and the
    # new logits are learned from scratch.
    #
    # Imported here (not at module load) so train_unified can run from any
    # CWD without forcing the preprocess script's sys.path tweaks.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from preprocess import VOCAB_EXTENSIONS
    added = tokenizer.add_tokens(sorted(VOCAB_EXTENSIONS))
    if added:
        print(f"Tokenizer vocab extended with {added} new tokens: "
              f"{sorted(VOCAB_EXTENSIONS)}")
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def load_asr_audit_exclusions(
    path: Path | None,
    *,
    min_per: float,
    min_cer: float,
    min_wer: float,
) -> dict[str, dict[str, str]]:
    """Load FLEURS clips whose Whisper audit disagreed with labels.

    The value is lang -> file -> audited expected-text SHA256. StressDataset
    only excludes a row if the current target still matches that hash, so
    repaired labels are not punished by stale audit results.
    """
    if path is None or not path.exists():
        return {}

    exclusions: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok", True):
                continue
            if "per" in rec:
                if float(rec["per"]) < min_per:
                    continue
            else:
                if float(rec.get("cer", 0.0)) < min_cer:
                    continue
                if float(rec.get("wer", 0.0)) < min_wer:
                    continue
            lang = rec.get("lang")
            file = rec.get("file")
            expected_hash = rec.get("expected_sha256")
            if expected_hash is None and rec.get("expected") is not None:
                expected_hash = hashlib.sha256(rec["expected"].encode()).hexdigest()
            if lang and file and expected_hash is not None:
                exclusions.setdefault(lang, {})[file] = expected_hash

    total = sum(len(v) for v in exclusions.values())
    by_lang = ", ".join(f"{lang}={len(exclusions[lang])}" for lang in sorted(exclusions))
    print(f"Loaded ASR-audit exclusions: {total}" + (f" ({by_lang})" if by_lang else ""))
    return exclusions


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
    """Confidence-weighted BCE between sigmoid(nonblank_logit) and VAD probabilities.

    VAD is at 16 ms stride (62.5 fps); wav2vec2 nonblank_logit is at 20 ms
    stride (50 fps). Interpolate per-sample with F.interpolate(mode='linear')
    from the clip's valid VAD length to its valid wav2vec2 length.

    Per-frame loss is weighted by the VAD model's confidence — frames where
    the VAD probability is near 0.5 (the model is unsure whether speech is
    present) contribute less to the loss than frames it's confident about
    (probability near 0 or 1). Weight = |p - 0.5| * 2, so a clear-signal
    frame gets full weight and an ambiguous frame gets ~0.

    Per-clip normalization divides by frame count rather than confidence
    sum, so a clip whose VAD signal is mostly uncertain naturally
    contributes less to the total loss (instead of having its few confident
    frames amplified to represent the whole clip).
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
        confidence = (target - 0.5).abs() * 2  # in [0, 1], 0 at p=0.5
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        losses.append((ce * confidence).sum() / n_f)
    if not losses:
        return torch.zeros((), device=nonblank_logit.device)
    return torch.stack(losses).mean()


def check_finite(name: str, value, *, enabled: bool) -> None:
    """Fail fast with a useful tensor name when debugging NaNs/Infs."""
    if not enabled or not torch.is_tensor(value):
        return
    if torch.isfinite(value).all():
        return
    finite = torch.isfinite(value)
    finite_vals = value.detach()[finite]
    if finite_vals.numel():
        detail = (
            f"finite_min={finite_vals.min().item():.6g} "
            f"finite_max={finite_vals.max().item():.6g}"
        )
    else:
        detail = "no finite values"
    raise FloatingPointError(
        f"Non-finite tensor: {name} shape={tuple(value.shape)} "
        f"dtype={value.dtype} {detail}"
    )


def check_ctc_log_probs(name: str, log_probs: torch.Tensor, model, *, enabled: bool) -> None:
    """Validate CTC log-probs while allowing intentionally masked vocab slots.

    Special tokens are not legal CTC emissions. The model may represent those
    impossible slots as -inf, so the useful debug invariant is: blank + real
    phoneme slots are finite, and the per-frame distribution normalizes.
    """
    if not enabled:
        return

    V = log_probs.shape[-1]
    allowed_nonfinite = torch.zeros(V, dtype=torch.bool, device=log_probs.device)
    masked_slots = getattr(model, "_masked_slots", None)
    if masked_slots is not None:
        allowed_nonfinite[masked_slots.to(log_probs.device)] = True
    allowed_nonfinite[int(model.blank_id)] = False

    bad = ~torch.isfinite(log_probs)
    unexpected = bad & ~allowed_nonfinite.view(*([1] * (log_probs.ndim - 1)), V)
    if unexpected.any():
        idx = unexpected.nonzero()[0].tolist()
        raise FloatingPointError(
            f"Non-finite real CTC slot: {name} shape={tuple(log_probs.shape)} "
            f"dtype={log_probs.dtype} first_bad_index={idx}"
        )

    row_lse = torch.logsumexp(log_probs.float(), dim=-1)
    check_finite(f"{name}/logsumexp", row_lse, enabled=enabled)
    max_err = (row_lse - 0.0).abs().max()
    if max_err > 1e-2:
        raise FloatingPointError(
            f"CTC log-probs not normalized: {name} "
            f"max_abs_logsumexp={max_err.item():.6g}"
        )


def check_parameter_grads(model, *, enabled: bool) -> None:
    """Fail with the parameter name when backward creates a NaN/Inf grad."""
    if not enabled:
        return
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None or torch.isfinite(grad).all():
            continue
        finite = torch.isfinite(grad)
        finite_vals = grad.detach()[finite]
        if finite_vals.numel():
            detail = (
                f"finite_min={finite_vals.min().item():.6g} "
                f"finite_max={finite_vals.max().item():.6g}"
            )
        else:
            detail = "no finite grad values"
        raise FloatingPointError(
            f"Non-finite gradient: {name} shape={tuple(grad.shape)} "
            f"dtype={grad.dtype} {detail}"
        )


def factorial_feature_ctc_loss(
    feature_log_probs: torch.Tensor,
    nonblank_logit: torch.Tensor,
    feature_targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """CTC over vector-valued feature labels with one shared alignment.

    feature_log_probs: (B, T, F, K) log P(feature_f=value | frame)
    nonblank_logit:    (B, T) shared blank/nonblank logit
    feature_targets:   (B, P, F) target feature values in {0, 1, 2}

    For target position p at frame t, the emit score uses the mean feature
    log-probability for that whole target vector:
        score(vec_p | t) = mean_f log P(feature_f = vec_p[f] | t)
    This is the log geometric mean of per-feature probabilities. It keeps the
    "all features should match" product-style signal while keeping the aux loss
    scale comparable to a single categorical head instead of growing with F.

    Then run the usual CTC forward recursion over interleaved blank / emit
    states, with skip transitions blocked for repeated feature vectors.
    """
    B, T_max, F_dim, K = feature_log_probs.shape
    P_max = feature_targets.shape[1]
    active = input_lengths > 0
    if not active.any():
        return feature_log_probs.new_zeros(())

    lp = feature_log_probs.float()
    blank_lp = F.logsigmoid(-nonblank_logit).float()                  # (B, T)
    emit_base = F.logsigmoid(nonblank_logit).float()                  # (B, T)
    neg_inf = torch.finfo(lp.dtype).min

    if P_max == 0:
        blank_mask = torch.arange(T_max, device=lp.device)[None, :] < input_lengths[:, None]
        losses = -(blank_lp * blank_mask).sum(dim=1)
        return losses[active].mean()

    targets = feature_targets.long().clamp(min=0, max=K - 1)           # (B, P, F)

    # emit_scores[b,t,p] = log P(nonblank_t) + mean_f log P(feature_f=target[p,f] | t)
    gather_idx = targets.unsqueeze(1).unsqueeze(-1).expand(B, T_max, P_max, F_dim, 1)
    lp_expanded = lp.unsqueeze(2).expand(B, T_max, P_max, F_dim, K)
    emit_scores = torch.gather(lp_expanded, dim=4, index=gather_idx).squeeze(-1).mean(dim=-1)
    emit_scores = emit_scores + emit_base.unsqueeze(-1)                # (B, T, P)

    p_idx = torch.arange(P_max, device=lp.device)
    p_valid = p_idx.unsqueeze(0) < target_lengths.unsqueeze(1)          # (B, P)

    repeated = torch.zeros(B, P_max, dtype=torch.bool, device=lp.device)
    if P_max > 1:
        repeated[:, 1:] = (targets[:, 1:] == targets[:, :-1]).all(dim=2)
        repeated &= p_valid
    min_input_lengths = target_lengths + repeated.sum(dim=1)
    possible = input_lengths >= min_input_lengths

    S_max = 2 * P_max + 1
    s_idx = torch.arange(S_max, device=lp.device)
    is_emit_state = s_idx % 2 == 1
    emit_pos = s_idx // 2
    state_valid = s_idx.unsqueeze(0) < (2 * target_lengths + 1).unsqueeze(1)

    state_scores = lp.new_full((B, T_max, S_max), neg_inf)
    state_scores[:, :, 0::2] = blank_lp.unsqueeze(-1)
    if P_max > 0:
        state_scores[:, :, 1::2] = emit_scores
    state_scores = torch.where(state_valid.unsqueeze(1), state_scores, state_scores.new_full((), neg_inf))

    skip_mask = torch.zeros(B, S_max, dtype=torch.bool, device=lp.device)
    if P_max > 1:
        emit_positions_for_state = emit_pos.unsqueeze(0).expand(B, S_max)
        emit_positions_safe = emit_positions_for_state.clamp(max=P_max - 1)
        repeated_for_state = torch.gather(repeated, dim=1, index=emit_positions_safe)
        skip_mask = (
            is_emit_state.unsqueeze(0)
            & (s_idx.unsqueeze(0) >= 3)
            & state_valid
            & ~repeated_for_state
        )

    alpha = lp.new_full((B, S_max), neg_inf)
    alpha[:, 0] = state_scores[:, 0, 0]
    if P_max > 0:
        alpha[:, 1] = torch.where(target_lengths > 0, state_scores[:, 0, 1], alpha[:, 1])
    alpha = torch.where(state_valid, alpha, alpha.new_full((), neg_inf))

    for t in range(1, T_max):
        from_prev_state = torch.cat([alpha.new_full((B, 1), neg_inf), alpha[:, :-1]], dim=1)
        from_skip = torch.cat([alpha.new_full((B, 2), neg_inf), alpha[:, :-2]], dim=1)
        from_skip = torch.where(skip_mask, from_skip, alpha.new_full((B, S_max), neg_inf))
        transitions = torch.logsumexp(
            torch.stack([alpha, from_prev_state, from_skip], dim=0),
            dim=0,
        )
        next_alpha = transitions + state_scores[:, t]
        next_alpha = torch.where(state_valid, next_alpha, next_alpha.new_full((), neg_inf))
        frame_active = t < input_lengths
        alpha = torch.where(frame_active.unsqueeze(1), next_alpha, alpha)

    last_emit_state = (2 * target_lengths - 1).clamp(min=0)
    last_blank_state = 2 * target_lengths
    final_emit = torch.gather(alpha, dim=1, index=last_emit_state.unsqueeze(1)).squeeze(1)
    final_blank = torch.gather(alpha, dim=1, index=last_blank_state.unsqueeze(1)).squeeze(1)
    loglik = torch.logsumexp(torch.stack([final_emit, final_blank], dim=0), dim=0)

    losses = -loglik / target_lengths.clamp(min=1).float()
    losses = torch.where(possible, losses, losses.new_zeros(()))
    return losses[active].mean()


def train_epoch(model, loader, optimizer, device, epoch, *, use_bf16,
                blank_id, stress_active: bool, stress_weight: float,
                vad_weight: float, invalid_mass_weight: float,
                feature_aux_weight: float, grad_clip_norm: float,
                debug_finite: bool, max_train_batches: int | None):
    model.train()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )

    total_ctc = 0.0
    total_stress = 0.0
    total_vad = 0.0
    total_invalid = 0.0
    total_aux = 0.0
    n_stress_batches = 0
    n_vad_batches = 0
    n_invalid_batches = 0
    n_aux_batches = 0
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
        check_finite("loss/ctc", ctc_loss, enabled=debug_finite)
        for name, value in outputs.items():
            if name == "log_probs":
                check_ctc_log_probs(f"outputs/{name}", value, model, enabled=debug_finite)
            else:
                check_finite(f"outputs/{name}", value, enabled=debug_finite)
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
                check_finite("loss/stress", stress_loss, enabled=debug_finite)
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
            check_finite("loss/vad", vl, enabled=debug_finite)
            total_vad += vl.item()
            n_vad_batches += 1

        im_loss = torch.zeros((), device=device)
        if invalid_mass_weight > 0 and "off_manifold" in outputs:
            # `off_manifold[b, t]` is -log P(features land in vocab) for the
            # unnormalized feature factorization. Minimize → features prefer
            # combinations that exist as real phonemes. Mask padded frames.
            om = outputs["off_manifold"]                              # (B, T)
            backbone = model.backbone
            n_frames_im = backbone._get_feat_extract_output_lengths(audio_mask.sum(-1)).to(torch.long)
            mask_im = (torch.arange(om.shape[1], device=om.device)[None] < n_frames_im[:, None])
            im_loss = (om * mask_im).sum() / mask_im.sum().clamp(min=1)
            check_finite("loss/off_manifold", im_loss, enabled=debug_finite)
            total_invalid += im_loss.item()
            n_invalid_batches += 1

        aux_loss = torch.zeros((), device=device)
        if feature_aux_weight > 0 and "feature_log_probs" in outputs:
            # Factorial feature CTC. Targets are feature vectors looked up from
            # the phoneme sequence; the dynamic program keeps one shared CTC
            # alignment over whole vectors instead of 24 independent alignments.
            flp = outputs["feature_log_probs"]         # (B, T, F, 3)
            backbone = model.backbone
            n_frames_feat = backbone._get_feat_extract_output_lengths(
                audio_mask.sum(-1)
            ).to(torch.long)
            ft = model.feature_table                  # (V, F), values in {0,1,2}
            safe_ids = phoneme_ids.clamp(min=0, max=ft.shape[0] - 1)
            feat_targets = ft[safe_ids]               # (B, P, F), values in {0,1,2}
            aux_loss = factorial_feature_ctc_loss(
                flp, outputs["nonblank_logit"], feat_targets,
                n_frames_feat, phoneme_lens,
            )
            check_finite("loss/feature_aux", aux_loss, enabled=debug_finite)
            total_aux += aux_loss.item()
            n_aux_batches += 1

        loss = (ctc_loss
                + stress_weight * stress_loss
                + vad_weight * vl
                + invalid_mass_weight * im_loss
                + feature_aux_weight * aux_loss)
        check_finite("loss/total", loss, enabled=debug_finite)

        t_lb = time.perf_counter()
        if debug_finite:
            with torch.autograd.detect_anomaly(check_nan=True):
                loss.backward()
        else:
            loss.backward()
        check_parameter_grads(model, enabled=debug_finite)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip_norm,
        )
        check_finite("grad/norm", grad_norm, enabled=debug_finite)
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
        postfix["g"] = f"{grad_norm.item():.1f}"
        pbar.set_postfix(**postfix)
        last_end = time.perf_counter()
        if max_train_batches is not None and n_batches >= max_train_batches:
            break

    epoch_sec = time.perf_counter() - epoch_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    nb = max(n_batches, 1)

    return {
        "ctc_loss": total_ctc / nb,
        "stress_loss": total_stress / max(n_stress_batches, 1) if n_stress_batches else 0.0,
        "stress_acc": stress_correct / stress_total if stress_total else 0.0,
        "vad_loss": total_vad / max(n_vad_batches, 1) if n_vad_batches else 0.0,
        "off_manifold": total_invalid / max(n_invalid_batches, 1) if n_invalid_batches else 0.0,
        "aux_ctc_loss": total_aux / max(n_aux_batches, 1) if n_aux_batches else 0.0,
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
def eval_epoch(model, loader, device, *, use_bf16, blank_id, stress_active: bool,
               debug_finite: bool = False, max_eval_batches: int | None = None):
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
        check_finite("eval/loss", outputs["loss"], enabled=debug_finite)
        for name, value in outputs.items():
            if name == "log_probs":
                check_ctc_log_probs(f"eval/outputs/{name}", value, model, enabled=debug_finite)
            else:
                check_finite(f"eval/outputs/{name}", value, enabled=debug_finite)

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
        if max_eval_batches is not None and n_batches >= max_eval_batches:
            break

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
    parser.add_argument("--use-aux-features", action="store_true",
                        help="Enable articulatory feature heads alongside the direct "
                             "phoneme head. By default, features participate as "
                             "weighted factors inside the main phoneme CTC emission "
                             "score via --feature-emission-weight. A legacy separate "
                             "factorial feature CTC can additionally be enabled with "
                             "--feature-aux-weight. Mutually exclusive with "
                             "--use-features.")
    parser.add_argument("--feature-aux-weight", type=float, default=0.0,
                        help="Coefficient on the factorial feature CTC loss. Only "
                             "meaningful with --use-aux-features. Normally 0 when "
                             "--feature-emission-weight is active, since features "
                             "already participate in the main CTC emission score.")
    parser.add_argument("--feature-emission-weight", type=float, default=0.3,
                        help="In --use-aux-features mode, include the articulatory "
                             "feature heads as weighted factors inside the main "
                             "phoneme CTC emission score. The direct phoneme head "
                             "has implicit weight 1.0; this weights the mean "
                             "feature-vector log-prob. 0 disables unified emission.")
    parser.add_argument("--regularized-heads", action="store_true",
                        help="Bundle of head-regularization tweaks: (1) K=5 learned softmax "
                             "mixtures over ALL encoder hidden states (49 for XLS-R 2B), each "
                             "concatenated along the feature axis — model picks its own 5 "
                             "depth views; (2) log-mel acoustic side-channel projected to 64 "
                             "dims and concatenated, giving heads direct waveform-derived "
                             "signal; (3) shared Linear(K*H+64, 768)→GELU→Dropout base feeding "
                             "all four heads, so phoneme and feature heads share a learned "
                             "projection (they predict overlapping information by construction).")
    parser.add_argument("--stress-warmup-steps", type=int, default=400,
                        help="Disable stress loss for this many steps so phoneme "
                             "model can converge enough for forced alignment to be meaningful. "
                             "At ~440 steps/epoch this gives 1 epoch of CTC-only warmup.")
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--min-whisper-logprob", type=float, default=-0.7,
                        help="Drop Pimsleur clips whose Whisper avg_logprob is "
                             "below this. Manual audit on eng Pimsleur (153 clips) "
                             "showed ~17%% wrong-rate below -0.7 and ~0%% above. "
                             "FLEURS/Tatoeba rows lack this field and always pass.")
    parser.add_argument("--audit-path", type=Path, action="append",
                        default=None,
                        help="JSONL audit file(s) from Groq/Whisper transcription. "
                             "Pass multiple times to merge exclusions from different "
                             "sources (FLEURS, Tatoeba, etc.). Rows whose current "
                             "target text still matches the audited expected_sha256 "
                             "and have CER/WER/PER above the corresponding threshold "
                             "are excluded from training. If not set, defaults to "
                             "fleurs_asr_exclusions.jsonl + tatoeba_asr_exclusions.jsonl "
                             "if those files exist.")
    parser.add_argument("--audit-min-per", type=float, default=1e-12)
    parser.add_argument("--audit-min-cer", type=float, default=1e-12)
    parser.add_argument("--audit-min-wer", type=float, default=1e-12)
    # Back-compat aliases — older sky yamls pass these.
    parser.add_argument("--fleurs-audit-path", type=Path, default=None,
                        help="Back-compat for --audit-path.")
    parser.add_argument("--fleurs-audit-min-per", type=float, default=None)
    parser.add_argument("--fleurs-audit-min-cer", type=float, default=None)
    parser.add_argument("--fleurs-audit-min-wer", type=float, default=None)
    parser.add_argument("--use-narrowed", action="store_true",
                        help="Train on the narrowed phonemes file (coda-nasal vowels "
                             "nasalized + English flaps) where it exists, else fall back "
                             "to phonemes.jsonl. A full drop-in from espeak_audit/narrow.py.")
    parser.add_argument("--narrowed-name", default="phonemes_narrowed.jsonl",
                        help="Filename of the narrowed file under each lang dir "
                             "(for A/B-ing narrowing variants, e.g. "
                             "phonemes_narrowed_acoustic.jsonl).")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints-unified"))
    parser.add_argument("--wandb-project", type=str, default="lexide-pronunciation")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--langs", nargs="*", default=None)
    parser.add_argument("--source-cap-second",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Per (lang, source) class balancing (default ON). "
                             "For each source class (fleurs / tatoeba / "
                             "pimsleur / tts) find the SECOND-most-populous "
                             "lang's count and use it as the cap for every "
                             "lang in that source. The most-populous lang "
                             "gets random-subsampled down; smaller langs are "
                             "untouched. Without this, English's 87k Pimsleur "
                             "clips would dominate (~38% of the corpus) and "
                             "the model would learn English-pronunciation-"
                             "mostly. With it on (default), eng pimsleur "
                             "drops to ~17k (matching the next biggest, fra). "
                             "Pass --no-source-cap-second to opt out.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=float("inf"),
                        help="Global grad clipping norm. Default preserves legacy no-clip behavior.")
    parser.add_argument("--debug-finite", action="store_true",
                        help="Fail fast if key losses/outputs/grad norm contain NaN/Inf.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Diagnostic: stop each training epoch after this many batches.")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Diagnostic: stop validation after this many batches.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Local path to a checkpoint dir (saved by save_to_dir) to resume "
                             "training from. Must be paired with --resume-epoch. Loads model "
                             "weights only (optimizer/scheduler are reinitialized fresh). "
                             "Use this to continue an in-flight run after a code change.")
    parser.add_argument("--resume-epoch", type=int, default=None,
                        help="Epoch number to start at when --resume-from is given. The "
                             "training loop runs epochs [resume_epoch, epochs] inclusive. "
                             "Both flags must be set together — there's no default, so this "
                             "path can't trigger by accident.")
    args = parser.parse_args()

    if (args.resume_from is None) != (args.resume_epoch is None):
        raise SystemExit("--resume-from and --resume-epoch must be provided together.")
    if args.resume_epoch is not None and args.resume_epoch > args.epochs:
        raise SystemExit(
            f"--resume-epoch {args.resume_epoch} > --epochs {args.epochs}; nothing to do."
        )

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    processor = load_processor(args.processor_source)

    if args.use_features and args.use_aux_features:
        raise SystemExit("--use-features and --use-aux-features are mutually exclusive.")

    feature_table = None
    aux_feature_table = None
    special_token_ids = None
    if args.use_features or args.use_aux_features:
        from .articulatory import build_feature_table, detect_special_token_ids
        tokens = [t for t, _ in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]
        table, fstats = build_feature_table(tokens)
        special_token_ids = detect_special_token_ids(tokens)
        mode_name = "factorized" if args.use_features else "auxiliary"
        print(f"Articulatory features ({mode_name}): {table.shape[1]}-dim. "
              f"covered={fstats['covered']} multi={fstats['multi_segment']} "
              f"empty={fstats['empty']} special={fstats['special']}. "
              f"unique signatures: {fstats['unique_signatures']}/{len(tokens)}")
        if args.use_features:
            feature_table = table
            print(f"Special token ids (masked from phoneme logits): {special_token_ids}")
        else:
            aux_feature_table = table

    if args.resume_from is not None:
        # Resume: load model with backbone + heads + feature_table + regularized
        # config restored from the checkpoint dir. CLI feature/regularized flags
        # are ignored (the checkpoint is the source of truth for arch).
        print(f"Resuming from {args.resume_from}, starting at epoch {args.resume_epoch}")
        model = FactorizedCTCModel.load_from_dir(args.resume_from).to(device)
    else:
        model = FactorizedCTCModel(
            model_name=args.model_name,
            vocab_size=len(processor.tokenizer),
            blank_id=processor.tokenizer.pad_token_id,
            feature_table=feature_table,
            aux_feature_table=aux_feature_table,
            special_token_ids=special_token_ids,
            regularized_heads=args.regularized_heads,
            feature_emission_weight=args.feature_emission_weight if args.use_aux_features else 0.0,
        ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # The CTC head size must match the (extension-augmented) tokenizer, or target
    # ids land outside the head. Fresh models size to len(tokenizer) by construction;
    # a RESUMED checkpoint keeps its own vocab_size — so if VOCAB_EXTENSIONS grew
    # since it was trained (e.g. the narrowed nasal symbols), resuming would feed
    # out-of-range targets. Fail early and clearly rather than deep in CTC.
    if model.vocab_size != len(processor.tokenizer):
        raise SystemExit(
            f"Vocab-size mismatch: model head={model.vocab_size} but tokenizer="
            f"{len(processor.tokenizer)}. The checkpoint predates the current "
            f"VOCAB_EXTENSIONS (narrowed symbols?). Train fresh or expand the head; "
            f"do not resume an old head with new target ids.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    print(f"Blank id: {model.blank_id}, vocab size: {model.vocab_size}")

    # Resolve audit-path inputs. The old `--fleurs-audit-path` is honored
    # as an alias; the new `--audit-path` is the canonical multi-file form.
    # Default: pick up both fleurs and tatoeba files from train/ if they exist.
    audit_paths = list(args.audit_path or [])
    if args.fleurs_audit_path is not None:
        audit_paths.append(args.fleurs_audit_path)
    if not audit_paths:
        train_dir = Path(__file__).resolve().parents[1]
        for fname in ("fleurs_asr_exclusions.jsonl", "tatoeba_asr_exclusions.jsonl",
                      "lang_exclusions.jsonl"):
            p = train_dir / fname
            if p.exists():
                audit_paths.append(p)
    min_per = args.fleurs_audit_min_per if args.fleurs_audit_min_per is not None else args.audit_min_per
    min_cer = args.fleurs_audit_min_cer if args.fleurs_audit_min_cer is not None else args.audit_min_cer
    min_wer = args.fleurs_audit_min_wer if args.fleurs_audit_min_wer is not None else args.audit_min_wer

    asr_exclusions: dict[str, dict[str, str]] = {}
    for p in audit_paths:
        partial = load_asr_audit_exclusions(p, min_per=min_per, min_cer=min_cer, min_wer=min_wer)
        for lang, files in partial.items():
            asr_exclusions.setdefault(lang, {}).update(files)

    datasets = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if args.langs is not None and lang_dir.name not in args.langs:
            continue
        phonemes_file = lang_dir / "phonemes.jsonl"
        if args.use_narrowed and phonemes_file.exists():
            # narrow.py writes a phonemes_narrowed.jsonl for EVERY lang (a broad
            # copy where nothing narrowed), so under --use-narrowed it must exist
            # for every trainable lang. Missing = setup error (narrow.py not run) —
            # fail loud rather than silently train this lang on broad labels.
            narrowed = lang_dir / args.narrowed_name
            if not narrowed.exists():
                raise SystemExit(
                    f"--use-narrowed but {narrowed} is missing for {lang_dir.name}. "
                    f"Run espeak_audit/narrow.py (it writes one per language).")
            phonemes_file = narrowed
        if phonemes_file.exists():
            ds = StressDataset(phonemes_file, processor.tokenizer,
                               max_audio_sec=args.max_audio_sec,
                               min_rms=args.min_rms,
                               excluded_target_hashes=asr_exclusions.get(lang_dir.name),
                               min_whisper_logprob=args.min_whisper_logprob)
            print(f"Loaded {lang_dir.name} from {phonemes_file.name}: {len(ds)} samples")
            datasets.append(ds)

    if args.source_cap_second and datasets:
        # Per-source per-lang second-place balancing. For each source class
        # (pimsleur / fleurs / tatoeba / tts), find the second-most-populous
        # lang's count C and cap every lang in that source at C. The biggest
        # lang in each source gets random-downsampled; smaller langs stay.
        #
        # Concrete: pimsleur has eng=87k, fra=17k → all langs capped at 17k
        # for pimsleur, dropping eng pimsleur by 70k. Other sources (fleurs/
        # tatoeba/tts) are already inherently balanced across langs so this
        # rarely cuts them.
        from collections import defaultdict
        import random as _random
        # Pass 1: count (source, lang)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ds in datasets:
            for s in ds.samples:
                counts[s.get("source") or "unknown"][s["lang"]] += 1
        # Pass 2: per-source caps (= second-highest lang count, or top if
        # only one lang has that source).
        caps: dict[str, int] = {}
        for source, lang_counts in counts.items():
            top = sorted(lang_counts.values(), reverse=True)
            caps[source] = top[1] if len(top) >= 2 else top[0]
        print(f"Source-class caps (second-highest lang count per source): {caps}")
        # Pass 3: build Subsets, sampling each (lang, source) down to its cap.
        rng = _random.Random(42)
        new_datasets = []
        for ds in datasets:
            if not isinstance(ds, StressDataset):
                new_datasets.append(ds)
                continue
            by_source: dict[str, list[int]] = defaultdict(list)
            for i, s in enumerate(ds.samples):
                by_source[s.get("source") or "unknown"].append(i)
            kept: list[int] = []
            for source, idxs in by_source.items():
                cap = caps.get(source, len(idxs))
                if len(idxs) > cap:
                    idxs = rng.sample(idxs, cap)
                kept.extend(idxs)
            kept.sort()
            if len(kept) < len(ds):
                lang = ds.samples[0]["lang"] if ds.samples else "?"
                print(f"  {lang}: subsampled to {len(kept)} (from {len(ds)})")
            new_datasets.append(Subset(ds, kept))
        datasets = new_datasets
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

    # Length-bucketed batch sampler — pre-compute audio lengths for train_ds
    # (cheap: reads cached `n_audio_samples` from each sample dict, no audio
    # I/O). Groups same-length clips into batches so the factorial CTC DP
    # doesn't OOM on a worst-case (long clip × long target) batch.
    train_lengths = get_audio_lengths(train_ds)
    print(f"Audio length stats: min={min(train_lengths)}, max={max(train_lengths)}, "
          f"median={sorted(train_lengths)[len(train_lengths)//2]}")
    train_batch_sampler = LengthBucketedBatchSampler(
        train_lengths, batch_size=args.batch_size, bucket_size_mul=100, seed=42,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_batch_sampler,
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
    # Split params into 4 groups: {backbone, head} × {decay, no_decay}.
    # No decay for: 1D params (biases, LayerNorm weights) and the
    # layer-selection logits. Decaying tiny selector parameters biases
    # them toward zero (= uniform softmax), which would confound the
    # diverse-init / collapse experiment in regularized-heads mode.
    # Biases and LN params are the standard "no decay" carve-out.
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = p.dim() < 2 or name.endswith("layer_weights")
        if is_backbone:
            (backbone_no_decay if no_decay else backbone_decay).append(p)
        else:
            (head_no_decay if no_decay else head_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_decay,    "lr": args.backbone_lr, "weight_decay": 0.01},
            {"params": backbone_no_decay, "lr": args.backbone_lr, "weight_decay": 0.0},
            {"params": head_decay,        "lr": args.head_lr,     "weight_decay": 0.01},
            {"params": head_no_decay,     "lr": args.head_lr,     "weight_decay": 0.0},
        ],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # When resuming, advance the cosine schedule to its value at the start of
    # the resume epoch (epoch numbering is 1-based, so step() runs (resume_epoch-1) times).
    if args.resume_epoch is not None:
        for _ in range(args.resume_epoch - 1):
            scheduler.step()

    wandb.init(project=args.wandb_project, name="unified-xls-r-2b", config=vars(args))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = args.resume_epoch if args.resume_epoch is not None else 1

    hf_api = None
    if args.hf_repo:
        from huggingface_hub import HfApi
        hf_api = HfApi()
        hf_api.create_repo(args.hf_repo, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Re-randomize the per-epoch shuffle inside each length-bucket and
        # across batches. Without this the sampler would yield the same order
        # every epoch (deterministic from the seeded generator).
        train_batch_sampler.set_epoch(epoch)
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
            feature_aux_weight=args.feature_aux_weight if args.use_aux_features else 0.0,
            grad_clip_norm=args.grad_clip_norm,
            debug_finite=args.debug_finite,
            max_train_batches=args.max_train_batches,
        )
        val_stats = eval_epoch(
            model, val_loader, device,
            use_bf16=args.bf16, blank_id=model.blank_id, stress_active=stress_active,
            debug_finite=args.debug_finite,
            max_eval_batches=args.max_eval_batches,
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
            "train/off_manifold": train_stats["off_manifold"],
            "train/aux_ctc_loss": train_stats["aux_ctc_loss"],
            "train/nonblank_prob_mean": train_stats["nonblank_prob_mean"],
            "val/ctc_loss": val_stats["ctc_loss"],
            "val/stress_loss": val_stats["stress_loss"],
            "val/stress_acc": val_stats["stress_acc"],
            "lr/backbone": lrs[0],   # backbone_decay (same LR as backbone_no_decay)
            "lr/heads": lrs[2],      # head_decay (same LR as head_no_decay)
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

        # Always save and push EVERY epoch's checkpoint. HF git history
        # preserves each revision, so worse-than-best epochs are still
        # recoverable by commit hash (`huggingface_hub.snapshot_download(...,
        # revision=<sha>)`). The commit message tags the new-best epochs so
        # they're easy to find. Why save them all: an apparent regression at
        # epoch N can still be the checkpoint we want (e.g. better on a
        # downstream task, or pre-overfitting on a sub-distribution we care
        # about). Cheap to keep, expensive to wish you had back.
        is_new_best = val_loss < best_val_loss
        if is_new_best:
            best_val_loss = val_loss
        model.save_to_dir(args.save_dir)
        processor.save_pretrained(args.save_dir)
        if hf_api is not None:
            tag = " [NEW BEST]" if is_new_best else ""
            print(f"Uploading epoch {epoch} (val_loss={val_loss:.4f}){tag} to HF...")
            hf_api.upload_folder(
                folder_path=str(args.save_dir),
                repo_id=args.hf_repo,
                commit_message=f"epoch {epoch} (val_loss={val_loss:.4f}){tag}",
            )
            print(f"Uploaded to https://huggingface.co/{args.hf_repo}")

    wandb.finish()


if __name__ == "__main__":
    main()
