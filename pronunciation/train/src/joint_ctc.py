"""One CTC objective over factorized, vector-valued pronunciation symbols.

The phone model already factorizes blank/nonblank from phone identity.  This
module extends each nonblank symbol with stress and, when applicable, a
language-specific factor.  The Cartesian alphabet is materialized only for a
same-language batch slice and handed to PyTorch's native CTC kernel:

    P(phone, stress, language_factor | nonblank, frame)
      = P(phone | nonblank, frame)
        P(stress | frame)
        P(language_factor | frame)

There is still exactly one blank and one CTC alignment.  Omitting a factor for
a sample marginalizes it out; crucially, unrelated-language examples never
touch that head's graph and cannot update its parameters.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F


def compose_joint_log_probs(
    phone_log_probs: torch.Tensor,
    *,
    phone_blank_id: int,
    factor_logits: list[torch.Tensor],
    factor_weights: list[float] | None = None,
) -> tuple[torch.Tensor, int]:
    """Compose normalized phone and factor distributions for native CTC.

    Returns `(log_probs, joint_blank_id)`.  Joint nonblank ids are a flattened
    Cartesian product in this order: phone, then each factor in `factor_logits`.
    The one joint blank occupies the final class.
    """
    if phone_log_probs.ndim != 3:
        raise ValueError("phone_log_probs must have shape (B,T,V)")

    nonblank = phone_log_probs.clone()
    # The base distribution stores its singular blank in the phone vocabulary.
    # Remove that slot before expanding nonblank symbols; the same blank is
    # appended exactly once after the Cartesian product.
    nonblank[..., phone_blank_id] = nonblank.new_tensor(-1e4)

    weights = factor_weights or [1.0] * len(factor_logits)
    if len(weights) != len(factor_logits):
        raise ValueError("joint factor logit/weight count mismatch")
    for logits, weight in zip(factor_logits, weights):
        if logits.shape[:2] != phone_log_probs.shape[:2]:
            raise ValueError("joint factor frame shape does not match phone logits")
        if weight <= 0:
            raise ValueError("joint factor weight must be positive")
        # A weight is an inverse temperature inside the normalized joint
        # distribution, never a coefficient on a second loss.
        factor_lp = F.log_softmax(logits.float() * weight, dim=-1).to(nonblank.dtype)
        nonblank = nonblank.unsqueeze(-1) + factor_lp.unsqueeze(-2)
        nonblank = nonblank.flatten(start_dim=-2)

    joint_blank_id = nonblank.shape[-1]
    blank = phone_log_probs[..., phone_blank_id].unsqueeze(-1)
    return torch.cat([nonblank, blank], dim=-1), joint_blank_id


def encode_joint_targets(
    phone_targets: torch.Tensor,
    factor_targets: list[torch.Tensor],
    factor_sizes: list[int],
) -> torch.Tensor:
    """Encode aligned target factors using the same Cartesian order."""
    if len(factor_targets) != len(factor_sizes):
        raise ValueError("factor target/size count mismatch")
    joint = phone_targets.long()
    for targets, size in zip(factor_targets, factor_sizes):
        if targets.shape != phone_targets.shape:
            raise ValueError("joint factor target shape does not match phones")
        if size < 1:
            raise ValueError("joint factor size must be positive")
        if ((targets < 0) | (targets >= size)).any():
            raise ValueError(f"joint factor target outside [0, {size})")
        joint = joint * size + targets.long()
    return joint


def joint_ctc_loss(
    *,
    phone_log_probs: torch.Tensor,
    phone_targets: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    phone_blank_id: int,
    stress_logits: torch.Tensor | None,
    stress_weight: float,
    stress_targets: torch.Tensor,
    language_head_logits: dict[str, torch.Tensor],
    language_head_specs: dict[str, dict],
    aligned_targets: dict[str, torch.Tensor],
    factor_available: dict[str, torch.Tensor],
    langs: list[str],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute one joint CTC, grouping only to avoid irrelevant head graphs.

    `stress_logits=None` deliberately omits the stress factor (used during the
    phone-only warmup).  Language heads are selected strictly by each spec's
    `lang`; a group for another language never evaluates or references them.

    `reduction="mean"` matches torch CTC's mean reduction (each example
    normalized by its target length, then averaged).  `reduction="none"`
    returns those per-example losses in original batch order — eval uses this
    for per-language reporting.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    batch_size = phone_log_probs.shape[0]
    if len(langs) != batch_size:
        raise ValueError("language count does not match batch size")

    heads_by_lang: dict[str, list[str]] = defaultdict(list)
    for name, spec in language_head_specs.items():
        heads_by_lang[str(spec["lang"])].append(name)

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, lang in enumerate(langs):
        active_heads = []
        for name in heads_by_lang.get(lang, []):
            target_name = str(language_head_specs[name]["target"])
            availability = factor_available.get(target_name)
            if availability is not None and bool(availability[index].item()):
                active_heads.append(name)
        groups[tuple(sorted(active_heads))].append(index)

    per_sample = phone_log_probs.new_zeros(batch_size, dtype=torch.float32)
    for head_names, indices_list in groups.items():
        indices = torch.tensor(indices_list, device=phone_log_probs.device)
        factor_logits = []
        factor_targets = []
        factor_sizes = []
        factor_weights = []

        if stress_logits is not None:
            factor_logits.append(stress_logits.index_select(0, indices))
            factor_targets.append(stress_targets.index_select(0, indices))
            factor_sizes.append(stress_logits.shape[-1])
            factor_weights.append(stress_weight)

        for name in head_names:
            spec = language_head_specs[name]
            logits = language_head_logits[name]
            target_name = str(spec["target"])
            if target_name not in aligned_targets:
                raise ValueError(f"missing aligned target {target_name!r} for {name}")
            factor_logits.append(logits.index_select(0, indices))
            factor_targets.append(aligned_targets[target_name].index_select(0, indices))
            factor_sizes.append(int(spec["num_labels"]))
            factor_weights.append(float(spec.get("weight", 1.0)))

        group_phone_lp = phone_log_probs.index_select(0, indices)
        joint_lp, joint_blank = compose_joint_log_probs(
            group_phone_lp,
            phone_blank_id=phone_blank_id,
            factor_logits=factor_logits,
            factor_weights=factor_weights,
        )
        group_phone_targets = phone_targets.index_select(0, indices)
        joint_targets = encode_joint_targets(
            group_phone_targets, factor_targets, factor_sizes,
        )
        group_target_lengths = target_lengths.index_select(0, indices)
        group_input_lengths = input_lengths.index_select(0, indices)

        losses = F.ctc_loss(
            joint_lp.transpose(0, 1).float(),
            joint_targets,
            group_input_lengths,
            group_target_lengths,
            blank=joint_blank,
            reduction="none",
            zero_infinity=True,
        )
        # Normalize each example by its target length; every batch index lands
        # in exactly one group, so this scatter fills the whole tensor.
        per_sample[indices] = losses / group_target_lengths.clamp(min=1).to(losses.dtype)

    if reduction == "none":
        return per_sample
    if batch_size == 0:
        return phone_log_probs.new_zeros(())
    return per_sample.mean()
