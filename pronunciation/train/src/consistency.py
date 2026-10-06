"""Detached symmetric consistency of the decoder's frame-factor distributions."""
import math

import torch
from torch.nn import functional as F


def symmetric_kl(a, b):
    """Half KL(stopgrad(a)||b) + half KL(stopgrad(b)||a), per frame."""
    a, b = a.float(), b.float()
    return 0.5 * ((a.detach().exp() * (a.detach() - b)).sum(-1)
                  + (b.detach().exp() * (b.detach() - a)).sum(-1))


def consistency_loss(first, second, lengths, batch, model, *, stress_active, stress_weight):
    """Joint-distribution KL without materializing its Cartesian alphabet.

    Phone log_probs already contain the Bernoulli nonblank factor. Conditional
    stress/prosody divergences are weighted by the *target* nonblank probability
    in each direction, as in the decoder's factored joint distribution. Language
    and availability select supervision only; they never condition the encoder.
    """
    a, b = first["log_probs"].float(), second["log_probs"].float()
    valid = torch.arange(a.shape[1], device=a.device)[None] < lengths[:, None]
    value = symmetric_kl(a, b)
    nb_a = first["nonblank_logit"].float().sigmoid().detach()
    nb_b = second["nonblank_logit"].float().sigmoid().detach()

    def factor(x, y, available, weight):
        x, y = F.log_softmax(x.float() * weight, -1), F.log_softmax(y.float() * weight, -1)
        xy = (x.detach().exp() * (x.detach() - y)).sum(-1)
        yx = (y.detach().exp() * (y.detach() - x)).sum(-1)
        return 0.5 * (nb_a * xy + nb_b * yx) * available.to(a.device)[:, None]

    if stress_active:
        value = value + factor(first["stress_logits"], second["stress_logits"],
                               batch["stress_available"], stress_weight)
        for name, x in first["language_head_logits"].items():
            spec = model.language_head_specs[name]
            available = batch[f"{spec['target']}_available"] & torch.tensor(
                [lang == spec["lang"] for lang in batch["langs"]])
            value = value + factor(x, second["language_head_logits"][name], available, spec["weight"])
    return value.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def warmup_cosine_multiplier(step, warmup_steps, total_steps):
    """LR for the next update, where step is the number already completed."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1 + math.cos(math.pi * progress))
