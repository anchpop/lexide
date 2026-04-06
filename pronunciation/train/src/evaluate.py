"""Evaluation: greedy CTC decoding and Phoneme Error Rate (PER)."""

from collections import defaultdict

import torch

from .vocab import IPAVocab


def edit_distance(hyp: list[int], ref: list[int]) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def ctc_greedy_decode(
    log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int
) -> list[list[int]]:
    """Greedy CTC decoding: collapse repeats, remove blanks.

    Args:
        log_probs: (T, B, vocab_size)
        lengths: (B,)
        blank_id: CTC blank token ID
    Returns:
        List of decoded token ID sequences.
    """
    predictions = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
    decoded = []
    for i in range(predictions.shape[0]):
        seq = predictions[i, : lengths[i]].tolist()
        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        decoded.append(collapsed)
    return decoded


@torch.no_grad()
def evaluate(model, val_loader, vocab: IPAVocab, device=None) -> dict:
    """Run evaluation, return overall PER and per-language breakdown."""
    model.eval()
    total_edits = 0
    total_ref_len = 0
    lang_edits: dict[str, int] = defaultdict(int)
    lang_ref_len: dict[str, int] = defaultdict(int)

    for batch in val_loader:
        mels = batch["mels"].to(device) if device else batch["mels"]
        mel_lengths = batch["mel_lengths"].to(device) if device else batch["mel_lengths"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]
        langs = batch["langs"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device is not None and device.type == "cuda"):
            log_probs, output_lengths = model(mels, mel_lengths)

        log_probs = log_probs.cpu()
        output_lengths = output_lengths.cpu()

        decoded = ctc_greedy_decode(log_probs, output_lengths, vocab.blank_id)

        for i in range(len(decoded)):
            ref = labels[i, : label_lengths[i]].tolist()
            hyp = decoded[i]
            edits = edit_distance(hyp, ref)
            total_edits += edits
            total_ref_len += len(ref)
            lang = langs[i]
            lang_edits[lang] += edits
            lang_ref_len[lang] += len(ref)

    overall_per = total_edits / max(total_ref_len, 1)
    per_by_lang = {
        lang: lang_edits[lang] / max(lang_ref_len[lang], 1)
        for lang in sorted(lang_edits)
    }
    return {"overall_per": overall_per, "per_by_lang": per_by_lang}
