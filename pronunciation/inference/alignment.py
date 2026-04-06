"""Viterbi forced alignment for CTC output matrices.

Given a CTC log-probability matrix and an expected phoneme sequence,
finds the optimal alignment using Viterbi decoding. Returns per-phoneme
confidence scores based on the frame-level probabilities along the
aligned path.
"""

import numpy as np


def viterbi_align(
    log_probs: np.ndarray,
    target_ids: list[int],
    blank_id: int = 0,
) -> list[dict]:
    """Viterbi forced alignment of CTC output to target sequence.

    Aligns the target phoneme sequence to the CTC output, allowing
    blank tokens between any phonemes and repeated phonemes within
    a segment. Returns per-phoneme alignment with confidence scores.

    Args:
        log_probs: (T, V) log-probability matrix from CTC model.
        target_ids: List of target token IDs (the expected pronunciation).
        blank_id: CTC blank token ID.

    Returns:
        List of dicts, one per target phoneme:
            {"token_id": int, "start_frame": int, "end_frame": int,
             "confidence": float, "frames": int}
    """
    T, V = log_probs.shape
    S = len(target_ids)

    # Build the CTC topology: interleave blanks between tokens
    # States: blank, tok_0, blank, tok_1, blank, ..., tok_{S-1}, blank
    # Total states: 2*S + 1
    n_states = 2 * S + 1
    state_tokens = []
    for i in range(S):
        state_tokens.append(blank_id)  # blank before token i
        state_tokens.append(target_ids[i])  # token i
    state_tokens.append(blank_id)  # final blank

    # Viterbi DP
    # dp[t][s] = log-probability of best path ending at time t in state s
    NEG_INF = -1e30
    dp = np.full((T, n_states), NEG_INF, dtype=np.float64)
    backptr = np.full((T, n_states), -1, dtype=np.int32)

    # Initialization: can start in state 0 (blank) or state 1 (first token)
    dp[0][0] = log_probs[0, state_tokens[0]]
    if n_states > 1:
        dp[0][1] = log_probs[0, state_tokens[1]]

    # Forward pass
    for t in range(1, T):
        for s in range(n_states):
            tok = state_tokens[s]

            # Stay in same state
            best = dp[t - 1][s]
            best_prev = s

            # Transition from previous state (s-1)
            if s >= 1 and dp[t - 1][s - 1] > best:
                best = dp[t - 1][s - 1]
                best_prev = s - 1

            # Skip blank: transition from s-2 (only if current and s-2 are
            # different non-blank tokens)
            if s >= 2 and state_tokens[s] != blank_id and state_tokens[s] != state_tokens[s - 2]:
                if dp[t - 1][s - 2] > best:
                    best = dp[t - 1][s - 2]
                    best_prev = s - 2

            dp[t][s] = best + log_probs[t, tok]
            backptr[t][s] = best_prev

    # Find best final state (last blank or last token)
    final_candidates = [n_states - 1, n_states - 2] if n_states >= 2 else [0]
    best_final = max(final_candidates, key=lambda s: dp[T - 1][s])

    # Backtrace
    path = [best_final]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t][path[-1]])
    path.reverse()

    # Extract per-phoneme segments from the path
    # Map state indices back to target token indices
    segments = []
    for target_idx in range(S):
        token_state = 2 * target_idx + 1  # state index for this token
        frames = [t for t, s in enumerate(path) if s == token_state]

        if frames:
            start = frames[0]
            end = frames[-1]
            # Confidence: average log-prob of the target token over its frames
            frame_probs = [log_probs[t, target_ids[target_idx]] for t in frames]
            confidence = float(np.mean(frame_probs))
        else:
            # Token was skipped (aligned through blanks only)
            start = end = -1
            confidence = NEG_INF

        segments.append({
            "token_id": target_ids[target_idx],
            "start_frame": start,
            "end_frame": end,
            "confidence": confidence,
            "frames": len(frames),
        })

    return segments


def compute_phoneme_scores(
    segments: list[dict],
    blank_threshold: float = -0.5,
) -> list[dict]:
    """Convert raw alignment segments into pronunciation scores.

    Args:
        segments: Output from viterbi_align.
        blank_threshold: Log-prob threshold below which a phoneme is
            flagged as poorly pronounced.

    Returns:
        Same list with added "score" (0-1) and "flagged" (bool) fields.
    """
    for seg in segments:
        if seg["frames"] == 0 or seg["confidence"] < -100:
            seg["score"] = 0.0
            seg["flagged"] = True
        else:
            # Convert log-prob confidence to 0-1 score
            # log_prob is negative; closer to 0 = more confident
            # Typical range: -0.01 (very confident) to -5 (very uncertain)
            score = float(np.clip(np.exp(seg["confidence"]), 0.0, 1.0))
            seg["score"] = score
            seg["flagged"] = seg["confidence"] < blank_threshold

    return segments


def filter_silence_frames(
    log_probs: np.ndarray,
    blank_id: int = 0,
    silence_threshold: float = 0.95,
) -> tuple[np.ndarray, list[int]]:
    """Remove leading/trailing frames where P(blank) is very high.

    These silence frames can distort alignment confidence scores.

    Args:
        log_probs: (T, V) log-probability matrix.
        blank_id: CTC blank token ID.
        silence_threshold: Probability threshold for blank to be
            considered silence.

    Returns:
        Filtered log_probs and list of original frame indices.
    """
    blank_probs = np.exp(log_probs[:, blank_id])
    log_threshold = np.log(silence_threshold)

    # Find first and last non-silence frames
    non_silence = np.where(log_probs[:, blank_id] < log_threshold)[0]
    if len(non_silence) == 0:
        return log_probs, list(range(len(log_probs)))

    start = max(0, non_silence[0] - 1)
    end = min(len(log_probs), non_silence[-1] + 2)

    kept_indices = list(range(start, end))
    return log_probs[start:end], kept_indices
