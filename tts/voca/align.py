"""Phoneme-level forced alignment for annotation (adapted from tiny-voca).

The recogniser is the lexide phonetizer (its frame matrix from the
wav2vec2-phoneme batch endpoint). It labels the training recordings only;
none of its weights or outputs enter the TTS model.

Per utterance:
1. CTC Viterbi alignment of the phone sequence (20 ms frames).
2. A label is taken to start its phone, with a measured onset offset
   (calibrated for the lexide phonetizer, see AlignmentSettings). A phone lasts
   from its own onset to the next phone's onset.
3. Silence is found from signal energy at acoustic resolution and given to the
   explicit pause-capable token at that position (BOS, EOS, punctuation, word
   boundary), from after the phone's last emission on. Tokens without a pause get zero frames.
4. Durations are counted on the acoustic frame grid, so they sum to the frame
   count exactly and every phoneme has at least one frame.
"""

from dataclasses import dataclass
import math
import numpy as np

from .text import PHONE_SET, PUNCTUATION, SP, UNK, VOICELESS, VOWEL_SET

ALIGNER_MODEL = "lexide phonetizer (wav2vec2-phoneme batch endpoint frame matrix)"
ALIGNER_RATE = 16000
ALIGNER_STRIDE = 320          # samples at 16 kHz: 20 ms
ALIGNER_CENTER = 200          # a frame's centre: 12.5 ms after its start
ALIGNMENT_VERSION = 4     # 4: a phone holds until its last emission (a stop's release is not the pause after its closure)
PAUSING = (",", ".", "?", "!", ";", ":", "—", "…")


@dataclass(frozen=True)
class AlignmentSettings:
    # Phone onset relative to the start of its first emission frame. -19 ms was
    # tiny-voca's recogniser; scripts/calibrate_onset.py measured -4 ms for the
    # lexide phonetizer on Achernar (2026-11-16, 400 clips; the eng-spike run used -19).
    onset_offset_seconds: float = -0.004
    min_pause_frames: int = 7              # ~75 ms of silence: less stays with the neighbouring phones (stop closures)
    punctuation_min_pause_frames: int = 3  # ~32 ms at punctuation, where a short silence is a real (if brief) pause
    silence_below_peak_db: float = 45.0
    silence_above_floor_db: float = 8.0
    silence_ceiling_below_peak_db: float = 30.0
    energy_window: int = 512

    def to_dict(self):
        return {"alignment_version": ALIGNMENT_VERSION, "aligner": ALIGNER_MODEL, **self.__dict__}


def ctc_viterbi(log_probs, labels, blank=0):
    """Best CTC path. Returns per-frame extended-state indices (token k is state 2k+1).

    Plain numpy reference, also used when torchaudio's implementation is absent.
    """
    frames, count = log_probs.shape[0], len(labels)
    if count < 1:
        raise ValueError("no labels to align")
    states = 2 * count + 1
    extended = np.full(states, blank, dtype=np.int64)
    extended[1::2] = labels
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = extended[3::2] != extended[1:-2:2]
    score = np.full(states, -np.inf)
    score[0] = log_probs[0, blank]
    score[1] = log_probs[0, extended[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        one = np.concatenate(([-np.inf], score[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], score[:-2]))
        two[~skip] = -np.inf
        options = np.stack((score, one, two))
        choice = options.argmax(0)
        score = options[choice, np.arange(states)] + log_probs[t, extended]
        back[t] = choice
    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    if not np.isfinite(score[state]):
        raise ValueError("audio is too short for this phoneme sequence")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return path


def spans_from_states(states, count):
    spans = []
    for index in range(count):
        where = np.flatnonzero(states == 2 * index + 1)
        if not where.size:
            raise ValueError("forced alignment did not visit every phoneme")
        spans.append((int(where[0]), int(where[-1])))
    return spans


def token_spans(log_probs, labels, blank=0):
    """Inclusive ``(first, last)`` emission frames of every label on the best CTC path."""
    count = len(labels)
    repeats = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a == b)
    if log_probs.shape[0] < count + repeats:
        raise ValueError("audio is too short for this phoneme sequence")
    try:
        import torch
        from torchaudio.functional import forced_align
    except ImportError:
        return spans_from_states(ctc_viterbi(log_probs, np.asarray(labels), blank), count)
    path, _ = forced_align(torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))[None],
                           torch.tensor([list(labels)], dtype=torch.long), blank=blank)
    spans, previous = [], blank
    for frame, label in enumerate(path[0].tolist()):
        if label != blank and (previous == blank or label != previous):
            spans.append([frame, frame])
        elif label != blank:
            spans[-1][1] = frame
        previous = label
    if len(spans) != count:
        raise ValueError("forced alignment did not visit every phoneme")
    return [(a, b) for a, b in spans]


def frame_energy_db(wave, hop, window=512):
    """RMS level per acoustic frame, window centred on the frame centre i*hop."""
    frames = len(wave) // hop
    padded = np.pad(np.asarray(wave, dtype=np.float64), (window // 2, window // 2))
    squares = np.concatenate(([0.0], np.cumsum(padded ** 2)))
    starts = np.arange(frames) * hop
    mean = (squares[starts + window] - squares[starts]) / window
    return 10 * np.log10(np.maximum(mean, 1e-12))


def silence_mask(level_db, settings):
    loud, floor = np.percentile(level_db, 95), np.percentile(level_db, 2)
    threshold = min(loud - settings.silence_ceiling_below_peak_db,
                    max(loud - settings.silence_below_peak_db, floor + settings.silence_above_floor_db))
    return level_db < threshold, float(threshold)


def pause_carrier(symbols, group):
    """Index of the token that receives a pause inside a run of non-phone tokens."""
    for wanted in (PAUSING, PUNCTUATION, (SP,)):
        for index in group:
            if symbols[index] in wanted:
                return index
    return group[0]


def assemble(symbols, spans, frames, silent, hop, sample_rate, settings):
    """Durations in acoustic frames for every token (sums to ``frames``).

    Acoustic frame i is centred on sample ``i * hop`` (the convention of the
    stored mel/pitch features and of the synthesis frames).
    """
    phones = [i for i, s in enumerate(symbols) if s in PHONE_SET or s == UNK]
    count = len(phones)
    if count != len(spans) or count < 1:
        raise ValueError("phoneme count differs from alignment")
    stride = ALIGNER_STRIDE / ALIGNER_RATE
    emission = np.array([first for first, _ in spans], dtype=np.float64) * stride
    seconds = np.arange(frames) * hop / sample_rate
    # One protected frame per phone at its emission centre; phones cannot lose it.
    keep = np.clip(np.round((emission + ALIGNER_CENTER / ALIGNER_RATE) * sample_rate / hop).astype(np.int64), 0, frames - 1)
    for k in range(1, count):
        keep[k] = max(keep[k], keep[k - 1] + 1)
    if keep[-1] > frames - 1:
        raise ValueError("audio is too short for this phoneme sequence")
    # ... and a phone the recognizer still hears later holds until then: the closure of a final stop is quiet, its
    # release is not, and the pause (or the end of the sentence) starts after the release.
    last = np.array([b for _, b in spans], dtype=np.float64) * stride
    held = np.clip(np.round((last + ALIGNER_CENTER / ALIGNER_RATE) * sample_rate / hop).astype(np.int64), keep, frames - 1)
    held[:-1] = np.minimum(held[:-1], keep[1:] - 1)
    starts = np.minimum(np.searchsorted(seconds, emission + settings.onset_offset_seconds, side="left"), keep)
    starts[0] = 0
    for k in range(1, count):
        starts[k] = max(starts[k], keep[k - 1] + 1)
    token = np.empty(frames, dtype=np.int64)
    bounds = np.append(starts, frames)
    for k in range(count):
        token[bounds[k]:bounds[k + 1]] = phones[k]
    # A pause runs from the first to the last silent frame between two protected
    # frames. Whatever lies between them (a breath, a stop release) is not part
    # of either phone's steady sound and stays with the pause.
    quiet = np.flatnonzero(silent[:keep[0]])
    if quiet.size:                                      # leading silence -> BOS
        token[:quiet[-1] + 1] = 0
    quiet = np.flatnonzero(silent[held[-1] + 1:]) + held[-1] + 1
    if quiet.size:                                      # trailing silence -> EOS
        token[quiet[0]:] = len(symbols) - 1
    for k in range(count - 1):                          # interior pauses -> pause-capable token
        left, right = phones[k], phones[k + 1]
        if right == left + 1:
            continue
        quiet = np.flatnonzero(silent[held[k] + 1:keep[k + 1]]) + held[k] + 1
        punctuated = any(symbols[i] in PUNCTUATION for i in range(left + 1, right))
        if quiet.size < (settings.punctuation_min_pause_frames if punctuated else settings.min_pause_frames):
            continue
        token[held[k] + 1:quiet[0]] = left
        token[quiet[0]:quiet[-1] + 1] = pause_carrier(symbols, range(left + 1, right))
        token[quiet[-1] + 1:keep[k + 1]] = right
    if np.any(np.diff(token) < 0):
        raise ValueError("alignment is not monotonic")
    durations = np.bincount(token, minlength=len(symbols)).astype(np.int64)
    if durations.sum() != frames or np.any(durations[phones] < 1):
        raise ValueError("durations do not cover the acoustic frames")
    return durations


def quality(symbols, durations, spans, log_probs, labels, voiced, level_db, hop, sample_rate):
    """Label-free checks of one alignment against the recording itself."""
    durations = np.asarray(durations)
    ends = np.cumsum(durations)
    starts = ends - durations
    token_probability = np.array([float(np.exp(log_probs[a:b + 1, label]).mean()) for (a, b), label in zip(spans, labels)])
    phone_frames = np.array([d for s, d in zip(symbols, durations) if s in PHONE_SET or s == UNK])

    def voiced_share(wanted):
        total = hits = 0
        for symbol, start, end in zip(symbols, starts, ends):
            if wanted(symbol) and end - start >= 3:      # interior frames only
                total += end - start - 2
                hits += float(voiced[start + 1:end - 1].sum())
        return round(hits / total, 4) if total else None

    pause = np.zeros(len(voiced), dtype=bool)
    for symbol, start, end in zip(symbols, starts, ends):
        if symbol not in PHONE_SET and symbol != UNK:
            pause[start:end] = True
    speech_seconds = float((~pause).sum() * hop / sample_rate)
    return {
        "token_probability_mean": round(float(token_probability.mean()), 4),
        "weak_phone_fraction": round(float((token_probability < 0.1).mean()), 4),
        "max_phone_frames": int(phone_frames.max()),
        "one_frame_phone_fraction": round(float((phone_frames == 1).mean()), 4),
        "phones_per_second": round(len(phone_frames) / max(speech_seconds, 1e-6), 2),
        "pause_frames": int(pause.sum()),
        "leading_frames": int(durations[0]),
        "trailing_frames": int(durations[-1]),
        "vowel_voiced": voiced_share(lambda s: s in VOWEL_SET),
        "voiceless_voiced": voiced_share(lambda s: s in VOICELESS),
        "pause_voiced": round(float(voiced[pause].mean()), 4) if pause.any() else None,
        "pause_level_db": round(float(level_db[pause].mean()), 1) if pause.any() else None,
        "speech_level_db": round(float(np.percentile(level_db[~pause], 95)), 1) if (~pause).any() else None,
    }


def align_utterance(recognizer, phonemized, wave, voiced, config, settings):
    """Return ``(durations, qc)`` for one recording or raise ``ValueError``."""
    symbols = phonemized.symbols
    frames = len(wave) // config.hop
    if len(wave) != frames * config.hop or len(voiced) != frames:
        raise ValueError("waveform length must be frames * hop")
    labels = [recognizer.label(s) for s in symbols if s in PHONE_SET or s == UNK]
    log_probs = recognizer.log_probs(wave, config.sample_rate)
    spans = token_spans(log_probs, labels, recognizer.blank)
    level = frame_energy_db(wave, config.hop, settings.energy_window)
    silent, threshold = silence_mask(level, settings)
    durations = assemble(symbols, spans, frames, silent, config.hop, config.sample_rate, settings)
    qc = quality(symbols, durations, spans, log_probs, labels, np.asarray(voiced) > 0.5, level, config.hop, config.sample_rate)
    qc["silence_threshold_db"] = round(threshold, 1)
    return durations.tolist(), qc
