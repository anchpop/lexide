"""The deployed path, scored per word against the recording.

Every validation interval, the weights that ship (the EMA) synthesize whole
held-out sentences the way the app does (``voca.speak``: predicted durations,
pitch, voicing and energy, the stored voice style, speak's pause floors). That
waveform and the recording are measured with the same tracker
(``voca.prepare.waveform_prosody``: Praat pitch, despiked as the training
labels are; RMS energy in training units), each at its own timing. Per word
occurrence (a repeated word is two occurrences):

* excursion: P90 - P10 of the voiced pitch in semitones (4+ voiced frames);
* prominence: within the sentence, the z-scored sum of peak energy, peak
  pitch and longest vowel, an acoustic proxy for the word that carries the accent.

Phrases end at punctuation (``voca.text.PHRASE_BREAKS``). Reported:

    main_word_agreement      share of phrases (2+ words) whose most prominent word is the teacher's
    rank_correlation         mean within-phrase Spearman correlation of prominence (phrases of 3+ words)
    accent_excursion_error   mean |synthesis - teacher| excursion on the teacher's main word, semitones
    other_excursion_error    the same on the phrases' other words
    accent_excursion         median excursion of the synthesis on the teacher's main word (teacher: ``reference``)
    final_movement_error     mean |synthesis - teacher| of the phrase-final word's signed pitch movement
                             (median of its last third of voiced frames minus its first third), semitones
    final_direction_agreement  share of phrase ends the teacher moves 1+ semitone where the synthesis moves the same way
                             (the more robust of the two: on the recordings, Praat and pYIN disagree by 6+ semitones
                             on the final movement at 11 of 60 validation phrase ends, often creaky, and on 30 of
                             366 word excursions)
    pause_presence_accuracy  pause or not at the pause-capable tokens inside the utterance, against the alignment
    pause_length_ratio       their total synthesized length over the recording's
    vowel_voicing            share of vowel frames the tracker finds voiced (teacher: ``reference``)

The first three are the selection measures; pauses and voicing are regression checks.

Crackle, on the waveform (``crackle``; after Codex's investigation of the generator's
background crackle). Events are peaks of the absolute second difference ("curvature")
8 robust standard deviations above its sliding local median and above -65 dBFS,
merged within 1 ms. Pause interiors are frames of non-phone tokens with two more
such frames on either side (the synthesis at its own durations, the teacher at the
alignment's).

    pause_events_per_second  events in pause interiors per second of them
    pause_event_level_db     their median curvature, dBFS
    pause_level_db           the pause interiors' median frame RMS, dBFS (a quieter background inflates ratios to it,
                             hence absolute levels)
    pause_event_frame_share  share of those events in the busiest 4 of the 32 8-sample positions of the
                             256-sample frame (12.5% if events ignore the frame grid; few events read higher by chance)
    speech_curvature_ratio   in speech-level frames (-30 to -15 dBFS), the mean frame-normalized curvature by
                             sample index mod 4, largest over smallest (the generator's iSTFT hop is 4)

    python -m voca.evaluate --checkpoint runs/x/inference.pt --validation features/eng_gemini_10h_v2/validation.jsonl
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy import ndimage, signal
import torch

from .data import AnnotatedDataset, despiked
from .model import PHONE_TABLE, word_layout
from .prepare import waveform_prosody
from .speak import COMMA_PAUSE, STOP_PAUSE, load_model
from .text import PHRASE_BREAKS, SYMBOLS, VOWEL_SET

SEMITONES = 12 / math.log(2)
PHONES = PHONE_TABLE.numpy()


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-20))


def sharp_events(curvature):
    """Sample indices of the peaks of ``curvature`` 8 robust standard deviations above
    the median of the 256 samples around them (taken every 24 samples), above -65 dBFS,
    at least 1 ms (24 samples) apart."""
    centres = np.arange(0, len(curvature), 24)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(curvature, (128, 127), mode="reflect"), 256)[::24]
    scale = 8 * 1.4826 * np.interp(np.arange(len(curvature)), centres, np.median(windows, axis=1))
    return signal.find_peaks(curvature, height=np.maximum(scale, 10 ** (-65 / 20)), distance=24)[0]


def crackle(wave, tokens, durations, hop):
    """One sentence's parts of the crackle measures (module docstring; ``summarize_crackle``)."""
    frames = min(int(durations.sum()), len(wave) // hop)
    wave = wave[:frames * hop]
    pause = np.convolve(~np.repeat(PHONES[tokens], durations)[:frames], np.ones(5), "same") == 5
    curvature = np.abs(np.diff(wave, 2, prepend=wave[:1].repeat(2)))
    events = sharp_events(curvature)
    events = events[pause[events // hop]]
    level = db(np.mean(wave.reshape(frames, hop) ** 2, axis=1))
    speech = curvature.reshape(frames, hop)[(level >= -30) & (level < -15)]
    return {"pause_levels": level[pause], "positions": events % hop, "levels": db(curvature[events] ** 2),
            "phases": (speech / np.maximum(speech.mean(1, keepdims=True), 1e-12)).reshape(-1, 4).sum(0)}


def summarize_crackle(parts, config):
    """The crackle measures over sentences' ``crackle`` parts."""
    pause_levels, positions, levels = (np.concatenate([p[key] for p in parts]) for key in ("pause_levels", "positions", "levels"))
    phases = sum(p["phases"] for p in parts)
    bins = np.bincount(positions // 8, minlength=config.hop // 8)
    busiest = np.sort(bins)[len(bins) - len(bins) // 8:]
    return {"pause_events_per_second": len(positions) / max(len(pause_levels) * config.hop / config.sample_rate, 1e-9),
            "pause_event_level_db": float(np.median(levels)) if len(levels) else math.nan,
            "pause_level_db": float(np.median(pause_levels)) if len(pause_levels) else math.nan,
            "pause_event_frame_share": float(busiest.sum() / len(positions)) if len(positions) else math.nan,
            "speech_curvature_ratio": float(phases.max() / phases.min()) if phases.min() > 0 else math.nan}


CLOCK_DELAYS = 4
SONORANT_LETTERS = set("nmŋɲɴlɹɾrwjʋɻɭʎɥ")


def clock_sensitivity(generator, generator_input, tokens, durations, voiced, config):
    """How much of a synthesized sentence follows the generator's own output grid (Codex's
    white-box detector, phase_score.py, with 4 delays instead of 16). The same excitation and
    input are rendered at ``CLOCK_DELAYS`` delays spread over the generator hop, and realigned
    (``Generator.render``): the voice is identical in all of them, images of it around the
    synthesis frame rate are not. Their variance in 1.2-4.4 kHz relative to the local
    broadband power, over 85 ms of voiced vowels and sonorants, in dB; the sentence's upper
    decile (NaN without such stretches). Codex reported -10 dB as "a tenth of local speech
    power is grid-dependent"; the 12k multi-voice English checkpoint peaked at -4.7 dB."""
    x, style, source = generator_input
    step = generator.hop // CLOCK_DELAYS
    renders = np.stack([generator.render(x, style, source, torch.tensor([d], device=x.device) if d else None)[0].float().cpu().numpy()
                        for d in range(0, generator.hop, step)])
    frames = min(int(durations.sum()), renders.shape[1] // config.hop)
    renders = renders[:, :frames * config.hop]
    band = signal.sosfiltfilt(signal.butter(4, [1200, 4400], fs=config.sample_rate, btype="bandpass", output="sos"), renders, axis=-1)
    smooth = lambda a: ndimage.uniform_filter1d(a.reshape(frames, config.hop).mean(1), 8)
    power, variance = smooth(np.square(renders).mean(0)), smooth(band.var(0))
    phone = np.repeat([SYMBOLS[t] in VOWEL_SET or SYMBOLS[t][:1] in SONORANT_LETTERS for t in tokens], durations)[:frames]
    use = (ndimage.uniform_filter1d((phone & voiced[:frames]).astype(float), 8) > 0.87) & (power > 10 ** (-33 / 10))
    return float(np.percentile(10 * np.log10(np.maximum(variance[use], 1e-17) / power[use]), 90)) if use.any() else math.nan


def occurrences(tokens, durations):
    """Word occurrences (``voca.model.word_layout``) in token order: their phrase, frame span and longest vowel (frames)."""
    word = word_layout(tokens[None], torch.ones(1, len(tokens), dtype=torch.bool)).index[0].tolist()
    tokens = tokens.tolist()
    bounds = np.concatenate(([0], np.cumsum(durations)))
    spans, phrase = {}, 0
    for i, (token, w) in enumerate(zip(tokens, word)):
        if SYMBOLS[token] in PHRASE_BREAKS and spans and spans[max(spans)]["phrase"] == phrase:
            phrase += 1
        if w < 0:
            continue
        span = spans.setdefault(w, {"phrase": phrase, "start": int(bounds[i]), "vowel": 0})
        span["end"] = int(bounds[i + 1])
        if SYMBOLS[token] in VOWEL_SET:
            span["vowel"] = max(span["vowel"], int(durations[i]))
    return [spans[w] for w in sorted(spans)]


def measure(prosody, spans, config):
    """Per word [N, 5]: excursion, peak pitch (semitones), peak energy (dB), longest vowel, final movement.
    Pitch the training labels' despiker rejects (isolated octave jumps) is left out."""
    trusted = despiked(torch.from_numpy(prosody[:, 0]), torch.from_numpy(prosody[:, 1] > 0.5)).numpy()
    rows = []
    for span in spans:
        frames = prosody[span["start"]:span["end"]]
        pitch = frames[trusted[span["start"]:span["end"]], 0] * SEMITONES
        log_rms = frames[:, 2] * config.energy_scale + config.energy_mean
        third = len(pitch) // 3
        rows.append((np.subtract(*np.quantile(pitch, [0.9, 0.1])) if len(pitch) >= 4 else np.nan,
                     pitch.max() if len(pitch) else np.nan,
                     log_rms.max() * 20 / math.log(10) if len(frames) else np.nan, span["vowel"],
                     np.median(pitch[-third:]) - np.median(pitch[:third]) if third >= 2 else np.nan))
    return np.array(rows, dtype=float).reshape(-1, 5)


def prominence(words):
    """Within-sentence z-scored sum of peak pitch, peak energy and longest vowel."""
    x = words[:, 1:4]
    x = np.where(np.isnan(x), np.fmin.reduce(x, axis=0), x)   # a word without pitch counts as the lowest
    return ((x - x.mean(0)) / (x.std(0) + 1e-9)).sum(1)


def main_words(words, phrases):
    """Index of the most prominent word of every phrase with 2+ words."""
    score = prominence(words)
    return [members[np.argmax(score[members])] for members in (np.flatnonzero(phrases == p) for p in np.unique(phrases))
            if len(members) >= 2]


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def compare(teacher, synthesis, phrases):
    """Sums and counts of the measures for one sentence (teacher and synthesis words aligned)."""
    out = {}

    def add(name, value, count=1):
        total = out.setdefault(name, [0.0, 0])
        total[0] += float(value)
        total[1] += count
    t_score, s_score = prominence(teacher), prominence(synthesis)
    for phrase in np.unique(phrases):
        members = np.flatnonzero(phrases == phrase)
        last = members[-1]
        if np.isfinite(teacher[last, 4]) and np.isfinite(synthesis[last, 4]):
            add("final_movement_error", abs(synthesis[last, 4] - teacher[last, 4]))
            if abs(teacher[last, 4]) >= 1:
                add("final_direction_agreement", np.sign(synthesis[last, 4]) == np.sign(teacher[last, 4]))
        if len(members) < 2:
            continue
        main = members[np.argmax(t_score[members])]   # as main_words picks it
        add("main_word_agreement", main == members[np.argmax(s_score[members])])
        if len(members) >= 3:
            add("rank_correlation", np.corrcoef(rank(t_score[members]), rank(s_score[members]))[0, 1])
        for i in members:
            if np.isfinite(teacher[i, 0]) and np.isfinite(synthesis[i, 0]):
                add("accent_excursion_error" if i == main else "other_excursion_error", abs(synthesis[i, 0] - teacher[i, 0]))
    return out


class DeployedEvaluation:
    """The measures above over a fixed set of ``AnnotatedDataset`` items. The
    recordings are measured once; each call synthesizes with ``model``.

    With more than one voice, the measures are pooled over all sentences and the
    ``VOICE_MEASURES`` are also reported per voice (``kore/...``). Every voice
    reads the same validation sentences, so voices compare on the same text."""

    PHRASE_MEASURES = ("main_word_agreement", "rank_correlation", "accent_excursion_error", "other_excursion_error",
                       "final_movement_error", "final_direction_agreement")
    VOICE_MEASURES = ("main_word_agreement", "accent_excursion_error", "final_direction_agreement", "vowel_voicing",
                      "pause_presence_accuracy", "pause_events_per_second", "clock_sensitivity_db")
    CLOCK_SENTENCES = 8      # per voice: ``clock_sensitivity`` renders each of them CLOCK_DELAYS times

    def __init__(self, dataset, indices, config):
        self.config, self.items, sounds, self.groups = config, [], [], {}
        for i in indices:
            row, item = dataset.rows[i], dataset[i]
            spans = occurrences(item["tokens"], row["durations"])
            teacher = waveform_prosody(item["wave"].numpy(), config)
            words, phrases = measure(teacher, spans, config), np.array([s["phrase"] for s in spans])
            voice = self.groups.setdefault(dataset.speakers[i], [])
            self.items.append({"tokens": item["tokens"], "lexical": item["lexical"], "speaker": item["speaker"],
                               "durations": np.array(row["durations"]), "teacher": teacher, "words": words, "phrases": phrases,
                               "main": main_words(words, phrases), "clock": len(voice) < self.CLOCK_SENTENCES})
            sounds.append(crackle(item["wave"].numpy(), item["tokens"].numpy(), self.items[-1]["durations"], config.hop))
            voice.append(len(self.items) - 1)
        if len(self.groups) == 1:     # one voice: the pooled measures are its own
            self.groups = {}
        self.reference = {"sentences": len(self.items),
                          "accent_excursion": float(np.nanmedian([e for item in self.items for e in item["words"][item["main"], 0]])),
                          "vowel_voicing": self.vowel_voicing([(item["tokens"].numpy(), item["durations"], item["teacher"]) for item in self.items]),
                          "comma_pause": COMMA_PAUSE, "stop_pause": STOP_PAUSE, **summarize_crackle(sounds, config)}

    @staticmethod
    def vowel_voicing(renders):
        voiced = frames = 0
        for tokens, durations, prosody in renders:
            vowel = np.repeat([SYMBOLS[t] in VOWEL_SET for t in tokens], durations)[:len(prosody)]
            voiced += int((prosody[:len(vowel), 1][vowel] > 0.5).sum())
            frames += int(vowel.sum())
        return voiced / max(frames, 1)

    @torch.inference_mode()
    def __call__(self, model, device):
        parts = [self.render(model, item, device) for item in self.items]
        metrics = self.summarize(parts)
        for name, members in self.groups.items():
            summary = self.summarize([parts[i] for i in members])
            metrics.update({f"{name}/{key}": summary[key] for key in self.VOICE_MEASURES})
        return metrics

    def render(self, model, item, device):
        """One sentence synthesized and measured against its recording: its parts of every measure."""
        c, tokens = self.config, item["tokens"][None].to(device)
        wave, lengths, details = model.synthesize(tokens, item["lexical"][None].to(device), torch.tensor([tokens.shape[1]], device=device),
                                                  torch.tensor([item["speaker"]], device=device), comma_pause=COMMA_PAUSE, stop_pause=STOP_PAUSE)
        durations = details["durations"][0].cpu().numpy()
        wave = wave[0, :int(lengths[0])].float().cpu().numpy()
        prosody = waveform_prosody(wave, c)
        synthesis = measure(prosody, occurrences(item["tokens"], durations), c)
        pausable = ~model.phone_tokens[tokens[0]].cpu().numpy()
        pausable[[0, -2, -1]] = False     # utterance edges, and the punctuation before EOS (no floor there)
        recorded = item["durations"]
        clock = (clock_sensitivity(model.decoder.generator, details["generator"], item["tokens"].numpy(), durations, prosody[:, 1] > 0.5, c)
                 if item["clock"] and "generator" in details else math.nan)
        return {"totals": compare(item["words"], synthesis, item["phrases"]), "main": synthesis[item["main"], 0], "clock": clock,
                "pauses": {"agree": int(((durations > 0) == (recorded > 0))[pausable].sum()), "count": int(pausable.sum()),
                           "synthesized": int(durations[pausable].sum()), "recorded": int(recorded[pausable].sum())},
                "render": (item["tokens"].numpy(), durations, prosody), "sound": crackle(wave, item["tokens"].numpy(), durations, c.hop)}

    def summarize(self, parts):
        totals, pauses = {}, {}
        for part in parts:
            for name, (value, count) in part["totals"].items():
                total = totals.setdefault(name, [0.0, 0])
                total[0] += value
                total[1] += count
            for name, value in part["pauses"].items():
                pauses[name] = pauses.get(name, 0) + value
        main, clock = np.concatenate([part["main"] for part in parts]), np.array([part["clock"] for part in parts])
        metrics = {name: totals[name][0] / totals[name][1] if name in totals else math.nan for name in self.PHRASE_MEASURES}
        metrics.update(accent_excursion=float(np.nanmedian(main)) if np.isfinite(main).any() else math.nan,
                       clock_sensitivity_db=float(np.nanmedian(clock)) if np.isfinite(clock).any() else math.nan,
                       vowel_voicing=self.vowel_voicing([part["render"] for part in parts]),
                       pause_presence_accuracy=pauses["agree"] / max(pauses["count"], 1),
                       pause_length_ratio=pauses["synthesized"] / max(pauses["recorded"], 1),
                       **summarize_crackle([part["sound"] for part in parts], self.config))
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    dataset = AnnotatedDataset(args.validation, model.config, cache=False)
    evaluation = DeployedEvaluation(dataset, range(len(dataset)), model.config)
    print(json.dumps({"reference": evaluation.reference, "deployed": evaluation(model, args.device)}, indent=1))


if __name__ == "__main__":
    main()
