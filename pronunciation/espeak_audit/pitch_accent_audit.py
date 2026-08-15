"""Check Japanese pitch-accent labels against the F0 that was actually produced.

The accent labels in ``phoneme_backend_pyopenjtalk.jsonl`` are OpenJTalk's
*citation* accent: what the dictionary says the words are accented like, given
a transcript. Two things routinely make that wrong for a given clip:

  - the transcript names the wrong word. Pitch accent is lexical and Japanese
    homophones are exactly where it contrasts (箸 HL vs 橋 LH), so a Whisper
    transcript that picked the kanji from its language-model prior yields the
    accent of a word that was never said;
  - the clip is a piece of a longer prosodic unit, so the accent phrase
    OpenJTalk builds never existed as spoken.

Either way the model is handed a target that contradicts the audio, and the
cheapest way for a head to survive contradictory targets is to stop listening
and emit the prior. This pass removes those rows.

Method, following the repo's anti-circularity rule: the **model may only place
time boundaries**, and the *decision* comes from measuring the signal. Modal
force-aligns each clip with the pinned aligner (the one GPU-only step); every
number that decides anything is computed locally by parselmouth. F0 is the one
measurement this pipeline can make robustly at the per-token level — unlike
formants, it is near-categorical and speaker-normalizing by construction, since
every test here compares two *adjacent moras of one speaker in one phrase*.

The test is deliberately local. Japanese has downdrift, so absolute F0 height
across a phrase means little; a labelled L→H rise or H→L fall between adjacent
moras is a large, direction-carrying event that downdrift cannot manufacture.
A clip loses its accent labels only on positive evidence that its contour is
wrong — too many labelled transitions measured moving the opposite way. A clip
the measurement cannot judge keeps its labels (see MAX_CONTRA_FRAC for why the
earlier require-confirmation rule was retired).

Two stages, so that retuning costs nothing:

  measure   align on Modal + measure F0 locally -> per-clip cache
  verdict   read the cache, apply thresholds -> train/jpn_pitch_accent_exclusions.jsonl

Run with scripts/py-linux.sh (needs modal + parselmouth + soundfile + numpy):

  scripts/py-linux.sh espeak_audit/pitch_accent_audit.py measure
  scripts/py-linux.sh espeak_audit/pitch_accent_audit.py verdict --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "train" / "scripts"))
sys.path.insert(0, str(REPO / "espeak_audit"))
import run_audit as RA  # noqa: E402  (load_vocab / MASKED_IDS, resolved from the pinned commit)
from modal_aligner import MODEL_REVISION  # noqa: E402  (single source of the pinned commit)

MODEL = f"vad-clean@{MODEL_REVISION[:12]}"
CACHE = REPO / "espeak_audit" / ".cache" / "pitch_accent" / MODEL
AUDIO = REPO / "data" / "audio"
LANG = "jpn"
DEFAULT_EXCLUSIONS = REPO / "train" / "jpn_pitch_accent_exclusions.jsonl"

# A mora needs this many voiced pitch frames inside its span before its median
# is worth anything. At the 5 ms step below that is 15 ms of voicing — enough
# to exclude a devoiced vowel or a span that is mostly the preceding closure.
MIN_VOICED_FRAMES = 3

# Semitones. A Tokyo downstep is a large event (commonly 4+ st); the phrase
# initial rise is smaller. AGREE is the bar for "the audio shows this", CONTRA
# the bar for "the audio shows the opposite". The gap between them is the
# abstain band, where the measurement simply is not telling us anything.
AGREE_ST = 0.75
CONTRA_ST = 1.5

# A clip loses its accent labels only when more than MAX_CONTRA_FRAC of the
# transitions the audio can judge move the wrong way — contradiction-ONLY
# gating. Positive confirmation is reported but not required.
#
# The first pass additionally required MIN_AGREE=1 confirmed transition and
# excluded "unverifiable" clips (agree=0) as f0_unverifiable. Measurement
# showed that rule selects for parselmouth-LEGIBLE prosody, not for correct
# prosody: clips whose falls are small, early, or realized a mora late (Tokyo
# peak delay — atamadaka's labelled mora-1→2 fall measures a median −0.18 st,
# with the real fall at 2→3) starve the test and get withheld. That enriched
# atamadaka 1.56–1.71× among dropped phrases corpus-wide and 4.19× on
# single-phrase clips — exactly the isolated-word shape of the minimal-pair
# eval. Dropping *contradicted* clips removes misleading targets and is cheap
# (data philosophy: tolerate false positives). Dropping *unverifiable* clips
# removed good data with a systematic pattern bias — shaping the training
# distribution around this file's measurement blind spots, which is the soft
# version of training the model to agree with parselmouth. On the 2026-08
# cache, contradiction-only at these thresholds keeps ~77% of clips vs ~69%
# under the old rule.
#
# Zero tolerance on contradictions does not work and the measured data says
# why: across the corpus a labelled transition moves the right way 63% of the
# time and the wrong way 9%, the rest landing in the abstain band. Some of
# that 9% is real mislabelling and some is microprosody, consonantal F0
# perturbation and alignment slop. With several testable transitions per clip,
# demanding that none contradict throws out two thirds of the corpus on
# measurement noise. A *rate* separates "this clip's contour is wrong" from
# "one mora wobbled".
MAX_CONTRA_FRAC = 0.20


def cache_path(file: str, phon_key: str) -> Path:
    h = hashlib.sha256(f"{MODEL}/{LANG}/{file}/{phon_key}".encode()).hexdigest()
    return CACHE / h[:2] / f"{h}.json"


def phon_key(phonemes: list[str]) -> str:
    return hashlib.sha256("\x00".join(phonemes).encode()).hexdigest()[:16]


def load_rows(labels_path: Path, limit: int | None = None):
    """Rows carrying usable accent labels — the only ones worth measuring."""
    rows = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not d.get("pitch_accent"):
            continue
        rows.append(d)
        if limit and len(rows) >= limit:
            break
    return rows


# ---------------------------------------------------------------- measurement

def clip_f0(audio, sr):
    """Voiced F0 track for a clip, in semitones, with the times of each frame.

    Two passes: the default 75-600 Hz window first, then a window re-centred on
    the clip's own median. That is standard Praat practice and it matters here,
    because a single octave error inside one mora would read as a 12-semitone
    "fall" and condemn a correctly-labelled clip.
    """
    import parselmouth

    snd = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.005)
    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]
    if voiced.size:
        median = float(np.median(voiced))
        pitch = snd.to_pitch(
            time_step=0.005,
            pitch_floor=max(50.0, median / 2.0),
            pitch_ceiling=min(800.0, median * 2.0),
        )
        values = pitch.selected_array["frequency"]
    times = pitch.xs()
    ok = values > 0
    return times[ok], 12.0 * np.log2(values[ok])


def measure_clip(row, spans, keep):
    """Per-mora median F0 (semitones) for one aligned clip.

    `spans` are aligned to `keep` (the non-masked token indices), so walk them
    together to get each *original* token index's time span.
    """
    audio, sr = sf.read(str(AUDIO / LANG / row["file"]))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    times, semitones = clip_f0(audio, sr)

    span_of = {orig: spans[i] for i, orig in enumerate(keep) if i < len(spans)}
    moras = []
    for index, accent in enumerate(row["pitch_accent"]):
        if accent is None:
            continue
        span = span_of.get(index)
        if span is None:
            continue
        t0, t1, score = span
        inside = (times >= t0) & (times < t1)
        n = int(inside.sum())
        moras.append({
            "index": index,
            "phrase": accent["phrase"],
            "mora": accent["mora"],
            "level": accent["level"],
            "t0": round(float(t0), 4),
            "t1": round(float(t1), 4),
            "align_score": round(float(score), 4),
            "n_voiced": n,
            "st": round(float(np.median(semitones[inside])), 3) if n else None,
        })
    return moras


def _measure_and_cache(task):
    row, spans, keep, align_error, reading = task
    key = phon_key(row["phonemes"])
    moras = [] if align_error else measure_clip(row, spans, keep)
    path = cache_path(row["file"], key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": MODEL, "lang": LANG, "file": row["file"],
        "align_error": align_error, "reading": reading, "moras": moras,
    }, ensure_ascii=False))
    return 1


def cmd_measure(args):
    import modal
    from multiprocessing import Pool

    labels = args.labels or (AUDIO / LANG / "phonemes.jsonl")
    rows = load_rows(Path(labels), args.limit)
    todo = [r for r in rows if not cache_path(r["file"], phon_key(r["phonemes"])).exists()]
    print(f"rows with accent labels: {len(rows)} | cached: {len(rows) - len(todo)} | "
          f"to measure: {len(todo)}")
    if not todo:
        print("nothing to do — all cached.")
        return

    vocab = RA.load_vocab()
    Aligner = modal.Cls.from_name("espeak-audit-aligner", "VadCleanAligner")
    aligner = Aligner()

    def align_batch(chunk):
        items = []
        for row in chunk:
            ids = [vocab.get(p, 3) for p in row["phonemes"]]
            ok = [i not in RA.MASKED_IDS for i in ids]
            keep = [j for j in range(len(row["phonemes"])) if ok[j]]
            audio, sr = sf.read(str(AUDIO / LANG / row["file"]))
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            import base64
            items.append((row, keep, {
                "key": f"{LANG}/{row['file']}",
                "audio_b64": base64.b64encode(audio.astype("<f4").tobytes()).decode(),
                "sample_rate": int(sr),
                "target_ids": [ids[j] for j in keep],
            }))
        res = None
        for attempt in range(3):
            try:
                res = aligner.align_batch_b64.remote([it[2] for it in items])
                break
            except Exception as e:  # noqa: BLE001 (transient gRPC/network)
                print(f"  [align retry {attempt + 1}/3] {type(e).__name__}: {e}", flush=True)
                time.sleep(3)
        if res is None:
            print(f"  [align gave up] deferring {len(chunk)} clips", flush=True)
            return []
        # Stale-deploy guard: the cache namespace claims a specific revision.
        for r in res:
            rev = r.get("model_revision")
            if rev is not None and rev != MODEL_REVISION:
                raise SystemExit(
                    f"deployed aligner is revision {rev} but cache namespace is "
                    f"{MODEL_REVISION} — redeploy modal_aligner.py."
                )
        by_key = {r["key"]: r for r in res}
        tasks = []
        for row, keep, sent in items:
            r = by_key.get(sent["key"])
            if r is None:
                continue
            tasks.append((row, r["spans"], keep, r["align_error"], r["reading"]))
        return tasks

    # Super-chunks: align a slice on Modal (threads), then measure it locally
    # (Pool), then the next. Caches incrementally, and keeps the Modal gRPC
    # client and the multiprocessing Pool from ever being live together.
    super_size = max(400, args.concurrency * args.batch * 4)
    chunks = [todo[i:i + super_size] for i in range(0, len(todo), super_size)]
    start, done = time.time(), 0
    for i, chunk in enumerate(chunks):
        batches = [chunk[j:j + args.batch] for j in range(0, len(chunk), args.batch)]
        tasks = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for fut in as_completed([pool.submit(align_batch, b) for b in batches]):
                tasks.extend(fut.result())
        with Pool(args.measure_workers) as pool:
            for _ in pool.imap_unordered(_measure_and_cache, tasks, chunksize=8):
                done += 1
        rate = done / max(time.time() - start, 1e-9)
        eta = (len(todo) - done) / rate / 60 if rate else 0
        print(f"  chunk {i + 1}/{len(chunks)}: {done}/{len(todo)} "
              f"({rate:.1f}/s, ETA {eta:.0f} min)", flush=True)
    print(f"done: {done} clips in {time.time() - start:.0f}s -> {CACHE}")


# ------------------------------------------------------------------- verdict

def clip_verdict(moras, agree_st=AGREE_ST, contra_st=CONTRA_ST):
    """Compare adjacent labelled moras against the F0 that was measured.

    Only transitions *within one accent phrase* and *between adjacent moras*
    are tested, and only where the two labels differ. Same-level pairs are
    skipped on purpose: downdrift makes a flat H…H sequence drift downward, so
    "no change" is not something the measurement can assert.
    """
    agree = contradict = abstain = 0
    for a, b in zip(moras, moras[1:]):
        if a["phrase"] != b["phrase"] or b["mora"] != a["mora"] + 1:
            continue
        if a["level"] == b["level"]:
            continue
        if a["n_voiced"] < MIN_VOICED_FRAMES or b["n_voiced"] < MIN_VOICED_FRAMES:
            continue
        if a["st"] is None or b["st"] is None:
            continue
        delta = b["st"] - a["st"]
        expected_rise = b["level"] > a["level"]
        signed = delta if expected_rise else -delta
        if signed >= agree_st:
            agree += 1
        elif signed <= -contra_st:
            contradict += 1
        else:
            abstain += 1
    return agree, contradict, abstain


def cmd_verdict(args):
    import collections

    records = []
    for path in CACHE.rglob("*.json"):
        records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit(f"no measurements cached under {CACHE} — run `measure` first.")

    stats = collections.Counter()
    deltas = []
    excluded = []
    for rec in records:
        if rec.get("align_error"):
            stats["align_error"] += 1
            excluded.append((rec["file"], "align_error"))
            continue
        moras = rec["moras"]
        agree, contradict, abstain = clip_verdict(moras, args.agree_st, args.contra_st)
        stats["transitions_agree"] += agree
        stats["transitions_contradict"] += contradict
        stats["transitions_abstain"] += abstain
        if args.report:
            for a, b in zip(moras, moras[1:]):
                if (a["phrase"] == b["phrase"] and b["mora"] == a["mora"] + 1
                        and a["level"] != b["level"]
                        and a["st"] is not None and b["st"] is not None
                        and min(a["n_voiced"], b["n_voiced"]) >= MIN_VOICED_FRAMES):
                    d = b["st"] - a["st"]
                    deltas.append(d if b["level"] > a["level"] else -d)
        judged = agree + contradict
        if judged and contradict / judged > args.max_contra_frac:
            stats["clip_contradicted"] += 1
            excluded.append((rec["file"], "f0_contradicts_citation_accent"))
        elif agree:
            stats["clip_confirmed"] += 1
        else:
            # No judgeable evidence either way. KEPT — excluding these turned
            # out to select against parselmouth-illegible (small/early/delayed-
            # fall) prosody, not against wrong labels. See MAX_CONTRA_FRAC.
            stats["clip_unverified_kept"] += 1

    total = len(records)
    print(f"measured clips: {total}")
    for key in ("clip_confirmed", "clip_unverified_kept", "clip_contradicted", "align_error"):
        n = stats[key]
        print(f"  {key:22s} {n:7d}  {100 * n / total:5.1f}%")
    tt = sum(stats[k] for k in
             ("transitions_agree", "transitions_contradict", "transitions_abstain"))
    if tt:
        print(f"  testable transitions: {tt}  "
              f"agree {100 * stats['transitions_agree'] / tt:.1f}% / "
              f"contradict {100 * stats['transitions_contradict'] / tt:.1f}% / "
              f"abstain {100 * stats['transitions_abstain'] / tt:.1f}%")
    if args.report and deltas:
        arr = np.array(deltas)
        print("\nsigned semitone change in the labelled direction "
              "(positive = audio agrees):")
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"    p{q:<3d} {np.percentile(arr, q):+7.2f}")
        print(f"    mean {arr.mean():+.2f}   frac>0 {100 * (arr > 0).mean():.1f}%")

    if args.dry_run:
        print("\n--dry-run: no exclusions written")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for file, reason in excluded:
            f.write(json.dumps({"lang": LANG, "file": file, "reason": reason}) + "\n")
    print(f"\nwrote {len(excluded)} exclusions to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="align on Modal + measure F0 locally")
    m.add_argument("--labels", type=Path, default=None,
                   help="phonemes.jsonl to read (default: data/audio/jpn/phonemes.jsonl)")
    m.add_argument("--limit", type=int, default=None, help="cap rows (testing)")
    m.add_argument("--batch", type=int, default=8, help="clips per Modal align call")
    m.add_argument("--concurrency", type=int, default=24, help="parallel Modal calls")
    m.add_argument("--measure-workers", type=int, default=8, help="local parselmouth processes")
    m.set_defaults(func=cmd_measure)

    v = sub.add_parser("verdict", help="apply thresholds to the cache")
    v.add_argument("--agree-st", type=float, default=AGREE_ST,
                   help="semitone move that counts as the audio agreeing")
    v.add_argument("--contra-st", type=float, default=CONTRA_ST,
                   help="semitone move the wrong way that counts as contradiction")
    v.add_argument("--max-contra-frac", type=float, default=MAX_CONTRA_FRAC,
                   help="largest tolerated contradict/(agree+contradict)")
    v.add_argument("--out", type=Path, default=DEFAULT_EXCLUSIONS)
    v.add_argument("--report", action="store_true", help="print the delta distribution")
    v.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    v.set_defaults(func=cmd_verdict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
