"""Phonemize every dataset language through the shared g2p API.

Writes a per-language JSONL with entries:
    {"file": "abc123.wav", "lang": "eng",
     "phonemes": ["h", "ɛ", "l", ...],
     "stress": [0, 0, 0, ...]}

Languages in OVERRIDE_LANGS get their per-record stress array replaced by
a sidecar (e.g. fra/stress_overrides.jsonl produced by relabel-french/). The
sidecar lists which words end a rhythmic group; we mark the last vowel of
each such word as primary stress and zero everything else, overriding
espeak's per-word stress for that language.

Training-time interleaving of stress into CTC token sequences happens in
dataset.py.
"""

import argparse
import ast
import json
import re
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Historical voice metadata, used only by engine-comparison/replay tools.
# Production labeling uses g2p combined language choices.
LANG_TO_ESPEAK = {
    # Languages we currently train on:
    "eng": "en-us",
    "deu": "de",
    "fra": "fr-fr",
    "ita": "it",
    "por": "pt-br",
    "spa": "es",
    "rus": "ru",
    "zho-hans": "cmn",  # Simplified-script Standard Mandarin
    # Languages added via Pimsleur — phonemes.jsonl gets regenerated for
    # these too so future training runs can opt into them just by adding
    # to --langs. espeak voice picked per language.
    "sqi": "sq",     # Albanian
    "ara": "ar",     # Arabic (Eastern + Egyptian + MSA all share)
    "hye": "hy",     # Armenian (also Western)
    "yue": "yue",    # Cantonese
    "hrv": "hr",     # Croatian
    "ces": "cs",     # Czech
    "dan": "da",     # Danish
    "fas": "fa",     # Persian (Dari + Farsi)
    "nld": "nl",     # Dutch
    "fin": "fi",     # Finnish
    "hat": "ht",     # Haitian Creole
    "heb": "he",     # Hebrew
    "hin": "hi",     # Hindi
    "hun": "hu",     # Hungarian
    "isl": "is",     # Icelandic
    "ind": "id",     # Indonesian
    "gle": "ga",     # Irish
    "jpn": "ja",     # Japanese
    "kor": "ko",     # Korean
    "ell": "el",     # Modern Greek
    "nor": "nb",     # Norwegian (Bokmål)
    "pol": "pl",     # Polish
    "pan": "pa",     # Punjabi
    "ron": "ro",     # Romanian
    "swa": "sw",     # Swahili
    "swe": "sv",     # Swedish
    "tha": "th",     # Thai
    "tur": "tr",     # Turkish
    "ukr": "uk",     # Ukrainian
    "urd": "ur",     # Urdu
    "vie": "vi",     # Vietnamese
    # No espeak voice (oji=Ojibwe, pus=Pashto, tgl=Tagalog, twi=Twi):
    # Audio + transcripts may still exist; g2p decides supported languages.
}


def resolve_espeak_voice(rec: dict, lang: str) -> str:
    """Select a clip's label voice without guessing its speaker's dialect.

    Explicit metadata wins (notably Pimsleur's Spanish/Castilian course
    mapping and per-film choices). Only Spanish FLEURS's es_419 dataset and
    locale-scoped Chirp3 voice names supply a missing dialect. Tatoeba's
    `voice` is a username, and Gemini's voice names carry no dialect.
    Unknown dialects retain the canonical `es` fallback; that is a label
    choice, not evidence that the speaker is Castilian.
    """
    if rec.get("espeak_voice"):
        return rec["espeak_voice"]
    if lang == "spa":
        if rec.get("source") == "fleurs":
            return "es-419"
        if (rec.get("source") == "tts"
                and rec.get("tts_backend") in (None, "chirp3")):
            voice = rec.get("voice") or ""
            if voice.startswith("es-US-Chirp3-HD-"):
                return "es-419"
            if voice.startswith("es-ES-Chirp3-HD-"):
                return "es"
    return LANG_TO_ESPEAK[lang]


STRESS_NONE = 0
STRESS_PRIMARY = 1
STRESS_SECONDARY = 2

# IPA vowels (monophthongs + near-variants used by espeak-ng across our languages)
IPA_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɚɝᵻ")

# The tokenizer that turns espeak's IPA into these labels (which diacritics
# fold onto the previous token, ʲ onto a preceding consonant only, word
# boundaries, language-switch markers) lives in the g2p crate — see
# `src/parse.rs` at github.com/anchpop/g2p. Change it there; yap and this
# pipeline both consume it.

# Languages whose espeak-emitted stress is systematically wrong and gets
# replaced from a sidecar (rhythmic-group stress for French: stress falls on
# the final syllable of each rhythmic group, not on every word). The sidecar
# is produced by train/relabel-french/ (LLM call). Languages not in this set
# keep espeak's per-word stress.
OVERRIDE_LANGS = {"fra"}


def _load_training_labels() -> dict:
    """Read the model vocabulary used by the training pipeline.

    Sky stages only pronunciation/, with an explicit file_mount of the canonical
    artifact beside this script. A full checkout reads it from the Rust crate.
    """
    script = Path(__file__).resolve()
    path = script.parents[3] / "tagging/lexide/data/training_labels.json"
    if not path.is_file():
        path = script.with_name("training_labels.json")
    return json.loads(path.read_text(encoding="utf-8"))


# Frozen model vocabulary; g2p owns pronunciation and phone segmentation.
_TRAINING_LABELS = _load_training_labels()
TOKENIZER_NAME = _TRAINING_LABELS["provenance"]["tokenizer_name"]
VOCAB_EXTENSIONS: set[str] = set(_TRAINING_LABELS["vocab_extensions"])


@cache
def _tokenizer_vocab() -> set[str]:
    """Frozen base accepted set; independent of network/tokenizer revisions."""
    return set(_TRAINING_LABELS["base_vocab"])


def check_training_label_vocab(model_name: str, vocab: set[str]) -> None:
    """Reject default tokenizer drift; custom processor sources stay independent."""
    if model_name != TOKENIZER_NAME:
        return
    expected = _tokenizer_vocab()
    if vocab != expected:
        raise ValueError(
            f"Training-label vocabulary drift for {model_name}: "
            f"missing={sorted(expected - vocab)}, unexpected={sorted(vocab - expected)}. "
            "Review the shared training_labels.json contract before training."
        )


def unknown_phonemes(phonemes: list[str]) -> set[str]:
    """Report unsupported model labels without rewriting g2p's output."""
    return set(phonemes) - (_tokenizer_vocab() | VOCAB_EXTENSIONS)


# Split text into atomic word tokens that align with espeak's IPA word spans.
# Hyphenated compounds ("passe-temps") are kept as one word — espeak treats
# them as a single unit and emits one IPA span for the whole compound.
_ATOMIC_WORD_RE = re.compile(r"\S+")


def _atomic_words(text: str) -> list[str]:
    return _ATOMIC_WORD_RE.findall(text)


def _merge_vowelless_spans(
    phonemes: list[str], word_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge any IPA word span with no vowel into the preceding span.

    French liaison is the motivating case: espeak emits "les amis" as
    `le_z_ami`, so the floating /z/ becomes its own span and breaks
    alignment with the 2 text words. Merging the vowelless /z/ span
    back into "le" restores alignment ("lez" + "ami" = 2 spans).

    If the first span is vowelless, merge it forward into the next span
    instead.
    """
    merged: list[tuple[int, int]] = []
    for start, end in word_spans:
        has_vowel = any(phonemes[i][:1] in IPA_VOWELS for i in range(start, end))
        if has_vowel or not merged:
            merged.append((start, end))
        else:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
    # If the very first span lacked a vowel and we kept it (no `merged` yet
    # at that point), fold it forward into the next vowel-bearing span.
    if len(merged) >= 2:
        first_start, first_end = merged[0]
        if not any(phonemes[i][:1] in IPA_VOWELS for i in range(first_start, first_end)):
            next_start, next_end = merged[1]
            merged = [(first_start, next_end)] + merged[2:]
    return merged


def _strip_punct(s: str) -> str:
    """Match the Rust binary's strip_punct: trim leading/trailing non-alnum
    except apostrophes and hyphens, lowercase. So "Berlin," → "berlin" and
    "qu'il" → "qu'il".
    """
    return s.strip("""!"#$%&()*+,./:;<=>?@[\\]^_`{|}~…—–""").lower()


def apply_stress_override(
    phonemes: list[str],
    word_spans: list[tuple[int, int]],
    sentence: str,
    stressed_words: list[str],
) -> list[int] | None:
    """Build a stress array from an LLM rhythmic-group sidecar entry.

    Aligns text words (whitespace/hyphen-split) to IPA word_spans by index.
    For each text word that matches one of the LLM-flagged `stressed_words`
    (left-to-right, with repeated targets consuming successive occurrences),
    marks the LAST vowel in the corresponding IPA span as primary stress.
    Returns None if word alignment fails or any requested stress cannot be
    placed on a matching word's vowel — caller should fall back to espeak's
    stress, not mark a partial/failed override as trusted. An empty sidecar
    list deliberately supplies all-zero stress.
    """
    text_words = _atomic_words(sentence)
    aligned_spans = _merge_vowelless_spans(phonemes, word_spans)
    if len(text_words) != len(aligned_spans):
        return None

    normalized = [_strip_punct(w) for w in text_words]
    stress = [STRESS_NONE] * len(phonemes)
    used = [False] * len(text_words)

    for stressed in stressed_words:
        atomics = _atomic_words(stressed)
        if not atomics:
            return None
        # LLM may return multi-token phrases or hyphenated compounds; rhythmic
        # stress falls on the final syllable of the final token.
        target = _strip_punct(atomics[-1])
        if not target:
            return None
        for i, w in enumerate(normalized):
            if used[i] or w != target:
                continue
            start, end = aligned_spans[i]
            for j in range(end - 1, start - 1, -1):
                if phonemes[j][:1] in IPA_VOWELS:
                    stress[j] = STRESS_PRIMARY
                    break
            else:
                return None
            used[i] = True
            break
        else:
            return None

    return stress


def load_stress_overrides(path: Path) -> dict[str, list[str]]:
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            overrides[rec["file"]] = rec["stressed_words"]
    return overrides


# g2p 0.4.0 compound classes (g2p/src/parse.rs vowel_unit/affricate);
# vocabulary membership does not mean the old aligner's rows were trained.
MERGED_TOKEN_BASES = frozenset({
    "aɪ", "aʊ", "eɪ", "oʊ", "əʊ", "ɔɪ", "ɔʏ", "ɔø",
    "ɑːɹ", "ɔːɹ", "ɛɹ", "ɪɹ", "ʊɹ",
    "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "tʂ", "dʐ",
    "pf", "bv", "tθ", "dð", "kx", "ɡɣ", "ʈʂ", "ɖʐ",
    "ɐ̃ʊ̃", "ɐ̃ɪ̃", "õɪ̃", "ũɪ̃", "ɐ̃j",
})
# Decorations can occur on either half of an affricate (tːs, t͡ʃʲ).
_MERGED_DECORATIONS = str.maketrans("", "", "ːʲʰ͜͡")
_MERGED_UNDECORATED = {token.translate(_MERGED_DECORATIONS) for token in MERGED_TOKEN_BASES}


def guard_narrowing_labels(lang: str, data_dir: Path, aligner_path: Path) -> None:
    """Inspect literal model pins without importing Modal or cloud clients."""
    pins = {}
    for node in ast.parse(aligner_path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "MODEL_ID", "MODEL_REVISION",
                }:
                    pins[target.id] = ast.literal_eval(node.value)
    revision = pins.get("MODEL_REVISION", "")
    old_revision = "2926e06f8092935f597e0018beb5d579b95b889a"
    if (pins.get("MODEL_ID") != "anchpop/lexide-pronunciation-unified-vad-clean"
            or not (len(revision) >= 7 and old_revision.startswith(revision))):
        return
    path = data_dir / lang / "phonemes.jsonl"
    with path.open() as labels:
        for line in labels:
            if not line.strip():
                continue
            for token in json.loads(line)["phonemes"]:
                if token.translate(_MERGED_DECORATIONS) in _MERGED_UNDECORATED:
                    raise ValueError(
                        f"Refusing narrowing: {path} contains merged token {token!r}, "
                        "but modal_aligner is pinned to vad-clean@2926e06, whose "
                        "merged-token rows are untrained. Use --skip-narrowing "
                        "until a merged-label model is pinned; see the narrowing "
                        "caveat in pronunciation/CLAUDE.md."
                    )


def run_narrowing(lang: str, data_dir: Path) -> None:
    """Regenerate phonemes_narrowed.jsonl for a language just labeled.

    Training reads the narrowed file (--use-narrowed is the default), so a
    stale one silently hides newly-labeled clips from training. Pass-through
    languages are a dependency-free broad copy; English and the nasal
    languages recompute their acoustic evidence from the measure cache, and
    clips the cache doesn't cover simply stay broad (narrow.py reports the
    abstain counts — re-run espeak_audit/measure_corpus.py to narrow them).
    """
    audit_dir = Path(__file__).resolve().parents[2] / "espeak_audit"
    guard_narrowing_labels(lang, data_dir, audit_dir / "modal_aligner.py")
    sys.path.insert(0, str(audit_dir))
    try:
        import narrow
    finally:
        sys.path.remove(str(audit_dir))
    # narrow's module-level audio root; the acoustic measure cache resolves
    # its own paths independently, so pointing this at --data-dir is safe.
    narrow.AUDIO = data_dir
    narrow.run([lang])


# The voice=null sources whose speaker identity comes from the embedding →
# clustering pipeline (train/speaker-embed/). Tatoeba/TTS carry a real
# `voice`; film carries a diarization-derived speaker_cluster from extraction.
SPEAKER_CLUSTERED_SOURCES = ("fleurs", "pimsleur")


def refresh_speaker_clusters(lang: str, data_dir: Path) -> None:
    """Bring the language's speaker_cluster labels up to date, in place.

    FLEURS/Pimsleur rows have voice=null; per-token acoustic analysis needs a
    speaker id per clip (see "Speaker identity" in CLAUDE.md), so those rows
    get a pseudo-speaker from ECAPA embeddings (Modal, per-clip cache — only
    new clips are embedded) + agglomerative clustering. Runs after
    phonemes.jsonl is written because clustering excludes silence-dropped
    clips through it. Needs the deployed `speaker-embed` Modal app; pass
    --skip-speaker-cluster for an offline run (the manifest keeps whatever
    labels it had).
    """
    embed_dir = Path(__file__).resolve().parents[2] / "train" / "speaker-embed"
    sys.path.insert(0, str(embed_dir))
    try:
        import embed
        import cluster
    finally:
        sys.path.remove(str(embed_dir))
    if data_dir.resolve() != embed.AUDIO.resolve():
        # embed's cache keys and cluster's manifest paths are rooted at the
        # canonical data/audio; silently clustering a different tree would
        # mix caches across corpora.
        raise ValueError(
            f"speaker clustering only supports the canonical data dir "
            f"{embed.AUDIO} (got {data_dir}); pass --skip-speaker-cluster"
        )
    clips = embed.clips_needing_embeddings(
        [lang], set(SPEAKER_CLUSTERED_SOURCES), limit=None
    )
    if not clips:
        return
    print(f"{lang}: refreshing speaker clusters "
          f"({len(clips)} {'/'.join(SPEAKER_CLUSTERED_SOURCES)} clips) ...")
    embed.embed_clips(clips)
    cluster.cluster_language(lang, sources=SPEAKER_CLUSTERED_SOURCES, write=True)


def load_accent_exclusions(path: Path) -> dict[str, str]:
    """Clips whose measured F0 contradicts their citation pitch accent.

    Written by espeak_audit/pitch_accent_audit.py. The phones are unaffected —
    only the accent factor is withheld, so the clip still trains everything
    else. Absent file means no acoustic pass has been run yet.
    """
    if not path.exists():
        return {}
    excluded = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            excluded[rec["file"]] = rec.get("reason", "f0_contradicts_citation_accent")
    return excluded


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from corpus_labels import LabelCache, training_fields, language_for_record  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAD_COMPUTE_BIN = REPO_ROOT / "vad_compare" / "target" / "release" / "vad_compute"
VAD_COMPUTE_MANIFEST = REPO_ROOT / "vad_compare" / "Cargo.toml"

# The single tarball every sky_*.yaml stages onto the GPU node. Lives under
# .work/ so it is gitignored, and matches .skyignore's `*.tar` so the workdir
# upload never carries it — file_mounts ships it deliberately instead.
DATASET_TAR = REPO_ROOT / ".work" / "pron_audio.tar"


# Some source corpora ship empty/corrupt recordings: Google FLEURS es_419
# has 490 clips (17.5% of the split) that are digital silence at the source,
# paired with valid transcripts — pure label noise if they reach training,
# and they fool speaker clustering into a fake "mega-cluster". We can't fix
# the upstream tar, so we drop any clip whose peak amplitude is below this
# floor. Real speech in our corpora peaks at 0.3-0.6; the silent clips peak
# at ~1e-4, so 1e-3 (-60 dBFS) separates them with two orders of magnitude
# of margin on each side. Source-agnostic: applies to every source/lang.
SILENCE_PEAK_FLOOR = 1e-3


def is_silent(path: Path) -> bool:
    """True if the recording is effectively digital silence (empty/corrupt
    source audio). Reads peak amplitude only — full read, since a clip can be
    silent at the head and have content later."""
    data, _ = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return len(data) == 0 or float(np.abs(data).max()) < SILENCE_PEAK_FLOOR


class SilenceCache:
    """Stat-validated cache for the full-file silence audit.

    The cache lives below ``data/audio/<lang>/.cache`` (excluded from the
    training tar). A hit is valid only while file size, nanosecond mtime, and
    the configured peak floor all match. Existing ``phonemes.jsonl`` rows can
    safely seed non-silent hits when that output is newer than the audio file:
    those rows could only have been written after passing this same guard.
    """

    def __init__(self, audio_dir: Path, phonemes_path: Path):
        cache_dir = audio_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self.audio_dir = audio_dir
        self.conn = sqlite3.connect(cache_dir / "silence.sqlite3")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS silence (
                file TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                peak_floor REAL NOT NULL,
                silent INTEGER NOT NULL
            )
            """
        )
        self.pending = 0
        count = self.conn.execute("SELECT COUNT(*) FROM silence").fetchone()[0]
        if count == 0 and phonemes_path.exists():
            self._seed_known_nonsilent(phonemes_path)

    def _seed_known_nonsilent(self, phonemes_path: Path) -> None:
        output_mtime_ns = phonemes_path.stat().st_mtime_ns
        rows = []
        with open(phonemes_path) as source:
            for line in source:
                filename = json.loads(line)["file"]
                path = self.audio_dir / filename
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if output_mtime_ns >= stat.st_mtime_ns:
                    rows.append((
                        filename, stat.st_size, stat.st_mtime_ns,
                        SILENCE_PEAK_FLOOR, 0,
                    ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO silence VALUES (?, ?, ?, ?, ?)", rows,
        )
        self.conn.commit()
        if rows:
            print(f"seeded silence cache with {len(rows):,} known-good clips")

    def is_silent(self, path: Path) -> bool:
        stat = path.stat()
        filename = path.name
        row = self.conn.execute(
            "SELECT size, mtime_ns, peak_floor, silent FROM silence WHERE file = ?",
            (filename,),
        ).fetchone()
        if row is not None and row[:3] == (
            stat.st_size, stat.st_mtime_ns, SILENCE_PEAK_FLOOR,
        ):
            return bool(row[3])

        silent = is_silent(path)
        self.conn.execute(
            "INSERT OR REPLACE INTO silence VALUES (?, ?, ?, ?, ?)",
            (filename, stat.st_size, stat.st_mtime_ns,
             SILENCE_PEAK_FLOOR, int(silent)),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.conn.commit()
            self.pending = 0
        return silent

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "SilenceCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def ensure_vad_compute_built() -> Path:
    """Build vad_compute if the release binary isn't there yet.

    vad_compute is a Rust binary (earshot framewise VAD). Phonemes and VAD
    are co-produced from the same audio so we build the binary lazily here
    rather than burdening the caller with a separate build step.
    """
    if VAD_COMPUTE_BIN.exists():
        return VAD_COMPUTE_BIN
    print("vad_compute binary not present; building (cargo build --release)...")
    subprocess.run(
        ["cargo", "build", "--release",
         "--manifest-path", str(VAD_COMPUTE_MANIFEST),
         "--bin", "vad_compute", "--quiet"],
        check=True,
    )
    if not VAD_COMPUTE_BIN.exists():
        raise RuntimeError(f"vad_compute build succeeded but binary not at {VAD_COMPUTE_BIN}")
    return VAD_COMPUTE_BIN


def regenerate_vad(phonemes_path: Path, audio_dir: Path) -> None:
    """Run vad_compute over a lang's phonemes.jsonl to (re)build vad.jsonl.

    vad_compute reads every file referenced by phonemes.jsonl and rewrites
    vad.jsonl from scratch, so this is safe to re-run after any extraction
    pass. Keeps vad coverage in lockstep with phonemes — otherwise newly-
    added clips silently train without the nonblank-head soft-regularizer.
    """
    vad_bin = ensure_vad_compute_built()
    vad_path = audio_dir / "vad.jsonl"
    subprocess.run(
        [str(vad_bin), str(phonemes_path), str(audio_dir), str(vad_path)],
        check=True,
    )


def newest_mtime(data_dir: Path) -> float:
    """Newest mtime under data_dir, ignoring .cache. Used to decide whether the
    staging tar is stale. Walks ~450k entries, which costs a second or two on
    SSD — cheap next to repacking 34 GB we already packed."""
    newest = 0.0
    stack = [data_dir]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.name == ".cache":
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
    return newest


def build_dataset_tar(data_dir: Path, output: Path = DATASET_TAR) -> Path:
    """Pack data/audio into the one tarball the training launchers stage.

    Half a million loose wavs defeat every transport we've tried: rsync crawls,
    and the Hub caps repository commits at 256/hour, which turns a loose-file
    dataset repo into a multi-day upload that stalls out. So the dataset moves
    as a single file — sky's file_mounts rsyncs it to the node, the run: block
    untars it into ~/data, and the loader still opens loose wavs from there.

    Always packs the WHOLE data_dir, even under --langs: training reads every
    language, so a tar of one language would be a footgun.

    Writes to a .partial path and renames, so an interrupted run leaves the
    previous good tar in place rather than a truncated one a launcher would
    cheerfully upload.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_mtime >= newest_mtime(data_dir):
        size_gb = output.stat().st_size / 1e9
        print(f"staging tar up to date: {output} ({size_gb:.1f} GB)")
        return output

    # Named ".partial.tar", not ".tar.partial", so it still matches .skyignore's
    # `*.tar` — a leftover partial must never ride along in the workdir upload.
    tmp = output.with_name(output.stem + ".partial.tar")
    tmp.unlink(missing_ok=True)
    print(f"packing {data_dir} -> {output} ...")
    subprocess.run(
        ["tar", "-cf", str(tmp), "-C", str(data_dir), "--exclude=.cache", "."],
        check=True,
    )
    tmp.replace(output)
    size_gb = output.stat().st_size / 1e9
    print(f"wrote staging tar: {output} ({size_gb:.1f} GB)")
    return output


def refresh_mixed_script_exclusions() -> None:
    """Regenerate train/mixed_script_exclusions.jsonl from the manifests.

    Deterministic, manifest-derived, no API calls — so preprocess keeps its
    "self-contained" contract: the sidecar can never go stale relative to the
    manifests a run just processed. train.sh passes it to the trainer via
    --audit-path alongside the ASR-audit sidecars.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import build_mixed_script_exclusions
    finally:
        sys.path.remove(str(scripts_dir))
    build_mixed_script_exclusions.main()


def _eligible_languages(
    data_dir: Path, requested: list[str] | None,
) -> list[str]:
    """Resolve processable language directories in deterministic order."""
    languages = []
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name == ".cache":
            continue
        lang = lang_dir.name
        if requested and lang not in requested:
            continue
        if not (lang_dir / "manifest.jsonl").exists():
            continue
        languages.append(lang)
    return languages


def _run_parallel_languages(
    args: argparse.Namespace, languages: list[str],
) -> None:
    """Run isolated per-language children, then pack once in the parent.

    Language outputs and caches are disjoint. Children always receive
    ``--no-pack``, preventing concurrent writes to the shared staging tar.
    Per-language logs keep concurrent tqdm output readable.
    """
    log_dir = REPO_ROOT / ".work" / "preprocess_parallel"
    log_dir.mkdir(parents=True, exist_ok=True)
    queued = list(languages)
    running: dict[str, tuple[subprocess.Popen, object, Path]] = {}
    failures: list[str] = []

    print(f"processing {len(languages)} languages with {args.jobs} workers")
    print(f"per-language logs: {log_dir}")
    while queued or running:
        while queued and len(running) < args.jobs:
            lang = queued.pop(0)
            log_path = log_dir / f"{lang}.log"
            log_file = open(log_path, "w")
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--data-dir", str(args.data_dir),
                "--langs", lang,
                "--jobs", "1",
                "--no-pack",
            ]
            if args.skip_vad:
                cmd.append("--skip-vad")
            if args.skip_speaker_cluster:
                cmd.append("--skip-speaker-cluster")
            if args.skip_narrowing:
                cmd.append("--skip-narrowing")
            if args.allow_noncommercial:
                cmd.append("--allow-noncommercial")
            process = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
            )
            running[lang] = (process, log_file, log_path)
            print(f"started {lang}: pid {process.pid} -> {log_path}", flush=True)

        completed = []
        for lang, (process, log_file, log_path) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            log_file.close()
            completed.append(lang)
            if returncode == 0:
                print(f"completed {lang}", flush=True)
            else:
                failures.append(lang)
                print(
                    f"FAILED {lang} (exit {returncode}); see {log_path}",
                    flush=True,
                )
        for lang in completed:
            del running[lang]
        if running and not completed:
            time.sleep(1)

    if failures:
        raise RuntimeError(
            "language preprocessing failed: " + ", ".join(failures)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "data" / "audio")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Restrict to these lang codes (default: all dirs). "
                             "Lets independent languages run as parallel jobs.")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Don't regenerate vad.jsonl after phonemization. "
                             "Use only when you're certain vad coverage is "
                             "current — by default we keep vad in lockstep "
                             "with phonemes.")
    parser.add_argument("--skip-narrowing", action="store_true",
                        help="Leave existing narrowed labels untouched; regenerate "
                             "them separately after this broad-label pass.")
    parser.add_argument("--skip-speaker-cluster", action="store_true",
                        help="Don't refresh manifest speaker_cluster labels "
                             "for the voice=null sources (FLEURS/Pimsleur). "
                             "Use for offline runs — embedding needs the "
                             "deployed speaker-embed Modal app.")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Process this many languages concurrently. Each language writes "
             "an isolated log and the parent packs the dataset exactly once.",
    )
    parser.add_argument(
        "--no-pack", action="store_true", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-noncommercial", action="store_true",
        help="Include CC BY-NC/noncommercial source rows. By default these "
             "remain in the auditable manifest/sidecars but are excluded from "
             "phonemes.jsonl so normal training output is commercially usable.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    languages = _eligible_languages(args.data_dir, args.langs)
    if args.jobs > 1 and len(languages) > 1:
        _run_parallel_languages(args, languages)
        if not args.no_pack:
            refresh_mixed_script_exclusions()
            build_dataset_tar(args.data_dir)
        return

    langs_with_unknowns: list[str] = []
    for lang_dir in sorted(args.data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if args.langs and lang not in args.langs:
            continue

        manifest_path = lang_dir / "manifest.jsonl"
        phonemes_path = lang_dir / "phonemes.jsonl"

        # download_pimsleur.py creates the lang dir before it knows whether
        # any clips will match the target language; some dirs may end up
        # holding only `pimsleur_processed.txt` and no manifest.
        if not manifest_path.exists():
            print(f"Skipping {lang} (no manifest.jsonl)")
            continue

        records = []
        with open(manifest_path) as f:
            for line in f:
                records.append(json.loads(line))

        # Acoustics get the last word on the accent factor: the sidecar's
        # accent is what the dictionary says, and this file lists the clips
        # where measured F0 says otherwise.
        accent_exclusions = load_accent_exclusions(
            Path(__file__).resolve().parents[1] / f"{lang}_pitch_accent_exclusions.jsonl"
        )
        if accent_exclusions:
            print(f"{lang}: withholding pitch accent on {len(accent_exclusions)} "
                  f"clips the acoustic audit rejected")

        # Load rhythmic-group sidecar for override languages. Missing sidecar
        # just means no overrides apply — espeak stress is used as-is.
        stress_overrides: dict[str, list[str]] = {}
        if lang in OVERRIDE_LANGS:
            override_path = lang_dir / "stress_overrides.jsonl"
            if override_path.exists():
                stress_overrides = load_stress_overrides(override_path)
                print(f"{lang}: loaded {len(stress_overrides)} stress overrides "
                      f"from {override_path.name}")
            else:
                print(f"{lang}: WARNING — no stress_overrides.jsonl; espeak's "
                      f"per-word stress will be used (systematically wrong for "
                      f"this language). Run train/relabel-french/ to generate it.")

        override_applied = 0
        override_align_failures = 0
        g2p_excluded = 0
        license_excluded = 0
        silent_dropped = 0
        prepared_records: list[dict] = []
        with SilenceCache(lang_dir, phonemes_path) as silence_cache:
            for rec in tqdm(records, desc=f"{lang} silence"):
                license_name = str(rec.get("license") or "")
                if not args.allow_noncommercial and (
                        "BY-NC" in license_name.upper()
                        or "NONCOMMERCIAL" in license_name.upper()):
                    license_excluded += 1
                    continue
                # Drop empty/corrupt source recordings. The stat-validated
                # cache avoids decoding unchanged WAVs on every label rebuild.
                if silence_cache.is_silent(lang_dir / rec["file"]):
                    silent_dropped += 1
                    continue
                prepared_records.append(rec)

        # token -> (count, first-example sentence). Buffered per-lang so we
        # can report all unknowns and skip writing the file if any are found
        # — partial output would silently train on a vocab-mismatched corpus.
        unknown_examples: dict[str, tuple[int, str]] = {}
        # Sentences that have letters but phonemized to nothing — see the
        # check further down where these are collected.
        empty_phoneme_examples: list[str] = []
        entries: list[dict] = []
        dispositions = []
        with closing(LabelCache(lang_dir / ".cache" / "g2p_labels.sqlite3")) as label_cache:
            for rec in tqdm(prepared_records, desc=f"{lang} phonemize"):
                g2p_language = language_for_record(rec, lang)
                labels = label_cache.phonemize(rec["sentence"], g2p_language)
                dispositions.append((rec, g2p_language, labels))
            build_identity = label_cache.build
        for rec, g2p_language, labels in dispositions:
            if labels.get("exclude_reason"):
                g2p_excluded += 1
                continue
            fields = training_fields(labels, rec)
            phonemes, stress = list(labels["phonemes"]), list(labels["stress"])
            word_spans = [tuple(span) for span in labels["word_spans"]]
            stress_source = fields.pop("stress_source")
            if rec["file"] in stress_overrides:
                new_stress = apply_stress_override(
                    phonemes, word_spans, rec["sentence"],
                    stress_overrides[rec["file"]],
                )
                if new_stress is not None:
                    stress = new_stress
                    stress_source = "override"
                    override_applied += 1
                else:
                    override_align_failures += 1
            unknowns = unknown_phonemes(phonemes)
            for u in unknowns:
                if u in unknown_examples:
                    count, example = unknown_examples[u]
                    unknown_examples[u] = (count + 1, example)
                else:
                    unknown_examples[u] = (1, rec["sentence"])
            # An utterance with letters in it must produce phonemes. When it
            # doesn't, that is a labeling failure, not a property of the
            # sentence — and it is invisible downstream, because dataset.py
            # drops empty-target rows as `no_phonemes` and training simply
            # proceeds with slightly less data. Count them here and report at
            # the end of the language, so the failure is attributable to its
            # cause instead of showing up as an unexplained row-count drift.
            if not phonemes and any(ch.isalpha() for ch in rec["sentence"]):
                empty_phoneme_examples.append(rec["sentence"])
            entry = {
                "file": rec["file"],
                "lang": lang,
                "sentence": rec["sentence"],
                "phonemes": phonemes,
                "stress": stress,
                "stress_source": stress_source,
                "source": rec.get("source"),
                "license": rec.get("license"),
                "phoneme_backend": "g2p",
                "g2p_identity": build_identity,
                "g2p_language": g2p_language,
                **fields,
            }
            acoustic_reason = accent_exclusions.get(rec["file"])
            if acoustic_reason is not None:
                entry.pop("pitch_accent", None)
                entry["pitch_accent_exclude_reason"] = acoustic_reason
            # Propagate Whisper signal fields from the manifest. Only
            # present for Pimsleur (extracted with download_pimsleur.py).
            # FLEURS / Tatoeba rows lack these and pass them through as None.
            for k in ("whisper_avg_logprob", "whisper_no_speech_prob",
                     "whisper_compression_ratio", "duration_sec"):
                if k in rec:
                    entry[k] = rec[k]
            entries.append(entry)

        if empty_phoneme_examples:
            # Loud, but not fatal: unlike an unknown token (which would train
            # a vocab mismatch), an empty row is merely lost. Refusing to write
            # the file would block a whole language over a handful of rows, so
            # report precisely instead and let the operator judge.
            print(f"\nWARNING: {lang} has {len(empty_phoneme_examples):,} "
                  f"sentence(s) with letters that phonemized to NOTHING. "
                  f"These rows are written but dataset.py will drop them as "
                  f"`no_phonemes`, so they are silently absent from training.")
            for example in empty_phoneme_examples[:5]:
                print(f"    {example[:100]!r}")
            if len(empty_phoneme_examples) > 5:
                print(f"    ... and {len(empty_phoneme_examples) - 5:,} more")
            print("  Usually an eSpeak invocation problem rather than a "
                  "property of the text — check the voice and that the text "
                  "reaches eSpeak after `--`.")

        if unknown_examples:
            total = sum(c for c, _ in unknown_examples.values())
            print(f"\nERROR: {lang} has {len(unknown_examples)} unknown token "
                  f"type(s) ({total:,} occurrences) outside the model vocabulary:")
            for tok, (count, example) in sorted(
                unknown_examples.items(), key=lambda kv: -kv[1][0]
            ):
                codepoints = " ".join(f"U+{ord(c):04X}" for c in tok)
                print(f"  {tok!r:>10}  {count:>6,}x   {codepoints}")
                print(f"             example sentence: {example[:100]!r}")
            print("\n  Review the examples: extend the model vocabulary for valid phones,")
            print("  fix pronunciation in g2p, or correct malformed source records.")
            print(f"  NOT writing {phonemes_path} — fix the above and re-run.")
            langs_with_unknowns.append(lang)
            continue

        with open(phonemes_path, "w") as out:
            for entry in entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        exclusion_path = lang_dir / "g2p_exclusions.jsonl"
        with exclusion_path.open("w") as out:
            for rec, g2p_language, labels in dispositions:
                if labels.get("exclude_reason"):
                    out.write(json.dumps({
                        "file": rec["file"], "sentence": rec["sentence"],
                        "g2p_language": g2p_language, "g2p_identity": build_identity,
                        "exclude_reason": labels["exclude_reason"],
                    }, ensure_ascii=False) + "\n")
        print(f"{lang}: wrote {len(entries)} entries to {phonemes_path}")
        if silent_dropped:
            print(f"{lang}: dropped {silent_dropped} silent/empty recording(s) "
                  f"(peak < {SILENCE_PEAK_FLOOR:g}, e.g. corrupt FLEURS source audio)")
        if g2p_excluded:
            print(f"{lang}: g2p explicitly excluded "
                  f"{g2p_excluded} recording(s)")
        if license_excluded:
            print(f"{lang}: excluded {license_excluded} noncommercial recording(s); "
                  f"pass --allow-noncommercial only for an explicitly NC run")
        if lang in OVERRIDE_LANGS and stress_overrides:
            print(f"{lang}: applied stress override to {override_applied} records, "
                  f"{override_align_failures} alignment failures "
                  f"(fell back to espeak stress)")

        if not args.skip_narrowing:
            run_narrowing(lang, args.data_dir)

        if not args.skip_vad:
            print(f"{lang}: regenerating vad.jsonl ...")
            regenerate_vad(phonemes_path, lang_dir)

        if not args.skip_speaker_cluster:
            refresh_speaker_clusters(lang, args.data_dir)

    if langs_with_unknowns:
        print(f"\n{len(langs_with_unknowns)} language(s) had unknown tokens "
              f"and were NOT written: {', '.join(langs_with_unknowns)}")
        sys.exit(1)

    # Last steps, unconditionally: refresh the manifest-derived exclusion
    # sidecar and pack the dataset. Doing this here is the whole point of
    # preprocess being self-contained — finishing this script means there is
    # nothing left to do before `sky launch`.
    if not args.no_pack:
        refresh_mixed_script_exclusions()
        build_dataset_tar(args.data_dir)


if __name__ == "__main__":
    main()
