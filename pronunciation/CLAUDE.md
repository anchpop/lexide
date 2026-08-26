# Pronunciation — project context

A multilingual **phonetic transcription model**: given speech, output a narrow IPA
transcription of *what was actually pronounced*. Built to power language learning
(the maintainer uses it to learn French) — so the bar is not "good WER on a
benchmark," it's "does it genuinely hear how a sound was produced."

> The root `../CLAUDE.md` holds repo-wide rules (e.g. never cancel a cloud job
> without asking). This file is the pronunciation-project context: philosophy,
> goals, architecture. Read it before making design decisions here.

## What we're really building

> **The true purpose is detecting accented speech** — helping a learner speak
> without an accent by faithfully reporting how their production deviates from
> native realization (stated by the maintainer 2026-08-24). Judge every label
> decision by "can the model report a learner's deviation?", not "is this
> segment informative in native speech?". Corollary: never label a variable
> phenomenon by uniform rule (the model learns the prior and hallucinates
> native detail over the learner's error — the pitch-accent lesson). The safe
> states are *omitted* or *per-clip acoustic*. Known rule-labeled-but-variable
> phenomena, measured by listening audits 2026-08-24 (all narrowing-roadmap
> candidates, none blocking): French facultative liaison (espeak inserts
> uniformly; real speakers ~50/50 — obligatory liaison is ~100% and fine),
> Hindi function-word ɦ-deletion (~16%) and cluster/coda voiced-aspirate
> weakening, i+V linking glides (~25-40%, all languages — correctly absent
> from labels since the espeak rebase).

espeak-ng gives a **broad / citation-form phonemic** transcription — the dictionary
ideal, not the realized sound. A model trained on espeak labels inherits that
broadness. The mission ("phonemizer → phonetizer") is to push the labels toward the
**actual realization** so the model learns to transcribe speech faithfully:
American /t/ flapping to [ɾ], vowels nasalizing before nasal codas, etc.

The product is a model that is *faithful to the audio*, usable by a learner to see
how they actually pronounced something vs. a target.

## Core philosophy (the load-bearing ideas)

These are hard-won and override generic ML instincts. Violating them has burned us.

1. **Faithful, not smoothing.** The model must transcribe what was said, not
   normalize toward the native/citation form. Reject anything that lets it "cheat"
   toward the expected answer: language-conditional heads, autoregressive decoding,
   LM rescoring. If a learner mispronounces, we must report the mispronunciation.

2. **The isolated minimal-pair eval is the gold standard.** Single words, no
   context — the model can't lean on a language model to guess; it must truly model
   the audio. That's the purest victory. Do **not** dismiss isolated-word failures as
   "eval artifacts" — good connected-speech numbers can just mean it's leaning on
   context (cheating). Optimize for genuinely hearing each segment. (Eval sets live
   in the sibling `yap` repo, `generate-data/`.)

3. **Acoustics are the arbiter for labels — never the model (anti-circularity).**
   When deciding a label, the model may only place *time boundaries*; the *decision*
   comes from measuring the signal (parselmouth: formants, A1–P0, voicing, burst).
   Never pick labels from the model's own likelihood — that's circular self-training
   (v1's errors become v2's training data). See `espeak_audit/`.

4. **Trust acoustics only where our measurement is robust; otherwise use systematic
   knowledge.** Our offline Python (4-formant snapshots, CTC-peaky alignment) is a
   *worse* per-clip analyst than the 2B-param model itself. So: gate on robust,
   near-categorical cues (voicing/burst → flapping); be skeptical of noisy gradient
   ones (raw vowel formants) and prefer deterministic/contextual remaps grounded in
   linguistics or *population*-level acoustics. The win from narrowing is **faithful
   + consistent labels across languages**, not "narrow for its own sake" — sometimes
   that means adding detail (English nasalization), sometimes removing spurious detail
   (Italian lax /ɪ ʊ/ → /i u/). Label the token's *in-clip* realization (what the
   model can recover from the one clip it sees), never a cross-clip speaker fact it
   can't.

5. **Filter data aggressively — tolerate false positives, never false negatives.**
   Data is abundant; dropping a good clip is cheap, letting a bad audio↔label pair
   into training is not. Never add defensive clamps that mask bugs — fix the root.

6. **The vocab is ours.** The xls-r-2b backbone never saw IPA; the CTC head is
   trained from scratch and the phoneme vocab is fully extensible
   (`preprocess.VOCAB_EXTENSIONS`). Add real phonemes as real classes — don't collapse
   them into near-neighbors to fit an old vocab.

7. **Stress is suprasegmental.** It has a separate factor head rather than inline
   tokenizer entries, but that factor is composed into the same joint CTC symbol
   and alignment as the phone. Tone/accent factors follow the same rule and are
   masked when the language or trustworthy label does not apply.

8. **A prosody target must be decidable from the frames it sits on.** Japanese
   pitch accent was first encoded as a *downstep marker* — one positive mora per
   accent phrase — and the head learned nothing but the prior (emitted the
   17%/83% base rate; gave 箸 and 橋 the same contour). Three things were wrong
   and each generalizes: the positive class was rare and the classes wildly
   unbalanced; the acoustic evidence for a downstep is the fall onto the
   *following* mora, so the label sat one mora away from its own cue; and for a
   phrase-final accent that cue is not in the clip at all. The fix is a per-mora
   **H/L level** (`tokyo_pitch_level`): balanced (52/48), dense, and local. It
   also makes odaka and heiban share a contour — correct, because in isolation
   they *are* the same, and the encoding must not assert a distinction the audio
   cannot carry. Contrast the tone heads, which always had this shape (every
   syllable carries a contrastive, intrasyllabic tone) and did learn.

## Architecture

- **Model** (`train/src/factorized_ctc.py`): `facebook/wav2vec2-xls-r-2b` backbone +
  **factorized CTC head** — a `nonblank_head` (is this frame a phone or blank?) and a
  `phoneme_head` (which phone?), plus stress and language-specific prosody factor
  heads in one Cartesian joint CTC alphabet. A `regularized_heads`
  variant learns a soft mixture over encoder layers; the simpler `mode=off` variant is
  the "vad-clean" model. VAD loss is confidence-weighted.
- **Published model (champion)**: `anchpop/lexide-pronunciation` (HF), pinned at commit
  `00a661934cdd`. This is the **mel-sidechannel + MLP-heads, degrade-augmented** recipe
  (`train/sky_vad_clean_sidechannel_degrade.yaml`); it beat the older `mode=off`
  `unified-vad-clean` model on the minimal-pair eval and was promoted to production
  2026-06-18. When anything force-aligns or measures against "the model," **pin the
  exact commit** — alignment depends on the weights (see `espeak_audit/modal_aligner.py`).
  - *Lineage*: the previous champion `unified-vad-clean` @`2926e06` is retained as a
    **private** HF repo (it's the distillation teacher), as is the tiny on-device student
    `distill-distilhubert`. All the other old pronunciation experiment repos were deleted.
- **Languages**: 7 established core (deu eng fra ita por rus spa), plus configured
  expansion targets (hin jpn tha zho-hans), with FLEURS+Tatoeba+TTS+Pimsleur;
  several Pimsleur-only langs (ara ces dan fas …) ride along.
- **Labels**: espeak-ng (the maintainer's **fork**, see gotchas) → `phonemes.jsonl`
  (broad) → optional `phonemes_narrowed.jsonl` (narrowed; see `espeak_audit/`).

## Data → training pipeline

1. **Acquire** (`data/`): `download_fleurs.py` (read sentences, multi-speaker),
   `download_tatoeba.py` (community), `download_pimsleur.py` (course audio, VAD-split +
   Whisper-transcribed), `generate_tts.py --backend {chirp3,gemini}` (Google
   Chirp3-HD voices, or Gemini `gemini-3.1-flash-tts-preview` with its ~29
   prebuilt voices). Each lang dir gets a `manifest.jsonl` (file, sentence,
   source, voice, …). Both backends write `source: "tts"`; Gemini rows also
   carry `tts_backend`/`tts_model`. `--sentence-offset` takes a deeper slice of
   the same seeded shuffle so a second backend records text the first did not.
   All 11 configured languages now have TTS (jpn/tha/zho-hans/hin were added
   2026-08-09; before that the four expansion languages had none, which is a
   large part of why the Japanese pitch-accent head never learned).
   **TTS buys correct text; its prosody is usable but errs decisively.**
   Measured against F0 on Japanese pitch accent: per labelled pitch transition
   TTS moves the right way about as often as human speech (66.7% Chirp3 /
   63.9% Gemini vs 68.3% Pimsleur) and produces the *largest* excursions in the
   corpus (median |move| 3.9 st vs Pimsleur's 2.6) — so it is not flattening
   accent, and it mostly knows the accent. But it contradicts more (15–16% vs
   9.5%), partly a threshold effect: a big wrong-direction move clears the
   contradiction bar where a human's smaller error lands in the abstain band.
   At clip level that compounds to ~31% of TTS clips rejected vs 14% of
   Pimsleur. Net: keep TTS accent supervision, but gate it through the acoustic
   audit rather than trusting it — and beware that the surviving clips are a
   *selected* sample, so check for pattern bias before leaning on them. Nobody
   has measured the equivalent for Thai/Mandarin tone.
2. **Preprocess** (`train/scripts/preprocess.py`): espeak-phonemize every sentence →
   `phonemes.jsonl`; framewise VAD via the `vad_compute` Rust binary (`vad_compare/`).
   Includes a silence guard, per-language phoneme remaps, and the vocab extensions.
3. **Data-quality filters** → sidecar exclusion files the trainer reads:
   - `scripts/audit_asr_groq.py --source {fleurs,tatoeba,tts}`: Groq-Whisper transcribe +
     phoneme-PER vs the label → `train/<source>_asr_exclusions.jsonl`. Run it on
     `tts` too: the Gemini backend is an LLM reading text, so unlike Chirp3 it
     *can* paraphrase or decline, and this is what catches a clip whose audio
     stopped matching its label. (Spot-checked 12/12 verbatim at introduction.)
   - `train/lang-filter/` (Rust + tysm + gpt-5.4-nano): flag clips whose transcript
     isn't entirely the target language → `train/lang_exclusions.jsonl`.
   - `train/relabel-french/`: LLM rhythmic-group stress → `fra/stress_overrides.jsonl`.
4. **Narrow** (`espeak_audit/`, optional): `measure_corpus.py` force-aligns each clip
   on Modal (pinned model) and **measures locally** (parselmouth) → cache. (Modal does
   ONLY alignment — the GPU thing that can't run locally; all DSP is local, so any
   measurement/param change re-runs with zero Modal. Caches incrementally per
   super-chunk.) `narrow.py` then rewrites tokens where the acoustics justify it →
   `phonemes_narrowed.jsonl` (contextual nasal) or `phonemes_narrowed_acoustic.jsonl`
   (`--mode acoustic`, per-token harmonic-A1–P0 within-speaker gating). Train with
   `--use-narrowed [--narrowed-name <file>]`. (See `espeak_audit/` docstrings;
   contextual-vs-acoustic nasal is an open A/B, decided by the minimal-pair eval.)
   `pitch_accent_audit.py` is the same shape for Japanese accent: `measure`
   (Modal align + local parselmouth F0) then `verdict` (thresholds → 
   `train/jpn_pitch_accent_exclusions.jsonl`, which `preprocess.py` reads and
   uses to withhold the accent factor while keeping the phones).
5. **Train** (`train/src/train_unified.py`) on SkyPilot/Modal GPUs; push to HF.
6. **Eval**: the isolated minimal-pair set (gold standard) + held-out clips.

`scripts/preprocess_and_upload.sh` chains the audit → filter → preprocess → upload steps.

## Speaker identity (every clip has one — the FIELD depends on source)

Any per-token acoustic analysis that normalizes within-speaker (e.g. the acoustic
nasalization detector: a vowel is nasal only if its A1–P0 is depressed vs *that
speaker's own* oral vowels of the same category) needs a speaker id per clip.
**Every clip has one — but do not assume which field carries it, and never fall
back to per-clip "no speaker" without checking all of them** (a single short clip
rarely has ≥3 same-vowel oral tokens, so per-clip baselines abstain on ~95% of
tokens — a silent coverage hole that *looks* like the analysis working):

- **Tatoeba** — `voice` is the **contributor username** (e.g. `CK`,
  `MisterTrouser`); a real, ground-truth speaker id. Prolific contributors have
  hundreds of clips → strong baselines. No clustering needed.
- **TTS** — `voice` is the synthetic voice name. *Not* one speaker per
  language: each language draws from all ~30 Chirp3-HD voices (~270 clips
  each). Gemini voice names are **not** language-scoped — the same "Kore"
  speaks every language — so those rows record `gemini:<lang>:<Voice>`,
  keeping the speaker key language-separated. Splitting one speaker is
  harmless; merging two corrupts within-speaker normalization.
- **FLEURS / Pimsleur** — `voice` is **null**; instead they carry
  `speaker_cluster`, a *pseudo-speaker* label from the embedding pipeline below.

So resolve the speaker as **`speaker_cluster or voice`** (cluster first — FLEURS/
Pimsleur; else `voice` — Tatoeba/TTS), and only treat a clip as speaker-less if
*both* are absent.

**Speaker-embedding → clustering pipeline** (`train/speaker-embed/`, run in the
preprocess phase; populates `speaker_cluster` for the `voice=null` sources):
- `modal_embed.py` — ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) 192-d
  speaker-verification embeddings on Modal (T4). `embed.py` orchestrates with a
  per-clip cache (key = `sha256("<lang>/<file>")`), so re-runs only embed new clips.
- `cluster.py` — agglomerative clustering of those embeddings into
  `speaker_cluster`, two ear-validated regimes: **FLEURS** per-language @ cosine
  **0.15** (10 s clips group tightly); **Pimsleur** per-**course** @ cosine
  **0.45** (1–3 s clips need a looser bar; course-scoping stops cross-recording
  merges). Deliberately *over*-segmented: merging two speakers corrupts
  within-speaker normalization, splitting one is harmless. Clusters only clips in
  `phonemes.jsonl` (so preprocess's silence-drop excludes degenerate silent clips).
- `cluster_review.py` — sanity/gender-consistency review of the clusters.

## Repo layout

- `data/` — downloaders + `data/audio/<lang>/` (wavs, `manifest.jsonl`, `phonemes.jsonl`).
- `train/` — `src/` (model, dataset, training), `scripts/preprocess.py`, the Rust
  `lang-filter`/`relabel-french`/`speaker-embed` crates, exclusion sidecars.
- `espeak_audit/` — the acoustics-as-arbiter pipeline: `phonetics.py` (parselmouth
  measures), `modal_aligner.py` (Modal forced-align+measure), `measure_corpus.py`,
  `narrow.py`, `nasal_acoustic.py`, `pitch_accent_audit.py`, REPORT*.md.
- `vad_compare/` — Rust `vad_compute` (framewise VAD).
- `inference/` — `infer.py`.
- `scripts/` — orchestration + one-off audits/backfills.

## Conventions & gotchas

- **espeak**: use the maintainer's fork at `~/coding/tmp/espeak-ng` (binary
  `build/src/espeak-ng`, `--path=build`). **Never install mainline espeak.** Point at
  it via `ESPEAK_NG_BIN` / `ESPEAK_NG_DATA_PATH` (in `.env`, which is gitignored
  and therefore per-machine — each host points at its own build).
  - Our patches live on branch **`french-phrase-stress-liaison`**
    (github.com/anchpop/espeak-ng): the French phrase-final stress/liaison
    work, the fr/de/ru modal-surface fixes, the Portuguese final-nasal
    endings, and the ru/it ipa-label canon fixes. As of 2026-08-23 the branch
    is **rebased onto upstream master `7d426728`** (tip `354bced1`). Build
    with CMake: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build
    build` (on NixOS, inside `nix-shell -p cmake gnumake gcc pkg-config`).
  - **Verify any new espeak build before regenerating labels.** Run
    `scripts/py-linux.sh scripts/verify_espeak_build.py` — it re-phonemizes a
    sample of every language's `phonemes.jsonl` through the whole path
    (`phonemize()` then the vocab/remap step, each row's own `espeak_voice`)
    and requires byte-identical output. Testing raw espeak output instead will
    "find" differences that are really `LANG_PHONEME_REMAP` (ita `ɪ ʊ`→`i u`,
    fra length marks) and the FLEURS per-clip dialect voices.
  - **The on-disk corpus labels reproduce at tag `corpus-v1-labels`
    (= `4dd31042`), NOT at the current branch tip.** To patch or reproduce
    existing labels, build the tag. The rebase (2026-08-23) pulled in
    upstream **#2457**, which fixed a long-standing bug where every
    `prevPh`/`prevPhW` condition short-circuited to false in IPA/`-x` output
    (the audio path was never affected) — so the corpus carries that bug's
    artifacts and the rebased build's IPA is the one that matches what
    espeak actually synthesizes. Known systematic diffs vs the corpus, all
    from #2457 unless noted: linking-`ʲ` after `i` dropped (all langs);
    ita tap→trill `r` outside intervocalic position (~66% of rows); deu
    post-consonant `r`→`ɾ` (~17%); pt final-nasal liaison `ŋ`→`w` before
    vowels; some spa `ɾ`→`r`; fas dictionary/stress updates (upstream, not
    #2457); rus number-reading (~1%). **Before the next retrain, regenerate
    every phonemes.jsonl with the rebased fork** and re-run `espeak_audit`
    narrowing (alignment-derived sidecars go stale); the ASR/lang exclusions
    are keyed by `sha256(sentence)`, not by phoneme labels, so they survive
    relabeling unchanged (verified in `train_unified.py`). Do NOT feed the
    rebased fork's references to anything scoring the *current* production
    model — it emits corpus-canon (`ʲ`, tap-everywhere ita) and will
    spuriously mismatch.
  - *Historical, for context*: the 2026-08-10 "Russian drift" (`y` for `ɨ`,
    `ɭ` for `ɫ`/`ɫʲ`) was a stale compiled build — `ph_russian`'s ipa labels
    were fixed in source but the phoneme data was never recompiled.
    `4dd31042` set `l^ → ɫʲ` (corpus canon) and gave ita's reduced `I`/`U`
    explicit `ipa i`/`ipa u`. The ita `LANG_PHONEME_REMAP` **stays** even
    though pure-Italian output no longer needs it — English `(en)…(it)`
    code-switch spans still emit real `ɪ ʊ` that the corpus normalized. 10
    ita rows with leftover `ɪː ʊː` (missed by the exact-token remap) were
    surgically relabeled in place 2026-08-23.
- **Python**:
  - *macOS*: the miniconda base interpreter
    (`/opt/homebrew/Caskroom/miniconda/base/bin/python3`) — it has parselmouth /
    modal / torch / soundfile. Bare `python3` is a different env missing these.
  - *the NixOS box*: `scripts/py-linux.sh` (venv at `~/.venv-lexide-data`, kept
    outside the repo so SkyPilot rsync mounts never pick it up). The wrapper
    exists because manylinux wheels link against libstdc++/libz, which NixOS
    does not put on the default loader path; it resolves those through `nix
    build` and sets `LD_LIBRARY_PATH`. Without it every C-extension import dies
    with `libstdc++.so.6: cannot open shared object file`. Its header documents
    the one-time venv setup.
- **Ignoring generated data**: large generated files (`phonemes.jsonl`,
  `phonemes_narrowed.jsonl`, caches) are git-ignored via **`.git/info/exclude`, NOT
  `.gitignore`** — SkyPilot `file_mounts` silently respect `.gitignore`, so a tracked
  ignore would stop the training data from uploading. Match the existing pattern.
- **Modal**: jobs (aligner, speaker-embed, allosaurus) scale to zero when idle
  (`deployed, Tasks 0` = normal/free, not down). Warm containers serve old code for
  ~300s after redeploy — `modal app stop <app>` before relying on new code. Any app
  pulling `lexide-pronunciation-unified-vad-clean` (it is **private**) must pass
  `secrets=[modal.Secret.from_name("huggingface-secret")]` on *both* the image
  build and the `@app.cls` — the secret supplies `HF_TOKEN`, which
  `huggingface_hub` picks up on its own. The aligner silently lacked this from
  the day the repo went private: `from_pretrained` 401s inside `@modal.enter`,
  the container never becomes ready, and callers just **hang** with no error
  (fixed 2026-08-09). If a Modal call hangs, read `modal app logs <app>` before
  suspecting the client. Pin model
  commits in Modal code so cache namespaces can't silently drift.
- **Secrets** (`.env`): `GROQ_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, espeak paths.
  `HF_TOKEN` lives in the *parent* `../.env`. (A stale quoted `GROQ_API_KEY` may shadow
  `.env` from the shell — read keys straight from `.env` if you hit a 401.)
- **Commits**: don't commit unless asked (permission is single-use). A `codex-review`
  hook reviews each commit; use judgement on its findings.
