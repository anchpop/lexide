# Pronunciation preprocessing

Run from `pronunciation/`:

```sh
cargo run --release --manifest-path preprocess/Cargo.toml -- \
  --skip-narrowing --python /opt/homebrew/Caskroom/miniconda/base/bin/python3
```

On NixOS, use `--python scripts/py-linux.sh`. Otherwise `--python` defaults to
`python3` from PATH. The interpreter needs the existing audio dependencies
(`numpy`, `soundfile`, `tqdm`; acoustic/speaker stages also use their existing
Praat, Modal and scikit-learn dependencies). The Rust build requires cmake and a
C compiler for g2p. Keep this crate in the checkout: it locates the Python helpers
and default corpus paths relative to its source directory.

Rust discovers language manifests and drives these stages in order:

1. Python filters noncommercial and silent recordings and resolves recording
   metadata to a combined g2p language.
2. Rust deserializes that language into `g2p::Language` and calls
   `g2p::phonemize(language, text)` directly. No installed g2p CLI, `g2p serve`,
   Python g2p client, or persistent pronunciation cache participates.
3. Python adapts the returned annotations to the training schema, applies stress
   overrides and acoustic/confidence masks, validates the model vocabulary and
   writes `phonemes.jsonl` plus explicit refusals in `g2p_exclusions.jsonl`.
4. Unless skipped, Rust invokes acoustic narrowing, builds/runs `vad_compute`,
   and invokes the existing speaker embedding/clustering helpers. The Modal
   service is unchanged.
5. Once every language succeeds, Rust refreshes mixed-script exclusions and packs
   the entire audio directory into `.work/pron_audio.tar` exactly once.

The Python helpers are individual stages in `train/scripts/preprocess_support.py`;
there is no second Python pipeline driver. Temporary JSONL files connect filtering,
Rust phonemization and training-schema adaptation, and are deleted after each job.
An engine failure stops that language before replacing its labels; explicit g2p
refusals exclude only those recordings. Vocabulary failures also preserve the
previous language output. Any failed language prevents packing. Build identity is
recorded on outputs as provenance, not used as a compatibility gate or cache key.

`--jobs N` runs language jobs concurrently, with logs under
`.work/preprocess_parallel/`. `--langs eng fra` selects languages; packing still
includes the whole corpus. `--allow-noncommercial` explicitly includes NC rows.
Silence and speaker-embedding caches remain because they avoid audio/GPU work.

The current aligner has untrained rows for merged labels, so use
`--skip-narrowing` until its pin is updated. This keeps existing narrowed files;
training that uses them continues to see those previous labels. The refusal guard
remains in place if narrowing is requested. Speaker refresh uses the deployed
Modal app and supports only the canonical corpus directory.

For an offline check on a scratch corpus:

```sh
cargo run --manifest-path preprocess/Cargo.toml -- \
  --data-dir /tmp/audio-sample --langs eng fra --jobs 2 \
  --skip-narrowing --skip-vad --skip-speaker-cluster --no-pack \
  --python /path/to/python3
```

Omit `--skip-vad` to exercise the local Rust VAD step. To test packing without
changing training artifacts, omit `--no-pack` and specify both
`--output-tar /tmp/sample.tar` and `--exclusions-output /tmp/exclusions.jsonl`.
Standalone ASR/engine audits still use the Python g2p client; removing that
remaining transport is tracked in YAP-26.
