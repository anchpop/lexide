# Phonemization through g2p

Production corpus labeling uses one API for every language:
`g2p_client.phonemize(text, language)`. g2p selects its implementation
and returns structured phonemes, stress, word spans, and any tone, pitch or
syllable annotations. Lexide does not choose a provider or maintain a list
of languages requiring a different labeling path.

`corpus_labels.py` adapts that shared response to the training file schema.
It selects the combined g2p language (including decoding historical
Spanish/Portuguese `variety` and `espeak_voice` metadata), validates factor alignment, and applies the
recording's transcription-confidence gate to pitch labels. Preprocessing
also applies acoustic accent exclusions and rhythmic-group stress overrides.
Those gates concern recordings, not phonemization engines.

## Cache and exclusions

Preprocessing stores raw responses and explicit g2p refusals in each language's
`.cache/g2p_labels.sqlite3`, keyed by exact text, combined language and g2p build
identity. Sentence edits and build changes therefore cannot reuse stale labels.
Infrastructure failures abort the run and are not cached as exclusions.

Every recording retained by the license/silence filters is phonemized or has an
explicit refusal. Refusals are written to `g2p_exclusions.jsonl` with the exact
sentence, combined language and build identity. Successful rows in `phonemes.jsonl` carry
`g2p_identity`, `g2p_language` and the generic source `g2p`. Existing corpus files are
not migrated until preprocessing is run. Resolved varieties are stored on
generated label rows; preprocessing does not duplicate them back into manifests.

ASR auditing calls the same API for both reference and recognized text. Pimsleur
ingestion saves audio and transcripts; labelability is decided in preprocessing
for all languages, so an ingestion-time engine check cannot discard recordings.

The `--phoneme-backend` override and unused `--espeak-batch-size` option are
removed. The old provider sidecars are not inputs to production preprocessing.
`audit_g2p_backends.py` and `build_external_phoneme_sidecars.py` remain independent
historical engine-comparison/reproduction tools, not production dependencies.

Use `--langs` to select a corpus subset. Unsupported language requests
fail at g2p rather than being silently skipped using a local capability table.
The existing `--skip-narrowing` requirement for merged-token labels still applies.

The Python transport still uses `g2p serve`; replacing that transport with a
native Rust corpus pipeline is tracked separately in YAP-26.
