# Pronunciation model vocabulary

Fresh training uses the explicit `phonemes` list in
`tagging/lexide/data/training_labels.json`, not a borrowed tokenizer's classes.
This is a subset of G2P's shared `Phoneme` inventory, not every phone G2P or
WikiPron can represent. Rust preprocessing checks both inventories; Python
finalization checks the model subset. Every JSON spelling must exactly match the
shared enum’s canonical NFC spelling; accepting an alias during Rust deserialization
is insufficient for Python’s exact string comparisons. Adding a legitimate model contrast belongs
to YAP-39 and requires training coverage, not just enum membership.

## Fresh training and checkpoint resume

The cleaned inventory has **347 segmental phones**. The tokenizer adds only
`<pad>` (CTC blank, ID 0) and `<unk>` (required unknown sentinel), for **349 slots**.
Both are masked out of the conditional phone logits, and neither is a valid
training target. Stress, tone and pitch remain separate factors.

Fresh training assigns deterministic IDs and sizes the new head to that tokenizer.
`--processor-source` supplies only the audio feature extractor. Resuming loads the
processor saved in `--resume-from`, without adding, removing or renumbering tokens;
a missing processor is an error. Inference already loads the checkpoint's own
processor. Distillation still rejects teachers whose classes are absent from the
student; use a teacher trained on the clean vocabulary rather than projecting away
its probability mass. No deployed checkpoint or inference decoder is modified by this cleanup.

The dataset loader errors on an unknown phone or control token, rather than
silently removing it and shortening the pronunciation. Regenerate old labels
before the next fresh train; never delete individual unknown phones to make a row
fit. Existing alignment/narrowing data must be regenerated or explicitly revalidated
against the refreshed sequences. No corpus rewrite or retraining was performed here.

## YAP-38 inventory audit (2026-09-18)

The prior 470 entries (393 inherited + 77 extensions) contained 123 entries outside
the shared segmental inventory. Retaining two required controls leaves 349 slots.
The 347 retained entries preserve every existing model phone recognized by the
shared enum; this change adds no acoustic classes and performs no phonetic mergers.

### Structural controls

`</s>`, `<pad>`, `<s>`, `<unk>`, `|`

Keep `<pad>` and `<unk>` internal. Fresh models do not need BOS/EOS or word-delimiter
classes: training consumes pre-tokenized phone lists and word boundaries are metadata.

### Tone-number labels

`1`, `a1`, `a2`, `a4`, `a5`, `ai2`, `ai5`, `ei2`, `ei5`, `i.1`, `i.2`, `i.4`, `i.5`, `i1`, `i2`, `i4`, `i5`, `iou1`, `iou2`, `iou4`, `iou5`, `iɑ1`, `iɑ2`, `iɑ5`, `iɛ1`, `iɛ2`, `iɛ4`, `iɛ5`, `iː1`, `i̪1`, `i̪2`, `i̪4`, `i̪5`, `o1`, `o2`, `o4`, `o5`, `onɡ2`, `onɡ5`, `ou1`, `ou2`, `ou5`, `u1`, `u2`, `u4`, `u5`, `ua1`, `ua2`, `ua4`, `ua5`, `uai5`, `uei2`, `uei5`, `uo1`, `uo2`, `uo5`, `uə2`, `uə5`, `y1`, `y2`, `y5`, `yu2`, `yu5`, `yæ2`, `yæ5`, `yə2`, `yə5`, `yɛ2`, `yɛ5`, `yɛ5ʲ`, `ɑ1`, `ɑ2`, `ɑ4`, `ɑ5`, `ɑu2`, `ɑu5`, `ə1`, `ə2`, `ə4`, `ə5`, `ər1`, `ər2`, `ər4`, `ər5`, `əː1`

These encode suprasegmental information or old mnemonic leakage, not new segmental
classes. Current G2P separates tones; Persian `q1` leakage was fixed at its source.
Do not strip digits from old rows: refresh the whole pronunciation so aligned tone
and span data come from the current G2P implementation.

### Standalone modifier

`ʲ` is not a standalone segmental class. Palatalized consonants such as `tʲ` and
`ɫʲ` remain valid. Historical linking-ʲ output needs relabeling from the current
engine, not blind deletion or substitution with a glide.

### Legacy notation and artifacts

`??`, `N`, `S`, `X`, `a.`, `a.ː`, `dZ`, `d[`, `d^`, `e:`, `i.`, `i.ɜ`, `i.ː`, `i:`, `o:`, `oe:`, `r.`, `s.`, `s^`, `tS`, `t[`, `t^`, `t^ː`, `ts.`, `ts.h`, `u"`, `u.`, `u.ː`, `u:`, `y:`, `ɑ:`, `ɪ^`

These are non-IPA mnemonic/ASCII or punctuation-bearing spellings, including `??`,
`d[`, `tS`, `dZ`, and colon-length forms. Their rejection does not eliminate the
underlying sounds: retain canonical IPA phones already in the inventory, and let
G2P emit the current pronunciation. No context-free alias remap is applied to
historical training rows.

### Corpus evidence

A read-only scan of 747,050 rows across 22 local broad/narrowed files found only
standalone `ʲ` (88,588 occurrences) and `1` (1,862) among the removed labels.
Broad/narrowed files overlap, so these are row/token counts, not distinct recordings.
`1` occurs in Persian (1,166) and English/mixed-language rows (696); `ʲ` occurs
across nine language directories. Other removed labels had zero occurrences in
this local snapshot. This is not a claim that every remote training dataset was audited.

Current G2P already rejects these spellings via `Phoneme`; Cantonese/Vietnamese
tones were moved into metadata in G2P 0.6.0. No further G2P output change is needed
for this inventory cleanup. Raw old checkpoint labels remain readable; typed
extraction continues to report an unsupported winner explicitly.
