# lexide tagging — project overview & status

Replacing the expensive autoregressive **Gemma 4 31B** tagger with a small, cheap model that
does the same job — **tokenization + POS + lemma + dependency (head & relation)** for 10
languages (deu eng fra hin ita jpn kor por rus spa) — to tag user-submitted sentences.

Component docs: `tagger/README.md` (models/training + the release pipeline), `tagger/LEMMA_LOOKUP.md`
(Wiktionary lemma floor), `modal/README.md` (parsley serve + cold-start notes).

**After training: `./release.sh`** — exports, verifies, publishes to HF, deploys the serve,
and gates on the Rust↔serve token-for-token parity test (details in `tagger/README.md`).

---

## Why

The old pipeline makes a 31B decoder re-emit the whole analysis as text — generation-length
compute on an always-warm A100 (`modal/modal_serve.py`, vLLM). Tagging is a per-token labeling
problem, so an **encoder with multi-task heads** does it in one forward pass, ~100× smaller and
CPU-servable. The one wrinkle — our token boundaries don't match any subword vocabulary — is
handled by offset-based subword→word pooling (for tagging) and a separate byte-level tokenizer
(for segmentation).

---

## Done

**Data** (`tagger/data_prep.py`). Normalized 2.89M sentences (silver from Gemma + gold
`cleaned_*`) into a unified schema with exact char offsets, POS/dep/lemma-script vocabs
(18 UPOS, 65 dep relations, 4001 lemma edit-scripts covering 99.4% of tokens), split
train/val/test.

**Models** (`tagger/model.py`). Two independently-trainable pieces:
- **MultiTaskTagger** — XLM-RoBERTa-base encoder + offset subword→word pooling, then a POS
  linear head, a lemma **edit-script** classifier, and a Dozat-Manning **biaffine** dependency
  head (head + relation).
- **CharBoundaryTagger** — a byte-level bidirectional **minGRU** predicting per-byte O/B/I token
  boundaries (0.99M params; emb 96 / hidden 192 / 4 BiMinGRU layers / 269-token vocab) —
  tokenization-free segmentation. No MLPs, residuals or attention: embedding → 4 bidirectional
  minGRU layers (each concatenating both directions, so information zigzags across depth) →
  one LayerNorm → Linear to 3 logits. The sentence segmenter is the same class.

Trained on Lambda (single A10, ~2.5h, bf16, 2 epochs). Both pushed to HF `anchpop/lexide-parsley`.

**Sentence segmenter** (`sentence-labeller/`). A *third*, independently-trained byte-minGRU with
the **same architecture** as the CharBoundaryTagger, but its O/B/I spans are **sentences** not
tokens (B = sentence begins, I = inside a sentence, O = a gap between sentences). It turns a raw
passage — or each entry of a list — into its sentences, so callers can feed one-sentence-at-a-time
to the tagger. Data is LLM/mechanically labelled passages: the Rust `sentence-labeller` binaries
generate synthetic prose (`generate`) and mechanically compose passages from the tokenization
sentence pools with varied gaps/wrappers/leaders (`augment`), and an LLM labels real Harry Potter
+ synthetic passages into `sentence`/`gap` sections (`label-sentences`). `sentence_data_prep.py`
flattens those into per-byte spans (real held-out val/test), `train_segmenter.py` trains it, and
`export_segmenter.py` dumps `sentence_segmenter.safetensors` + parity fixtures. Exposed as
`Lexide::segment_sentences[_in]` (local, in-process) / the parsley serve `/segment` endpoint
(remote, optional `lang`).

**Segmenter v2 (2026-07-24, deployed).** The 0.31M v1 split abbreviations ("Mr." became its own
sentence), ate quote attributions, and choked on headings — its train set had ~245 abbreviation
counter-examples in 263k sentences. v2 fixes this with (a) `augment` v2: per-language
abbreviation-sentence templates, quote-attribution wrappers ("…?" she asked. + analogues), and
heading-like gaps, scaled to 12k passages/lang from the silver pools (122k passages / 1.17M
sentences); (b) a **language-conditioned BOS** — ids 259-268 replace BOS when the language is
known (15% dropout keeps lang-free use working; old 259-row checkpoints still load everywhere);
(c) a bigger net: emb 96 / hidden 192 / 4 layers ≈ **0.99M params**. Val F1 **82.6 vs 80.6**
(real held-out; lang-free 82.56), test 84.2; all targeted abbreviation/attribution/heading cases
verified fixed. Trained on Lambda (`sky_byte_models.yaml`) in ~35 min after rewriting the minGRU
scan as a **Hillis-Steele doubling scan** (`model.py` — the sequential python loop was
kernel-launch-bound: 0.06 → 2.7 it/s). The Rust `byte_bio.rs` forward was likewise rewritten
(axpy over transposed weights, timestep tiling) — bit-identical output, ~3x faster native, ~8x in
wasm.

**Segmenter v3–v6 (2026-07-26, v6 deployed).** v3/v4 kept fixing *data*: `{N}`-ized templates
drawing street/company names from mined pools (v4 winner 40/40 on the pattern holdout). v5 and v6
then fixed the thing that was actually capping F1 — **label-policy noise**, not model capacity.
The jpn/kor/spa gap (81/77/81 against 92+ elsewhere) traced to the LLM labeller applying
contradictory sentence-splitting policies to identical constructions; rewriting the ambiguous
semantic instructions as **mechanical** ones (e.g. "check for the binder particle と/って/라고",
not "is this an attribution?") took labelling consistency from ~80% to ~97-99%. v6 added
script/screenplay formats to the mechanical augmenter and four **conditional** prompt bullets
appended only when the record contains the trigger (quote chars, dash, ellipsis, `lang==hin`), so
out-of-scope records keep byte-identical prompts and stay cached in tysm. Test F1 **92.70** (v5
scored 92.11 on the same new gold), **48/49** pattern cases; per language deu 95.1, eng 94.4,
fra 91.0, hin 91.5, ita 92.3, jpn 91.2, kor 91.2, por 96.8, rus 86.0, spa 98.2. Seed sweeps are
standard now (`SEG_SEEDS` in the sky yaml): seeds fail on *different* marginal cases, and the
6-seed sweep is what caught seed 2 regressing a dash-parenthetical guard case that seed 5 passes.
Remaining known bias: English attribution with no comma (`"…!" Hermione urged Neville.`)
over-splits — cross-language interference from the jpn/kor juxtaposition examples.

**Char tokenizer v2 (2026-07-25, deployed).** Growing the net alone never beat v1 (99.70 vs
99.78 token F1) — the *data* was the limit. 43k Gemma-labelled synthetic sentences (per-language
abbreviation templates, multi-word proper nouns incl. mined entities, ~25 frames/lang) plus a 3k
held-out eval set (`data/aug_holdout/`, committed as the yardstick) fixed it. Shipped tokenizer
is now the same 0.99M architecture as the segmenter (emb 96 / hidden 192 / 4 layers,
language-conditioned BOS): in-distribution token F1 **99.80** (but see the per-language table
below — that figure was deu+eng only), augmented holdout **93.6 vs 69.6**, dropped-text
**0.28% vs 4.18%** (v1 lost 54% of characters on abbreviation-dense Korean). "Eiffel Tower"
merges as one token again.

**Web demo** (`web-demo/`, live at <https://anchpop.github.io/lexide/>). The two byte-minGRUs
compiled to wasm (~195KB + 7.9MB weights, 3.96MB each), running fully in-browser: paste a passage, see
sentences/gaps/token spans live, with a `lang:` hint selector. Reuses `byte_bio.rs` verbatim via
`#[path]`; parity-tested in Node against the Python fixtures. Deployed from the `gh-pages`
branch (`build.sh` + copy `www/` there).

**Results** — held-out **silver** test (measures agreement with the Gemma teacher, not gold):
overall POS 98.3 / lemma 97.9 / UAS 93.1 / **LAS 91.7**.

| tier | langs | POS | LAS |
|------|-------|-----|-----|
| European | fra ita spa por deu eng rus | ~99 | 92–95 |
| low-resource | kor, hin | 95–96 | ~84 |
| **weak** | **jpn** | **85** | **65** |

Quality tracks per-language data volume: jpn/kor/hin (22–96k sentences) lag the European
languages (300–400k each).

**Char tokenizer, per language (2026-07-28).** The long-quoted "99.8% token-span F1" was
measured on `val_records[:2000]` — and the val file is written language-by-language, so that
prefix is **1500 German + 500 English**. Japanese was never in the number. Re-scored on a
language-stratified sample of the silver test split (400 sentences/language, shipped weights,
language token on):

| deu | eng | fra | hin | ita | **jpn** | kor | por | rus | spa |
|-----|-----|-----|-----|-----|---------|-----|-----|-----|-----|
| 99.9 | 99.7 | 99.4 | 97.6 | 99.8 | **86.6** | 91.8 | 99.9 | 99.9 | 99.8 |

Japanese is the floor by a wide margin (Korean second), which is unsurprising in hindsight:
they are the two languages with no whitespace to anchor a boundary, so every span has to be
inferred from context — and jpn has 52k training sentences against 300–420k for each European
language. `evaluate()` in `train_tokenizer.py` now stratifies by language and reports per-lang
F1 so this cannot hide again. Dropped-text is ~0 for jpn/kor; the errors are boundary
placement, not lost characters. Sampling the jpn disagreements (6 of 10 test sentences differ
somewhere): kanji compounds split down the middle (`翻訳` → `翻|訳`, `求め` → `求|め`) are
outright wrong, while others are teacher-policy calls — auxiliary attachment (`ました` vs
`まし|た`), compound verbs (`苛立ち始め` vs `苛立ち|始め`), and the `って` binder (`トム|って` vs
`トムって`) — so part of the 13-point gap is silver noise from the teacher whose own jpn is
weakest (POS 85 / LAS 65). Fixing jpn tokenization is the same lever as fixing jpn tagging:
more/better jpn silver, or a weighted sampler.

**Boundary prior (2026-07-29).** That last sentence was wrong, and the diagnosis it rested
on was too. Three hypotheses were tested: label noise (real — jpn gold disagrees with itself
1.39% of the time against deu's 0.02% — but fixing it moved F1 ~0), data volume (projected
10x more jpn data ≈ +3.5 points, then saturating), and **lexical knowledge**. The third one
is it, shown by a German control: bucketing errors by how often a token appeared in training,
an unseen German word is 3.0% wrong while an unseen Japanese word is **30.5%** wrong. German
gets both boundaries free from whitespace no matter how strange the word; Japanese has to
already know it.

So the model is now handed a per-byte proposal alongside the bytes — whitespace where that
is exact, a dictionary + Viterbi where it is not (`tagger/prior.py`, `segment::prior`). One
extra embedding, summed into the byte embedding, +384 params:

| | jpn | kor | overall |
|---|---|---|---|
| before | 86.6 | 91.8 | 97.17 |
| with prior | **94.5** | **95.2** | **98.66** |

Those two rows are **not** a fair comparison, and the honest version is bigger. The 86.6 was
scored against the *unmerged* Japanese gold the old model was trained for (食べ|まし|た);
we then changed policy to merge a predicate with its auxiliary chain. Re-scoring both models
on the same 150 validation sentences with the same code, against both policies:

| model | vs merged gold (what we target) | vs unmerged gold |
|---|---|---|
| old, no prior at all | 74.57 | 82.30 |
| curriculum model, no dictionary | 80.38 | 71.49 |
| curriculum model, dictionary | **92.11** | 84.46 |

Each model scores best against the policy it was trained for, which is what makes the
headline pair misleading. Against the policy we actually target the work is worth
**74.6 -> 92.1**, of which **+11.7 is the dictionary alone** (80.4 -> 92.1, same model, same
gold, same sentences).

**A caution on every number in this section.** byte-v12 and byte-v13 differ only in whether
the inert `B_SOFT` symbol exists, yet they score jpn 93.06 and 90.94 — so run-to-run
variance on Japanese is around **2 points**, and the per-language eval is 400 sentences.
Single-run differences smaller than that (which is most of the ones recorded above) are not
evidence of anything. The effects that survive this are the large ones: the prior itself,
the dictionary, and the Korean wordbank regression.

The rare-frequency error buckets collapse 2.4x. Two findings worth keeping:

- **A prior helps only where the model has no other route to the answer.** Giving Korean a
  wordbank raises boundary recall from 62.6% to 98.9% — strictly more information — and
  **costs 5.6 F1** (97.74 -> 92.2 on the jpn+kor harness). With whitespace anchors and 96k
  sentences the model already reads eojeol-internal splits from context better than a unigram
  Viterbi proposes them, so the bank only supplies a confident-looking opinion to defer to.
  Japanese has no anchor and genuinely lacks the lexicon, which is why the same mechanism is
  worth ~8 F1 there. Korean and Hindi therefore ship with plain whitespace.

  Worth recording as a wrong turn: the first diagnosis was *precision* (whitespace 100%,
  bank 92.6%), and the fix was to encode certain and proposed boundaries as different
  symbols (`B` / `B_SOFT`). Measured, it does nothing — soft 91.80 vs hard 92.24. Counting
  the symbols shows why: with no wordbanks shipped, `B_SOFT` appears only in Japanese, where
  a sentence is one whitespace-free run, so the first token is `B` and all the rest are
  `B_SOFT` (1.07 `B` per sentence). It encodes "not sentence-initial", and across languages
  "is Japanese" — which the language token already says. There was never any signal in it to
  measure. It would carry information for a language with both whitespace and a dictionary,
  which is the configuration this same measurement rejects. Drop to `PRIOR_VOCAB=4` at the
  next retrain.
- **The prior is load-bearing, and that is a liability as much as a feature.** Measured on
  the shipped v11 weights over 150 Japanese validation sentences:

  | | jpn F1 |
  |---|---|
  | language token + dictionary prior | 93.33 |
  | no language + dictionary prior | 93.19 |
  | no language + whitespace prior | **22.41** |
  | language token + whitespace prior | **21.17** |

  Take the dictionary away and Japanese collapses by ~71 points — the language token cannot
  rescue it. The model has not learned Japanese boundaries *plus* a hint; it has learned to
  trust the hint. That is why `CharTokenizer::load` refuses a prior-trained checkpoint whose
  prior data is missing instead of falling back to whitespace, and why a caller who supplies
  no language gets one inferred from script (`prior.infer_lang`) rather than a whitespace
  proposal that actively asserts the sentence is one word.

  It also went unnoticed because the lang-free eval dropped the language *token* while still
  handing the prior the true language, so it reported 93.79 for a configuration real callers
  never saw. `evaluate()` now drops both. A training run that also dropped the prior's
  language occasionally would make the failure graceful rather than catastrophic; that is
  the obvious next change.
- **The prior gets its own coordinates.** Layer 0 is linear, so adding the prior into the
  byte embedding computes `W(emb + prior)`, forcing it through the byte projection;
  concatenating computes `W_byte·emb + W_prior·prior`. With 5 prior symbols the narrow
  dedicated channel expresses everything the wide additive one could, without perturbing
  byte identity. +0.33 F1 for 5,792 params.
- **The Japanese dictionary is bundled, not a dependency.** MeCab's Viterbi is reimplemented
  in Rust over a packed 83MB UniDic artifact (570k surfaces, the full 5981x5981 connection
  matrix, unk.def unknown-word rules), reproducing fugashi's segmentation exactly on the test
  split. Training priors are precomputed by that same binary, so the proposal a model trains
  against cannot drift from the one it sees at inference.

**Lemma quality investigation.** A blind `claude-fable-5` judge on 100 hard tokens found the
silver lemmas are **~99% accurate** (only ~1 real error; most disagreements are annotation
*policy*, e.g. jpn です→だ). Conclusion: the lemma lever is small and external gold would import
a *different* policy — so **don't override in-distribution labels**. But a Wiktionary
`(form,POS)→lemma` table is an excellent **out-of-distribution floor** (`tagger/lemma_lookup.py`,
built by `tagger/parse_wiktextract.py`): on out-of-training content-word forms it lifts lemma
accuracy over copy-the-form by **+23 (deu) / +39 (rus) / +28 (spa) / +21 (fra) / +12 (eng)**,
agreeing with/correcting Gemma 93–96%. Applied to content POS only (proper nouns copy).
Multi-candidate entries are resolved by **training-data priors** (`tagger/build_lemma_priors.py`:
prefer training's lemmatization of the exact form, then training lemma frequency, then closest
length — fixes homograph picks like eng `love→lofe`). Tables built for 9 languages (jpn omitted
— gated + weakest fit); in `data/lemma_tables/` (gitignored).

**Deployment — `parsley` 🌿** (`modal/modal_serve_tagger.py`). CPU Modal serve, scale-to-zero,
**live** at `https://anchpop--lexide-parsley-parsley-tag.modal.run`. `POST {sentences, lang}` →
JSON tokens. Reuses the existing `huggingface-secret`. All 9 languages. ~0.4s warm, ~26s cold.
(Memory snapshots don't help here — Modal rebuilds them each cold start; see `modal/README.md`.)

**Rust client** (`lexide/src/`). The `lexide` crate speaks parsley's JSON format alongside
the Gemma text format: `Lexide::from_parsley_server(url)`, a `ResponseFormat` dispatch, and
whitespace rebuilt exactly from char offsets (shared with the local backend in `src/raw.rs`).

**ONNX export** (`tagger/export_onnx.py`, `tagger/export_modal.py`). The tagger exports to a
single ONNX graph (encoder + pooling + all heads), **verified to match PyTorch to ~1e-5** at
multiple shapes — the biaffine + gather ops survive the export. 1129 MB fp32; on the `lexide-onnx`
Modal volume + local `data/onnx/`. The char-minGRU doesn't ONNX-export (sequential scan), so
`tagger/export_char_modal.py` dumps its weights as safetensors + reference fixtures instead.

**Rust local inference** (`lexide/src/local/`). The `local` feature now runs the whole parsley
pipeline in-process on CPU — mistralrs/Gemma-E2B is retired. Stack as planned: **`ort`** for the
ONNX tagger, HF **`tokenizers`** for XLM-R subwords, a pure-Rust **byte-minGRU** reimpl
(verified bit-for-bit against Python on multilingual fixtures), edit-script decode, and the
Wiktionary lemma tables compiled to **`fst`** (`build-lemma-fst` bin: 185 MB JSON → 30 MB total,
candidate selection resolved at build time). Verified **token-for-token against the live parsley
serve** on 25 sentences × 10 languages (`lexide/tests/parsley_parity.rs`): text, POS, lemma,
dep, head all identical. ~55 ms warm per sentence on CPU; load is ~10-15 s on this box
(disk-bound — the fp32 graph is 1.1 GB, so int8 quantization is also the load-time fix).
`Lexide::from_pretrained(LocalConfig)` reads `LEXIDE_MODEL_DIR` (see `lexide/README.md`).

---

## Where we're going

1. **Fly service** — thin Rust binary reusing the crate's `local` backend; ~ms cold starts.
2. **int8 / ONNX quantization** — encoder → ~280MB, faster load + inference (also shrinks the
   Fly image).
3. ~~Distribute the ONNX artifacts~~ — done: everything the Rust backend needs (graph,
   tokenizer, vocab, char-minGRU safetensors, fst lemma tables) is on HF
   `anchpop/lexide-parsley/onnx/` (repo renamed from `lexide-tagger`; old name redirects)
   and mirrored on the `lexide-onnx` Modal volume.

Quality follow-ups (separate from the Rust work):
- **Fix Japanese** — the biggest POS+lemma win, and (newly measured) the biggest *tokenizer* win
  too: jpn token F1 86.6 / kor 91.8 against ~99.8 elsewhere. Rebalance the training mix (weighted
  sampler so 35k Hindi / 52k Japanese aren't drowned by 300–420k European) and/or get more
  low-resource silver.
- **Measure real accuracy** — everything so far is teacher-agreement; a small gold set (or a
  morphological analyzer for jpn/kor) would give true numbers.

---

## The old Gemma tagger (teacher) — not forgotten

The first-generation tagger (`train/`, LoRA on Gemma 4 31B; see `train/README.md`) is replaced
*online* by parsley but stays load-bearing: it's the **teacher** that generates silver (the jpn
fix needs it), `train/data/cleaned_*.jsonl` is the **gold** set data_prep mixes in (11 langs
incl. zho, which parsley doesn't cover), and it's the only shippable tagger for **Japanese**.
Its vLLM serves stay deployed on Modal (scale-to-zero → idle cost ≈ 0). Decommission the serve
only after yap switches to parsley (the lexide crate's `RemoteConfig` **default URL is still the
Gemma endpoint**) and jpn is fixed or gated.

---

## Key facts

- **HF model:** `anchpop/lexide-parsley` (renamed from `lexide-tagger`, old name redirects):
  `tagger/best`, `tagger/final`, `tokenizer/`, and `onnx/` (the complete Rust local-backend
  artifact set incl. fst lemma tables). The Modal secret's token is read-only (fine for
  serving); a write token exists locally for uploads.
- **Live endpoint:** `https://anchpop--lexide-parsley-parsley-tag.modal.run` (Modal app `lexide-parsley`, workspace `anchpop`).
- **Not shipped:** Japanese (POS/LAS 85/65; char tokenizer 86.6 token F1 — see the per-language
  table above).
- **Cold starts:** ~26s scale-to-zero on Modal; `min_containers=1` for zero cold starts (ongoing
  warm-container cost). The Rust `local` backend runs in-process (load ~10-15s, disk-bound on the
  1.1 GB fp32 graph — quantization shrinks it); a Fly deploy of it is the remaining serving work.
- **Toolchain on this box:** sky/Lambda via `~/.sky-venv` (+ gcc `LD_LIBRARY_PATH`); Modal via `~/.modal-venv`; Rust via `direnv exec /data/coding/yap`.
- **Gitignored (regenerate, don't commit):** `data/big/`, `data/processed/`, `data/lemma_tables/`, `data/onnx/`, `tagger/output/`.
