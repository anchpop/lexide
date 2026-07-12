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
  boundaries (~0.31M params) — tokenization-free segmentation.

Trained on Lambda (single A10, ~2.5h, bf16, 2 epochs). Both pushed to HF `anchpop/lexide-parsley`.

**Sentence segmenter** (`sentence-labeller/`). A *third*, independently-trained byte-minGRU with
the **same architecture** as the CharBoundaryTagger, but its O/B/I spans are **sentences** not
tokens (B = sentence begins, I = inside a sentence, O = a gap between sentences). It turns a raw
passage — or each entry of a list — into its sentences, so callers can feed one-sentence-at-a-time
to the tagger. Data is LLM/mechanically labelled passages: the Rust `sentence-labeller` binaries
generate synthetic prose (`generate`) and mechanically compose passages from the tokenization
sentence pools with varied gaps/wrappers/leaders (`augment`), and an LLM labels real Harry Potter
+ synthetic passages into `sentence`/`gap` sections (`label-sentences`). `sentence_data_prep.py`
flattens those into per-byte spans (22k train passages / 263k sentences; real held-out val/test),
`train_segmenter.py` trains locally on the box's GPU (~0.31M params), and `export_segmenter.py`
dumps `sentence_segmenter.safetensors` + parity fixtures. Exposed as `Lexide::segment_sentences`
(local, in-process) / the parsley serve `/segment` endpoint (remote).

**Results** — held-out **silver** test (measures agreement with the Gemma teacher, not gold):
overall POS 98.3 / lemma 97.9 / UAS 93.1 / **LAS 91.7**.

| tier | langs | POS | LAS |
|------|-------|-----|-----|
| European | fra ita spa por deu eng rus | ~99 | 92–95 |
| low-resource | kor, hin | 95–96 | ~84 |
| **weak** | **jpn** | **85** | **65** |

Quality tracks per-language data volume: jpn/kor/hin (22–96k sentences) lag the European
languages (300–400k each). Char tokenizer: **99.78% token-span F1**.

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
- **Fix Japanese** — the biggest POS+lemma win: rebalance the training mix (weighted sampler so
  22k Hindi / 38k Japanese aren't drowned by 400k European) and/or get more low-resource silver.
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
- **Not shipped:** Japanese (POS/LAS 85/65).
- **Cold starts:** ~26s scale-to-zero on Modal; `min_containers=1` for zero cold starts (ongoing
  warm-container cost). The Rust `local` backend runs in-process (load ~10-15s, disk-bound on the
  1.1 GB fp32 graph — quantization shrinks it); a Fly deploy of it is the remaining serving work.
- **Toolchain on this box:** sky/Lambda via `~/.sky-venv` (+ gcc `LD_LIBRARY_PATH`); Modal via `~/.modal-venv`; Rust via `direnv exec /data/coding/yap`.
- **Gitignored (regenerate, don't commit):** `data/big/`, `data/processed/`, `data/lemma_tables/`, `data/onnx/`, `tagger/output/`.
