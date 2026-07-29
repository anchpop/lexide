# lexide tagger — small deployable replacement for the Gemma tagging pipeline

Goal: replace the expensive autoregressive Gemma 4 31B tagger with a small, cheap model
that does the same job — **tokenization + POS + lemma + dependency (head & relation)**
across 10 languages (deu, eng, fra, hin, ita, jpn, kor, por, rus, spa).

## Why this design

The old pipeline makes a decoder re-emit the whole analysis as text, paying
generation-length compute and risking format drift. Tagging is a *per-token labeling*
problem, so instead we use an **encoder with multi-task heads**: one parallel forward
pass, four cheap heads reading off the same representation. ~100× smaller than the 31B
teacher and CPU-servable when quantized.

The one real wrinkle — *our token boundaries don't line up with any subword vocabulary* —
is handled two ways:

1. **Tagging** uses **offset-based subword→word pooling**: the encoder tokenizes however
   it likes; for each of *our* tokens we pool the encoder subword whose character offsets
   cover that token's first char. Boundaries never need to agree.
2. **Tokenization itself** is a separate tiny **byte-level minGRU** that predicts token
   boundaries (per-byte O/B/I) directly from raw text — no subword vocabulary involved.

## Components

| File | What |
|------|------|
| `data_prep.py` | Normalizes silver+gold into a unified schema with exact char offsets; builds POS/DEP/lemma-script vocabs; writes train/val/test splits. |
| `model.py` | `MultiTaskTagger` (encoder + POS head + lemma edit-script head + Dozat-Manning biaffine dependency head) and `CharBoundaryTagger` (byte-level bidirectional minGRU). |
| `dataset.py` | Torch datasets + collation; the offset→word pooling map; byte O/B/I labels. |
| `train.py` | Trains the multi-task tagger. Runs a short on-GPU **smoke phase** first, then full training. Metrics: POS acc, lemma acc, UAS, LAS, per language. |
| `train_tokenizer.py` | Trains the char boundary tagger; reports token-span F1. |
| `prior.py` | The **boundary prior** — a per-byte proposal fed alongside the bytes (below). |
| `build_wordbanks.py` / `build_unidic_artifact.py` | Corpus unigram wordbanks, and the packed UniDic artifact the Rust Viterbi reads. |
| `predict.py` | End-to-end inference: raw text → tokens with POS/lemma/head/dep. |
| `sky_tagger.yaml` + `run_node.sh` | Lambda launch (single x86 GPU, autostop/autodown) + node orchestration (train → push to HF → train tokenizer → push). |
| `export_onnx.py` / `export_modal.py` | Tagger → single ONNX graph, numerically verified vs PyTorch; the Modal wrapper runs it with the HF weights. |
| `export_char_modal.py` | Char-minGRU weights → safetensors + multilingual reference fixtures (for the Rust reimpl's bit-parity test). |
| `parse_wiktextract.py` / `build_lemma_priors.py` / `lemma_lookup.py` | Wiktionary lemma tables, training-data candidate priors, and the layered OOD lemma floor (see `LEMMA_LOOKUP.md`). |
| `record_parity_fixtures.py` | Records live-serve outputs as the Rust parity-test fixtures. |
| `../release.sh` | The whole post-training chain: export → verify → publish to HF → deploy → parity-gate (below). |

## The boundary prior

The tokenizer has to decide where words start from raw bytes. With whitespace that is
nearly free — an *unseen* German word is still 97% correct, because the spaces hand over
both boundaries. Japanese has no such signal, so every rare word has to be *known* rather
than copied: a Japanese token seen fewer than 500 times was 24–30% wrong, against 0% for
German. That gap is lexical knowledge, not label noise or data volume, and no amount of
either closes it.

So each sentence is accompanied by a per-byte proposal:

| language | source | precision |
|---|---|---|
| spaced languages | whitespace | 100% (Korean), 96.4% (Hindi) |
| Japanese | bundled UniDic + Viterbi (`segment::unidic`) | 84.4% |
| optional | corpus unigram wordbank + Viterbi | 92.6% (Korean) |

Feeding it lifted Japanese 86.6 → 94.5 and Korean 91.8 → 95.2.

Two things this design gets right, both learned the hard way:

* **A prior only helps where the model has no other route to the answer.** Giving Korean a
  wordbank raises boundary *recall* from 62.6% to 98.9% — strictly more information than
  whitespace alone — and costs **5.6 F1** (97.74 -> 92.2). The model, with whitespace anchors
  and 96k sentences, already infers eojeol-internal splits from context better than a unigram
  Viterbi proposes them, so the bank only adds a confident-looking opinion to partly defer
  to. Japanese has no anchor and genuinely lacks the lexicon, which is why the same mechanism
  is worth ~8 F1 there. Coverage is not the test; marginal information is.

  A caveat on how this was diagnosed: the first explanation was precision (whitespace is
  100% right, the bank 92.6%), and the fix was to encode "certain" and "proposed" as
  different symbols (`B` / `B_SOFT`). Measured, **it does nothing** — soft scores 91.80
  against hard's 92.24 — and counting the symbols says why. In the shipped configuration
  `B_SOFT` is inert:

  | lang | B | B_SOFT | B per sentence |
  |---|---|---|---|
  | jpn | 595 | 4940 | 1.07 |
  | all others | 1816-8846 | 0 | 3.4-5.9 |

  Japanese has no internal whitespace, so a sentence is a single run: the first token gets
  `B` and every other token gets `B_SOFT`. The symbol therefore encodes "not
  sentence-initial" within Japanese, and "is Japanese" across languages — which the language
  token already carries. It would mean something for a language with *both* whitespace and a
  dictionary (Korean with a wordbank: `B` the exact eojeol boundary, `B_SOFT` the 92.6%
  internal split), which is exactly the configuration the measurement above rejects. Worth
  dropping to `PRIOR_VOCAB=4` at the next retrain.
* **The prior gets its own coordinates, not the byte's.** Layer 0 is linear, so summing the
  prior into the byte embedding computes `W(emb + prior)` — the prior's contribution is
  forced through the *byte* projection. Concatenating computes `W_byte·emb + W_prior·prior`,
  which with only 5 prior symbols can express anything the additive form could and cannot
  perturb byte identity. Worth +0.33 F1 for 5,792 params (`--prior-mode concat`).
* **Train and inference must share the implementation.** Training priors are precomputed by
  the same Rust binary that runs at inference (`emit-priors --out`, read via
  `PriorSidecar`), so the proposal distribution a model learns against cannot drift from the
  one it later sees. It is also much faster than a Viterbi in every dataloader worker.

## Data

`data_prep.py` produced `data/processed/{train,val,test}.jsonl` + `vocab.json` from
2.89M sentences (silver `data/big/*` + gold `train/data/cleaned_*`). Key facts:

- 18 UPOS tags, 65 dependency relations (UD subtypes kept).
- Lemmas are handled as **edit scripts** (`"p|s|ins"`: keep p-char prefix + s-char
  suffix of the form, splice `ins` in the middle). A global 4000-script vocab covers
  **99.4%** of tokens; the rest fall back to copy-the-form.
- Heavy language skew (deu/fra/por ~400k each; hin 31k, jpn 48k, kor 96k). Chinese has
  no data. Eval sets are capped per language for balance.
- Char offsets are exact by construction (token text + whitespace reconstructs the
  sentence), so pooling alignment is never fuzzy.

## Labels & metrics

- **POS**: token-classification accuracy.
- **Lemma**: edit-script classification accuracy (then applied to the surface form).
- **Dependencies**: biaffine arc scorer over `[ROOT, w1, …, wN]` → **UAS**; relation
  scorer at the chosen head → **LAS**. Greedy argmax decoding (MST decoding is a
  possible refinement).

## Running

Local data prep (pure python, no GPU):

```bash
python3 data_prep.py --big-dir data/big --gold-dir train/data --out-dir data/processed
```

Launch training on Lambda (single GPU, autostop 20 min / autodown):

```bash
export HF_TOKEN=... WANDB_API_KEY=...
sky launch -c lexide-parsley tagger/sky_tagger.yaml \
  --secret HF_TOKEN --secret WANDB_API_KEY -i 20 --down --retry-until-up -y
```

Artifacts are pushed to `anchpop/lexide-parsley` on HF (`tagger/` and `tokenizer/`
subfolders) because autodown wipes the node disk.

Inference:

```bash
python3 predict.py --tagger-dir output/tagger/best --tokenizer output/tokenizer/tokenizer.pt \
  --text "Eine Fundgrube."
```

## Release pipeline (after training)

Once training has pushed `tagger/best` + `tokenizer/tokenizer.pt` to HF, one command
turns them into a verified, published, deployed release:

```bash
cd tagging && ./release.sh
```

It chains, stopping at the first failure: ONNX export (numerically verified against
PyTorch, on Modal) → char-minGRU safetensors + reference fixtures (Modal) → pull to
`data/onnx/` → rebuild training-data lemma priors (`build_lemma_priors.py`) → compile
Wiktionary tables + priors to fst (`build-lemma-fst`) → Rust unit tests (bit-for-bit
char-tokenizer parity against the fresh fixtures) → upload the complete `onnx/` set to
HF `anchpop/lexide-parsley` → deploy the parsley Modal serve → re-record parity fixtures
from the live endpoint (`record_parity_fixtures.py`) → run the Rust↔serve
token-for-token parity test. Green at the end means: what's on HF, what the serve runs,
and what the Rust `local` backend computes are all the same model producing identical
tokens. Commit the refreshed `lexide/tests/fixtures/parsley_reference.json` afterwards.

Prereqs on the box: `~/.modal-venv` (Modal CLI), cargo (direnv exec of the yap flake is
auto-detected), a write-role `HF_TOKEN` in the repo-root `.env`, and — for the priors
step — `data/processed/` + `data/lemma_tables/` (it warns and keeps existing priors if
the training data isn't present).

## Choices worth revisiting

- **Encoder = xlm-roberta-base** was chosen for reliability (rock-solid fast tokenizer
  with offsets). `mmBERT` / `mdeberta-v3-base` are worth racing on dev — pass `--encoder`.
- **Dependency labels are silver** (from the teacher). Mixing in real **UD treebank gold**
  for these languages is the highest-leverage next step to push UAS/LAS past the teacher.
- If deployment is CPU-only, a follow-up distillation to a ~30–60M student is the next
  size cut after this proves out.
