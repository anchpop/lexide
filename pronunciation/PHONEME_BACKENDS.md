# Where training labels come from, per language

Every phoneme label in this corpus comes from exactly one G2P source, and
which source is not a preference — it is a correctness constraint. A model
trained on labels from engine A cannot be scored against targets from engine
B: the two disagree about the phoneme inventory, about tokenization, and
about what counts as one segment. The mismatch is silent. Shapes match,
hashes match, training converges, and the damage only shows up as a language
that is mysteriously worse than its neighbours.

That failure has already been observed downstream. yap scored Hindi film
audio against eSpeak `hi` targets and measured hin as by far its worst
language (AUC 0.75–0.80 against 0.93–0.97 for Spanish and French). The model
was fine. eSpeak was simply never Hindi's label source, so ~290 standalone
`ʰ` tokens in the targets did not exist in the model's vocabulary at all.

## The table

| lang | source | provider | why not eSpeak |
|---|---|---|---|
| `eng` `deu` `fra` `ita` `por` `spa` `rus` | eSpeak (fork) | `LANG_TO_ESPEAK` | — |
| `jpn` | g2p crate (`src/japanese`) | `g2p-jpn` | kanji readings are lexical, not derivable from the glyphs; pitch accent needs a dictionary. OpenJTalk via the `jpreprocess` Rust rewrite with the NAIST dictionary bundled; `pyopenjtalk` stays in `PROVIDERS`. 99.2% of corpus sentences label identically; the rest differ in Latin-abbreviation / digit+counter readings and accent-phrase chaining |
| `zho-hans` | g2p crate (`src/mandarin`) | `g2p-zho` | polyphone disambiguation is context-dependent; tone must come from the reading, not the character. g2pM + pinyin_to_ipa ported (`g2pm-ipa` stays in `PROVIDERS`); labels identical on every row both label, and rows with digits / Latin / out-of-dictionary characters are excluded instead of trained with a hole |
| `tha` | g2p crate (`src/thai`, Python inside) | `g2p-tha` | unwritten vowels and implicit syllable boundaries make rule-based G2P unreliable; espeak `th` matches 0 of 3,000 Wiktionary words. The crate embeds a pinned `uv` project running vachana-thai (`vachana-thai` stays in `PROVIDERS`) and does `thai_labels`' parsing in Rust; identical labels. Needs `uv` on PATH |
| `hin` | g2p crate (`src/hindi`) | `g2p-hin` | schwa deletion is morphologically conditioned; aspiration must be one segment with its consonant (`t̪ʰ`, `bʱ`), not a standalone `ʰ`. The crate is a port of `schwa-stress-hin` (still in `PROVIDERS` for reproduction; `canon: legacy` is byte-identical to it) plus the corrections from the 2026-09-02 audit: ə→[ɛ] beside ɦ, ŋ before velars, ज्ञ→[ɡj], final ɪ/ʊ neutralized, no deletion into impossible clusters; digit/Latin rows are excluded (`hindi_digits:` / `hindi_latin_script:`) instead of labeled with a hole |
| `kor` | g2p crate (`src/korean`, Python inside) | `g2p-kor` | espeak `ko` has no tense consonants at all (달/딸/탈 collapse), splits affricates and aspirates into letters, and applies none of the implicit sound changes; 47% of Wiktionary words at the phonemic level (2026-09-03 audit). The crate embeds a pinned `uv` project running g2pk2 + mecab-ko (95.6% of Wiktionary words; the only candidate with ㄴ-insertion and morphological tensification) and maps post-sandhi Hangul to phones in Rust. Labels are connected speech: sound changes apply across the spaces within a clause (못 만났어 [몬만나써]), and punctuation splits clauses. Digit / Latin / hanja / bare-jamo rows are excluded. Needs `uv` on PATH. **No Korean audio has been collected yet** — the model has never been trained on Korean; this is the chain for when it is. The remaining ~4% (lexical Sino-Korean tensification, 결점 [결쩜]) is a planned 표준국어대사전 overlay |

Everything else in `LANG_TO_ESPEAK` (Pimsleur-era languages) is eSpeak-labeled
and has not been through a backend audit.

## How it is enforced

`BACKEND_REQUIRED_LANGS` in `train/scripts/preprocess.py` is the set that must
not be labeled from eSpeak. Two checks keep it true:

1. `preprocess.main()` builds the sidecar itself (`ensure_backend_sidecar`) for
   any backend-required language, and raises if it somehow ends up without one
   rather than falling through to eSpeak.
2. `load_phoneme_backend(path, lang)` requires every row's `backend` field to
   name that language's provider from
   `build_external_phoneme_sidecars.CONFIG`. This is what stops a
   `--phoneme-backend jpn=/tmp/whatever.jsonl` override from quietly
   substituting a different engine.

Both fail closed. If you are adding a language whose labels should not come
from eSpeak, add it to `BACKEND_REQUIRED_LANGS` **and** to `CONFIG`, in the
same change — `CONFIG` is what the check reads.

## Hindi flat-response contract

The `g2p-hin` provider requests `lang="hin", canon="current"` through
`g2p_client.request`. Audit schema **2** preserves g2p's flat phones, numeric
stress, absolute syllables and word spans, plus the canon and build identity
(on refusals too). Schema-1 word-local audits must be regenerated; the
standalone sidecar builder refuses them rather than interpreting the old shape.

`g2p_labels` with `HINDI_SPEC` keeps the flat arrays and absolute indices. The only schema
mappings are syllable `stressed` → numeric `stress`, word-span membership →
syllable `word`, and the existing `roy-2017-rules-on-schwa-hin` provenance.
The sidecar schema is unchanged: it does **not** gain `raw` or `word_spans`.
`hindi_labels` remains for historical word-shaped audit reproduction, not the
production request path.

`train/tests/test_hindi_flat_labels.py` checks exact serialized sidecar bytes
against goldens captured through the old production path with pinned g2p 0.4.0
and the current canon. Fixture provenance lives beside the goldens. From
`pronunciation/train` on Linux:

```bash
LEXIDE_DATA_VENV=~/.venv-lexide-tests \
G2P_BIN=/data/coding/g2p/target/release/g2p \
LEXIDE_HINDI_FULL_SHADOW=1 \
direnv exec /data/coding/yap bash ../scripts/py-linux.sh \
  -m pytest tests/test_hindi_flat_labels.py -q -s
```

The full shadow checks every Hindi manifest disposition, including exclusions,
in temporary directories. It compares old and direct adapters against the
**same binary/canon**; it never updates goldens or corpus files. Any difference
between newly generated labels and an older on-disk sidecar is a separate
relabel diff, not evidence that the adapter changed behavior.

## Shared g2p sidecar adapter

`build_external_phoneme_sidecars.CONFIG` pairs each provider with a partial of
`g2p_labels(rec, audit, spec)`. Frozen specs declare the empty disposition and
ordered factor builders; the common adapter preserves `phonemes`, `stress`,
then the language's structured annotations. It does not homogenize schemas:

| languages | appended fields, in order | empty phones |
|---|---|---|
| `tha`, `zho-hans`, `kor` | `tone` | `exclude_reason: g2p_no_phonemes` |
| `hin` | `syllables`, `stress_source` | `exclude_reason: hindi_no_devanagari_phones` |
| `jpn` | `pitch_accent` **or** `pitch_accent_exclude_reason` | retain empty arrays / accent withholding |

The Japanese empty behavior deliberately follows the **original converter**.
Provider exclusions precede validation. Hindi absolute word/syllable coverage
and tone alignment are validated **before** empty dispositions. Japanese stress
and present pitch arrays must now align with phones, even when accent is
withheld, except for the provider's existing `pitch_accent: []` sentinel with
an explicit withholding reason. This rejects malformed inputs but leaves valid
serialized bytes unchanged.
A provider accent-withholding reason (including an empty string) takes priority
over Whisper confidence; otherwise only logprobs **below** `-0.35` withhold
accent. Missing optional factors stay missing; no null fields are added.
Korean's provider still fabricates aligned `[None, ...]` tones for compatibility.
`BACKEND_REQUIRED_LANGS` and the eSpeak-fallback guards remain unchanged.

`train/tests/test_g2p_superset_labels.py` compares all five languages with frozen
original-converter oracles and exact sidecar goldens. Hindi reuses its earlier
fixtures; `train/tests/fixtures/g2p_superset/` adds Thai, Mandarin, Japanese and
Korean captures with pinned-binary SHA256/build identity and current canon.
Audio-language samples take four filenames from **every** source/backend
stratum (including both TTS backends), plus synthetic edge cases. **Korean has
no audio**: its verification is text fixtures and synthetic adapter cases only.
Empty Thai requests fail inside vachana and empty Mandarin responses omit the
provider-required tone field; their empty-adapter cases are synthetic, not
claims of successful binary requests. No provider behavior is changed here.

Run the offline gate from `/data/coding/lexide/pronunciation/train`:

```bash
LEXIDE_DATA_VENV=~/.venv-lexide-tests \
G2P_BIN=/data/coding/g2p/target/release/g2p \
direnv exec /data/coding/yap bash ../scripts/py-linux.sh \
  -m pytest tests/test_g2p_superset_labels.py tests/test_hindi_flat_labels.py -q
```

Add `LEXIDE_G2P_LIVE_SHADOW=1` to replay binary-generated fixture cases;
synthetic adapter-only cases remain frozen. `LEXIDE_G2P_FULL_SHADOW=1` opts
into fresh pinned-binary conversion of **every** Hindi/Thai/Mandarin/Japanese
manifest row (potentially slow). `LEXIDE_G2P_CACHED_SHADOW=1` instead compares
both adapters over existing full audits, checking manifest hashes and schema;
those audits may use an older binary and this is **not** a live-provider gate.
Every mode writes only temporary sidecars, reports drift rather than refreshing
goldens, and never regenerates production audits/labels.

## Consumers outside this repo

Anything scoring audio against this model must generate targets from the same
source, tokenized the same way. The eSpeak side of that is now one shared
artifact: the `g2p` crate (github.com/anchpop/g2p) embeds the fork and owns
the tokenizer (continuation diacritics fold onto the previous token, `ʲ` folds
onto a preceding *consonant* only, language-switch markers stripped). This
repo calls its binary; yap links the crate. Which *build* each side runs is
still a choice — each pins its own g2p rev — so a relabel here does not move
yap until yap bumps.

yap mirrors this table in `language-utils` (`PhonemeLabelSource`) and fails
closed the same way. Keep the two in sync; they describe one fact.
