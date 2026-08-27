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
| `jpn` | backend | `pyopenjtalk` | kanji readings are lexical, not derivable from the glyphs; pitch accent needs a dictionary |
| `zho-hans` | backend | `g2pm-ipa` | polyphone disambiguation is context-dependent; tone must come from the reading, not the character |
| `tha` | backend | `vachana-thai` | unwritten vowels and implicit syllable boundaries make rule-based G2P unreliable |
| `hin` | backend | `schwa-stress-hin` | schwa deletion is morphologically conditioned; aspiration must be one segment with its consonant (`t̪ʰ`, `bʱ`), not a standalone `ʰ` |
| `kor` | eSpeak (`ko`) | — | **unvalidated.** Uses eSpeak today because nothing better is wired up, not because eSpeak has been checked. Validate before trusting Korean labels. |

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

## Consumers outside this repo

Anything scoring audio against this model must generate targets from the same
source, tokenized the same way (`phonemize`'s parser: continuation diacritics
fold onto the previous token, `ʲ` folds onto a preceding *consonant* only).
yap mirrors this table in `language-utils` (`PhonemeLabelSource`) and fails
closed the same way. Keep the two in sync; they describe one fact.
