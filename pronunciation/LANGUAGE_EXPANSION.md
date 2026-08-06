# Thai, Mandarin, Hindi, and Japanese expansion

The repository uses these canonical ids:

| Product id | Language | FLEURS | Tatoeba | eSpeak | Google TTS |
|---|---|---|---|---|---|
| `tha` | Thai | `th_th` | `tha` | `th` | `th-TH` |
| `zho-hans` | Standard Mandarin, Simplified script | `cmn_hans_cn` | `cmn` | `cmn` | `cmn-CN` |
| `hin` | Hindi | `hi_in` | `hin` | `hi` | `hi-IN` |
| `jpn` | Japanese | `ja_jp` | `jpn` | `ja` | `ja-JP` |

`him` is not used: it is an ISO 639-2 collective code for Himachali/Western
Pahari languages. Hindi is `hin` in this repository.

## Data recipe

Start each language with all four existing source classes:

1. FLEURS for clean, multi-speaker human read speech.
2. Tatoeba for short human utterances and isolated-word-like examples.
3. Pimsleur for learner-domain, short native prompts (where available).
4. Multi-voice Chirp TTS for phonetic and lexical coverage, capped by the
   existing per-source balancer so synthetic speech cannot dominate.

Do not train immediately after acquisition. Run the existing Whisper content
audit and language-mixing filter, then audit a stratified sample of the eSpeak
labels against a second G2P source and the audio. The expansion should fail
closed: a language is admitted only after its unknown-token inventory is zero
and its common phone confusions have been reviewed.

Suggested second label sources are Open JTalk/UniDic full-context labels for
Japanese, a word-aware pinyin source for Mandarin, and independent G2P tools or
lexicons for Thai and Hindi. They are auditors, not automatic truth: systematic
disagreements should be settled with linguistic rules or population-level
acoustics, consistent with this repository's anti-circularity rule.

Candidates tested in the bake-off:

- Japanese: `pyopenjtalk`, retaining its full-context labels so mora boundaries
  and accent metadata are not lost when converting its phones to our IPA.
- Mandarin: use Apache-2.0 `g2pM` as the primary Mainland contextual polyphone
  frontend, followed by the explicit Pinyin-to-IPA mapping. In a 500-sentence
  comparison, g2pW disagreed with g2pM on 4.44% of syllables and repeatedly
  selected Taiwan-standard readings (for example 和 as `han4`), which does not
  match `cmn_hans_cn`. Plain pypinyin disagreed on 1.86% and remains a useful
  dictionary fallback. Preserve tone separately from the segment sequence.
- Thai: use the BSD-3-Clause TLTK rule/lexicon frontend (screened through the
  lightweight Vachana packaging) as the practical primary candidate. It
  produced IPA plus lexical-tone diacritics without runtime failures. It is
  not safe on mixed-script input, however: it passed at least one Latin span
  through unchanged in 402 of the 426 mixed Latin/Thai FLEURS sentences. Those
  rows must be explicitly excluded or independently transcribed, never parsed
  as IPA. PyThaiNLP `thaig2p_v2` remains an Apache-2.0 comparison
  model, but its legacy Marian checkpoint is pathologically slow through the
  installed Transformers 5 cache layer and is not currently auditable at
  corpus scale.
- Hindi: Epitran `hin-Deva` is retained only as an independent baseline: it
  keeps schwas that standard Hindi deletes. The preferred candidate combines
  the MIT-licensed ACL 2020 supervised schwa-deletion classifier (94–95% held-
  out accuracy reported by its authors) with an explicit Devanagari-to-IPA
  map. Its complete pass also acts as a Unicode-corruption detector.
  Do not pair those phones with an all-zero stress vector. Hindi stress is
  quantity-sensitive and schwa deletion can trigger resyllabification and a
  stress shift. Roy and Pandey's government-funded Akshara-to-Sound converter
  is a substantially better candidate than the initially located GitHub copy:
  the official SourceForge release is GPLv3 and emits Standard Hindi surface
  IPA and prosodic structure. Headless inspection exposed two out-of-range
  loops and a defective `ऑ` path (the code introduces an extra /u/ before
  /ɔ/); fixing that vowel changes both syllabification and failure behavior for
  many loans. The executable is therefore retained as an auditor, not copied
  into the training path.

  The production candidate instead applies Roy's published *surface* rules to
  the ACL-2020 schwa-aware phone stream: syllabify after deletion, compute mora
  weight, stress every superheavy syllable, stress non-final heavy syllables,
  and apply the paper's bisyllabic weak-syllable rule. It preserves the inferred
  syllable spans and rule source in every sidecar row. This gives a coherent,
  inspectable segment-plus-stress proposal without transferring vowel ordinals
  between disagreeing phone streams. It still needs Pimsleur/acoustic review,
  especially for loans and words whose stress varies by part of speech. See
  [Roy (ICON 2017)](https://aclanthology.org/W17-7502/) and the
  [official PLS release](https://sourceforge.net/projects/pls-for-indic-languages/files/).

## Acquired snapshot

The reproducible report is `python scripts/report_expansion_data.py`. The
current normalized human-audio snapshot is:

| Language | Sources | Clips | Hours | Known speakers |
|---|---|---:|---:|---:|
| `tha` | FLEURS + Tatoeba | 3,190 | 8.73 | 2 known Tatoeba uploaders |
| `zho-hans` | FLEURS + Tatoeba | 3,330 | 9.76 | 2 known Tatoeba uploaders |
| `hin` | FLEURS + Kathbath validation shard | 3,696 | 8.78 | 10 Kathbath ids |
| `jpn` | FLEURS + Tatoeba | 3,264 | 8.04 | 4 known Tatoeba uploaders |

All files are readable and present. Complete-manifest native backend audits
have zero failures for Thai/TLTK, Mandarin/g2pM, Hindi/Epitran, and Japanese/
Open JTalk. The schwa-aware Hindi pass found four malformed source transcripts;
all four were checked against their recordings and repaired in the downloaders
before being re-audited. The Thai Tatoeba import quarantined 133 files whose
export used the null license marker rather than a reusable license. Its 588
remaining clips are CC BY 4.0. All 84 Mandarin Tatoeba clips are CC BY-NC 4.0,
so normal preprocessing excludes them. `--allow-noncommercial` must be passed
explicitly for a permitted research/NC run; the rows remain in the manifest and
complete sidecar for provenance and reproducibility.

Text-only forced-language Whisper audits are stored separately from phonetic
audits. They currently cover roughly 240 FLEURS clips per language and 200
Kathbath clips. Median character error is approximately 9.5% Thai, 5.4%
Mandarin, 11.4% Hindi FLEURS, 2.1% Japanese, and 8.1% Hindi Kathbath. These
scores screen transcript/audio mismatches; they are not phonetic truth and high
CER rows require listening before exclusion, especially for Thai and names.

The Japanese Tatoeba audit covers 200 clips: median CER 0, 90th percentile
22.2%, no wrong-language detections, and only four rows above 50%. Three are
ordinary kanji-vs-kana renderings. Both Whisper Large v3 and Turbo decoded the
fourth as `まさしく` while its source transcript said `正しく`; the manifest
preserves the original and records the audio-verified correction.

All 10,260 FLEURS manifest rows carry the upstream `CC BY 4.0` license and
dataset attribution URL. This is recorded per row rather than inferred from the
source name. Japanese Tatoeba contributes 972 recovered clips: all 27 CC BY
4.0 candidates plus 945 of the CC BY-NC 4.0 candidates. Acquisition stopped
after commercial coverage was complete instead of downloading the remaining
NC-only tail. Normal preprocessing retains the 27 CC BY clips and excludes the
945 NC clips. Every recovered legacy RIFF header was repaired losslessly, all
audio is readable, and Tatoeba rows carry an uploader plus sentence attribution
URL even when the export's optional attribution field was empty.
Kathbath rows likewise carry the official AI4Bharat dataset attribution. Across
the four manifests there are no rows missing a license or attribution URL.

### External backend contract

`preprocess.py` accepts a repeatable argument such as:

```sh
python3 train/scripts/preprocess.py --langs jpn zho-hans \
  --phoneme-backend jpn=/tmp/jpn-openjtalk.jsonl \
  --phoneme-backend zho-hans=/tmp/zho-g2pw.jsonl
```

Each JSONL row is bound to one manifest record:

```json
{"file":"clip.wav","sentence_sha256":"…","phonemes":["n","i"],"stress":[0,1],"backend":"g2pw+pinyin-ipa","tone":[null,3]}
```

The sidecar must account for every non-silent manifest clip for that language.
A row may carry an explicit `exclude_reason` instead of phonemes/stress; this
is how known backend failures remain visible without entering training.
Missing rows, duplicate files, stale sentence hashes, mismatched phoneme/stress
lengths, and unknown IPA tokens are fatal. This deliberately prevents an
unnoticed eSpeak fallback or a half-regenerated label set. eSpeak is also
closed off at the entry point: for these four languages `preprocess.py`
refreshes the chain itself — it runs the incremental G2P audit (free when the
manifest is unchanged; the language's G2P tool is only imported for
new/changed rows) and rebuilds the sidecar before labeling, so a plain
preprocess run after new data lands is complete on its own. The
`--phoneme-backend LANG=JSONL` flag remains only as an explicit override, and
the `LANG_TO_ESPEAK` entries remain for audits and tooling. `tone` and
`pitch_accent` are preserved as optional aligned metadata and consumed as
factors of the same CTC symbol as phone and stress. Thai and Mandarin use
distinct tone heads; Japanese uses a mora/nucleus accent head. A factor is
omitted when the language or trustworthy label does not apply.
Per-row licenses are propagated to `phonemes.jsonl`; CC BY-NC rows are excluded
by default even when their labels pass every phonetic gate.

## Prosody design

Keep the existing stress factor. Tone and pitch accent are additional
suprasegmental properties, not replacements for stress. They are separate
prediction heads but not separate objectives: nonblank emissions form a
Cartesian phone × stress × applicable-language-factor alphabet with one blank
and one native CTC loss. A normalized inverse-temperature weight controls each
factor without creating a second loss or alignment.

Thai and Mandarin deliberately use different five-tone heads because their
numbered categories are not interchangeable. Japanese uses three aligned
states: non-mora phone, mora without an accent nucleus, and nucleus. Citation
labels retain the known sandhi/connected-speech caveats; no pitch-tracker-derived
targets are introduced. Twenty Japanese clips where NJD and HTS disagree on a
punctuation/foreign-name phrase boundary retain phone/stress supervision but
explicitly withhold the accent factor.

The generated sidecars currently pass the contract and the phone-inventory
gate with zero unknown tokens:

- Thai: 2,764 trainable rows and 426 explicit mixed-script exclusions. Tone
  1--5 is aligned to the vowel-bearing phone. The TLTK word boundaries are
  retained long enough to mark the final syllable of each Standard Thai lexical
  word for the existing stress head (53,142 primary-stress phone labels). Tone
  and stress remain separate targets; connected-speech weakening still needs
  acoustic/Pimsleur review.
- Mandarin: all 3,330 rows trainable, with g2pM's tone 1--5 aligned to the
  syllable nucleus and stress kept separate.
- Japanese: all 3,264 manifest rows have complete Open JTalk labels; normal
  commercial-safe preprocessing emits 2,319 after excluding 945 NC clips.
  Phones are mapped to IPA, including devoicing and gemination, while accent-phrase nuclei come from
  the pre-full-context NJD features. This distinction matters because the HTS
  label conversion rewrites heiban `acc=0` in a way that otherwise makes it
  indistinguishable from final-mora accent. Twenty commercial rows with an
  NJD/HTS phrase-boundary mismatch explicitly withhold only accent supervision.
- Hindi: all 3,696 rows trainable with the schwa-aware segment stream plus
  published surface-weight stress rules. The aligned sidecar contains 120,886
  syllable spans and no unknown phones. These are rule-derived lexical labels;
  phrase-level prominence and optional schwa realization must still be judged
  from audio rather than forced into this target.

`scripts/validate_expansion_prosody.py` protects interpretable anchor cases:
the five-way Thai and Mandarin /a/ tone series, Tokyo Japanese `箸/橋/端`
(initial/final/heiban accent), and nasal-vowel participation in Hindi surface
syllabification. These are semantic regression checks, not substitutes for the
source-stratified acoustic audit.

Commercial-safe `phonemes.jsonl` and `vad.jsonl` are now in lockstep for all
four languages (2,764 Thai; 3,246 Mandarin; 3,696 Hindi; 2,319 Japanese), with
no duplicate, missing, extra, or empty VAD rows. Controlled multi-voice Google
TTS remains unavailable in this environment because no application-default
credentials are configured. It is not replaced with an unreviewable synthetic
source; when credentials become available it should enter as a separate split.

## Promotion gates

- A hand-reviewed, source-stratified native-backend audit for every new
  language; eSpeak is not the label source for these four languages.
- Dedicated isolated minimal-pair sets: Mandarin and Thai tone, Japanese accent
  contrasts, and Hindi aspiration/voicing/retroflex contrasts.
- Human-only and TTS-only validation reported separately.
- No regression on the existing language-agnostic minimal-pair suite.
- Prosody evaluation reports categorical tone/accent performance separately by
  language and source, with the citation-form caveats above.

## Remaining Pimsleur value

The importer recognizes `Thai`, `Hindi`, `Japanese`, and `Mandarin Chinese` /
`Chinese Mandarin`; Cantonese is deliberately mapped elsewhere. All four sets
remain useful, but they close different evidence gaps:

1. **Hindi (highest priority):** careful short words/phrases for optional schwa
   realization, the published mora-weight stress proposal, aspiration,
   breathy-voice, dental/retroflex, and the reviewed loanword edge cases.
2. **Mainland/Standard Mandarin:** commercially usable native tone and sandhi
   examples. The current extra human source is entirely NC, so Pimsleur has
   unusually high licensing as well as phonetic value.
3. **Thai:** connected-speech weakening versus the final-syllable lexical
   stress rule, five-tone minimal contexts, and names/code-switching that the
   Thai frontend intentionally excludes.
4. **Japanese:** isolated `箸/橋/端`-style accent contrasts and phrase-boundary
   realizations to compare citation Open JTalk nuclei with observed downsteps.

Pimsleur should remain a separate source split, with Whisper used only for
content/language screening. Its pronunciations must go through the same native
backend sidecars and audio-first prosody checks before entering training.
