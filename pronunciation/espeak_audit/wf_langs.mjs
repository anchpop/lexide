export const meta = {
  name: 'espeak-audit-all-langs',
  description: 'Per-language phonetician analysis of where espeak diverges from real audio, adversarially verified, synthesized into a cross-language report',
  phases: [
    { title: 'Interpret', detail: 'one phonetician agent per language reads its summary.json' },
    { title: 'Verify', detail: 'adversarial check of each language\'s findings against the numbers' },
    { title: 'Synthesize', detail: 'cross-language report' },
  ],
}

const OUT = '/Users/andrepopovitch/coding/lexide/pronunciation/espeak_audit/out'

const LANGS = [
  { code: 'eng', name: 'English (US)', tag: 'eng_tts_en-US-Chirp3-HD-Algieba',
    hints: 'AmE flapping /t,d/->[ɾ] intervocalically (voiced, no burst); /u/ & /oʊ/ fronting (high F2); aspiration of onset /p t k/; vowel reduction to schwa (unstressed shorter/centralised); dark/velarised coda /l/; vowel nasalisation before nasal codas.' },
  { code: 'deu', name: 'German', tag: 'deu_tts_de-DE-Chirp3-HD-Fenrir',
    hints: 'final devoicing (Auslautverhärtung): word-final /b d ɡ/ should be voiceless (low voiced_frac at word_final); coda /r/ vocalised to [ɐ]; ich-Laut [ç] vs ach-Laut [x]; tense/lax vowel quality+length; glottal-stop onset before vowel-initial syllables.' },
  { code: 'fra', name: 'French', tag: 'fra_tts_fr-FR-Chirp3-HD-Schedar',
    hints: 'nasal vowels (B1 widening + A1-P0 drop vs oral); schwa deletion/epenthesis; liaison; uvular /ʁ/; no aspiration; mid-vowel laxing e/ɛ, o/ɔ, ø/œ; schwa vs ø.' },
  { code: 'ita', name: 'Italian', tag: 'ita_tts_it-IT-Chirp3-HD-Leda',
    hints: 'consonant gemination (geminate stops much longer dur, voiceless geminates with long closure); open vs close mid vowels /e ɛ/ and /o ɔ/ (F1 split); no vowel reduction (peripheral unstressed vowels); intervocalic /s/ voicing; clear /l/.' },
  { code: 'por', name: 'Portuguese (BR)', tag: 'por_tts_pt-BR-Chirp3-HD-Leda',
    hints: 'strong nasalisation (nasal vowels + vowels before nasal codas); unstressed vowel reduction/raising (final -o->[u], -e->[i]); palatalisation /t d/ before /i/ -> [tʃ dʒ] (look at intervocalic/pre-i /t d/ with high-freq burst/affrication); coda /l/ -> [w] vocalisation; coda /s/; tap vs /r/ realisations [h x ʁ].' },
  { code: 'rus', name: 'Russian', tag: 'rus_tts_ru-RU-Chirp3-HD-Fenrir',
    hints: 'vowel reduction (akanye: unstressed /o/->[ɐ/ə], /e/->[ɪ]); palatalisation (soft consonants Cʲ before front vowels); final obstruent devoicing (word-final /b d ɡ z .../ voiceless); strong stressed/unstressed duration+quality split.' },
  { code: 'spa', name: 'Spanish (ES)', tag: 'spa_tts_es-ES-Chirp3-HD-Sulafat',
    hints: 'spirantisation/lenition: intervocalic /b d ɡ/ -> approximants [β ð ɣ] (voiced_frac ~1 AND weak/absent closure burst, longer continuant); no vowel reduction (stable 5-vowel system, stressed≈unstressed quality); intervocalic /d/ esp. -ado often [ð] or elided; tap vs trill /ɾ r/.' },
]

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    language: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          phenomenon: { type: 'string' },
          espeak_says: { type: 'string' },
          audio_shows: { type: 'string' },
          evidence: { type: 'string', description: 'the actual numbers from summary.json (with n)' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          is_espeak_error: { type: 'boolean', description: 'true if espeak is too broad/wrong here; false if espeak is acoustically correct' },
        },
        required: ['phenomenon', 'espeak_says', 'audio_shows', 'evidence', 'confidence', 'is_espeak_error'],
      },
    },
    caveats: { type: 'string' },
  },
  required: ['language', 'findings', 'caveats'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    language: { type: 'string' },
    verified: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          phenomenon: { type: 'string' },
          verdict: { type: 'string', enum: ['supported', 'weak', 'refuted'] },
          critique: { type: 'string' },
          is_espeak_error: { type: 'boolean' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          one_line: { type: 'string', description: 'espeak X -> audio Y (numbers)' },
        },
        required: ['phenomenon', 'verdict', 'critique', 'is_espeak_error', 'confidence', 'one_line'],
      },
    },
  },
  required: ['language', 'verified'],
}

const interpretPrompt = (L) => `You are an expert phonetician auditing the espeak-ng phonemizer for ${L.name}.

espeak emits a BROAD, CITATION-FORM transcription. We measured what a real ${L.name} Google-TTS voice actually produced, per phone, with parselmouth (the independent acoustic arbiter). A model trained on espeak labels CANNOT arbitrate (it reproduces espeak), so trust ONLY the acoustic numbers.

Read the machine summary (within-speaker acoustic aggregates + detectors):
  ${OUT}/${L.tag}.summary.json
You may also read a few rows of ${OUT}/${L.tag}.segments.jsonl and ${OUT}/${L.tag}.clips.jsonl for examples.

Keys: vowel_table (per espeak vowel: F1/F2/F3/B1/A1-P0/dur medians + n); obstruents (voiced/voiceless stops by position {intervocalic,onset,word_final,coda_other}: voiced_frac on the RAW stop span, burst_hi_lo_db, dur_ms); reduction (stressed vs unstressed F1/F2/dur); nasalization (oral_ref B1/A1-P0 vs nasal vowels); overspecification (phones squeezed to ~0 frames = candidate deletions/insertions); model_agreement (model reading vs espeak — exact match frac + substitutions).

Known ${L.name} phenomena espeak may mis-handle: ${L.hints}

TASK: identify where espeak's broad/citation output systematically DIVERGES from the measured audio. For each, give espeak_says vs audio_shows and cite the ACTUAL NUMBERS (with n). Mark is_espeak_error=true when espeak is too broad/wrong, false when espeak is acoustically justified (also valuable). Be a skeptic: require adequate n; respect the documented confounds — durations are midpoint-boundary (raw CTC spans are peaky); A1-P0 is only valid for F1>400 Hz; stop voicing/burst are on the raw span. Anchor everything within THIS speaker. Don't invent phenomena the numbers don't support.`

const verifyPrompt = (L, f) => `You are an adversarial reviewer checking another phonetician's findings for ${L.name} against the same data. Default to skepticism.

Their findings JSON:
${JSON.stringify(f, null, 2)}

Re-read ${OUT}/${L.tag}.summary.json. For EACH finding, decide: supported (numbers clearly back it, adequate n, no confound), weak (right direction but small n / modest effect / partial confound), or refuted (not supported, confounded, or an artifact of CTC peakiness / A1-P0 low-F1 confound / tiny n). Write a one-line critique and a crisp one_line "espeak X -> audio Y (numbers)". Keep is_espeak_error and confidence (adjust confidence if warranted). Do not add new findings; only adjudicate these.`

phase('Interpret')
const perLang = await pipeline(
  LANGS,
  (L) => agent(interpretPrompt(L), { schema: FINDINGS_SCHEMA, phase: 'Interpret', label: `interpret:${L.code}` }),
  (findings, L) => agent(verifyPrompt(L, findings), { schema: VERDICT_SCHEMA, phase: 'Verify', label: `verify:${L.code}` })
    .then((v) => ({ code: L.code, name: L.name, tag: L.tag, verified: v })),
)

const clean = perLang.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  `You are writing the cross-language espeak audit report. Below are adversarially-verified per-language findings (verdicts: supported/weak/refuted; is_espeak_error true=espeak wrong, false=espeak correct).

${JSON.stringify(clean, null, 2)}

Write a clear, honest Markdown report and return it AS YOUR RESPONSE (the entire
response should be the markdown — do not use any file tools, just return the text).
Structure:
1. TL;DR — the strongest CONFIRMED (verdict=supported) espeak errors across languages.
2. A compact table: language | phenomenon | espeak -> audio | verdict | espeak error?
   Include only supported+weak findings; drop refuted (but note how many were refuted).
3. Per-language sections (only supported/weak findings, with the numbers).
4. Cross-cutting patterns (e.g. final devoicing in DE+RU; nasalisation in FR+PT; intervocalic lenition in EN-flap + ES-spirant).
5. Recommendations for non-circular label enrichment: which divergences are rule-governed (fixable with deterministic allophonic rules on espeak output) vs speaker/variant-specific (need acoustic adjudication, never model likelihood).
Keep claims tied to numbers. End the markdown with a "## Summary" section of 6-10 lines on the biggest confirmed cross-language findings and anything that CHANGED now that n is ~4x larger (240 clips/lang vs the earlier 60).`,
  { phase: 'Synthesize', label: 'synthesize' },
)

return { languages: clean.map((c) => c.code), report: '/Users/andrepopovitch/coding/lexide/pronunciation/espeak_audit/REPORT_all_langs.md', synthesis: synth }
