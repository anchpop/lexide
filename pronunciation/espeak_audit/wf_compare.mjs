export const meta = {
  name: 'espeak-audit-tts-vs-human',
  description: 'Per-language comparison of a TTS voice vs a human Tatoeba speaker: which espeak divergences replicate, which TTS under-shows, which are TTS artifacts; adversarially verified and synthesized',
  phases: [
    { title: 'Compare', detail: 'one phonetician per language reads BOTH speakers\' summaries' },
    { title: 'Verify', detail: 'adversarial check of each comparison against the numbers' },
    { title: 'Synthesize', detail: 'TTS-vs-human cross-language report' },
  ],
}

const OUT = '/Users/andrepopovitch/coding/lexide/pronunciation/espeak_audit/out'

const LANGS = [
  { code: 'eng', name: 'English (US)', tts: 'eng_tts_en-US-Chirp3-HD-Algieba', human: 'eng_tatoeba_CK',
    hints: 'AmE flapping /t,d/->[ɾ] intervocalic (voiced, no burst); /uː/ & /oʊ/ fronting (high F2); onset /p t k/ aspiration; unstressed reduction (shorter/centralised); dark coda /l/; pre-nasal vowel nasalisation. Flapping is the canary: human CK flapped harder (vf 1.0) than TTS (0.6) in prior runs.' },
  { code: 'deu', name: 'German', tts: 'deu_tts_de-DE-Chirp3-HD-Fenrir', human: 'deu_tatoeba_driini',
    hints: 'final devoicing (word-final /b d ɡ/ voiceless); coda /r/->[ɐ]; ich [ç] vs ach [x]; tense/lax vowel quality+length; glottal-stop hard onset of vowel-initial syllables.' },
  { code: 'fra', name: 'French', tts: 'fra_tts_fr-FR-Chirp3-HD-Schedar', human: 'fra_tatoeba_Phoenix',
    hints: 'nasal vowels (B1 widening + A1-P0 drop); schwa deletion/epenthesis; liaison; uvular /ʁ/; no aspiration; mid-vowel laxing e/ɛ, o/ɔ, ø/œ; spurious English vowels from espeak.' },
  { code: 'ita', name: 'Italian', tts: 'ita_tts_it-IT-Chirp3-HD-Leda', human: 'ita_tatoeba_NM1',
    hints: 'consonant gemination (geminate duration); open/close mid /e ɛ/, /o ɔ/ (F1 split); spurious lax /ɪ ʊ/ from espeak (Italian merges tense/lax); no unstressed reduction; intervoc /s/ voicing; trill/tap /r ɾ/.' },
  { code: 'por', name: 'Portuguese (BR)', tts: 'por_tts_pt-BR-Chirp3-HD-Leda', human: 'por_tatoeba_Lemmy',
    hints: 'strong nasalisation; unstressed reduction/raising (-o->[u], -e->[i]); /t d/ before /i/ -> [tʃ dʒ]; coda /l/->[w]; spurious /ʲ/ glide from espeak; glide /w j/ vocalisation; /r/ realisations.' },
  { code: 'rus', name: 'Russian', tts: 'rus_tts_ru-RU-Chirp3-HD-Fenrir', human: 'rus_tatoeba_fjay69',
    hints: 'standalone /ʲ/ palatalisation over-specified by espeak (should be Cʲ); akanye (unstressed /o/->[ɐ/ə]); final devoicing; stress dur+quality; /ɨ/ vs /i/; velarised /ɫ/.' },
  { code: 'spa', name: 'Spanish (ES)', tts: 'spa_tts_es-ES-Chirp3-HD-Sulafat', human: 'spa_tatoeba_arh',
    hints: 'spirantisation intervoc /b d ɡ/->[β ð ɣ] (espeak already emits β ð ɣ — check if realized); no vowel reduction (stable 5-vowel); tap vs trill /ɾ r/; intervoc /d/ in -ado often elided.' },
]

const COMPARE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    language: { type: 'string' },
    comparisons: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          phenomenon: { type: 'string' },
          tts_evidence: { type: 'string', description: 'numbers from the TTS summary (with n)' },
          human_evidence: { type: 'string', description: 'numbers from the human summary (with n)' },
          presence: { type: 'string', enum: ['both', 'human_only', 'tts_only', 'neither'] },
          is_espeak_error: { type: 'boolean' },
          tts_vs_human_note: { type: 'string', description: 'does TTS under-show this vs the human?' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['phenomenon', 'tts_evidence', 'human_evidence', 'presence', 'is_espeak_error', 'tts_vs_human_note', 'confidence'],
      },
    },
    caveats: { type: 'string' },
  },
  required: ['language', 'comparisons', 'caveats'],
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
          presence: { type: 'string', enum: ['both', 'human_only', 'tts_only', 'neither'] },
          is_espeak_error: { type: 'boolean' },
          critique: { type: 'string' },
          one_line: { type: 'string', description: 'phenomenon: TTS=… human=… -> presence (numbers)' },
        },
        required: ['phenomenon', 'verdict', 'presence', 'is_espeak_error', 'critique', 'one_line'],
      },
    },
  },
  required: ['language', 'verified'],
}

const comparePrompt = (L) => `You are an expert phonetician testing whether espeak's divergences from real ${L.name} speech are ROBUST, by comparing two speakers of the same language:
  TTS (Google Chirp3-HD, careful/synthetic):  ${OUT}/${L.tts}.summary.json
  HUMAN (Tatoeba reader, real vocal tract):    ${OUT}/${L.human}.summary.json
Both are 240 clips, measured identically with parselmouth (the independent arbiter). A model trained on espeak labels CANNOT arbitrate; trust the acoustic numbers. Anchor WITHIN each speaker (never compare absolute Hz across the two — different vocal tracts; compare PATTERNS, e.g. intervocalic vs elsewhere, stressed vs unstressed, nasal vs oral).

Summary keys: vowel_table (per espeak vowel F1/F2/F3/B1/A1-P0/dur + n); obstruents (voiced/voiceless stops by position {intervocalic,onset,word_final,coda_other}: voiced_frac on RAW span, burst_hi_lo_db, dur_ms); reduction (stressed vs unstressed); nasalization (oral_ref vs nasal vowels); overspecification (phones squeezed to ~0 frames); model_agreement.

Known ${L.name} phenomena: ${L.hints}

TASK: for each candidate espeak-divergence, classify presence:
  - both       = appears in TTS AND human (robust, speaker-independent)
  - human_only = real in human but TTS under-shows it (TTS over-articulates/smooths connected-speech allophony — IMPORTANT, these are findings the TTS-only report MISSED)
  - tts_only   = in TTS but not human (likely synthesis artifact — the TTS-only report may have OVER-claimed)
  - neither
Cite the actual numbers from BOTH speakers (with n). Mark is_espeak_error (espeak too broad/wrong) vs espeak correct. In tts_vs_human_note say explicitly whether TTS under-shows vs human. Respect confounds: durations are midpoint-boundary (raw CTC peaky); A1-P0 only valid F1>400; stop voicing/burst on raw span. Don't invent phenomena the numbers don't support.`

const verifyPrompt = (L, f) => `Adversarial reviewer. Check this TTS-vs-human comparison for ${L.name} against the two summaries. Default skeptical.

${JSON.stringify(f, null, 2)}

Re-read ${OUT}/${L.tts}.summary.json and ${OUT}/${L.human}.summary.json. For each: verdict supported/weak/refuted, confirm the presence class (both/human_only/tts_only/neither) is justified by adequate n in EACH speaker (a "both" needs evidence in both; a "human_only" needs the TTS number to genuinely lack it, not just smaller n). Kill artifacts (CTC peakiness for duration, A1-P0 low-F1, tiny n, model-substitution-only claims). Write a crisp one_line "phenomenon: TTS=… human=… -> presence". Do not add findings.`

phase('Compare')
const perLang = await pipeline(
  LANGS,
  (L) => agent(comparePrompt(L), { schema: COMPARE_SCHEMA, phase: 'Compare', label: `compare:${L.code}` }),
  (cmp, L) => agent(verifyPrompt(L, cmp), { schema: VERDICT_SCHEMA, phase: 'Verify', label: `verify:${L.code}` })
    .then((v) => ({ code: L.code, name: L.name, tts: L.tts, human: L.human, verified: v })),
)

const clean = perLang.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  `You are writing the TTS-vs-HUMAN espeak audit report (7 languages, 240 clips each per speaker). Below are adversarially-verified per-language comparisons. presence: both=robust across speakers, human_only=TTS under-showed it (the TTS-only report MISSED), tts_only=likely TTS artifact (TTS-only report OVER-claimed), neither.

${JSON.stringify(clean, null, 2)}

Write a clear, honest Markdown report and return it AS YOUR RESPONSE (the entire response is the markdown; do not use file tools). Structure:
1. TL;DR — the point of this pass: which prior TTS-only conclusions HELD UP on human speech, which were TTS artifacts, and what TTS MISSED (human_only).
2. Table: language | phenomenon | presence | TTS vs human (numbers) | verdict | espeak error?
3. The three lists that matter most: (a) ROBUST espeak errors confirmed in BOTH speakers (act on these); (b) HUMAN_ONLY — espeak errors TTS hid (e.g. flapping/reduction strength); (c) TTS_ONLY — claims to retract.
4. Per-language notes (only supported/weak).
5. Updated recommendation: which deterministic rules are now safe (confirmed in both), and which need the human signal.
End with a "## Summary" of 6-10 lines: the headline being how much the TTS-only audit generalised to real speech.`,
  { phase: 'Synthesize', label: 'synthesize' },
)

return { languages: clean.map((c) => c.code), synthesis: synth }
