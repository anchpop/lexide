export const meta = {
  name: 'espeak-audit-4source',
  description: 'Validate each headline espeak-divergence across 4 independent sources (tatoeba human, google TTS, FLEURS multi-speaker, Pimsleur didactic) per language; adversarially verified; final cross-source report',
  phases: [
    { title: 'CrossSource', detail: 'one agent per language scores each phenomenon across all 4 sources' },
    { title: 'Verify', detail: 'adversarial check' },
    { title: 'Synthesize', detail: 'final 4-source report' },
  ],
}

const OUT = '/Users/andrepopovitch/coding/lexide/pronunciation/espeak_audit/out'

// 4 sources per language. fleurs/pimsleur are speaker-POOLED (voice=null):
// absolute vowel formants mix vocal tracts/genders -> read with caution;
// speaker-independent contrasts (voicing-by-position, deletion, nasalization-
// vs-own-oral, reduction) stay valid. Pimsleur is a hyper-careful didactic
// register that may SUPPRESS connected-speech allophony (flapping/reduction).
const LANGS = [
  { code: 'eng', name: 'English (US)',
    sources: { tatoeba: 'eng_tatoeba_CK', tts: 'eng_tts_en-US-Chirp3-HD-Algieba', fleurs: 'eng_fleurs_ALL', pimsleur: 'eng_pimsleur_ALL' },
    checklist: 'intervocalic /t,d/ flapping (voiced_frac + burst, intervoc vs elsewhere); /uː/ fronting (F2 uː vs ʊ); word-final voiced-stop devoicing; unstressed reduction (phone_dur stressed vs unstressed).' },
  { code: 'deu', name: 'German',
    sources: { tatoeba: 'deu_tatoeba_driini', tts: 'deu_tts_de-DE-Chirp3-HD-Fenrir', fleurs: 'deu_fleurs_ALL', pimsleur: 'deu_pimsleur_ALL' },
    checklist: 'final devoicing (voiceless word_final vf~0, absence of voiced word_final); lenis onset devoicing (voiced onset vf < intervoc — TTS hid this, do humans/fleurs/pimsleur show it?); tense/lax vowel quality (iː/ɪ, uː/ʊ); stress reduction (dur+F1).' },
  { code: 'fra', name: 'French',
    sources: { tatoeba: 'fra_tatoeba_Phoenix', tts: 'fra_tts_fr-FR-Chirp3-HD-Schedar', fleurs: 'fra_fleurs_ALL', pimsleur: 'fra_pimsleur_ALL' },
    checklist: 'nasal-vowel B1 widening vs oral_ref; e/ɛ and o/ɔ F1 splits; schwa over-emission/deletion (ə deletion rate); spurious English vowels (uː/ɔː/ɪ… each tiny n); no aspiration / full voicing.' },
  { code: 'ita', name: 'Italian',
    sources: { tatoeba: 'ita_tatoeba_NM1', tts: 'ita_tts_it-IT-Chirp3-HD-Leda', fleurs: 'ita_fleurs_ALL', pimsleur: 'ita_pimsleur_ALL' },
    checklist: 'spurious lax /ɪ ʊ/ (espeak splits, Italian merges with /i u/); e/ɛ and o/ɔ F1 splits; /r/→tap (model subs only); no unstressed reduction.' },
  { code: 'por', name: 'Portuguese (BR)',
    sources: { tatoeba: 'por_tatoeba_Lemmy', tts: 'por_tts_pt-BR-Chirp3-HD-Leda', fleurs: 'por_fleurs_ALL', pimsleur: 'por_pimsleur_ALL' },
    checklist: 'spurious /ʲ/ over-specification (deletion rate, model deletions); nasal-vowel B1 elevation vs oral; unstressed duration reduction; voiced/voiceless stop voicing.' },
  { code: 'rus', name: 'Russian',
    sources: { tatoeba: 'rus_tatoeba_fjay69', tts: 'rus_tts_ru-RU-Chirp3-HD-Fenrir', fleurs: 'rus_fleurs_ALL', pimsleur: 'rus_pimsleur_ALL' },
    checklist: '/ʲ/ over-specification (deletion ~0.09, all at frame floor, model deletes); akanye/reduction; final devoicing; onset/intervocalic voiced-obstruent devoicing leakage (TTS over-voiced and HID this — do the 3 human-ish sources show partial devoicing, p10 low?).' },
  { code: 'spa', name: 'Spanish (ES)',
    sources: { tatoeba: 'spa_tatoeba_arh', tts: 'spa_tts_es-ES-Chirp3-HD-Sulafat', fleurs: 'spa_fleurs_ALL', pimsleur: 'spa_pimsleur_ALL' },
    checklist: 'spirantization: espeak already emits β ð ɣ — are intervocalic voiced obstruents realized as continuants (few true /b d ɡ/ stops, β/ð/ɣ present)? no vowel reduction (stressed≈unstressed); voiced/voiceless stop voicing.' },
]

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    language: { type: 'string' },
    phenomena: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          phenomenon: { type: 'string' },
          per_source: {
            type: 'object', additionalProperties: false,
            properties: {
              tatoeba: { type: 'string' }, tts: { type: 'string' },
              fleurs: { type: 'string' }, pimsleur: { type: 'string' },
            },
            required: ['tatoeba', 'tts', 'fleurs', 'pimsleur'],
            description: 'present/absent/NA + the key number(with n) for each source',
          },
          n_sources_supporting: { type: 'integer' },
          robustness: { type: 'string', enum: ['all_four', 'three', 'two', 'one', 'none'] },
          is_espeak_error: { type: 'boolean' },
          caveat: { type: 'string', description: 'pooled-formant or pimsleur-register caveat if relevant' },
        },
        required: ['phenomenon', 'per_source', 'n_sources_supporting', 'robustness', 'is_espeak_error', 'caveat'],
      },
    },
  },
  required: ['language', 'phenomena'],
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
          robustness: { type: 'string', enum: ['all_four', 'three', 'two', 'one', 'none'] },
          is_espeak_error: { type: 'boolean' },
          verdict: { type: 'string', enum: ['supported', 'weak', 'refuted'] },
          one_line: { type: 'string', description: 'phenomenon: tato=… tts=… fleurs=… pims=… -> robustness' },
          critique: { type: 'string' },
        },
        required: ['phenomenon', 'robustness', 'is_espeak_error', 'verdict', 'one_line', 'critique'],
      },
    },
  },
  required: ['language', 'verified'],
}

const crossPrompt = (L) => `You are a phonetician validating espeak divergences for ${L.name} across FOUR independent data sources, each 240 clips, measured identically with parselmouth:
  tatoeba (one human reader): ${OUT}/${L.sources.tatoeba}.summary.json
  tts (Google Chirp3-HD):     ${OUT}/${L.sources.tts}.summary.json
  fleurs (MANY speakers pooled, voice=null): ${OUT}/${L.sources.fleurs}.summary.json
  pimsleur (didactic, pooled): ${OUT}/${L.sources.pimsleur}.summary.json
Read all four. A finding that appears across ALL sources — different speakers, registers, recording conditions — is bulletproof.

CRITICAL caveats:
- fleurs & pimsleur POOL speakers (and genders) -> their absolute vowel-formant tables mix vocal tracts; treat vowel-quality numbers there as LOW confidence. Speaker-independent contrasts (voicing-by-position from obstruents, deletion rates from overspecification, nasalization B1/A1-P0 vs that source's own oral_ref, stressed-vs-unstressed reduction) remain valid pooled.
- pimsleur is a hyper-careful didactic register and may SUPPRESS connected-speech allophony (flapping, reduction). Weaker effect in pimsleur is expected and is a register fact, not a refutation.
- TTS over-voices obstruents (known): if a devoicing finding is absent in tts but present in the 3 human-ish sources, that's a TTS artifact, not absence.

For each phenomenon in this checklist: ${L.checklist}
report per_source (present/absent/NA + the key number with n), count n_sources_supporting, assign robustness, mark is_espeak_error, and note the relevant caveat. Anchor within each source. Cite numbers.`

const verifyPrompt = (L, f) => `Adversarial reviewer for ${L.name} 4-source validation. Re-read the four summaries:
  ${OUT}/${L.sources.tatoeba}.summary.json, ${OUT}/${L.sources.tts}.summary.json, ${OUT}/${L.sources.fleurs}.summary.json, ${OUT}/${L.sources.pimsleur}.summary.json
Findings:
${JSON.stringify(f, null, 2)}

For each: confirm robustness (does each claimed-supporting source ACTUALLY have adequate n and the right direction?), down-weight pooled-formant vowel claims, treat pimsleur weakness as register (not refutation), treat tts-absent-but-humans-present devoicing as supported (TTS over-voicing). Assign verdict supported/weak/refuted and a crisp one_line with the four source numbers. No new phenomena.`

phase('CrossSource')
const perLang = await pipeline(
  LANGS,
  (L) => agent(crossPrompt(L), { schema: SCHEMA, phase: 'CrossSource', label: `cross:${L.code}` }),
  (f, L) => agent(verifyPrompt(L, f), { schema: VERDICT_SCHEMA, phase: 'Verify', label: `verify:${L.code}` })
    .then((v) => ({ code: L.code, name: L.name, verified: v })),
)
const clean = perLang.filter(Boolean)

phase('Synthesize')
const synth = await agent(
  `Write the FINAL 4-source espeak audit report (tatoeba human / google TTS / FLEURS multi-speaker / Pimsleur didactic; 7 languages, 240 clips each). Adversarially-verified per-language results:

${JSON.stringify(clean, null, 2)}

Return the COMPLETE markdown as your response (no file tools). Structure:
1. TL;DR — which espeak divergences are confirmed in ALL FOUR sources (bulletproof, act on these), in 3, etc.
2. Master table: language | phenomenon | tato | tts | fleurs | pims | robustness | espeak error?  (use ✓/✗/~/NA per source).
3. The bulletproof list (all_four or three with a clear reason the 4th is NA/register).
4. Source-character lessons: what each source is good/bad for (TTS over-voices; Pimsleur suppresses connected-speech allophony; FLEURS pooled-formant noise; tatoeba single-human variance).
5. Final label-enrichment recommendation: deterministic rules safe to ship (bulletproof + rule-governed), vs needs-acoustic-adjudication, vs still-unmeasured.
End with "## Summary" (6-10 lines): the degree to which the audit's findings are source-independent.`,
  { phase: 'Synthesize', label: 'synthesize' },
)

return { languages: clean.map((c) => c.code), synthesis: synth }
