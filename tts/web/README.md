# lexide-speech

`loadVoice(language, source, options)` loads a browser worker and warms up the voice. The espeak voices use the following language and espeak codes. English also accepts yap grams. All voices return mono Float32 PCM at 24 kHz. The engine is the Rust crate in `../runtime`, compiled to WebAssembly; the worker caches it and the voice in OPFS by hash.

| Language | Package | espeak-ng |
| --- | --- | --- |
| English | `eng` | `en-us` |
| French | `fra` | `fr-fr` |
| German | `deu` | `de` |
| Italian | `ita` | `it` |
| Spanish (Spain) | `spa-es` | `es` |
| Spanish (Latin America) | `spa` | `es-419` |
| Portuguese (Brazil) | `por` | `pt-br` |
| Russian | `rus` | `ru` |

```ts
import {loadVoice, type Gram} from 'lexide-speech'

const yap = await loadVoice('eng', 'yap')
const grams: Gram[] = [{words: ['go'], pos: ['VERB'], count: 120, phonemes: 'ɡ ˈoʊ'}]
const audio = await yap.speak(grams)
yap.release()

const espeak = {phonemize: (text: string) => runEspeak(text, 'en-us')}
const voice = await loadVoice('eng', espeak)
const {samples, sampleRate} = await voice.speak('Wait, are you sure?', {take: 0})
voice.release()
```

Supply your own `runEspeak` implementation (`../web-espeak` is one). The callback receives one stretch of text. Configure your wrapper for the model language (`en-us` for English). Return stdout, including word spaces, stress marks and underscore separators. The training build used upstream espeak-ng commit `ee9fa7cc` with the argument list `["-v", voice, "-q", "--ipa", "--sep=_", "--", text]`. The package keeps punctuation from the original text and maps espeak units into the voice inventory.

All voices take `onProgress` at load time. At synthesis time they take `speaker` (a name from `voice.speakers`), `take` (the delivery's seed: set one to reproduce a reading), `seed` (source noise and pitch jitter), `variety` (spread around the typical reading, 1; 0 is the typical reading), `speed` (0.25 to 4), `pitchShift`, `pitchRange` (0 to 3, 1), `pitchJitter` (0, off), `commaPause` and `stopPause` (extra seconds, 0.15 and 0.3), `voicedVowels` (true) and `fixEnding` (pull a rising ending toward the typical one, true). Each English utterance must fit the model’s 512-token limit. Serve the catalog’s hashed voice files in `voices/` next to the package files.

## Build and test

`pnpm build` needs `cargo` (with the `wasm32-unknown-unknown` target) and `wasm-pack` on `PATH`; it builds the engine into `dist/` and then compiles the TypeScript. `pnpm test:parity <scripts dir>` and `pnpm test:espeak <scripts dir>` check the Rust tokenizer and espeak mapping against voca/text.py and plain/stock_espeak.py (they need `cargo` too). `pnpm test:browser <yap.voca> [espeak.voca]` runs the built package in Chromium (`CHROMIUM_PATH`).
