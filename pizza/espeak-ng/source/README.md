# eSpeak NG for browser phonemization

This GPL-3.0-or-later module supplies the caller's phonemizer for a distilled
lexide-speech voice. Serve the contents of `dist/` together, with
`application/wasm` for the WASM file. Load the module on first text playback.

```js
const {loadEspeak} = await import('./espeak/index.mjs')
const espeak = await loadEspeak('en-us')
const english = await loadVoice('eng', espeak)
await english.speak('Hello, can you hear me?')
```

`await espeak.phonemize(text)` returns the UTF-8 CLI trace, including clause
newlines, for `espeak-ng -v en-us -q --ipa --sep=_ -- "$text"`.
The speech package splits punctuation and parses that trace. The browser
fetches additional dictionaries on first use, including for language switches
within a stretch. Supported voice names are `en-us` (default), `it`, `fr-fr`,
`de`, `es`, `es-419`, `pt-br`, and `ru`.

## Build

Use Emscripten, CMake, Make and a native C/C++ compiler. The script reuses
compiled tables in the upstream checkout's `build/espeak-ng-data/` if present.
A clean checkout builds its own native tables.

```sh
git clone https://github.com/espeak-ng/espeak-ng.git
git -C espeak-ng checkout ee9fa7cc4704d683020526f47a9655f14c8d5f4c
nix shell nixpkgs#{emscripten,cmake,gnumake,gcc} -c ./build.sh ./espeak-ng
```

The adapter and upstream license notices accompany the distribution.
[Upstream source at the pinned revision](https://github.com/espeak-ng/espeak-ng/tree/ee9fa7cc4704d683020526f47a9655f14c8d5f4c)
