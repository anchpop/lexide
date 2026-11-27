#!/usr/bin/env bash
# Lay out the Pizza TTS page in $1: the page, the built package (run `pnpm build` in tts/web first) as lexide-speech/.
# The voice bundles (lexide-speech/voices/) and the espeak-ng build (espeak-ng/, from tts/web-espeak) are large,
# immutable and built elsewhere, so they are published once and not copied here.
set -euo pipefail
out=$(realpath -m "${1:?pass the output directory}")
cd "$(dirname "$0")"
mkdir -p "$out/lexide-speech"
cp -r index.html demo.js speeds.json languages.json clips "$out/"
cp ../dist/*.js ../dist/*.mjs ../dist/*.wasm "$out/lexide-speech/"
