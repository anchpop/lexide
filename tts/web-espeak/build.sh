#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
src=$(realpath "${1:?pass the pinned upstream espeak-ng checkout}")
[[ $(git -C "$src" rev-parse HEAD) == ee9fa7cc4704d683020526f47a9655f14c8d5f4c ]]
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Reuse native tables when available, otherwise compile them on the host.
data="$src/build/espeak-ng-data"
if [[ ! -f "$data/phondata" ]]; then
    cmake -S "$src" -B "$tmp/native" -DENABLE_TESTS=OFF
    cmake --build "$tmp/native" --target data -j4
    data="$tmp/native/espeak-ng-data"
fi
emcmake cmake -S "$src" -B "$tmp/cross" -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_C_FLAGS=-flto -DCMAKE_CXX_FLAGS=-flto -DENABLE_TESTS=OFF \
    -DUSE_ASYNC=OFF -DUSE_KLATT=OFF -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF \
    -DUSE_LIBPCAUDIO=OFF -DUSE_SPEECHPLAYER=OFF -DCOMPILE_INTONATIONS=OFF
cmake --build "$tmp/cross" --target espeak-ng -j4
base="$tmp/espeak-ng-data"
mkdir -p "$base" dist
cp "$data"/{phondata,phonindex,phontab,intonations,en_dict} "$base/"
# Embed the small voice definitions. Only English needs to download by default.
for lang in gmw/{en,en-US,de} roa/{it,fr,es,es-419,pt-BR} zle/ru; do
    mkdir -p "$base/lang/${lang%/*}"
    cp "$data/lang/$lang" "$base/lang/$lang"
done
# Voice dictionaries, plus language switches in their rules and espeak's
# foreign-alphabet table. These stay separate from the English download.
cp "$data"/{de,es,fr,it,pt,ru,ar,bn,el,gu,he,hi,hy,ka,kn,ko,ml,pa,pl,ta,te,cmn}_dict dist/
emcc -Oz -flto phonemize.c -I"$src/src/include" -I"$src/src/libespeak-ng" \
    -I"$src/src/ucd-tools/src/include" -I"$tmp/cross/src/libespeak-ng/include" \
    "$tmp/cross/src/libespeak-ng/libespeak-ng.a" "$tmp/cross/src/ucd-tools/libucd.a" \
    --embed-file "$base@/espeak-ng-data" --no-entry -sMODULARIZE -sENVIRONMENT=web \
    -Wl,--wrap=LoadDictionary -sASYNCIFY -sALLOW_MEMORY_GROWTH -sEXPORTED_FUNCTIONS=_initialize,_phonemize,_free \
    -sEXPORTED_RUNTIME_METHODS=ccall,FS,stringToNewUTF8,UTF8ToString -o dist/espeak.mjs
cp index.mjs dist/
mkdir -p dist/source
cp build.sh phonemize.c index.mjs COPYING* README.md dist/source/
