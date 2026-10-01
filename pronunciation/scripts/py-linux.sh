#!/usr/bin/env bash
# Python for the data/audio tooling on the NixOS box.
#
# The interpreter named in CLAUDE.md is the maintainer's macOS conda base. On
# this machine the equivalent is a plain venv, but manylinux wheels (numpy,
# scipy, parselmouth, soundfile) link against libstdc++/libz, which NixOS does
# not put on the default loader path — so the venv only works with
# LD_LIBRARY_PATH pointing at the nix store. Paths are resolved through `nix
# build` rather than hardcoded so a store GC can't leave a dangling reference.
#
#   scripts/py-linux.sh data/generate_tts.py --backend gemini --langs jpn
#
# Create the venv once with:
#   python3 -m venv ~/.venv-lexide-data
#   ~/.venv-lexide-data/bin/pip install google-cloud-texttospeech soundfile \
#       numpy scipy tqdm praat-parselmouth
set -euo pipefail

VENV="${LEXIDE_DATA_VENV:-$HOME/.venv-lexide-data}"
if [ ! -x "$VENV/bin/python3" ]; then
    echo "no venv at $VENV — see the header of $0 for the one-time setup" >&2
    exit 1
fi

# Cache the resolved store paths; `nix build` is ~1s and this wrapper is used
# per-invocation, not per-clip.
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lexide-py-linux-ldpath"
if [ ! -s "$CACHE" ] || [ ! -d "$(head -c 60 "$CACHE")" ]; then
    mkdir -p "$(dirname "$CACHE")"
    nix build --no-link --print-out-paths \
        nixpkgs#zlib nixpkgs#stdenv.cc.cc.lib nixpkgs#libgcc 2>/dev/null \
        | sed 's|$|/lib|' | paste -sd: > "$CACHE"
fi

export LD_LIBRARY_PATH="$(cat "$CACHE")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$VENV/bin/python3" "$@"
