#!/usr/bin/env bash
# Exercise the same Rust bindings as build.sh, with nodejs glue for node:test.
set -euo pipefail
cd "$(dirname "$0")/.."
direnv exec /data/coding/yap wasm-pack build --target nodejs --out-dir target/node-pkg
direnv exec /data/coding/yap node --test tests/*.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py'
