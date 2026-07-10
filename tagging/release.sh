#!/usr/bin/env bash
# Post-training release pipeline: turn freshly trained weights on HF into a verified,
# published, deployed parsley release. Run from tagging/ on the box after a training run
# (run_node.sh has pushed tagger/best + tokenizer/tokenizer.pt to anchpop/lexide-parsley):
#
#     ./release.sh
#
# Steps (each gates the next; the script stops on the first failure):
#   1. ONNX-export the tagger on Modal, numerically verified against PyTorch    -> volume
#   2. Export the char-minGRU weights + reference fixtures on Modal             -> volume
#   3. Pull the exported artifacts to data/onnx/
#   4. Rebuild training-data lemma priors (skipped if data/processed is absent)
#   5. Compile the Wiktionary tables + priors to fst                            -> data/onnx/lemma_fst
#   6. Rust unit tests — includes bit-for-bit char-tokenizer parity vs step 2's fixtures
#   7. Upload the complete data/onnx/ set to HF anchpop/lexide-parsley/onnx/
#      (write token from the repo-root .env; the Modal secret's token is read-only)
#   8. Deploy the parsley Modal serve (bakes the new weights + tables)
#   9. Re-record parity fixtures from the live serve
#  10. Run the token-for-token parity test: Rust local pipeline vs the live serve
#
# After it passes, commit the updated tests/fixtures/parsley_reference.json.
set -euo pipefail
cd "$(dirname "$0")"

MODAL="${MODAL:-$HOME/.modal-venv/bin/modal}"
VENV_PY="tagger/.venv/bin/python"          # has huggingface_hub (for the upload)
HF_REPO="anchpop/lexide-parsley"
ENV_FILE="../.env"                         # repo root .env with the write-role HF_TOKEN

# cargo: direct if on PATH, else via the yap flake devshell (how this box provides it)
if command -v cargo >/dev/null; then
    CARGO=(cargo)
else
    CARGO=(direnv exec /data/coding/yap cargo)
fi

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "1/10 ONNX export + PyTorch verification (Modal)"
"$MODAL" run tagger/export_modal.py

step "2/10 char-minGRU safetensors + fixtures export (Modal)"
"$MODAL" run tagger/export_char_modal.py

step "3/10 pull exported artifacts -> data/onnx/"
mkdir -p data/onnx
for f in tagger.onnx tokenizer.json vocab.json char_tokenizer.safetensors char_tokenizer_fixtures.json; do
    "$MODAL" volume get --force lexide-onnx "$f" data/onnx/
done

step "4/10 rebuild training-data lemma priors"
if [ -f data/processed/train.jsonl ]; then
    PYTHONPATH=tagger python3 tagger/build_lemma_priors.py
else
    echo "WARNING: data/processed/train.jsonl not found — keeping existing wikt_priors_*.json." >&2
    echo "         (Rebuild with data_prep.py + build_lemma_priors.py if training data changed.)" >&2
fi

step "5/10 compile lemma tables + priors -> fst"
(cd lexide && "${CARGO[@]}" run --release --no-default-features --features local \
    --bin build-lemma-fst -- --in ../data/lemma_tables --out ../data/onnx/lemma_fst)

step "6/10 Rust unit tests (incl. char-tokenizer bit-parity vs the fresh fixtures)"
(cd lexide && "${CARGO[@]}" test --lib --no-default-features --features local,remote --release)

step "7/10 upload data/onnx/ -> HF ${HF_REPO}/onnx/"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE with a write-role HF_TOKEN is required" >&2; exit 1; }
set -a; source "$ENV_FILE"; set +a
"$VENV_PY" - <<'PY'
import os
from huggingface_hub import upload_folder
r = upload_folder(
    repo_id="anchpop/lexide-parsley",
    folder_path="data/onnx",
    path_in_repo="onnx",
    token=os.environ["HF_TOKEN"],
    commit_message="release.sh: refresh onnx/ artifacts",
)
print("uploaded:", r.commit_url)
PY

step "8/10 deploy the parsley serve (Modal)"
"$MODAL" deploy modal/modal_serve_tagger.py

step "9/10 re-record parity fixtures from the live serve"
python3 tagger/record_parity_fixtures.py

step "10/10 token-for-token parity: Rust local pipeline vs the live serve"
(cd lexide && "${CARGO[@]}" test --test parsley_parity --no-default-features --features local --release -- --nocapture)

printf '\n\033[1mRelease verified.\033[0m Commit the refreshed fixtures:\n'
printf '  git add lexide/tests/fixtures/parsley_reference.json && git commit\n'
