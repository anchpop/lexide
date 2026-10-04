#!/usr/bin/env bash
# Release joint parsley without touching the legacy onnx/ artifacts or publishing crates.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
PYTHON="${PYTHON:-$HOME/.venv-lexide-tests/bin/python}"
if [[ -f /tmp/pyenv.sh ]]; then source /tmp/pyenv.sh; fi
export HF_HUB_DISABLE_XET=1
REVISION=09a1f8b32248cc303132026ec488f4ad65085ecb
CHECKPOINT="$ROOT/tagger/output/hf-joint-v2/training-runs/joint-v2-24L/best"
ARTIFACTS="$ROOT/tagger/output/joint-onnx"
CARGO=(direnv exec /data/coding/yap cargo)
# MODAL_PYTHON can select a working interpreter if the venv's symlink broke after a Nix upgrade.
MODAL_PYTHON="${MODAL_PYTHON:-/nix/store/60m4rxhg2fldqaak400c0lry96ijrzqn-python3-3.13.13/bin/python3.13}"
modal_cli() { PYTHONPATH="$HOME/.modal-venv/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}" "$MODAL_PYTHON" -m modal "$@"; }

"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
snapshot_download('anchpop/lexide-parsley', revision='$REVISION',
                  allow_patterns='training-runs/joint-v2-24L/best/*',
                  local_dir='$ROOT/tagger/output/hf-joint-v2')
PY
"$PYTHON" -m unittest discover -s tagger -p test_joint.py
"$PYTHON" tagger/export_joint_onnx.py --checkpoint "$CHECKPOINT" --out "$ARTIFACTS"
"$PYTHON" tagger/verify_joint_onnx.py --checkpoint "$CHECKPOINT" --onnx "$ARTIFACTS" \
    --test data/processed-joint/test.jsonl --report tagger/output/joint-onnx-verification.json

# Upload only the six fp32 runtime files; no int8, checkpoints, segmenter or v1 deletions.
"$PYTHON" - "$ARTIFACTS" <<'PY'
import os
from pathlib import Path
import re
import sys
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if not token:
    token = next(line.split('=', 1)[1].strip().strip('\"\'')
                 for line in Path('../.env').read_text().splitlines() if line.startswith('HF_TOKEN='))
api = HfApi(token=token)
result = api.upload_folder(
    repo_id='anchpop/lexide-parsley', folder_path=sys.argv[1], path_in_repo='joint',
    allow_patterns=['encoder.onnx', 'encoder.onnx.data', 'heads.onnx', 'tokenizer.json', 'vocab.json', 'config.json'],
    commit_message='Publish verified fp32 joint parsley artifacts; preserve v1 onnx')
api.upload_file(repo_id='anchpop/lexide-parsley', path_or_fileobj='tagger/model_card.md',
                path_in_repo='README.md', commit_message='Document current joint parsley')
path = Path('lexide/src/local/mod.rs')
updated, count = re.subn(r'pub const MODEL_REVISION: &str = "[0-9a-f]{40}";',
                        f'pub const MODEL_REVISION: &str = "{result.oid}";', path.read_text())
assert count == 1, 'Expected one Rust artifact revision constant'
path.write_text(updated)
print('Rust artifact revision:', result.oid)
PY

SEGMENT_URL=https://anchpop--lexide-parsley-parsley-segment.modal.run
PASSAGES='{"texts":["Dr. Smith arrived at 3 p.m. He was early. Really?","こんにちは。元気ですか？","Eine Fundgrube. Das ist gut!"],"lang":"eng"}'
curl --fail --silent --show-error --max-time 300 "$SEGMENT_URL" -H 'Content-Type: application/json' \
    --data "$PASSAGES" > tagger/output/segment-before.json
JOINT_HF_REVISION="$REVISION" modal_cli deploy modal/modal_serve_tagger.py
curl --fail --silent --show-error --max-time 300 "$SEGMENT_URL" -H 'Content-Type: application/json' \
    --data "$PASSAGES" > tagger/output/segment-after.json
cmp tagger/output/segment-before.json tagger/output/segment-after.json
modal_cli run modal/verify_parsley.py

"$PYTHON" tagger/record_parity_fixtures.py --checkpoint "$CHECKPOINT"
export LEXIDE_MODEL_DIR="$ARTIFACTS"
"${CARGO[@]}" test --manifest-path lexide/Cargo.toml --features local --test parsley_parity -- --nocapture
"${CARGO[@]}" fmt --manifest-path lexide/Cargo.toml --check
printf 'Verified. Review the artifact revision and fixtures; nothing was committed or published to crates.io.\n'
