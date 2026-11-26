#!/usr/bin/env bash
# One language's single words and short phrases (data/scripts/<lang>_words.jsonl, Achernar): audio, frames, features
# (features/<lang>_words_v1, same gates as the sentences). Resumable.
#   set -a; . /data/coding/yap/.env; . ../.env; set +a; scripts/words_data.sh eng
set -euo pipefail
cd "$(dirname "$0")/.."
l=$1; set=data/${l}_gemini_achernar_words; PY=${PYTHON:-python}
$PY scripts/generate.py data/scripts/${l}_words.jsonl $set --provider gemini --language $l --voice Achernar --workers ${WORKERS:-16}
$PY scripts/fetch_frames.py $set --language $l --parallel 3
VOCA_LANGUAGE=$l PYTHONPATH=. $PY -m voca.build --set $set --output features/${l}_words_v1 --workers 12
