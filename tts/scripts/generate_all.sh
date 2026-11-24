#!/usr/bin/env bash
# Every language's audio, frame matrices and feature set (built and uploaded), unattended and resumable (scripts/language_data.py),
# French first. Before a language's full generation, 200 Achernar clips are generated, aligned and
# built: a language whose alignment accepts under 75% of them is skipped, not paid for.
#
#   setsid nohup scripts/generate_all.sh LOG_DIR > LOG_DIR/generate_all.log 2>&1 &
#
# Two chains: Gemini generation, and the phonetizer's frames (then build and upload) one language behind it.
# Needs PYTHON (an interpreter with the data dependencies), GEMINI_API_KEY in /data/coding/yap/.env, HF_TOKEN in ../.env.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=$1
PY=${PYTHON:-python}
ORDER="fra jpn spa deu ita por por-pt spa-es rus kor zho-hans tha hin"
mkdir -p "$LOG"
: > "$LOG/generated"
set -a; . /data/coding/yap/.env; . ../.env; set +a

gate() {   # 200 clips through alignment: is the language usable?
  local l=$1 out=runs/gate-$1
  $PY scripts/generate.py data/scripts/${l}_achernar.jsonl data/${l}_gemini_achernar --provider gemini --language $l --voice Achernar \
    --limit 200 --workers 12 > "$LOG/gate-$l.log" 2>&1 &&
  $PY scripts/fetch_frames.py data/${l}_gemini_achernar --language $l --parallel 2 >> "$LOG/gate-$l.log" 2>&1 &&
  rm -rf $out && VOCA_LANGUAGE=$l $PY -m voca.build --set data/${l}_gemini_achernar --output $out --workers 16 >> "$LOG/gate-$l.log" 2>&1 &&
  $PY -c "import sys; n = sum(1 for s in ('train', 'validation') for _ in open(f'$out/{s}.jsonl')); print('accepted', n); sys.exit(n < 150)" >> "$LOG/gate-$l.log"
  local ok=$?
  rm -rf $out
  return $ok
}

(
  for l in $ORDER; do
    if ! gate $l; then echo "SKIPPED $l: alignment gate failed ($(tail -n 1 "$LOG/gate-$l.log")) $(date +%T)"; echo "skip $l" >> "$LOG/generated"; continue; fi
    until $PY scripts/language_data.py $l --stages generate --workers 12 >> "$LOG/generate-$l.log" 2>&1; do sleep 60; done   # resumes
    echo "generated $l $(date +%T)"
    echo "$l" >> "$LOG/generated"
  done
  echo "generation done $(date +%T)"
) &

(
  $PY scripts/language_data.py eng --stages frames > "$LOG/frames-eng.log" 2>&1; echo "frames eng rc=$? $(date +%T)"
  for l in $ORDER; do
    until grep -qx "$l\|skip $l" "$LOG/generated"; do sleep 60; done
    grep -qx "skip $l" "$LOG/generated" && continue
    parallel=2; [ $l = fra ] && parallel=4
    $PY scripts/language_data.py $l --stages frames --frames-parallel $parallel > "$LOG/frames-$l.log" 2>&1 &&
    $PY scripts/language_data.py $l --stages build push > "$LOG/build-$l.log" 2>&1
    echo "frames, build and upload $l rc=$? $(date +%T)"
  done
  echo "frames done $(date +%T)"
) &
wait
