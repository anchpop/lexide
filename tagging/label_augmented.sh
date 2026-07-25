#!/usr/bin/env bash
# Label the augmented tokenization sentences with the Gemma teacher (two passes: the
# second retries transient failures). Train portions land next to the main silver files
# (data_prep picks up the _augmented name); holdouts go to data/aug_holdout/ for eval.
set -euo pipefail
cd "$(dirname "$0")"

BIN=lexide/target/release/label-tokenization
mkdir -p data/aug_holdout

for pass in 1 2; do
    echo "=== labeling pass $pass ==="
    for lang in deu eng fra hin ita jpn kor por rus spa; do
        "$BIN" "$lang" "data/aug_sentences/$lang.txt" \
            "data/big/$lang/target_language_sentences_tokenization_augmented.jsonl" 300
        "$BIN" "$lang" "data/aug_sentences/${lang}_holdout.txt" \
            "data/aug_holdout/$lang.jsonl" 300
    done
done
echo "=== all labeled ==="
wc -l data/big/*/target_language_sentences_tokenization_augmented.jsonl data/aug_holdout/*.jsonl
