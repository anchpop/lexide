# Parsley model

`JointTagger` is the production tokenizer/tagger: 24-layer bge-m3 → character
BiLSTM → O/B/I boundaries, followed by POS, lemma edit-script and biaffine arc/relation
heads over [start character, end character, start subword] states. Training uses gold
word spans; inference uses predicted spans and exact single-root CLE (`mst.py`).
There are no boundary dictionaries or lemma-table overrides.

## Train and evaluate

From `tagging/`:

```bash
python tagger/data_prep_joint.py
python -m unittest discover -s tagger -p test_joint.py
sky launch -c parsley-joint tagger/sky_joint.yaml --secret HF_TOKEN
```

`run_joint.sh` runs smoke training, the full recipe, test predictions and scoring,
and publishes checkpoints under `training-runs/$RUN_NAME/`. `save_checkpoint` bundles
the encoder config, state dict, vocab and exact tokenizer. No retraining is needed
for the ONNX export.

`data_prep_joint.py` deduplicates by text and language, excludes v1-seen text from
validation/test candidates, and holds out up to 1000 test / 400 validation sentences
per language. Per-epoch language-cap sampling always includes all gold.
`eval_e2e.py` measures span-based token/POS/lemma/UAS/LAS F1, not gold-tokenization
accuracy. The shipped checkpoint is `training-runs/joint-v2-24L/best` at
`09a1f8b32248cc303132026ec488f4ad65085ecb` in `anchpop/lexide-parsley`.

## Export and release

Use a Python environment with torch, transformers, onnx and onnxruntime.

```bash
python tagger/export_joint_onnx.py --checkpoint "$CHECKPOINT" --out "$ARTIFACTS"
python tagger/verify_joint_onnx.py --checkpoint "$CHECKPOINT" --onnx "$ARTIFACTS" \
  --test data/processed-joint/test.jsonl --report "$ARTIFACTS/verification.json"
./release.sh
```

- `encoder.onnx` (+ `encoder.onnx.data`): input_ids, attention_mask, char_to_sub,
  char_ids, char_features, lang_id → boundary_logits, chars, sub_at_char.
- `heads.onnx`: chars, sub_at_char, starts, ends → POS/lemma logits and arc/rel scores.
- Batch size is one, character sequences are unpadded. The ordinary packed LSTM
  remains for batched training/serving. Sentences exceeding 8192 subwords fail clearly;
  segment passages first.
- `verify_joint_onnx.py` checks numerical differences and identical final predictions
  on length-spread examples in every language, including the longest sentences.
- `record_parity_fixtures.py` records **CPU fp32 PyTorch**, never live GPU responses.
- `bench_joint_onnx.py` compares fp32 and dynamic-int8 CPU throughput, size and full
  12k-test metrics. Only fp32 is published. Dynamic-int8 MatMul/Gemm (embeddings and
  LSTM stay fp32) measured 84.94 macro LAS versus 85.11 fp32: nearly free in measured
  accuracy, but 1,691/12,000 sentences have some prediction difference. Encoder size
  is 1.44 GB versus 2.35 GB; an uncontended 4-thread ORT run over 120 length-spread
  sentences measured 19.17/s versus 9.91/s (tokenization excluded). No int8 artifacts
  are shipped.

`release.sh` publishes `joint/*`, preserves `onnx/*` for 0.4.x, checks the immutable
Rust revision, deploys, and tests Rust parity. `lexide/examples/tag_jsonl.rs` scores
Rust predictions with the same evaluator; `bench_local.rs` measures local inference.

## V1 history

`train.py`, `train_tokenizer.py`, `predict.py`, `prior.py` and their data/teacher tools
remain for historical experiments. V1 chained a byte-minGRU tokenizer and XLM-R word
heads, with optional boundary priors and Wiktionary lemmas. It is no longer the local
or Modal parsley backend. The browser demo retains those byte models separately;
`predict.py` also provides the unchanged sentence-segmenter loader.
