# Parsley

Parsley jointly tokenizes and tags raw sentences in 12 languages: deu, eng, fra, hin,
ita, jpn, kor, por, rus, spa, tha and zho-hans. One 24-layer bge-m3 encoder feeds a
character BiLSTM, an O/B/I boundary head, and POS, lemma-edit-script and biaffine
dependency heads. Dependencies use exact single-root Chu–Liu–Edmonds decoding.
Predicted spans cover every non-whitespace character; words sharing an encoder
piece have distinct character representations.

## Key facts

- **Checkpoint:** `anchpop/lexide-parsley`, revision
  `09a1f8b32248cc303132026ec488f4ad65085ecb`, `training-runs/joint-v2-24L/best/`.
- **Rust 0.5 local:** fp32 `joint/encoder.onnx` (+ `.data` sidecar), `heads.onnx`,
  tokenizer, vocab and config. Downloads pin `MODEL_REVISION` in
  `lexide/src/local/mod.rs`; `LEXIDE_MODEL_DIR` overrides the download.
  The old `onnx/` artifacts remain for released 0.4.x crates.
- **Tag URL:** `https://anchpop--lexide-parsley-parsley-tag.modal.run`.
  Modal app `lexide-parsley`, L4, shared batching queue, bf16 encoder/heads and fp32 LSTM.
- **Segment URL:** `https://anchpop--lexide-parsley-parsley-segment.modal.run`.
  Independent CPU byte-minGRU; Rust's `segment` feature remains independent too.
- **Release:** `./release.sh` exports and verifies ONNX, uploads only `joint/`, deploys
  the pinned checkpoint, records CPU fp32 fixtures and checks Rust parity. It does
  not publish a crate or commit.
- **Parity:** 288 length-spread ONNX/PyTorch cases and 293 Rust/PyTorch fixtures across
  all 12 languages. GPU bf16 outputs are not the Rust parity reference.

## Quality

End-to-end F1 on 12k held-out sentences, with tokenization errors charged to every
word task; an arc counts only when both dependent and head spans match:

| macro | token | POS | lemma | UAS | LAS | jpn LAS | zho-hans LAS | kor LAS | tha LAS |
|---|---|---|---|---|---|---|---|---|---|
| v1 (byte tokenizer + XLM-R tagger) | 95.2 | 90.7 | 92.0 | 74.9 | 71.3 | 58.2 | 52.0 | 60.3 | 63.1 |
| joint, 18 layers, 2 epochs | 99.1 | 97.7 | 98.0 | 86.8 | 84.3 | 78.9 | 76.8 | 81.3 | 81.4 |
| **parsley (joint, 24 layers, 3 epochs)** | **99.1** | **97.9** | **98.1** | **87.4** | **85.1** | **80.2** | **77.2** | **82.0** | **82.3** |

These are agreement with the Gemma teacher, not independent gold. The test sentences appear
nowhere in v1's data, so both models are scored on the same unseen split. Part of v1's
deficit is label drift since August (e.g. English contractions are now split `did|n't`), but
that is what Gemma emits today. The old gold-tokenization metrics overstated v1 badly.

**Why v1 was weak exactly where it was weak.** v1 represented a word by its *first subword's*
vector. In Japanese, Chinese and Thai one SentencePiece piece often spans several of our
tokens, so distinct words got identical vectors: 22.5% of jpn words, 16.5% zho-hans, 13.2%
tha, 5.9% kor, ~0% elsewhere. That is the ranking of v1's failures. The character layer gives
every word its own state, and moving tokenization into the encoder gives segmentation
bge-m3's lexical knowledge instead of a 1M-param byte model plus dictionary priors.

## Cost and speed

| | throughput | cost |
|---|---|---|
| parsley, Modal L4 | ~850 sentences/s | ~$0.26 / M sentences (GPU) |
| Gemma 4 31B serve, A100-80GB, 600 in flight | 7.5 sentences/s, 317s cold start | ~$90 / M |
| Rust local, fp32, Ryzen 9 3900X | 6/s on 1 thread, 12/s on 12 | 2.2 GB RSS, 2.2s cached load |
| int8 encoder (not shipped) | ~2x fp32 on CPU | LAS 84.94 vs 85.11, 1.44 GB |

Both training runs together cost ~$11 on a Lambda A100 (1,300-1,450 sentences/s training).

## What still limits it

The remaining disagreement is mostly the teacher being inconsistent, not missing knowledge.
Measured on the 24-layer model's test predictions (2026-10-04):
- **Errors sit on frequent words.** 77% of POS errors are on forms seen 50+ times in training
  (AUX/VERB, NOUN/PROPN, DET/PRON). For jpn/zho-hans/tha tokenization, 40-56% of misses are
  on words seen 50+ times and only 8-27% on unseen words.
- **The teacher flips coins on conventions.** 不用 is one token 47% of the time, 有人 57%,
  ので 48%, with no usage difference between the two choices. German modal + infinitive
  makes the modal the head 33-45% of the time and the infinitive the rest. The same split
  shows up in both the LLM-cleaned gold and Gemma's silver. One string that *does* split by
  usage, and is labelled correctly: どうか (one token as "please", か|どう|か as "whether").
- **Dictionaries don't help.** Even a dictionary that fixed every miss on unseen and rare
  words would leave token F1 at roughly jpn 98.9 / zho-hans 97.4 / tha 98.1. v1's Wiktionary
  lemma floor applied to parsley's predictions is net-negative in 6 of 9 languages.

## Where we're going

- **Pick one convention per ambiguous construction**, chosen for what a learner should see as
  one word or card. Apply it mechanically in yap's `token_corrections`, relabel, and retrain
  (~$7). This is the lever the error analysis points at.
- **Switch yap** from the Gemma serve to the parsley tag URL (the client already parses it).
- **int8 encoder**: nearly free in accuracy and halves CPU time. Ship it if local speed
  matters more than bit-for-bit parity with the fp32 reference.

## Training and history

See `tagger/README.md` for joint training and export, `modal/README.md` for serving,
and `lexide/README.md` for Rust APIs. `HISTORY.md` keeps v1's and the segmenter's
development notes and lessons. V1 training and prior tools remain as historical
and teacher tooling; its byte-tokenizer browser demo retains its own implementation.
The Gemma teacher in `train/` remains available for silver labels. This release does
not change yap, retire its teacher endpoint, or redeploy the browser demo.

On this box: Python uses `/tmp/pyenv.sh` and `~/.venv-lexide-tests`; Rust uses
`direnv exec /data/coding/yap`. Model downloads, predictions and measurements live
under gitignored `tagger/output/`.
