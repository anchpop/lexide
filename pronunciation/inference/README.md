Pronunciation batch inference
=============================

`infer.py` supports length-bucketed offline batches, limits both clip count and
padded audio duration, and restores results to input order. The CLI loads at most
`--sort-window` files at once (128 by default). GPU out-of-memory errors split the
batch and retry; an oversized singleton still raises rather than disappearing.

```sh
python -m pronunciation.inference.infer \
  --repo /path/to/checkpoint --batch-size 4 --max-batch-seconds 60 \
  --format jsonl *.wav > predictions.jsonl
```

`--batch-size 1` remains the default. Batching can change predictions and emission timing
through floating-point arithmetic, even with correct attention masks. GroupNorm
and Cohere checkpoints retain singleton execution until their padding behavior
can be validated. The decoder's bulk GPU-to-CPU transfers apply in either mode.
The existing `--bf16` flag is independent of batching and changes precision.

Python callers can use `transcribe_batch(paths, model, processor, device,
batch_size=4)`, or `transcribe_audio_batch(audios, model, processor, device)` for
one already-loaded batch of mono 16 kHz float32 arrays. Bound the latter's input
size yourself, using `plan_batches` if needed. Acoustic-sidechannel models keep
raw waveforms; other models normalize each utterance before padding. Decoding
trims each sample to its actual encoder frame count.

Local regression checks:

```sh
bash pronunciation/scripts/py-linux.sh -m pytest pronunciation/inference/tests -q
```
