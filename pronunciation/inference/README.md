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

Source-stratified endpoint PER
-----------------------------

`eval_per.py` evaluates existing canonical token labels without loading a model,
implementing a decoder, relabeling, or reconstructing training splits. Its two
commands separate outcome-independent sampling from inference. For example, from
this checkout on NixOS (replace URLs/identity with actual deployment output):

```sh
LEXIDE_DATA_VENV=/home/andrep/.venv-lexide-tests \
bash /data/coding/lexide/pronunciation/scripts/py-linux.sh \
  /data/coding/lexide/pronunciation/inference/eval_per.py prepare \
  --audio-root /data/coding/lexide/pronunciation/data/audio \
  --languages kor tha hin jpn zho-hans spa --sample-size 200 --seed 20260915 \
  --output /data/coding/lexide/.work/korean-eval/stage1-plan.json

LEXIDE_DATA_VENV=/home/andrep/.venv-lexide-tests \
bash /data/coding/lexide/pronunciation/scripts/py-linux.sh \
  /data/coding/lexide/pronunciation/inference/eval_per.py run \
  --plan /data/coding/lexide/.work/korean-eval/stage1-plan.json \
  --output-dir /data/coding/lexide/.work/korean-eval \
  --single-url "$SINGLE_URL" --batch-url "$BATCH_URL" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --deploy-marker "$DEPLOY_MARKER" --decoder-version nonblank_v1
```

The caller owns deployment and teardown; this harness cannot deploy/stop anything.
It reads only `phonemes_narrowed.jsonl` references. Preparation audits row shape,
nonempty token lists, duplicate paths, audio existence, source, and positive finite
duration; manifest fallback supplies only missing source/duration. All excluded
rows and reasons are retained in the plan. Missing duration in both labels and
manifest is measured from the audio header (tracked separately); it does not exclude
otherwise-valid FLEURS/Tatoeba rows. It samples uniformly without replacement
inside proportional source strata, using largest-remainder allocation (source-name
tie break) and independent seeded source/language streams. No training-filter or
split reconstruction is applied. Selected unreadable/nonfinite audio fails rather
than being replaced. Population labels, selected raw rows, audio files, and decoded
mono float32 payloads are SHA-256 pinned. Selected full label rows are retained.

Run sends sequential batches of 32 (configurable 1–64), never concurrent requests.
Soundfile decodes float32, stereo channels are averaged, and little-endian float32
mono samples—not WAV bytes—are sent at the original sample rate; the existing
service handles resampling/normalization/decoding. All four identity fields must
match the probe and every batch envelope. Model-load, malformed response, identity,
and per-item errors abort without dropping or replacing clips. Four attempts with
bounded exponential backoff cover only transport failures and transient HTTP
408/429/500/502/503/504; application load errors are never retried. Raw response
bodies, headers, request hashes, attempts, and item ordering are flushed to
`stage1-responses.jsonl` before validation. Scored rows and full item responses go to
`stage1-per-clip.jsonl`; `stage1-summary.json` contains metrics and provenance.
Outputs are exclusive-create: use a new directory after failures, preserving the
original evidence. No silent resume, overwrite, or selection on model outcome.

PER compares canonical tokens exactly, ignoring the separate stress factor and
removing only the API's leading `ˈ`/`ˌ` stress prefix; it does not normalize Unicode,
merge tokens, or collapse repeated emissions. Unit-cost Levenshtein backtrace ties
prefer diagonal, then deletion, then insertion. Confusions and tense correct/plain/
deletion/other counts depend on this deterministic alignment; raw reference and
prediction supports do not. JSON null denotes insertion/deletion, not an IPA token.

Headline PER is total edits / total reference tokens, alongside clip mean/median
and per-source counts/PER. Approximate SE uses a **stratified clip-level ratio delta
method**: with `R = sum edits / sum refs`, `z_i = edits_i - R * refs_i`,
`SE = sqrt(sum_h n_h * sample_variance(z_h)) / sum refs`. This conditions on actual
largest-remainder sample counts, omits the finite-population correction, and reports
absolute PER units (multiply by 100 for percentage points). If any source stratum
is singleton, the whole metric falls back to unstratified clip-level ratio delta
`SE = sqrt(n * sample_variance(z)) / sum refs`, retaining between-source variation
rather than imputing a singleton variance. Each metric records `se_method` and
`se_singleton_sources`. Fewer than two clips give null SE. It is not token-binomial
uncertainty and does not account for speaker/title clustering, label errors, or
training contamination. Rounded allocations can differ
slightly from exact population source weights; the headline is the pooled sample
ratio, not an exactly source-weighted population estimator.

Stage 1 samples from a corpus with approximately 95% training / 5% validation
contamination. These are **not heldout results**. Cross-language/source comparisons
are diagnostics of native-label fit, not proof that a model hears learners' accent
errors or isolated tense/plain minimal pairs. No Stage 2 is performed here.

Tests (including hand-calculated alignments, sampling, SE, and HTTP failures):

```sh
LEXIDE_DATA_VENV=/home/andrep/.venv-lexide-tests \
bash /data/coding/lexide/pronunciation/scripts/py-linux.sh \
  -m pytest /data/coding/lexide/pronunciation/inference/tests -q
```
