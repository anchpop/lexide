"""Reproducible source-stratified token PER through an existing batch endpoint.

No model/decoder or split reconstruction lives here. ``prepare`` fixes the sample
before predictions; ``run`` probes identity and checks every response envelope.
See README.md for usage, uncertainty, and the Stage 1 contamination caveat.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

IDENTITY_FIELDS = ("model_id", "model_revision", "deploy_marker", "decoder_version")
TENSE = {"k͈": "k", "p͈": "p", "t͈": "t", "s͈": "s", "tɕ͈": "tɕ"}
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json_text(value) + "\n")


def append_json(stream, value):
    stream.write(json_text(value) + "\n")
    stream.flush()


def allocate(counts, size):
    """Largest remainder, source-name tie break, no replacement."""
    total = sum(counts.values())
    if not 0 < size <= total or any(n <= 0 for n in counts.values()):
        raise ValueError("sample size must be positive and not exceed population")
    result = {source: size * n // total for source, n in counts.items()}
    ordered = sorted(counts, key=lambda s: (-(size * counts[s] % total), s))
    for source in ordered[:size - sum(result.values())]:
        result[source] += 1
    return result


def stratified_sample(rows, size, seed, language):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    counts = {s: len(rows) for s, rows in groups.items()}
    allocation = allocate(counts, size)
    selected = []
    for source in sorted(groups):
        population = sorted(groups[source], key=lambda r: (r["file"], r["label_line"]))
        # Independent streams keep one language/source unchanged if another is added.
        rng = random.Random(f"{seed}:{language}:{source}")
        selected.extend(rng.sample(population, allocation[source]))
    return selected, counts, allocation


def inspect_population(root, language):
    directory = root / language
    labels = directory / "phonemes_narrowed.jsonl"
    raw = labels.read_bytes()
    parsed, excluded = [], []
    for line, data in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(data)
            if not isinstance(row, dict):
                raise ValueError("label row is not an object")
            parsed.append((line, data, row))
        except (ValueError, UnicodeDecodeError) as error:
            excluded.append({"label_line": line, "reason": "invalid_json_object", "detail": str(error)})
    file_counts = Counter(r.get("file") for _, _, r in parsed if isinstance(r.get("file"), str))
    needs_manifest = any(not r.get("source") or r.get("duration_sec") is None for _, _, r in parsed)
    manifest_path = directory / "manifest.jsonl"
    manifest = {}
    manifest_sha = None
    if needs_manifest and manifest_path.exists():
        manifest_sha = file_digest(manifest_path)
        for data in manifest_path.read_text().splitlines():
            row = json.loads(data)
            if row["file"] in manifest:
                raise ValueError(f"ambiguous manifest file: {row['file']}")
            manifest[row["file"]] = row
    valid = []
    fallback_counts = Counter()
    for line, data, row in parsed:
        name = row.get("file")
        reason = None
        if not isinstance(name, str) or not name:
            reason = "invalid_file_field"
        elif file_counts[name] != 1:
            reason = "duplicate_file"
        elif not (directory / name).resolve().is_relative_to(directory.resolve()):
            reason = "file_outside_language_directory"
        elif not (directory / name).is_file():
            reason = "missing_audio"
        elif not isinstance(row.get("phonemes"), list) or not row["phonemes"] or any(
            not isinstance(t, str) or not t or t.isspace() for t in row["phonemes"]
        ):
            reason = "invalid_or_empty_reference"
        fallback = manifest.get(name, {}) if isinstance(name, str) else {}
        source = row.get("source") or fallback.get("source")
        duration = row.get("duration_sec")
        if duration is None:
            duration = fallback.get("duration_sec")
        duration_from_header = False
        if reason is None and duration is None:
            import soundfile as sf
            try:
                info = sf.info(directory / name)
                duration = info.duration
                duration_from_header = True
            except (RuntimeError, OSError):
                reason = "unreadable_audio_header"
        if reason is None and (not isinstance(source, str) or not source.strip()):
            reason = "missing_source"
        if reason is None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                               or not math.isfinite(duration) or duration <= 0):
            reason = "invalid_duration"
        if reason:
            excluded.append({"label_line": line, "file": name, "reason": reason})
            continue
        if not row.get("source"):
            fallback_counts["source"] += 1
        if row.get("duration_sec") is None and not duration_from_header:
            fallback_counts["duration_sec"] += 1
        if duration_from_header:
            fallback_counts["audio_header_duration"] += 1
        valid.append(dict(language=language, file=name, source=source, duration_sec=duration,
                          duration_origin="audio_header" if duration_from_header else
                          "label" if row.get("duration_sec") is not None else "manifest",
                          label_line=line, label_row_sha256=digest(data), label_row=row,
                          reference_tokens=row["phonemes"], audio_path=str((directory / name).resolve())))
    return valid, dict(labels_path=str(labels), labels_sha256=digest(raw),
                       manifest_sha256=manifest_sha, total_rows=len(raw.splitlines()),
                       eligible_rows=len(valid), exclusion_counts=dict(Counter(x["reason"] for x in excluded)),
                       exclusions=excluded, manifest_fallback_counts=dict(fallback_counts))


def prepare(root, languages, size, seed, output):
    import numpy as np
    import soundfile as sf

    selected, populations = [], {}
    for language in languages:
        rows, audit = inspect_population(root, language)
        sample, counts, allocation = stratified_sample(rows, size, seed, language)
        audit.update(source_population=counts, source_sample=allocation)
        populations[language] = audit
        for row in sample:
            # A corrupt selected input fails preparation, never gets replaced.
            samples, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
            if samples.size == 0 or not np.isfinite(samples).all() or sr <= 0:
                raise ValueError(f"invalid selected audio: {row['audio_path']}")
            row.update(audio_sha256=file_digest(row["audio_path"]), sample_rate=sr,
                       channels=samples.shape[1], sample_count=samples.shape[0])
        selected.extend(sample)
        print(json_text({"language": language, **{k: v for k, v in audit.items() if k != "exclusions"}}), flush=True)
    plan = dict(schema_version=1, stage="Stage 1: contaminated native-label diagnostic, not heldout",
                seed=seed, sample_per_language=size, languages=languages, populations=populations,
                selected=selected, created_at=datetime.now(timezone.utc).isoformat(),
                selection="largest remainder proportional source strata; uniform without replacement; source-name ties",
                code_sha256=file_digest(__file__))
    write_json(output, plan)
    print(f"Prepared {len(selected)} clips; plan_sha256={file_digest(output)}", flush=True)


def align(reference, prediction):
    """Unit token Levenshtein; backtrace ties: diagonal, deletion, insertion.

    Return (reference token | None, prediction token | None) pairs. Null means
    an insertion/deletion, never a literal token. No normalization is performed.
    """
    n, m = len(reference), len(prediction)
    distance = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(distance[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]),
                                 distance[i - 1][j] + 1, distance[i][j - 1] + 1)
    pairs = []
    i, j = n, m
    while i or j:
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            pairs.append((reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, prediction[j - 1]))
            j -= 1
    return list(reversed(pairs))


def extract_tokens(result):
    if not isinstance(result, dict) or "error" in result or result.get("load_error"):
        raise ValueError(f"prediction error: {result}")
    phones = result.get("phonemes")
    if not isinstance(phones, list):
        raise ValueError("prediction must contain a phonemes list")
    tokens = []
    for phone in phones:
        if not isinstance(phone, dict) or not isinstance(phone.get("phoneme"), str):
            raise ValueError("invalid phoneme object")
        token = phone["phoneme"]
        # Only the API's one prefixed stress mark is removed; canonical IPA stays exact.
        if token.startswith(("ˈ", "ˌ")):
            token = token[1:]
        if not token or token.isspace() or token.startswith(("ˈ", "ˌ")):
            raise ValueError("empty or malformed predicted token")
        tokens.append(token)
    return tokens


def validate_identity(envelope, expected):
    if not isinstance(envelope, dict):
        raise ValueError("response envelope is not an object")
    for key in IDENTITY_FIELDS:
        if not isinstance(expected.get(key), str) or not expected[key] or envelope.get(key) != expected[key]:
            raise ValueError(f"identity mismatch for {key}: expected {expected.get(key)!r}, got {envelope.get(key)!r}")
    if envelope.get("load_error") is not None:
        raise ValueError(f"model load failed: {envelope['load_error']}")


def validate_batch(envelope, expected, count):
    validate_identity(envelope, expected)
    if not isinstance(envelope.get("results"), list) or len(envelope["results"]) != count:
        raise ValueError("batch result count/order contract violated")
    # Validate all items before any metric computation; raw envelope is already saved.
    return [extract_tokens(result) for result in envelope["results"]]


def score(reference, prediction):
    if not reference:
        raise ValueError("PER requires a nonempty reference")
    alignment = align(reference, prediction)
    counts = Counter("insertion" if r is None else "deletion" if p is None else "substitution"
                     for r, p in alignment if r != p)
    edits = sum(counts.values())
    return dict(edits=edits, reference_count=len(reference), prediction_count=len(prediction),
                per=edits / len(reference), substitutions=counts["substitution"],
                deletions=counts["deletion"], insertions=counts["insertion"], alignment=alignment)


def metrics(rows):
    if not rows:
        raise ValueError("cannot summarize zero clips")
    references = sum(r["reference_count"] for r in rows)
    edits = sum(r["edits"] for r in rows)
    ratio = edits / references
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    # Stratified CLIP-level ratio delta method, conditioned on realized n_h.
    # z_i = edits_i - R * ref_i; var(R) ~= sum_h n_h*s_h(z)^2 / (sum refs)^2.
    # No finite-population correction (conservative); not token-binomial SE.
    singleton_sources = sorted(source for source, group in groups.items() if len(group) < 2)
    estimable = len(rows) >= 2
    method = "stratified_clip_delta"
    if singleton_sources:
        # Include between-source variation rather than impute a singleton variance.
        method = "unstratified_clip_delta_singleton_fallback"
        residuals = [r["edits"] - ratio * r["reference_count"] for r in rows]
        variance = len(rows) * statistics.variance(residuals) if estimable else 0.0
    else:
        variance = sum(len(group) * statistics.variance(
            r["edits"] - ratio * r["reference_count"] for r in group
        ) for group in groups.values())
    return dict(scored=len(rows), reference_tokens=references, edits=edits, per=ratio,
                mean_clip_per=statistics.mean(r["per"] for r in rows),
                median_clip_per=statistics.median(r["per"] for r in rows),
                per_se=math.sqrt(variance) / references if estimable else None,
                se_singleton_sources=singleton_sources,
                se_method=method if estimable else "unavailable_fewer_than_two_clips",
                substitutions=sum(r["substitutions"] for r in rows),
                deletions=sum(r["deletions"] for r in rows), insertions=sum(r["insertions"] for r in rows))


def summarize(rows, plan):
    languages = {}
    for language in plan["languages"]:
        group = [r for r in rows if r["language"] == language]
        languages[language] = {**metrics(group), "sources": {
            source: {**metrics([r for r in group if r["source"] == source]),
                     "population": plan["populations"][language]["source_population"][source]}
            for source in sorted({r["source"] for r in group})}}
    korean = [r for r in rows if r["language"] == "kor"]
    confusions = Counter((ref, pred) for row in korean for ref, pred in row["alignment"] if ref != pred)
    tense = {}
    for token, plain in TENSE.items():
        pairs = [p for row in korean for r, p in row["alignment"] if r == token]
        tense[token] = dict(plain_counterpart=plain, reference_support=len(pairs),
                            prediction_support=sum(row["prediction_tokens"].count(token) for row in korean),
                            correct_matches=pairs.count(token), plain_substitutions=pairs.count(plain),
                            deletions=pairs.count(None),
                            other_substitutions=sum(p not in (token, plain, None) for p in pairs))
    return dict(languages=languages, korean_tense=tense, korean_top_15_confusions=[
        dict(reference=r, prediction=p, count=n) for (r, p), n in sorted(
            confusions.items(), key=lambda x: (-x[1], x[0][0] or "", x[0][1] or ""))[:15]],
        per_se_method="stratified clip-level delta; z=edits-PER*reference_count; sqrt(sum_h n_h*sample_variance(z_h))/sum_refs; no finite-population correction; absolute PER units; if any singleton source, fall back to unstratified clip delta sqrt(n*sample_variance(z))/sum_refs; null for n<2; each metric records se_method",
        estimand="pooled sample token PER; source allocation proportional with largest-remainder rounding; mean/median are unweighted clip PER",
        limitation=plan["stage"] + "; approximately 95% train / 5% validation, not reconstructed; native-label fit cannot establish learner/minimal-pair discrimination")


class Client:
    def __init__(self, expected, raw_log, attempts=4, timeout=240, session=None):
        import requests
        self.expected, self.raw_log = expected, raw_log
        self.attempts, self.timeout = attempts, timeout
        self.session = session or requests.Session()

    def post(self, url, payload, clip_ids):
        import requests
        request_hash = digest(json_text(payload).encode())
        for attempt in range(1, self.attempts + 1):
            entry = dict(url=url, clip_ids=clip_ids, request_sha256=request_hash, attempt=attempt,
                         timestamp=datetime.now(timezone.utc).isoformat())
            try:
                response = self.session.post(url, json=payload, timeout=(30, self.timeout))
            except (requests.Timeout, requests.ConnectionError) as error:
                append_json(self.raw_log, {**entry, "transport_error": str(error)})
                if attempt == self.attempts:
                    raise
            else:
                append_json(self.raw_log, {**entry, "status_code": response.status_code,
                                          "response_headers": dict(response.headers), "response_text": response.text})
                try:
                    envelope = response.json()
                except ValueError:
                    envelope = None
                identity = envelope
                if isinstance(envelope, dict) and isinstance(envelope.get("detail"), dict):
                    identity = envelope["detail"]
                # Explicit application envelopes must match even on error responses.
                if isinstance(identity, dict) and any(k in identity for k in (*IDENTITY_FIELDS, "load_error")):
                    validate_identity(identity, self.expected)
                if 200 <= response.status_code < 300:
                    validate_identity(envelope, self.expected)
                    return envelope
                if response.status_code not in TRANSIENT_STATUS or attempt == self.attempts:
                    response.raise_for_status()
            time.sleep(min(2 ** attempt, 20))
        raise RuntimeError("unreachable retry state")


def run(plan_path, output, single_url, batch_url, expected, batch_size, attempts, timeout):
    import numpy as np
    import soundfile as sf

    if not 1 <= batch_size <= 64:
        raise ValueError("batch size must be 1..64")
    plan = json.loads(plan_path.read_text())
    if not output.is_dir():
        output.mkdir(parents=True)
    # Exclusive files prevent accidental mixing across runs; failure leaves all raw evidence.
    with (output / "stage1-responses.jsonl").open("x") as raw, (output / "stage1-per-clip.jsonl").open("x") as clips:
        client = Client(expected, raw, attempts, timeout)
        for population in plan["populations"].values():
            if file_digest(population["labels_path"]) != population["labels_sha256"]:
                raise ValueError("reference file changed after sampling")
        probe = client.post(single_url, {"marker_only": True}, [])
        print("MODEL VALIDATED " + json_text(probe), flush=True)
        rows = []
        for start in range(0, len(plan["selected"]), batch_size):
            selected = plan["selected"][start:start + batch_size]
            requests, inputs = [], []
            for row in selected:
                path = row["audio_path"]
                if file_digest(path) != row["audio_sha256"]:
                    raise ValueError(f"audio changed after sampling: {path}")
                samples, sr = sf.read(path, dtype="float32", always_2d=True)
                if not samples.size or not np.isfinite(samples).all():
                    raise ValueError(f"invalid audio: {path}")
                samples = samples.mean(axis=1, dtype=np.float32)
                pcm = np.asarray(samples, dtype="<f4").tobytes()
                requests.append(dict(audio_f32_b64=base64.b64encode(pcm).decode("ascii"),
                                     sample_rate=sr, language=row["language"]))
                inputs.append(dict(pcm_f32le_sha256=digest(pcm), sample_rate=sr, sample_count=len(samples)))
            envelope = client.post(batch_url, {"requests": requests}, [f"{r['language']}/{r['file']}" for r in selected])
            predictions = validate_batch(envelope, expected, len(selected))
            for index, (row, prediction, encoded) in enumerate(zip(selected, predictions, inputs)):
                scored = {**row, **encoded, **score(row["reference_tokens"], prediction),
                          "prediction_tokens": prediction, "response": envelope["results"][index],
                          "identity": {key: envelope[key] for key in IDENTITY_FIELDS},
                          "batch_start": start, "batch_index": index}
                append_json(clips, scored)
                rows.append(scored)
            print(f"Scored {len(rows)}/{len(plan['selected'])}", flush=True)
        summary = {**summarize(rows, plan), "identity": expected, "plan_sha256": file_digest(plan_path),
                   "code_sha256": file_digest(__file__), "per_clip_sha256": file_digest(output / "stage1-per-clip.jsonl"),
                   "responses_sha256": file_digest(output / "stage1-responses.jsonl"),
                   "populations": plan["populations"], "seed": plan["seed"],
                   "batch_size": batch_size, "single_url": single_url, "batch_url": batch_url,
                   "python": sys.version, "numpy_version": np.__version__, "soundfile_version": sf.__version__,
                   "finished_at": datetime.now(timezone.utc).isoformat()}
        write_json(output / "stage1-summary.json", summary)
        print("EVAL FINISHED " + json_text(summary["languages"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--audio-root", type=Path, required=True)
    prep.add_argument("--languages", nargs="+", default=["kor", "tha", "hin", "jpn", "zho-hans", "spa"])
    prep.add_argument("--sample-size", type=int, default=200)
    prep.add_argument("--seed", type=int, default=20260915)
    prep.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("run")
    evaluation.add_argument("--plan", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--single-url", required=True)
    evaluation.add_argument("--batch-url", required=True)
    for field in IDENTITY_FIELDS:
        evaluation.add_argument("--" + field.replace("_", "-"), required=True)
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--attempts", type=int, default=4)
    evaluation.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.audio_root.resolve(), args.languages, args.sample_size, args.seed, args.output)
    else:
        if args.attempts < 1 or args.timeout <= 0:
            parser.error("attempts and timeout must be positive")
        run(args.plan, args.output_dir, args.single_url, args.batch_url,
            {key: getattr(args, key) for key in IDENTITY_FIELDS}, args.batch_size, args.attempts, args.timeout)


if __name__ == "__main__":
    main()
