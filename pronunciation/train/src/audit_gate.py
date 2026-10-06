"""Fail-closed transcript coverage for newly acquired read-speech sources."""
import hashlib
import json
from pathlib import Path

STRICT_SOURCES = frozenset({"mls", "cv", "aishell1", "aishell3"})


def load_coverage(paths):
    coverage = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source") or Path(path).stem.removesuffix("_asr_exclusions")
            if source in STRICT_SOURCES:
                coverage[(source, row["lang"], row["file"])] = row
    return coverage


def strict_verdict(record, lang, coverage):
    """Return None for legacy sources, otherwise require a current boolean verdict."""
    source = record.get("source")
    if source not in STRICT_SOURCES:
        return None
    key = (source, lang, record["file"])
    audit = coverage.get(key)
    expected = hashlib.sha256(record["sentence"].encode()).hexdigest()
    if "phonemes" in record:
        selection_current = (audit is not None
                             and bool(record.get("g2p_language"))
                             and bool(record.get("g2p_identity"))
                             and all(audit.get(k) == record[k] for k in ("g2p_language", "g2p_identity")))
    else:
        selection_current = audit is not None and audit.get("g2p_selection") == {
            key: record.get(key) for key in ("g2p_language", "variety", "espeak_voice")
        }
    if (audit is None or audit.get("ok") is not True
            or not selection_current
            or audit.get("expected_sha256") != expected
            or audit.get("phone_match_version") != 1
            or type(audit.get("phone_match")) is not bool):
        raise ValueError(f"{lang}/{record['file']}: missing, stale or failed {source} phone-match audit; run preprocess audit --sources {source}")
    return audit["phone_match"]
