"""Compare the built tokenizer with Python using the local training scripts."""
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB.parent))
from voca import text

scripts = Path(sys.argv[1]) if len(sys.argv) > 1 else WEB.parent / "data/scripts"
available = sorted(scripts.glob("*_achernar.jsonl"))
if not available:
    raise SystemExit(f"No *_achernar.jsonl scripts in {scripts}")
for script in available:
    lang = script.name.removesuffix("_achernar.jsonl")
    if not (WEB.parent / "voca/inventories" / f"{lang}.json").exists():
        continue
    os.environ["VOCA_LANGUAGE"] = lang
    importlib.reload(text)
    rows = [json.loads(line) for line in script.open()]
    training = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    if lang == "eng":
        for other in sorted(scripts.glob("eng_*.jsonl")):
            if other == script:
                continue
            validation.extend(row for row in map(json.loads, other.open()) if row.get("split") == "validation")
    sample = random.Random(11).sample(training, min(1000, len(training)))
    with tempfile.TemporaryFile(mode="w+") as fixtures:
        print(json.dumps([text.frontend_spec(), list(text.LEXICAL)]), file=fixtures)
        tokens = 0
        for row in validation + sample:
            value = text.tokenize(row["grams"])
            tokens += len(value)
            print(json.dumps([row["id"], row["grams"], value.ids, value.lexical()]), file=fixtures)
        fixtures.seek(0)
        subprocess.run(["node", WEB / "test/parity.mjs"], stdin=fixtures, check=True)
    print(f"{lang}: {len(validation)} validation, {len(sample)} training, {tokens} tokens", flush=True)

