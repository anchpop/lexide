"""ESPEAK and ESPEAK_DATA_PATH select the pinned training build. Pass the scripts directory. Needs cargo on PATH."""
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB.parent))
from plain import stock_espeak as stock
from voca import text

CHECK = ["cargo", "run", "-q", "--release", "--manifest-path", WEB.parent / "runtime/Cargo.toml",
         "--example", "frontend_parity", "--"]
scripts = Path(sys.argv[1])
run = subprocess.run
for language in stock.LANGUAGES:
    os.environ.update(LANGUAGE_CODE=language, VOCA_LANGUAGE=language)
    importlib.reload(stock)
    importlib.reload(text)
    training, validation = [], []
    achernar = scripts / f"{language}_achernar.jsonl"
    for script in sorted(scripts.glob(f"{language}_*.jsonl")):
        if language != "eng" and script != achernar:
            continue
        for row in map(json.loads, script.open()):
            if row.get("split") == "validation":
                validation.append(row["text"])
            elif script == achernar and row.get("split") == "train":
                training.append(row["text"])
    assert validation and training, language
    sample = random.Random(11).sample(training, min(len(training), 1000 if language == "eng" else 100))
    with tempfile.TemporaryFile(mode="w+") as fixtures:
        print(json.dumps([stock.parser_spec(language), text.frontend_spec(), list(text.LEXICAL)]), file=fixtures)
        tokens = unsupported = 0
        for sentence in validation + sample + ["‘Don’t’ – d'Artagnan... (hello)!", "𐐨\u001c--\u001c₂", "_", "?!"]:
            calls = []

            def record(args, **kwargs):
                result = run(args, **kwargs)
                calls.append([args[-1], result.stdout])
                return result

            with patch.object(stock.subprocess, "run", record):
                grams = stock.grams(sentence)
            try:
                value = text.tokenize(grams)
            except KeyError:
                expected = None  # Python also rejects phones outside the checkpoint inventory.
                unsupported += 1
            else:
                lexical = value.lexical()
                for row in lexical:
                    for col in text.HINTS:
                        row[col] = 0
                expected = [value.ids, lexical]
                tokens += len(value)
            print(json.dumps([sentence, calls, grams, expected]), file=fixtures)
        fixtures.seek(0)
        run(CHECK + ["espeak"], stdin=fixtures, check=True)
    print(f"{language}: {len(validation)} validation, {len(sample)} training, 4 edge cases, "
          f"{tokens} tokens, {unsupported} outside inventory", flush=True)
