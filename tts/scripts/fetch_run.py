"""Download a run from the private HF repo anchpop/lexide-tts-runs into runs/
(HF_TOKEN). Skips last.pt (optimizer state) unless --all.

    python scripts/fetch_run.py eng-nsf
"""
import json, os, sys, urllib.request
from pathlib import Path

REPO, name = "anchpop/lexide-tts-runs", sys.argv[1]
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300).read()


files = json.loads(get(f"https://huggingface.co/api/models/{REPO}/tree/main/{name}?recursive=true"))
for f in files:
    if f["type"] != "file" or (f["path"].endswith("last.pt") and "--all" not in sys.argv):
        continue
    target = Path("runs") / f["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(get(f"https://huggingface.co/{REPO}/resolve/main/{f['path']}"))
print(name, len(files), "entries")
