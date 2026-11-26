"""A feature set without its clips under ``MIN_FRAMES`` (voca.build now rejects those itself), as a manifests-only set over the
original's features: python scripts/drop_short.py features/eng_gemini_multi_v4 features/eng_gemini_multi_v5"""
import sys, json, hashlib, os
from pathlib import Path
from voca.align import AlignmentSettings
from voca.build import MIN_FRAMES
from voca.config import Config
from voca.text import frontend_fingerprint
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); c = Config()
rows = {s: [r for r in map(json.loads, open(src / f"{s}.jsonl", encoding="utf-8")) if r["frames"] >= MIN_FRAMES] for s in ("train", "validation")}
ann = hashlib.sha256(json.dumps({"rows": [r["id"] for s in rows.values() for r in s], "frontend": frontend_fingerprint(),
                                 "acoustic": c.acoustic_fingerprint, "alignment": AlignmentSettings().to_dict()}, sort_keys=True).encode()).hexdigest()
dst.mkdir()
for s, rs in rows.items():
    with open(dst / f"{s}.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({**r, "annotation": ann}, ensure_ascii=False) + "\n" for r in rs)
root = json.loads((src / "annotations.json").read_text())["acoustic_root"]
(dst / "annotations.json").write_text(json.dumps({"fingerprint": ann, "complete": True, "acoustic_root": os.path.relpath(src / root, dst)}, indent=1))
(dst / "rejected.jsonl").write_text("")
print({s: len(r) for s, r in rows.items()})
