#!/usr/bin/env python3
"""Modal deployment for `parsley` — the small CPU tagger that replaces the Gemma pipeline.

Serves the encoder multi-task tagger (POS + lemma + dependency) plus the byte-minGRU
tokenizer from anchpop/lexide-tagger, on CPU, with scale-to-zero. One forward pass per
sentence — no GPU — so idle cost is ~nothing and a warm container answers in ms.

    modal deploy modal/modal_serve_tagger.py     # deploy the web endpoint
    modal run    modal/modal_serve_tagger.py     # smoke-test locally

The model files are pulled from HF at container start into a cached volume; the tagger
source and the Wiktionary lemma tables are baked into the image at deploy time (so deploy
from a checkout that has data/lemma_tables/ populated — see tagger/LEMMA_LOOKUP.md).
"""
import glob
import os
from pathlib import Path

import modal

APP_NAME = "lexide-parsley"
HF_REPO = "anchpop/lexide-tagger"
MODEL_CACHE = "/models"                 # volume mount, caches the HF snapshot
APP_SRC = "/root/parsley"               # baked-in tagger source
TABLES_DIR = "/root/lemma_tables"       # baked-in Wiktionary tables

_here = Path(__file__).resolve().parent
_tagger_src = _here.parent / "tagger"
_tables_src = _here.parent / "data" / "lemma_tables"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Pinned to the versions validated by the first smoke deploy (2026-07-10). torch is the
        # default (CUDA) wheel but runs on CPU fine here; a `--index-url .../whl/cpu` torch would
        # shrink the image / speed cold starts — optimize later if cold start matters.
        "torch==2.13.0",
        "transformers==5.13.0",
        "tokenizers>=0.19",
        "sentencepiece",
        "huggingface_hub",
        "numpy<2",
        "fastapi[standard]",
    )
    # tagger source (predict.py, model.py, dataset.py, data_prep.py, lemma_lookup.py, ...)
    .add_local_dir(str(_tagger_src), APP_SRC, copy=True,
                   ignore=["output/**", ".venv/**", "__pycache__/**", "wandb/**", "*.pyc"])
)
# Wiktionary lemma tables (the OOD lemma floor) — baked in if present; serving still works
# without them (model-only lemmas). Populate data/lemma_tables/ before deploy for the floor.
if _tables_src.exists():
    image = image.add_local_dir(str(_tables_src), TABLES_DIR, copy=True)

volume = modal.Volume.from_name("lexide-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.cls(
    image=image,
    cpu=2.0,
    memory=4096,
    volumes={MODEL_CACHE: volume},
    secrets=[hf_secret],
    scaledown_window=300,   # keep a warm container ~5 min after the last request
    min_containers=0,       # scale to zero when idle; set to 1 to kill cold starts
    timeout=120,
)
class Parsley:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, APP_SRC)
        from huggingface_hub import snapshot_download
        from lemma_lookup import LemmaTable
        from predict import Pipeline

        local = snapshot_download(
            HF_REPO,
            allow_patterns=["tagger/best/*", "tokenizer/*"],
            local_dir=os.path.join(MODEL_CACHE, "lexide-tagger"),
            token=os.environ.get("HF_TOKEN"),
        )
        # single model in memory; lemma tables are applied per-language in tag() below
        self.pipe = Pipeline(
            tagger_dir=os.path.join(local, "tagger", "best"),
            tokenizer_path=os.path.join(local, "tokenizer", "tokenizer.pt"),
            device="cpu",
        )
        self.tables = {}
        for p in glob.glob(os.path.join(TABLES_DIR, "wikt_*.json")):
            lang = os.path.basename(p)[len("wikt_"):-len(".json")]
            self.tables[lang] = LemmaTable.load(p)
        print(f"[parsley] loaded tagger + tokenizer; lemma tables: {sorted(self.tables)}")

    def _tag_one(self, text, lang):
        toks = self.pipe(text)                     # char-tokenize -> tag (model lemmas)
        table = self.tables.get(lang)
        if table is not None:
            for t in toks:
                t["lemma"] = table.resolve(t["text"], t["pos"], t["lemma"])
        return toks

    @modal.method()
    def run(self, text, lang=""):
        """Callable path for `modal run` smoke tests (the HTTP path is tag() below)."""
        return self._tag_one(text, lang)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def tag(self, request: dict):
        """POST {"sentences": ["...", ...], "lang": "deu"} -> [[{text,pos,lemma,head,dep,...}], ...].

        `lang` is optional and only selects the Wiktionary lemma floor (the tagger itself is
        multilingual); omit it or pass an unbuilt language to get model-only lemmas.
        """
        sentences = request.get("sentences") or ([request["sentence"]] if request.get("sentence") else [])
        lang = request.get("lang", "")
        return {"results": [self._tag_one(s, lang) for s in sentences]}


@app.local_entrypoint()
def main():
    """Smoke test: `modal run modal/modal_serve_tagger.py`."""
    samples = [("Eine Fundgrube.", "deu"), ("The cats were sleeping.", "eng")]
    parsley = Parsley()
    for text, lang in samples:
        out = parsley.run.remote(text, lang)
        print(f"\n{lang}: {text!r}")
        for t in out:
            print(f"  {t['text']!r:16} pos={t['pos']:6} lemma={t['lemma']!r:14} "
                  f"head={t['head']} dep={t['dep']}")
