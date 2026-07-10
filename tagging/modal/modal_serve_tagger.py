#!/usr/bin/env python3
"""Modal deployment for `parsley` — the small CPU tagger that replaces the Gemma pipeline.

Serves the encoder multi-task tagger (POS + lemma + dependency) plus the byte-minGRU
tokenizer from anchpop/lexide-parsley, on CPU, with scale-to-zero. One forward pass per
sentence — no GPU — so idle cost is ~nothing and a warm container answers in ms.

    modal deploy modal/modal_serve_tagger.py     # deploy the web endpoint
    modal run    modal/modal_serve_tagger.py     # smoke-test locally

The tagger model, its source, and the Wiktionary lemma tables are all baked into the image
at build/deploy time (the model via a run_function download). This matters for memory
snapshots: the snapshotted load must read only from local disk — Modal will not reuse a
snapshot whose snap=True method did network I/O. Deploy from a checkout with
data/lemma_tables/ populated (see tagger/LEMMA_LOOKUP.md).
"""
import os
from pathlib import Path

import modal

APP_NAME = "lexide-parsley"
HF_REPO = "anchpop/lexide-parsley"
MODEL_DIR = "/model"                    # tagger weights baked into the image (see _download_model)
APP_SRC = "/root/parsley"               # baked-in tagger source
TABLES_DIR = "/root/lemma_tables"       # baked-in Wiktionary tables

_here = Path(__file__).resolve().parent
_tagger_src = _here.parent / "tagger"
_tables_src = _here.parent / "data" / "lemma_tables"

app = modal.App(APP_NAME)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _download_model():
    """Bake the tagger weights into the image at build time (needs the HF token; runs once)."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        HF_REPO,
        allow_patterns=["tagger/best/*", "tokenizer/*"],
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU-only torch (the default wheel bundles ~2.5GB of unused CUDA libs; this is a CPU serve,
    # so the smaller image means faster cold-start container provisioning). Installed first so
    # transformers sees torch already satisfied and doesn't pull the CUDA build.
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
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
# Bake the model weights into the image so the snapshotted load is local-only.
image = image.run_function(_download_model, secrets=[hf_secret])


@app.cls(
    image=image,
    cpu=2.0,
    memory=4096,
    scaledown_window=300,   # keep a warm container ~5 min after the last request
    # Scale to zero (cheapest) — cold starts pay a model load (~15-20s). For user-facing
    # latency with no cold starts, set min_containers=1 (one always-warm CPU container).
    min_containers=0,
    timeout=180,
)
class Parsley:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, APP_SRC)
        from predict import Pipeline

        # Only the model loads at startup; lemma tables load lazily per language (see _table)
        # so a cold start doesn't parse all ~185MB of table JSON up front.
        self.pipe = Pipeline(
            tagger_dir=os.path.join(MODEL_DIR, "tagger", "best"),
            tokenizer_path=os.path.join(MODEL_DIR, "tokenizer", "tokenizer.pt"),
            device="cpu",
        )
        self._tables = {}
        print("[parsley] loaded tagger + tokenizer (lemma tables load lazily per language)")

    def _table(self, lang):
        """Load (and cache) the lemma table for a language on first use; None if not built."""
        if lang not in self._tables:
            from lemma_lookup import LemmaTable
            p = os.path.join(TABLES_DIR, f"wikt_{lang}.json")
            priors = os.path.join(TABLES_DIR, f"wikt_priors_{lang}.json")
            self._tables[lang] = (
                LemmaTable.load(p, priors_path=priors) if lang and os.path.exists(p) else None
            )
        return self._tables[lang]

    def _tag_one(self, text, lang):
        toks = self.pipe(text)                     # char-tokenize -> tag (model lemmas)
        table = self._table(lang)
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
