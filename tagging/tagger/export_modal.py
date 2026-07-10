"""Run the ONNX export on Modal (has torch) and push the graph to HF.

    modal run tagger/export_modal.py

Downloads anchpop/lexide-tagger/tagger/best, runs export_onnx.py (exports + numerically
verifies against PyTorch), and uploads the graph to anchpop/lexide-tagger/onnx/tagger.onnx.
"""
import os
from pathlib import Path

import modal

_tagger_src = Path(__file__).resolve().parent
app = modal.App("lexide-tagger-export")
hf_secret = modal.Secret.from_name("huggingface-secret")
onnx_vol = modal.Volume.from_name("lexide-onnx", create_if_missing=True)  # export lands here

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==5.13.0", "tokenizers>=0.19", "sentencepiece",
        "onnx", "onnxruntime", "onnxscript", "huggingface_hub", "numpy<2",
    )
    .add_local_dir(str(_tagger_src), "/root/tagger", copy=True,
                   ignore=["output/**", ".venv/**", "__pycache__/**", "wandb/**", "*.pyc"])
)


@app.function(image=image, cpu=4.0, memory=8192, secrets=[hf_secret],
              volumes={"/vol": onnx_vol}, timeout=1200)
def export():
    import shutil
    import subprocess
    from huggingface_hub import snapshot_download

    tok = os.environ["HF_TOKEN"]
    d = snapshot_download("anchpop/lexide-tagger", allow_patterns=["tagger/best/*"],
                          local_dir="/tmp/model", token=tok)
    out = "/tmp/tagger.onnx"
    subprocess.run(
        ["python", "export_onnx.py", "--model-dir", os.path.join(d, "tagger", "best"), "--out", out],
        cwd="/root/tagger", check=True,
    )
    # also copy the tokenizer + label vocab the Rust side needs, next to the graph
    shutil.copy(out, "/vol/tagger.onnx")
    for fn in ("tokenizer.json", "vocab.json"):
        src = os.path.join(d, "tagger", "best", fn)
        if os.path.exists(src):
            shutil.copy(src, f"/vol/{fn}")
    onnx_vol.commit()
    print(f"[export] wrote /vol/tagger.onnx ({os.path.getsize(out)/1e6:.1f} MB) + tokenizer/vocab to volume lexide-onnx")


@app.local_entrypoint()
def main():
    export.remote()
