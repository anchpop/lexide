"""Parsley: joint L4 tagger and independent CPU sentence segmenter."""
import os
from pathlib import Path
import re

import modal

app = modal.App("lexide-parsley")
SOURCE = "/root/joint"
MODEL = "/model"
secret = modal.Secret.from_name("huggingface-secret")


def download_checkpoint():
    from huggingface_hub import snapshot_download
    snapshot_download(
        os.environ["JOINT_HF_REPO"], revision=os.environ["JOINT_HF_REVISION"],
        allow_patterns=[os.environ["JOINT_HF_PATH"] + "/*"],
        local_dir=MODEL, token=os.environ["HF_TOKEN"],
    )
    # save_checkpoint bundles the exact training tokenizer, not latest upstream.
    checkpoint = Path(MODEL) / os.environ["JOINT_HF_PATH"]
    for name in ("state.pt", "config.json", "vocab.json", "tokenizer/tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)


revision = os.environ.get("JOINT_HF_REVISION", "09a1f8b32248cc303132026ec488f4ad65085ecb")
if modal.is_local() and not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise ValueError("Set JOINT_HF_REVISION to an immutable 40-character HF commit SHA")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.14.1", "transformers==5.18.0", "sentencepiece", "fastapi[standard]")
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "tagger"), SOURCE,
                   copy=True, ignore=["output/**", ".venv/**", "__pycache__/**", "wandb/**", "*.pyc"])
    # Env layers (NOT run_function arguments) invalidate Modal's baked-weight cache.
    .env({"JOINT_HF_REPO": os.environ.get("JOINT_HF_REPO", "anchpop/lexide-parsley"),
          "JOINT_HF_PATH": os.environ.get("JOINT_HF_PATH", "training-runs/joint-v2-24L/best"),
          "JOINT_HF_REVISION": revision, "HF_HUB_DISABLE_XET": "1"})
    .run_function(download_checkpoint, secrets=[secret])
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.cls(image=image, gpu="L4", cpu=4, memory=16384, min_containers=0,
         max_containers=4, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=64)
class Parsley:
    @modal.enter()
    def load(self):
        import sys
        import torch
        sys.path.insert(0, SOURCE)
        from predict_joint import load_checkpoint
        torch.set_num_threads(4)
        self.model, self.tokenizer, self.vocab = load_checkpoint(
            Path(MODEL) / os.environ["JOINT_HF_PATH"], "cuda")
        self.queue = None

    def predict(self, records):
        from predict_joint import predict_batch
        predictions, _, _ = predict_batch(self.model, self.tokenizer, self.vocab, records, "cuda")
        return [[{**t, "text": p["text"][t["start"]:t["end"]]} for t in p["tokens"]]
                for p in predictions]

    async def batch_loop(self):
        import asyncio
        while True:
            first = await self.queue.get()
            await asyncio.sleep(0.01)
            batch = [first]
            # Bound both sentence count and padded character length (quadratic arc heads).
            longest = len(first[0]["text"])
            while len(batch) < 256 and not self.queue.empty():
                candidate = self.queue.get_nowait()
                length = max(longest, len(candidate[0]["text"]))
                if length * (len(batch) + 1) > 32768:
                    self.queue.put_nowait(candidate)
                    break
                batch.append(candidate)
                longest = length
            try:
                results = await asyncio.to_thread(self.predict, [r for r, _ in batch])
                for (_, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    @modal.fastapi_endpoint(method="POST", docs=True, label="lexide-parsley-parsley-tag")
    async def tag(self, request: dict):
        import asyncio
        from fastapi import HTTPException
        sentences = request.get("sentences")
        if sentences is None:
            sentences = [request["sentence"]] if "sentence" in request else []
        lang = request.get("lang") or ""
        if (not isinstance(sentences, list) or len(sentences) > 1000
                or not all(isinstance(s, str) and len(s) <= 4096 for s in sentences)
                or not isinstance(lang, str)):
            raise HTTPException(422, "Expected up to 1000 strings of <=4096 characters and a string lang")
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=64000)
            self.worker = asyncio.create_task(self.batch_loop())
        futures = []
        for text in sentences:
            future = asyncio.get_running_loop().create_future()
            if not text.strip():
                future.set_result([])
            else:
                await self.queue.put(({"text": text, "lang": lang}, future))
            futures.append(future)
        return {"results": await asyncio.gather(*futures)}


def download_segmenter():
    from huggingface_hub import hf_hub_download
    hf_hub_download("anchpop/lexide-parsley", "segmenter/segmenter.pt",
                    revision=os.environ["SEGMENTER_REVISION"], local_dir=MODEL,
                    token=os.environ["HF_TOKEN"])


# Keep segment-only calls off the GPU and pin the original public URL explicitly.
segment_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("huggingface_hub", "fastapi[standard]", "numpy")
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "tagger"), SOURCE,
                   copy=True, ignore=["output/**", ".venv/**", "__pycache__/**", "wandb/**", "*.pyc"])
    .env({"SEGMENTER_REVISION": "09a1f8b32248cc303132026ec488f4ad65085ecb", "HF_HUB_DISABLE_XET": "1"})
    .run_function(download_segmenter, secrets=[secret])
)


@app.cls(image=segment_image, cpu=2, memory=4096, min_containers=0,
         scaledown_window=300, timeout=180)
class SentenceSegmenter:
    @modal.enter()
    def load(self):
        import sys
        import torch
        sys.path.insert(0, SOURCE)
        from predict import load_char_model
        torch.set_num_threads(2)
        self.model = load_char_model(Path(MODEL) / "segmenter/segmenter.pt", "cpu")

    @modal.fastapi_endpoint(method="POST", docs=True, label="lexide-parsley-parsley-segment")
    def segment(self, request: dict):
        import torch
        from predict import byte_encode, spans_from_byte_labels
        texts = request.get("texts") or ([request["text"]] if request.get("text") else [])
        lang = request.get("lang") or None
        results = []
        with torch.inference_mode():
            for text in texts:
                x = torch.tensor([byte_encode(text, lang, self.model)])
                labels = self.model(x)[0].argmax(-1).tolist()
                results.append([text[s:e] for s, e in spans_from_byte_labels(text, labels)])
        return {"results": results}
