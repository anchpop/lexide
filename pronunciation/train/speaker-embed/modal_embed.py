"""Speaker embeddings (ECAPA-TDNN) on Modal, for deriving pseudo-speaker labels
on the sources that lack them (FLEURS + Pimsleur, voice=null).

Returns a 192-dim speaker-verification embedding per clip. The local orchestrator
(embed.py) caches these per-clip (keyed by filename), so re-runs only embed NEW
clips — making this safe to drop into the preprocess phase.
"""
import modal

app = modal.App("speaker-embed")

MODEL = "speechbrain/spkrec-ecapa-voxceleb"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "speechbrain", "soundfile", "numpy")
    .run_commands(
        # Pre-fetch the ECAPA checkpoint into the image so cold start is fast.
        "echo 'ECAPA=v1' && python -c \""
        "from speechbrain.inference.speaker import EncoderClassifier; "
        f"EncoderClassifier.from_hparams(source='{MODEL}', savedir='/model')\""
    )
)


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=300,
    timeout=900,
    enable_memory_snapshot=True,
)
class SpeakerEmbedder:
    @modal.enter()
    def load(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.model = EncoderClassifier.from_hparams(
            source=MODEL, savedir="/model", run_opts={"device": "cuda"}
        )

    @modal.method()
    def embed_batch(self, items: list) -> list:
        """items: [{key, audio_b64}] where audio_b64 is base64 of the WAV file
        bytes (compact — ~10x smaller over the wire than a float list). Returns
        [{key, embedding: [192 floats]}]. ECAPA wants 16 kHz mono."""
        import base64
        import io
        import soundfile as sf
        import torch
        import torchaudio.functional as AF

        out = []
        for it in items:
            data, sr = sf.read(io.BytesIO(base64.b64decode(it["audio_b64"])), dtype="float32")
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data)
            if sr != 16000:
                wav = AF.resample(wav, sr, 16000)
            with torch.no_grad():
                emb = self.model.encode_batch(wav.unsqueeze(0).to("cuda"))  # (1,1,192)
            out.append({"key": it["key"], "embedding": emb.squeeze().detach().cpu().tolist()})
        return out
