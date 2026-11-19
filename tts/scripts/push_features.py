"""Upload a built feature set as one tar to the private HF dataset
anchpop/lexide-tts-features (HF_TOKEN), for sets too big to send with sky's
workdir. sky_train.yaml downloads it when the set is not in features/.

    python scripts/push_features.py features/eng_gemini_10h
"""
import sys, tarfile, tempfile
from pathlib import Path
from huggingface_hub import HfApi

REPO = "anchpop/lexide-tts-features"
features = Path(sys.argv[1])
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=features.parent) as scratch:
    tar = Path(scratch) / f"{features.name}.tar"
    with tarfile.open(tar, "w") as archive:
        archive.add(features, arcname=features.name)
    api.upload_file(path_or_fileobj=tar, path_in_repo=tar.name, repo_id=REPO, repo_type="dataset",
                    commit_message=f"features {features.name}")
