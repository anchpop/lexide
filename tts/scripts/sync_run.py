"""Push a run directory to the private Hugging Face repo anchpop/lexide-tts-runs
(HF_TOKEN). Run every few minutes on a training box, so a cluster tearing
itself down never takes checkpoints, metrics or samples with it.

    python scripts/sync_run.py runs/eng-nsf
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO = "anchpop/lexide-tts-runs"
run = Path(sys.argv[1])
api = HfApi()
api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(repo_id=REPO, folder_path=run, path_in_repo=run.name, ignore_patterns=["step-*.pt"],
                  commit_message=f"sync {run.name}")
