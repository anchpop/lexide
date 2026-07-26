"""Upload a training-output directory to the HF model repo (used by sky_byte_models.yaml).

    python3 tagger/hf_upload_run.py <local_dir> <subdir> [run_name]

Puts <local_dir> at anchpop/lexide-parsley/training-runs/<run_name>/<subdir>/
(run_name defaults to byte-v2). Reads the write token from HF_TOKEN (set via
`sky launch --secret HF_TOKEN`).
"""
import sys

from huggingface_hub import HfApi

local, sub = sys.argv[1], sys.argv[2]
run = sys.argv[3] if len(sys.argv) > 3 else "byte-v2"
HfApi().upload_folder(folder_path=local, repo_id="anchpop/lexide-parsley",
                      path_in_repo=f"training-runs/{run}/{sub}")
print(f"uploaded {local} -> training-runs/{run}/{sub}")
