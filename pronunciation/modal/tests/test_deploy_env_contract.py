"""yap's `compare-audio-models deploy` picks the checkpoint purely through these
environment variables. If one is hard-coded back, an eval deploy would quietly
serve production's pin; the per-request marker check catches that, but only
after a full deploy."""
import importlib.util
from pathlib import Path

MODAL_FILE = Path(__file__).resolve().parents[1] / "wav2vec2_phoneme.py"


def test_identity_comes_from_the_env_deploy_sets(monkeypatch):
    env = {
        "WAV2VEC2_APP_NAME": "wav2vec2-phoneme-contract-test",
        "WAV2VEC2_MODEL_ID": "example/contract-test",
        "WAV2VEC2_MODEL_REVISION": "0123456789abcdef0123456789abcdef01234567",
        "WAV2VEC2_DEPLOY_MARKER": "contract-marker",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # A private copy, so the other tests keep the module loaded with defaults.
    spec = importlib.util.spec_from_file_location("wav2vec2_phoneme_env_contract", MODAL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.APP_NAME == env["WAV2VEC2_APP_NAME"]
    assert module.MODEL_ID == env["WAV2VEC2_MODEL_ID"]
    assert module.MODEL_REVISION == env["WAV2VEC2_MODEL_REVISION"]
    assert module.DEPLOY_MARKER == env["WAV2VEC2_DEPLOY_MARKER"]
