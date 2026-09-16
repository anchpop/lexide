"""Exercise a deployed EVAL endpoint; save small reproducible Rust input artifacts.

Does not deploy or stop apps. The operator must authorize/manage the eval app.
Run verify_frame_matrix.rs on every *-score.json artifact after this script.
"""
import argparse
import base64
import collections
import json
from pathlib import Path
import zlib

import numpy as np
import requests
import soundfile as sf


G2P_IDENTITY = "g2p/0.4.0 espeak-ng/aa907af78d5665d8 thai/ad66331eca29d4ea korean/9e4bc6b854f6a903"
MODEL = "anchpop/lexide-pronunciation-merged"
REVISION = "95f4b185676627ffe566e8760349ebb42cc55dde"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://anchpop--wav2vec2-phoneme-eval-wav2vec2phoneme-predict.modal.run")
    parser.add_argument("--marker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", action="append", required=True, help="language=/absolute/path.wav")
    args = parser.parse_args()
    if "--wav2vec2-phoneme-eval-" not in args.url:
        parser.error("this verification tool only calls the eval app")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(model_id=MODEL, model_revision=REVISION, deploy_marker=args.marker, decoder_version="nonblank_v1")

    def post(request, batch=False):
        response = requests.post(args.url.replace("-predict.modal.run", "-predict-batch.modal.run") if batch else args.url,
                                 json={"requests": [request]} if batch else request, timeout=300)
        response.raise_for_status()
        value = response.json()
        assert all(value.get(key) == expected for key, expected in identity.items()), value.keys()
        assert "load_error" not in value, value
        return (value["results"][0] if batch else value), len(response.content)

    probe, _ = post({"marker_only": True})
    print(json.dumps({"probe": probe}, ensure_ascii=False), flush=True)
    for spec in args.clip:
        language, path = spec.split("=", 1)
        samples, sr = sf.read(path, dtype="float32")
        assert samples.ndim == 1
        request = dict(audio_f32_b64=base64.b64encode(samples.astype("<f4").tobytes()).decode(),
                       sample_rate=sr, language=language, return_frame_matrix=True)
        response, response_bytes = post(request)
        singleton, batch_bytes = post(request, batch=True)
        matrix = response["frame_matrix"]
        assert matrix == singleton["frame_matrix"]  # Including every compressed byte.
        assert response["phonemes"] == singleton["phonemes"]
        assert matrix["producer"] == identity
        assert matrix["trained_against_g2p"] == G2P_IDENTITY
        assert matrix["schema_version"] == 1
        heads = matrix["heads"]
        expected_aux = {"jpn": "jpn_pitch_accent", "tha": "tha_tone", "zho-hans": "zho_hans_tone"}[language]
        assert set(heads) == {"phone", "nonblank", "stress", expected_aux}
        stats = {}
        for name, head in heads.items():
            values = np.frombuffer(zlib.decompress(base64.b64decode(head["data"])), dtype="<f2").reshape(head["shape"]).astype("float32")
            assert values.shape[0] == heads["phone"]["shape"][0]
            if name not in ("phone", "nonblank"):
                assert np.allclose(values.sum(axis=-1), 1, atol=.001)
                stats[name] = dict(labels=head["labels"], argmax_counts=dict(collections.Counter(map(int, values.argmax(-1)))),
                                   mean_probability=values.mean(0).tolist(), min_probability=values.min(0).tolist(),
                                   max_probability=values.max(0).tolist())
        full, full_bytes = post({**request, "return_all_heads": True})
        assert set(full["frame_matrix"]["heads"]) == {"phone", "nonblank", "stress", "jpn_pitch_accent", "tha_tone", "zho_hans_tone"}
        assert full["phonemes"] == response["phonemes"]
        no_language = dict(request)
        no_language.pop("language")
        neutral, neutral_bytes = post(no_language)
        assert set(neutral["frame_matrix"]["heads"]) == {"phone", "stress", "nonblank"}
        # Explicitly a projection of actual bytes, not a server request option.
        phone_only = {**matrix, "heads": {name: head for name, head in heads.items() if name in ("phone", "nonblank")}}
        def sizes(value):
            return dict(compressed_bytes=sum(len(base64.b64decode(head["data"])) for head in value["heads"].values()),
                        matrix_json_bytes=len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()))
        print(json.dumps(dict(language=language, clip=path, duration_ms=1000 * len(samples) / sr,
                              frames=heads["phone"]["shape"][0], frame_rate_ms=matrix["frame_rate_ms"],
                              singleton_byte_identical=True, language_matrix=sizes(matrix),
                              without_suprasegmentals_projection=sizes(phone_only),
                              all_heads=sizes(full["frame_matrix"]), no_language=sizes(neutral["frame_matrix"]),
                              http_response_bytes=response_bytes, singleton_batch_response_bytes=batch_bytes,
                              all_heads_response_bytes=full_bytes, no_language_response_bytes=neutral_bytes,
                              distributions=stats), ensure_ascii=False), flush=True)
        # Same target is saved alongside endpoint score for offline Rust scoring.
        target = [phone["phoneme"].lstrip("ˈˌ") for phone in response["phonemes"]]
        assert target and all(token in heads["phone"]["labels"] for token in target)
        for impossible in [False, True]:
            tokens = [target[0]] * heads["phone"]["shape"][0] if impossible else target
            scored, _ = post({**request, "target_phonemes": tokens})
            assert scored["frame_matrix"] == matrix
            artifact = dict(response=scored, target=tokens, impossible=impossible,
                            duration_ms=1000 * len(samples) / sr, clip=path)
            (args.output / f"{language}-{'impossible' if impossible else 'normal'}-score.json").write_text(
                json.dumps(artifact, ensure_ascii=False))
        alternative = next(phone["phoneme"] for phone in response["phonemes"][0]["top_k"]
                           if phone["phoneme"] != target[0] and phone["phoneme"] in heads["phone"]["labels"])
        changed_target = [alternative, *target[1:]]
        changed, _ = post({**request, "target_phonemes": changed_target})
        assert changed["frame_matrix"] == matrix
        (args.output / f"{language}-substitution-score.json").write_text(json.dumps(dict(
            response=changed, target=changed_target, impossible=False,
            duration_ms=1000 * len(samples) / sr, clip=path,
            target_source="deterministic first-phone alternative substitution"), ensure_ascii=False))
        # Keep every full-inventory field available for independent Rust decoding too.
        full_scored, _ = post({**request, "return_all_heads": True, "target_phonemes": target})
        assert full_scored["frame_matrix"] == full["frame_matrix"]
        (args.output / f"{language}-all-heads-score.json").write_text(json.dumps(dict(
            response=full_scored, target=target, impossible=False, duration_ms=1000 * len(samples) / sr, clip=path), ensure_ascii=False))


if __name__ == "__main__":
    main()
