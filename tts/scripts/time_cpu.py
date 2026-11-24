"""Time CPU synthesis: the current TinyVoca checkpoint, StyleVoca (random
weights) at several generator sizes, and Kokoro-82M with its public weights.

Compute does not depend on weight values, but an untrained duration head would
emit garbage lengths, so every model is forced to the validation sentences'
aligned durations: a forward hook makes the duration head output logits that
decode to them, and the models' own synthesis paths run unchanged. Kokoro gets
our token count with random ids from its vocabulary and our durations converted
to its 600-sample frames. Stages are split with forward pre-hooks:
encoder + prosody | decoder | waveform generator.

Needs torch, and for Kokoro: transformers, loguru and `pip install --no-deps kokoro`
(its G2P dependency misaki is skipped; tokens are given directly).

    PYTHONPATH=. python scripts/time_cpu.py --output runs/cpu-timing.json
"""
import argparse, dataclasses, importlib.util, json, statistics, sys, time, types
from pathlib import Path
import torch

from voca.config import Config
from voca.model import build_model
from voca.speak import load_model


def sentences(manifest, count):
    rows = sorted((json.loads(l) for l in open(manifest, encoding="utf-8")), key=lambda r: r["frames"])
    rows = [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]
    return [{key: torch.tensor([r[key]]) for key in ("tokens", "lexical", "durations")} | {"seconds": r["frames"] * 256 / 24000} for r in rows]


class Stages:
    """Wall-clock stage boundaries from forward pre-hooks on the decoder and generator modules."""

    def __init__(self, decoder, generator):
        self.marks = {}
        for name, module in (("decoder", decoder), ("generator", generator)):
            module.register_forward_pre_hook(lambda *_, name=name: self.marks.setdefault(name, time.perf_counter()) and None)

    def run(self, synthesize):
        self.marks.clear()
        start = time.perf_counter()
        output = synthesize()
        end = time.perf_counter()
        return output, {"encoder_prosody": self.marks["decoder"] - start, "decoder": self.marks["generator"] - self.marks["decoder"],
                "generator": end - self.marks["generator"], "total": end - start}


def force(module, logits):
    """Replace a duration head's output with ``logits()`` (computed from the current sentence)."""
    module.register_forward_hook(lambda _module, _inputs, output: logits().to(output.dtype))


def voca_runner(model):
    current = {}
    if model.config.architecture == "styletts":
        bins = model.config.style_max_duration

        def logits():
            d = current["durations"][..., None]
            return torch.cat((30.0 * (2 * (torch.arange(bins) < d).float() - 1), 10.0 * (2 * (d > 0).float() - 1)), dim=-1)
        force(model.predictor.duration_proj, logits)
        stages = Stages(model.decoder.encode, model.decoder.generator.m_source)   # the first modules each stage calls
    else:
        force(model.duration, lambda: torch.stack((torch.log1p(current["durations"].double()).float(),
                                                   10.0 * (2 * (current["durations"] > 0).float() - 1)), dim=-1))
        stages = Stages(model.acoustic_stack, model.vocoder)

    def run(sentence):
        current.update(sentence)
        output, times = stages.run(lambda: model.synthesize(sentence["tokens"], sentence["lexical"], torch.tensor([sentence["tokens"].shape[1]]),
                                                            torch.tensor([0])))
        assert torch.equal(output[2]["durations"], sentence["durations"])
        return times
    return run


def kokoro_runner():
    # Import kokoro.model without kokoro/__init__.py, which pulls in the misaki G2P.
    spec = importlib.util.find_spec("kokoro")
    package = types.ModuleType("kokoro")
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules["kokoro"] = package
    from kokoro.model import KModel
    from loguru import logger
    logger.disable("kokoro")
    from huggingface_hub import hf_hub_download
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    voice = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"), weights_only=True)
    ids = sorted(set(model.vocab.values()))
    current = {}

    def frames(durations):
        # Our frames (256 samples) -> Kokoro's (600), rounding the running total so the length is kept.
        ends = torch.round(durations.cumsum(1) * 256 / 600)
        return torch.diff(ends, prepend=torch.zeros(1, 1)).clamp_min(1).long()

    def logits():
        bins = torch.arange(model.predictor.duration_proj.linear_layer.out_features)
        return 30.0 * (2 * (bins < frames(current["durations"])[..., None]).float() - 1)
    force(model.predictor.duration_proj, logits)
    stages = Stages(model.decoder, model.decoder.generator)
    generator = torch.Generator().manual_seed(0)

    def run(sentence):
        current.update(sentence)
        n = sentence["tokens"].shape[1]
        tokens = torch.tensor([[0, *[ids[i] for i in torch.randint(1, len(ids), (n - 2,), generator=generator)], 0]])
        (_, durations), times = stages.run(lambda: model.forward_with_tokens(tokens, voice[n - 3], 1.0))
        assert torch.equal(durations, frames(sentence["durations"])[0])
        return times
    return model, run


def measure(run, data, repeats):
    run(data[len(data) // 2])   # warm-up
    return [{key: statistics.median(r[key] for r in results) for key in results[0]}
            for results in ([run(sentence) for _ in range(repeats)] for sentence in data)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("features/eng_gemini_10h/validation.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/eng-nsf-10h/inference.pt"))
    parser.add_argument("--sentences", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--generators", type=int, nargs="+", default=[144, 224, 320], help="StyleVoca generator_channels to time")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_flush_denormal(True)
    data = sentences(args.validation, args.sentences)
    audio = sum(s["seconds"] for s in data)
    short = min(range(len(data)), key=lambda i: abs(data[i]["seconds"] - 2.0))
    models = {"tinyvoca nsf (" + args.checkpoint.parent.name + ")": load_model(args.checkpoint)}
    for channels in args.generators:
        models[f"styletts gen{channels}"] = build_model(dataclasses.replace(Config(), architecture="styletts", generator_channels=channels)).eval()
    runners = {name: (model, voca_runner(model)) for name, model in models.items()}
    runners["kokoro-82m"] = kokoro_runner()
    report = {"sentences": len(data), "audio_seconds": audio, "short_sentence_seconds": data[short]["seconds"], "models": {}}
    for name, (model, run) in runners.items():
        parameters = sum(p.numel() for p in model.parameters())
        if hasattr(model, "decoder"):
            split = {"generator": model.decoder.generator}
        else:
            split = {"generator": model.vocoder}
        entry = {"parameters": parameters, "generator_parameters": sum(p.numel() for p in split["generator"].parameters())}
        for threads in args.threads:
            torch.set_num_threads(threads)
            with torch.inference_mode():
                times = measure(run, data, args.repeats)
            totals = {key: sum(t[key] for t in times) for key in times[0]}
            entry[f"threads_{threads}"] = {"rtf": totals["total"] / audio, "stage_share": {k: v / totals["total"] for k, v in totals.items() if k != "total"},
                                           "stage_rtf": {k: v / audio for k, v in totals.items() if k != "total"},
                                           "short_sentence_seconds": times[short]["total"]}
            print(f"{name:32s} {threads}t  RTF {entry[f'threads_{threads}']['rtf']:.3f}  short {times[short]['total'] * 1000:.0f} ms  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in entry[f"threads_{threads}"]["stage_rtf"].items()), flush=True)
        report["models"][name] = entry
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
