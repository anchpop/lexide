"""Each language's multi-voice TTS data, end to end. Resumable: run it again
after any failure and finished work is skipped.

    set -a; . /data/coding/yap/.env; . /data/coding/lexide/.env; set +a    # GEMINI_API_KEY, HF_TOKEN
    python scripts/language_data.py jpn                            # every stage
    python scripts/language_data.py deu fra --stages select generate frames

Per language (its pack dump at data/<language>_sentences.jsonl, from yap's
generate-data output tts_sentences.jsonl; its inventory voca/inventories/<language>.json, from
scripts/build_inventory.py), every stage run with VOCA_LANGUAGE=<language>:

    select    data/scripts/<language>_<voice>.jsonl (scripts/select_sentences.py): Achernar 10 h,
              each other voice 30 min of its own sentences, plus quoted, shared and validation sentences;
              an existing script is extended with the roles it lacks, never rewritten
    generate  data/<language>_gemini_<voice>/ (scripts/generate.py, Gemini, about $0.80 per hour of audio)
    frames    the phonetizer's frame matrices (scripts/fetch_frames.py)
    build     features/<language>_gemini_multi_v1/ (voca.build), all voices in one set
    push      that set to HF (scripts/push_features.py), which refuses to overwrite one
"""
import argparse, os, pathlib, subprocess, sys

from generate import LANGUAGES

VOICES = {"Achernar": 10.0, "Kore": 0.5, "Leda": 0.5, "Puck": 0.5, "Iapetus": 0.5, "Orus": 0.5}
STAGES = ("select", "generate", "frames", "build", "push")
# Any frame matrix the phonetizer wrote: its labels tell the selection which phones can be aligned.
ALIGNER_LABELS = "data/eng_gemini_achernar/frames/0000eef7710dae6d.json"
FEATURES = "{}_gemini_multi_v1"


def script(language, voice):
    return pathlib.Path(f"data/scripts/{language}_{voice.lower()}.jsonl")


def clips(language, voice):
    return pathlib.Path(f"data/{language}_gemini_{voice.lower()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("languages", nargs="+", choices=sorted(LANGUAGES))
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--workers", type=int, default=12, help="concurrent Gemini requests")
    parser.add_argument("--frames-parallel", type=int, default=4, help="concurrent phonetizer batches (the endpoint is production's)")
    args = parser.parse_args()
    for language in args.languages:
        features = pathlib.Path("features") / FEATURES.format(language)

        def run(*command):
            print("+", " ".join(command), flush=True)
            subprocess.run([sys.executable, *command], check=True, env={**os.environ, "VOCA_LANGUAGE": language})
        for stage in (s for s in STAGES if s in args.stages):
            print(f"== {language} {stage}", flush=True)
            if stage == "select":
                run("scripts/select_sentences.py", f"data/{language}_sentences.jsonl", "data/scripts", "--aligner-labels", ALIGNER_LABELS,
                    "--voices", *(f"{v.lower()}={h}" for v, h in VOICES.items()))
            elif stage == "generate":
                for voice in VOICES:
                    run("scripts/generate.py", str(script(language, voice)), str(clips(language, voice)), "--provider", "gemini",
                        "--language", language, "--voice", voice, "--workers", str(args.workers))
            elif stage == "frames":
                for voice in VOICES:
                    run("scripts/fetch_frames.py", str(clips(language, voice)), "--language", language, "--parallel", str(args.frames_parallel))
            elif stage == "build" and not (features / "annotations.json").exists():
                run("-m", "voca.build", "--set", *(str(clips(language, v)) for v in VOICES), "--output", str(features), "--workers", "22")
            elif stage == "push" and not (features / "pushed").exists():
                run("scripts/push_features.py", str(features))
                (features / "pushed").touch()


if __name__ == "__main__":
    main()
