"""Demo script for pronunciation grading.

Usage:
    # Grade a file:
    python -m pronunciation.inference.demo --audio test.wav --text "bonjour le monde" --language fr

    # Record from microphone (requires sounddevice):
    python -m pronunciation.inference.demo --record --text "hello world" --language en
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Pronunciation grading demo")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--record", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--text", type=str, required=True, help="Expected text")
    parser.add_argument("--language", type=str, required=True, help="Language code (en, fr, de, es, ...)")
    parser.add_argument("--threshold", type=float, default=-0.5, help="Confidence threshold for flagging")
    parser.add_argument("--device", type=str, default=None, help="torch device (cpu/cuda)")
    args = parser.parse_args()

    from .grader import PronunciationGrader

    print("Loading model...")
    grader = PronunciationGrader(device=args.device)
    print(f"Model loaded. Vocab size: {len(grader.vocab)}")

    if args.record:
        import numpy as np
        try:
            import sounddevice as sd
        except ImportError:
            print("Install sounddevice for recording: pip install sounddevice")
            sys.exit(1)

        print(f"\nRecording {args.duration}s... Say: \"{args.text}\"")
        audio = sd.rec(int(args.duration * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        audio = audio.squeeze()
        print("Recording done.\n")

        result = grader.grade_from_array(audio, args.text, args.language, confidence_threshold=args.threshold)
    elif args.audio:
        result = grader.grade(args.audio, args.text, args.language, confidence_threshold=args.threshold)
    else:
        print("Provide --audio or --record")
        sys.exit(1)

    # Display results
    print(f"Text:    {result['text']}")
    print(f"IPA:     {result['ipa']}")
    print(f"Overall: {result['overall_score']:.1%}")
    print()

    print("Per-phoneme breakdown:")
    print(f"  {'IPA':>4}  {'Score':>6}  {'Conf':>7}  {'Status'}")
    print(f"  {'---':>4}  {'-----':>6}  {'------':>7}  {'------'}")
    for p in result["phonemes"]:
        status = "WEAK" if p["flagged"] else "ok"
        print(f"  {p['ipa']:>4}  {p['score']:>6.1%}  {p['confidence']:>7.2f}  {status}")

    if result["flagged_phonemes"]:
        flagged_ipa = " ".join(p["ipa"] for p in result["flagged_phonemes"])
        print(f"\nFlagged phonemes: {flagged_ipa}")
    else:
        print("\nAll phonemes pronounced well!")


if __name__ == "__main__":
    main()
