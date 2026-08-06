import json
import jsonlines
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from inference import MultilingualNLPInference
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


LANGUAGE_NAMES = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "spa": "Spanish",
    "kor": "Korean",
    "por": "Portuguese",
    "ita": "Italian",
    "rus": "Russian",
    "jpn": "Japanese",
    "hin": "Hindi",
    "tha": "Thai",
    "zho-hans": "Chinese",
}


def reconstruct_sentence(tokens: List[str], whitespaces: List[str]) -> str:
    """Reconstruct original sentence from tokens and whitespace info."""
    parts = []
    ws_map = {
        "_": " ",
        "none": "",
        "nbsp": "\u00a0",
        "thinsp": "\u2009",
        "hairsp": "\u200a",
        "zwsp": "\u200b",
        "narnbsp": "\u202f",
        "ideogrp": "\u3000",
    }
    for i, token in enumerate(tokens):
        parts.append(token)
        if i < len(whitespaces):
            ws = ws_map.get(whitespaces[i], whitespaces[i])
            parts.append(ws)
    return "".join(parts)


def load_test_data(data_dir: str = "data", languages: Optional[List[str]] = None, max_samples_per_lang: int = 200):
    """Load test data from cleaned JSONL files."""
    if languages is None:
        languages = list(LANGUAGE_NAMES.keys())

    test_data = []
    for lang in languages:
        file_path = Path(data_dir) / f"cleaned_{lang}.jsonl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {lang}")
            continue

        with jsonlines.open(file_path) as reader:
            samples = list(reader)

        # Take last N samples as test set (to avoid overlap with training)
        test_samples = samples[-max_samples_per_lang:]
        for sample in test_samples:
            test_data.append((sample, lang))

    return test_data


def evaluate_model(
    inferencer: MultilingualNLPInference,
    test_data: List[Tuple[Dict, str]],
    verbose: bool = False,
) -> Dict:
    """Evaluate model on test data, computing Recon, POS, UAS, LAS, and Lemma accuracy."""
    lang_metrics = defaultdict(lambda: {
        "recon_correct": 0,
        "recon_total": 0,
        "pos_correct": 0,
        "pos_total": 0,
        "uas_correct": 0,
        "las_correct": 0,
        "dep_total": 0,
        "lemma_correct": 0,
        "lemma_total": 0,
    })

    for sample, lang in tqdm(test_data, desc="Evaluating"):
        sentence = sample["sentence"]
        true_tokens = sample.get("tokens", sample.get("doc", []))

        try:
            analysis = inferencer.analyze_sentence(sentence, lang)
            parsed = inferencer.parse_analysis(analysis)
        except Exception as e:
            if verbose:
                print(f"Error processing '{sentence}': {e}")
            continue

        metrics = lang_metrics[lang]

        # Reconstruction: can we rebuild the sentence from predicted tokens + whitespace?
        metrics["recon_total"] += 1
        if parsed:
            pred_tokens_text = [p["token"] for p in parsed]
            pred_ws = [p.get("whitespace", " ") for p in parsed]
            # Map whitespace representations
            ws_map = {"_": " ", "none": "", " ": " "}
            pred_ws_mapped = [ws_map.get(w, w) for w in pred_ws]
            reconstructed = reconstruct_sentence(pred_tokens_text, pred_ws_mapped)
            if reconstructed.strip() == sentence.strip():
                metrics["recon_correct"] += 1
            elif verbose:
                print(f"Recon mismatch: '{sentence}' vs '{reconstructed}'")

        # Token-level metrics (align by position)
        n = min(len(true_tokens), len(parsed))
        for i in range(n):
            true_tok = true_tokens[i]
            pred_tok = parsed[i]

            # POS
            metrics["pos_total"] += 1
            if true_tok["pos"] == pred_tok.get("pos"):
                metrics["pos_correct"] += 1

            # UAS (head correct) and LAS (head + dep label correct)
            metrics["dep_total"] += 1
            true_head = str(true_tok["head"])
            pred_head = pred_tok.get("head", "")
            if true_head == pred_head:
                metrics["uas_correct"] += 1
                if true_tok["dep"] == pred_tok.get("dep"):
                    metrics["las_correct"] += 1

            # Lemma
            metrics["lemma_total"] += 1
            if true_tok["lemma"] == pred_tok.get("lemma"):
                metrics["lemma_correct"] += 1

    return dict(lang_metrics)


def compute_percentages(lang_metrics: Dict) -> Dict[str, Dict[str, float]]:
    """Convert raw counts to percentages."""
    results = {}
    for lang, m in lang_metrics.items():
        results[lang] = {
            "Recon": m["recon_correct"] / max(m["recon_total"], 1) * 100,
            "POS": m["pos_correct"] / max(m["pos_total"], 1) * 100,
            "UAS": m["uas_correct"] / max(m["dep_total"], 1) * 100,
            "LAS": m["las_correct"] / max(m["dep_total"], 1) * 100,
            "Lemma": m["lemma_correct"] / max(m["lemma_total"], 1) * 100,
        }
    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print results in a nice table format."""
    metrics = ["Recon", "POS", "UAS", "LAS", "Lemma"]
    col_width = 7

    # Header
    header = f"{'Language':<12}" + "".join(f"{m:>{col_width}}" for m in metrics)
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Per-language rows
    all_values = {m: [] for m in metrics}
    lang_order = list(LANGUAGE_NAMES)
    for lang in lang_order:
        if lang not in results:
            continue
        scores = results[lang]
        name = LANGUAGE_NAMES.get(lang, lang)
        row = f"{name:<12}"
        for m in metrics:
            val = scores[m]
            all_values[m].append(val)
            row += f"{val:>{col_width}.1f}%"
        print(row)

    # Average row
    print(separator)
    avg_row = f"{'Average':<12}"
    for m in metrics:
        vals = all_values[m]
        avg = sum(vals) / len(vals) if vals else 0
        avg_row += f"{avg:>{col_width}.1f}%"
    print(avg_row)
    print(separator)


def log_to_wandb(results: Dict[str, Dict[str, float]]):
    """Log results to W&B as a table and summary metrics."""
    if wandb is None or wandb.run is None:
        print("W&B not available, skipping logging")
        return

    metrics = ["Recon", "POS", "UAS", "LAS", "Lemma"]

    # Create W&B table
    table = wandb.Table(columns=["Language"] + metrics)
    lang_order = list(LANGUAGE_NAMES)

    all_values = {m: [] for m in metrics}
    for lang in lang_order:
        if lang not in results:
            continue
        scores = results[lang]
        name = LANGUAGE_NAMES.get(lang, lang)
        row = [name] + [round(scores[m], 1) for m in metrics]
        table.add_data(*row)
        for m in metrics:
            all_values[m].append(scores[m])

    # Add average row
    avg_row = ["Average"] + [round(sum(all_values[m]) / len(all_values[m]), 1) if all_values[m] else 0 for m in metrics]
    table.add_data(*avg_row)

    wandb.log({"eval/results_table": table})

    # Also log scalar summaries
    for m in metrics:
        vals = all_values[m]
        if vals:
            wandb.log({f"eval/avg_{m.lower()}": sum(vals) / len(vals)})

    # Per-language scalars
    for lang, scores in results.items():
        name = LANGUAGE_NAMES.get(lang, lang).lower()
        for m in metrics:
            wandb.log({f"eval/{name}_{m.lower()}": scores[m]})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./final_model")
    parser.add_argument("--base-model", default=None, help="Base model name (auto-detected from adapter if not set)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Auto-detect base model from environment or default
    base_model = args.base_model
    if base_model is None:
        model_name = os.getenv("MODEL_NAME", "gemma-4-31B-it")
        base_model = f"google/{model_name}"

    # Optionally init wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project, job_type="eval")

    print(f"Loading model from {args.model_path} (base: {base_model})...")
    inferencer = MultilingualNLPInference(
        base_model_name=base_model,
        adapter_path=args.model_path
    )

    print("Loading test data...")
    test_data = load_test_data(
        data_dir=args.data_dir,
        max_samples_per_lang=args.max_samples
    )
    print(f"Loaded {len(test_data)} test samples")

    print("\nEvaluating model...")
    raw_metrics = evaluate_model(inferencer, test_data, verbose=args.verbose)
    results = compute_percentages(raw_metrics)

    print_results_table(results)
    log_to_wandb(results)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")

    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
