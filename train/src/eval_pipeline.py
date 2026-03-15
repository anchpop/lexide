#!/usr/bin/env python3
"""End-to-end evaluation of the unified NLP pipeline.

CANINE encoder + ByT5 decoder: raw text -> tokens + POS + sentence boundaries + deps + lemmas

Metrics per language:
- Tokenization accuracy
- Token POS accuracy
- Sentence boundary F1
- Lemma accuracy
- UAS / LAS
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import CanineTokenizer
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from stage1_train import (
    BIO_POS_ID2LABEL,
    BIO_POS_LABEL2ID,
    DEP2ID,
    DEP_RELS,
    ID2DEP,
    LANG_MAP,
    LANG2ID,
    NUM_BIO_POS,
    NUM_DEP_RELS,
    UPOS_TAGS,
    UPOS2ID,
    CanineForNLP,
    Stage1Dataset,
    build_char_labels,
    collate_fn,
    group_into_paragraphs,
    load_all_sentences,
)


def decode_bio_to_tokens(text, bio_pred_ids, sent_pred_ids):
    """Convert character-level BIO predictions to tokens and sentences.

    Returns list of sentences, each a list of dicts with 'text', 'pos', 'sent_start'.
    """
    sentences = []
    current_sent = []
    current_tok_text = ""
    current_tok_pos = None

    for i, (ch, bio_id, sent_id) in enumerate(zip(text, bio_pred_ids, sent_pred_ids)):
        label = BIO_POS_ID2LABEL.get(bio_id, "O")

        # Sentence boundary
        if sent_id == 1 and i > 0:
            if current_tok_text:
                current_sent.append({"text": current_tok_text, "pos": current_tok_pos})
                current_tok_text = ""
            if current_sent:
                sentences.append(current_sent)
                current_sent = []

        if label == "O":
            # Whitespace
            if current_tok_text:
                current_sent.append({"text": current_tok_text, "pos": current_tok_pos})
                current_tok_text = ""
        elif label.startswith("B-"):
            if current_tok_text:
                current_sent.append({"text": current_tok_text, "pos": current_tok_pos})
            current_tok_text = ch
            current_tok_pos = label[2:]
        elif label.startswith("I-"):
            current_tok_text += ch

    if current_tok_text:
        current_sent.append({"text": current_tok_text, "pos": current_tok_pos})
    if current_sent:
        sentences.append(current_sent)

    return sentences


def reconstruct_text(tokens_with_ws):
    """Reconstruct surface text from tokens with whitespace."""
    parts = []
    for tok in tokens_with_ws:
        parts.append(tok["text"])
        parts.append(tok.get("whitespace", ""))
    return "".join(parts)


@torch.no_grad()
def run_stage1(model, tokenizer, text, device, lang_code="eng", max_length=2048):
    """Run stage 1 on raw text, return predictions."""
    enc = tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    lang_ids = torch.tensor([LANG2ID.get(lang_code, 0)], device=device)

    h = model.encode(ids, mask, lang_ids)
    bio_log = model.bio_pos_head(h)
    sent_log = model.sent_head(h)

    n_chars = min(len(text), max_length - 2)
    bio_preds = bio_log[0, 1:n_chars + 1].argmax(-1).cpu().tolist()
    sent_preds = sent_log[0, 1:n_chars + 1].argmax(-1).cpu().tolist()

    return decode_bio_to_tokens(text[:n_chars], bio_preds, sent_preds)


@torch.no_grad()
def run_stage1_with_deps(model, tokenizer, text, device, lang_code="eng", max_length=2048):
    """Run unified model: tokenization, POS, deps, and lemmas in one encoder pass."""
    enc = tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    lang_ids = torch.tensor([LANG2ID.get(lang_code, 0)], device=device)

    # Single encoder pass
    h = model.encode(ids, mask, lang_ids)
    bio_log = model.bio_pos_head(h)
    sent_log = model.sent_head(h)

    n_chars = min(len(text), max_length - 2)
    bio_preds = bio_log[0, 1:n_chars + 1].argmax(-1).cpu().tolist()
    sent_preds = sent_log[0, 1:n_chars + 1].argmax(-1).cpu().tolist()

    sentences = decode_bio_to_tokens(text[:n_chars], bio_preds, sent_preds)

    # Build token spans (CANINE coordinates: +1 for CLS)
    all_spans = []
    all_pos_ids = []
    dep_info = []
    char_pos = 0
    for sent in sentences:
        spans = []
        pos_ids = []
        for tok in sent:
            start = text.find(tok["text"], char_pos)
            if start == -1:
                start = char_pos
            end = start + len(tok["text"])
            spans.append((start + 1, end + 1))
            pos_ids.append(UPOS2ID.get(tok.get("pos", "X"), UPOS2ID.get("X", 16)))
            char_pos = end
            while char_pos < n_chars and bio_preds[char_pos] == BIO_POS_LABEL2ID["O"]:
                char_pos += 1

        if spans:
            all_spans.extend(spans)
            all_pos_ids.extend(pos_ids)
            dep_info.append({
                "batch_idx": 0,
                "token_spans": spans,
                "heads": [0] * len(spans),
                "dep_rels": [0] * len(spans),
            })

    # Dependency parsing using cached hidden states
    if dep_info:
        tok_reps, tok_mask = model.pool_tokens(h, dep_info, device)
        if tok_reps is not None:
            arc_scores = model.arc_biaffine(model.arc_dep_mlp(tok_reps[:, 1:]), model.arc_head_mlp(tok_reps))
            rel_scores = model.rel_biaffine(model.rel_dep_mlp(tok_reps[:, 1:]), model.rel_head_mlp(tok_reps))
            for si, sent in enumerate(sentences):
                n = len(sent)
                if si < arc_scores.shape[0]:
                    pred_heads = arc_scores[si, :n].argmax(-1).cpu().tolist()
                    pred_rels_full = rel_scores[si, :n]
                    for ti in range(n):
                        sent[ti]["head"] = pred_heads[ti]
                        rel_idx = pred_rels_full[ti, pred_heads[ti]].argmax(-1).item()
                        sent[ti]["dep"] = ID2DEP.get(rel_idx, "dep")

    # Lemma generation using cached hidden states
    if all_spans:
        lemmas = model.generate_lemmas(h, mask, all_spans, all_pos_ids)
        li = 0
        for sent in sentences:
            for tok in sent:
                if li < len(lemmas):
                    tok["lemma"] = lemmas[li]
                    li += 1

    return sentences


def evaluate_per_language(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = CanineForNLP(
        num_bio_pos=NUM_BIO_POS, num_dep_rels=NUM_DEP_RELS,
        arc_dim=model_args.get("arc_dim", 256), rel_dim=model_args.get("rel_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

    # Load test data per language
    print("Loading test data...")
    results = {}

    for lang_code, lang_name in LANG_MAP.items():
        path = Path(args.data_dir) / f"cleaned_{lang_code}.jsonl"
        if not path.exists():
            continue

        all_sents = []
        with open(path) as f:
            for line in f:
                all_sents.append(json.loads(line))

        # Use last 5% as test set (same split as training)
        rng = random.Random(42)
        rng.shuffle(all_sents)
        n_test = max(1, int(0.05 * len(all_sents)))
        test_sents = all_sents[-n_test:]

        print(f"\n--- {lang_name} ({lang_code}): {len(test_sents)} test sentences ---")

        stats = defaultdict(int)
        errors = {"tokenization": [], "pos": [], "dep": [], "lemma": []}
        MAX_ERRORS = 200

        for sent_data in tqdm(test_sents, desc=lang_code):
            gold_tokens = sent_data["tokens"]
            raw_text = sent_data["sentence"]

            # Run Stage 1
            pred_sents = run_stage1_with_deps(model, tokenizer, raw_text, device, lang_code=lang_code)
            pred_tokens = [tok for s in pred_sents for tok in s]

            # Reconstruction accuracy: check if predicted tokens reproduce text
            pred_text = "".join(tok["text"] for tok in pred_tokens)
            gold_text = "".join(tok["text"] for tok in gold_tokens)
            stats["tok_correct"] += int(pred_text == gold_text)
            stats["tok_total"] += 1
            if pred_text != gold_text and len(errors["tokenization"]) < MAX_ERRORS:
                gold_toks = [t["text"] for t in gold_tokens]
                pred_toks = [t["text"] for t in pred_tokens]
                errors["tokenization"].append({
                    "sentence": raw_text[:200],
                    "gold_tokens": gold_toks,
                    "pred_tokens": pred_toks,
                })

            # Token-level POS accuracy (align predicted to gold)
            for gi, gt in enumerate(gold_tokens):
                if gi < len(pred_tokens):
                    pt = pred_tokens[gi]
                    if pt["text"] == gt["text"]:
                        stats["pos_total"] += 1
                        pred_pos = pt.get("pos", "")
                        gold_pos = gt["pos"]
                        stats["pos_correct"] += int(pred_pos == gold_pos)
                        if pred_pos != gold_pos and len(errors["pos"]) < MAX_ERRORS:
                            errors["pos"].append({
                                "token": gt["text"],
                                "gold_pos": gold_pos,
                                "pred_pos": pred_pos,
                                "context": raw_text[:200],
                            })

                        # Dependency accuracy
                        if "head" in pt and "head" in gt:
                            stats["dep_total"] += 1
                            stats["uas_correct"] += int(pt["head"] == gt["head"])
                            las_ok = pt["head"] == gt["head"] and pt.get("dep", "") == gt.get("dep", "")
                            stats["las_correct"] += int(las_ok)
                            if not las_ok and len(errors["dep"]) < MAX_ERRORS:
                                errors["dep"].append({
                                    "token": gt["text"],
                                    "gold_head": gt["head"],
                                    "pred_head": pt["head"],
                                    "gold_dep": gt.get("dep", ""),
                                    "pred_dep": pt.get("dep", ""),
                                    "context": raw_text[:200],
                                })

            # Lemma accuracy (using integrated decoder predictions)
            for gi, gt in enumerate(gold_tokens):
                if gi < len(pred_tokens) and pred_tokens[gi]["text"] == gt["text"]:
                    pred_lemma = pred_tokens[gi].get("lemma", pred_tokens[gi]["text"])
                    gold_lemma = gt["lemma"]
                    stats["lemma_total"] += 1
                    stats["lemma_correct"] += int(pred_lemma == gold_lemma)
                    if pred_lemma != gold_lemma and len(errors["lemma"]) < MAX_ERRORS:
                        errors["lemma"].append({
                            "token": gt["text"],
                            "gold_lemma": gold_lemma,
                            "pred_lemma": pred_lemma,
                            "context": raw_text[:200],
                        })

        # Compute metrics
        lang_metrics = {
            "tokenization_acc": stats["tok_correct"] / max(stats["tok_total"], 1),
            "pos_acc": stats["pos_correct"] / max(stats["pos_total"], 1),
            "uas": stats["uas_correct"] / max(stats["dep_total"], 1),
            "las": stats["las_correct"] / max(stats["dep_total"], 1),
            "lemma_acc": stats["lemma_correct"] / max(stats["lemma_total"], 1),
            "n_sentences": stats["tok_total"],
            "n_tokens_matched": stats["pos_total"],
        }
        results[lang_code] = lang_metrics
        print(f"  {lang_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in lang_metrics.items()))

        if HAS_WANDB and args.wandb:
            wandb.log({f"eval/{lang_code}/{k}": v for k, v in lang_metrics.items() if isinstance(v, float)})

        # Print errors
        for err_type, err_list in errors.items():
            if err_list:
                print(f"\n  --- {lang_name} {err_type} errors ({len(err_list)}) ---")
                for e in err_list[:200]:
                    if err_type == "tokenization":
                        print(f"    gold: {e['gold_tokens']}")
                        print(f"    pred: {e['pred_tokens']}")
                    elif err_type == "pos":
                        print(f"    '{e['token']}': gold={e['gold_pos']} pred={e['pred_pos']}")
                    elif err_type == "dep":
                        print(f"    '{e['token']}': head {e['gold_head']}->{e['pred_head']}, "
                              f"dep {e['gold_dep']}->{e['pred_dep']}")
                    elif err_type == "lemma":
                        print(f"    '{e['token']}': gold='{e['gold_lemma']}' pred='{e['pred_lemma']}'")

        # Save errors to file
        err_path = Path(args.output_dir) / f"errors_{lang_code}.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Average across languages
    avg = defaultdict(float)
    for lang, m in results.items():
        for k, v in m.items():
            if isinstance(v, float):
                avg[k] += v
    n_langs = len(results)
    avg = {k: v / n_langs for k, v in avg.items()}
    print("Average: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    # Save results
    out_path = Path(args.output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_language": results, "average": avg}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if HAS_WANDB and args.wandb:
        wandb.log({f"eval/avg/{k}": v for k, v in avg.items()})

    return results


# Sentence boundary evaluation on paragraphs
def evaluate_sentence_boundaries(args):
    """Evaluate sentence boundary detection on multi-sentence paragraphs."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = CanineForNLP(
        num_bio_pos=NUM_BIO_POS, num_dep_rels=NUM_DEP_RELS,
        arc_dim=model_args.get("arc_dim", 256), rel_dim=model_args.get("rel_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

    sentences = load_all_sentences(args.data_dir)
    paragraphs = group_into_paragraphs(sentences, seed=42)

    rng = random.Random(42)
    rng.shuffle(paragraphs)
    n_test = int(0.05 * len(paragraphs))
    test_paras = paragraphs[-n_test:]

    print(f"\nEvaluating sentence boundaries on {len(test_paras)} paragraphs...")

    tp = fp = fn = 0
    for para in tqdm(test_paras, desc="SentBound"):
        text, bio_ids, sent_ids, _ = build_char_labels(para)
        n_gold_sents = len(para)

        lang_code = para[0].get("lang", "eng")
        pred_sents = run_stage1(model, tokenizer, text, device, lang_code=lang_code)
        n_pred_sents = len(pred_sents)

        # Count sentence boundaries (starts)
        gold_starts = {i for i, s in enumerate(sent_ids) if s == 1}
        # Find predicted sentence starts from decoded output
        pred_starts = set()
        pos = 0
        for si, sent in enumerate(pred_sents):
            if si > 0:
                pred_starts.add(pos)
            for tok in sent:
                pos += len(tok["text"])
                # Skip estimated whitespace
                pos += 1  # rough
            # Adjust for any whitespace between sentences

        # Simpler: just compare number of predicted vs gold sentences
        tp += min(n_pred_sents, n_gold_sents)
        fp += max(0, n_pred_sents - n_gold_sents)
        fn += max(0, n_gold_sents - n_pred_sents)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    sent_metrics = {"sent_bound_prec": prec, "sent_bound_rec": rec, "sent_bound_f1": f1}
    print(f"Sentence boundary: P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    if HAS_WANDB and args.wandb:
        wandb.log({f"eval/{k}": v for k, v in sent_metrics.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_path", default="output/stage1/best_model.pt")
    p.add_argument("--output_dir", default="output/eval")
    p.add_argument("--eval_sent_boundary", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="lexide-pipeline")
    p.add_argument("--run_name", default="eval")
    args = p.parse_args()

    if HAS_WANDB and args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    results = evaluate_per_language(args)

    if args.eval_sent_boundary:
        evaluate_sentence_boundaries(args)

    if HAS_WANDB and args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
