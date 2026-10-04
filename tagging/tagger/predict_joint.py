"""Load a self-contained joint checkpoint and predict raw-text JSONL in batches."""
import argparse
import json
from pathlib import Path
import time

import torch

from data_prep import apply_script
from dataset_joint import collate, encode_records
from eval_e2e import read_jsonl
from model import JointTagger
from mst import single_root_mst


def spans_from_char_labels(text, labels):
    """O/B/I char labels -> token spans that always reconstruct the text.

    Every non-space char lands in a token (only B, or text after a gap, opens one; a stray O
    continues the current token), and a token never
    starts or ends with whitespace: a space stays inside only when the next non-space char
    continues the token (I). So the labels can only move boundaries, never drop text.
    """
    label = lambda i: labels[i] if i < len(labels) else 0
    spans, start = [], None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is None:
                continue
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            if not (label(i) == 2 and j < len(text) and label(j) == 2):
                spans.append((start, i))
                start = None
        elif start is None or label(i) == 1:
            if start is not None:
                spans.append((start, i))
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def span_tensors(spans, device):
    width = max(1, max(map(len, spans)))
    starts = torch.zeros(len(spans), width, dtype=torch.long, device=device)
    ends = starts.clone()
    mask = torch.zeros_like(starts, dtype=torch.bool)
    for i, sentence in enumerate(spans):
        if sentence:
            starts[i, :len(sentence)] = torch.tensor([s for s, e in sentence], device=device)
            ends[i, :len(sentence)] = torch.tensor([e for s, e in sentence], device=device)
            mask[i, :len(sentence)] = True
    return starts, ends, mask


@torch.inference_mode()
def predict_batch(model, tokenizer, vocab, records, device="cpu"):
    """Return predictions plus reusable character states for gold-span diagnostics."""
    model.eval()
    items = encode_records(records, tokenizer, vocab, char_buckets=model.char_buckets, training=False)
    batch = collate(items, tokenizer.pad_token_id, device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=torch.device(device).type == "cuda"):
        encoded = model.encode_chars(batch)
        labels = encoded["boundary_logits"].argmax(-1).cpu().tolist()
        spans = [spans_from_char_labels(r["text"], l) for r, l in zip(records, labels)]
        starts, ends, mask = span_tensors(spans, device)
        heads = model.word_heads(encoded, starts, ends, mask)
    pos = heads["pos_logits"].argmax(-1).cpu().tolist()
    lemma = heads["lemma_logits"].argmax(-1).cpu().tolist()
    arcs = heads["arc_scores"].float().cpu().numpy()
    rels = heads["rel_scores"].argmax(-1).cpu().tolist()
    output = []
    for i, (record, sentence) in enumerate(zip(records, spans)):
        parents = single_root_mst(arcs[i, :len(sentence), :len(sentence) + 1])
        tokens = []
        for j, (s, e) in enumerate(sentence):
            tokens.append({"start": s, "end": e, "pos": vocab["pos"][pos[i][j]],
                           "lemma": apply_script(record["text"][s:e], vocab["lemma_scripts"][lemma[i][j]]),
                           "dep": vocab["dep"][rels[i][j][parents[j]]], "head": parents[j]})
        output.append({"lang": record.get("lang", "unknown"), "text": record["text"], "tokens": tokens})
    return output, encoded, batch


@torch.inference_mode()
def predict_records(model, tokenizer, vocab, records, batch_size=16, device="cpu"):
    output = []
    for offset in range(0, len(records), batch_size):
        predictions, _, _ = predict_batch(model, tokenizer, vocab, records[offset:offset + batch_size], device)
        output.extend(predictions)
    return output


def save_checkpoint(path, model, tokenizer, vocab, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = {**config, "encoder_config": model.encoder.config.to_dict()}
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    tokenizer.save_pretrained(path / "tokenizer")
    # Atomic state replacement protects the last good model from interrupted writes.
    temporary = path / "state.pt.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(path / "state.pt")


def load_checkpoint(path, device="cpu"):
    from transformers import AutoTokenizer
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    vocab = json.loads((path / "vocab.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    model = JointTagger(**config)
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tokenizer, vocab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, vocab = load_checkpoint(args.checkpoint, args.device)
    records = read_jsonl(args.input)
    start = time.perf_counter()
    predictions = predict_records(model, tokenizer, vocab, records, args.batch_size, args.device)
    elapsed = time.perf_counter() - start
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"sentences": len(records), "seconds": elapsed,
                      "sentences_per_second": len(records) / elapsed}))
