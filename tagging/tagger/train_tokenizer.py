"""Train the char/byte-level boundary tagger (the tokenizer) with a bidirectional minGRU.

Predicts per-byte O/B/I; token spans are recovered from B...I runs. Reports token-level
boundary F1 (a token is correct iff its exact [start,end) span is predicted).
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (BOS_BYTE, CHAR_VOCAB_SIZE, CharBoundaryDataset, char_collate,
                     encode_bytes_and_labels, maybe_merge_inflection, read_jsonl)
from model import CharBoundaryTagger
from prior import PRIOR_NONE, PRIOR_VOCAB, PriorSidecar, Wordbank, prior_ids_for


def spans_from_labels(labels):
    """labels: list[int] over bytes (0=O,1=B,2=I) -> set of (start,end) byte spans."""
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 1:  # B
            j = i + 1
            while j < n and labels[j] == 2:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return set(spans)


def stratified(records, per_lang):
    """Up to `per_lang` records of each language, in file order.

    The val file is written language-by-language, so `records[:n]` is not a sample of the
    corpus — it is whichever languages happen to come first alphabetically. Scoring that
    prefix reported 99.8 token F1 for years while Japanese sat at 86.6, unmeasured.
    """
    seen = Counter()
    out = []
    for r in records:
        lang = r.get("lang", "und")
        if seen[lang] >= per_lang:
            continue
        seen[lang] += 1
        out.append(r)
    return out


def fmt_metrics(m):
    langs = " ".join(f"{lang} {d['token_f1']}" for lang, d in m["per_lang"].items())
    return (f"F1={m['token_f1']} prec={m['prec']} rec={m['rec']} "
            f"byte_acc={m['byte_acc']} | {langs}")


@torch.no_grad()
def evaluate(model, records, device, per_lang=400, max_bytes=512, use_lang=True,
             merge_inflection=True, use_prior=False, wordbanks=None, prior_soft=True,
             sidecar=None, blank_prior=False):
    """Micro-averaged token-span F1 overall and per language."""
    model.eval()
    agg = dict(tp=0, fp=0, fn=0, bc=0, bt=0)
    per = {}
    for r in stratified(records, per_lang):
        lang = r.get("lang") if use_lang else None
        tokens = maybe_merge_inflection(r, merge_inflection)
        byte_ids, labels = encode_bytes_and_labels(r["text"], tokens, max_bytes, lang)
        x = torch.tensor([byte_ids], device=device)
        p = None
        if use_prior:
            # The prior sees the same language the *caller* would supply, so the lang-free
            # eval measures what a lang-free caller actually gets. Passing r["lang"] here
            # regardless — as this did — reported jpn 93.79 lang-free while real lang-free
            # inference scored 22.4: whitespace on Japanese asserts the whole sentence is
            # one word, and the model leans on the proposal hard enough to obey.
            # prior.infer_lang recovers "jpn" from script, which is why the two now agree.
            if blank_prior:
                # what the model sees when no proposal is available at all
                ids = [PRIOR_NONE] * min(len(r["text"].encode("utf-8")) + 2, max_bytes)
            elif sidecar is not None:
                ids = sidecar[r["_idx"]]
            else:
                ids = prior_ids_for(r["text"], r.get("lang") if use_lang else None,
                                    max_bytes, wordbanks=wordbanks, soft=prior_soft)
            p = torch.tensor([ids], device=device)
        pred = model(x, p)[0].argmax(-1).tolist()
        gold_spans, pred_spans = spans_from_labels(labels), spans_from_labels(pred)
        counts = dict(
            tp=len(gold_spans & pred_spans),
            fp=len(pred_spans - gold_spans),
            fn=len(gold_spans - pred_spans),
            bc=sum(int(a == b) for a, b in zip(pred, labels) if b != -100),
            bt=sum(1 for b in labels if b != -100),
        )
        d = per.setdefault(r.get("lang", "und"), dict(tp=0, fp=0, fn=0, bc=0, bt=0))
        for k, v in counts.items():
            d[k] += v
            agg[k] += v

    def score(d):
        prec = d["tp"] / max(1, d["tp"] + d["fp"])
        rec = d["tp"] / max(1, d["tp"] + d["fn"])
        return {"token_f1": round(200 * prec * rec / max(1e-9, prec + rec), 2),
                "prec": round(prec * 100, 2), "rec": round(rec * 100, 2),
                "byte_acc": round(d["bc"] / max(1, d["bt"]) * 100, 2)}

    model.train()
    out = score(agg)
    out["per_lang"] = {lang: score(d) for lang, d in sorted(per.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="output/tokenizer")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--lang-dropout", type=float, default=0.15,
                    help="fraction of training examples encoded with the generic BOS "
                         "instead of their language token (keeps lang-free use working)")
    ap.add_argument("--max-bytes", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--no-merge-inflection", dest="merge_inflection",
                    action="store_false", default=True,
                    help="keep the old finer Japanese split (食べ|まし|た) instead of "
                         "merging a predicate with its auxiliary chain")
    ap.add_argument("--use-prior", action="store_true",
                    help="feed a per-byte boundary proposal alongside the bytes: "
                         "whitespace for spaced languages, a dictionary+Viterbi "
                         "analysis (fugashi/UniDic) for Japanese")
    ap.add_argument("--prior-mode", choices=["add", "concat"], default="add",
                    help="how the proposal enters the network: summed into the byte "
                         "embedding, or given its own coordinates")
    ap.add_argument("--prior-dim", type=int, default=8,
                    help="width of the prior embedding when --prior-mode=concat")
    ap.add_argument("--langs", default=None,
                    help="comma-separated subset to train on. jpn,kor is 2%% of the "
                         "corpus and trains in minutes — the right harness for "
                         "iterating on prior designs")
    ap.add_argument("--wordbank-langs", default="kor,hin",
                    help="languages whose proposal comes from a unigram wordbank + Viterbi. "
                         "Japanese is absent by default because the analyzer measurably "
                         "beats a bank there (99.4%% vs 98.8%% boundary coverage, 1.1 F1); "
                         "kor/hin gain nothing from an analyzer and cost <1MB as banks")
    ap.add_argument("--wordbank-dir", default="data/wordbanks")
    ap.add_argument("--prior-sidecar", default=None,
                    help="priors precomputed by the Rust `emit-priors` binary, as "
                         "<path>.{train,val}. This is the shipping path: the proposal the "
                         "model trains against then comes from the same code that produces "
                         "it at inference, and the Viterbi leaves the dataloader.")
    ap.add_argument("--prior-warmup-frac", type=float, default=0.5,
                    help="fraction of training during which the model gets neither the "
                         "language token nor the boundary proposal, so it first learns "
                         "boundaries from bytes alone. 0 disables the curriculum.")
    ap.add_argument("--prior-dropout", type=float, default=0.1,
                    help="fraction of training examples whose boundary proposal is blanked "
                         "to PRIOR_NONE, so the model keeps a non-prior route to the answer. "
                         "Without it the prior becomes a hard dependency: v11 scores jpn "
                         "93.3 with the dictionary and 21.2 without.")
    ap.add_argument("--no-prior-soft", dest="prior_soft", action="store_false",
                    help="collapse B_SOFT onto B, so a proposed boundary is encoded the "
                         "same as one whitespace guarantees. The pre-36b37e9 behavior, "
                         "kept as a control arm.")
    ap.add_argument("--no-group-unknown", dest="group_unknown", action="store_false",
                    default=True,
                    help="charge unknown cost per character instead of grouping a run of "
                         "one script into a single proposal (ザルツブルク -> ザル|ツ|ブ|ル|ク)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    wordbanks = {}
    if args.use_prior:
        for lang in filter(None, args.wordbank_langs.split(",")):
            path = os.path.join(args.wordbank_dir, f"{lang}.tsv")
            if os.path.exists(path):
                wordbanks[lang] = Wordbank.load(path, group_unknown=args.group_unknown)
            else:
                print(f"WARNING: no wordbank at {path}; {lang} falls back to whitespace")
        others = "analyzer (jpn)" if "jpn" not in wordbanks else ""
        print(f"prior: wordbanks={sorted(wordbanks)} {others} "
              f"group_unknown={args.group_unknown} soft={args.prior_soft}")

    train_records = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.train_limit)
    val_records = read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    if args.langs:
        keep = set(args.langs.split(","))
        train_records = [r for r in train_records if r.get("lang") in keep]
        val_records = [r for r in val_records if r.get("lang") in keep]
    print(f"train={len(train_records)} val={len(val_records)}"
          + (f" langs={args.langs}" if args.langs else ""))

    model = CharBoundaryTagger(vocab_size=CHAR_VOCAB_SIZE, emb_dim=args.emb_dim,
                               hidden_dim=args.hidden, layers=args.layers,
                               prior_vocab=PRIOR_VOCAB if args.use_prior else 0,
                               prior_mode=args.prior_mode,
                               prior_dim=args.prior_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"char tokenizer params: {n_params/1e6:.2f}M")

    train_side = val_side = None
    if args.prior_sidecar:
        train_side = PriorSidecar(f"{args.prior_sidecar}.train")
        val_side = PriorSidecar(f"{args.prior_sidecar}.val")
        print(f"prior sidecar: {len(train_side)} train / {len(val_side)} val records")

    # lang and prior dropout are applied per batch in the loop, not here, so they can follow
    # a schedule. 0.0 means "never drop" — the dataset yields the true language token every
    # time and the loop decides what to blank. (None would mean something else entirely:
    # language-blind, generic BOS always, which the loop could not undo.)
    ds = CharBoundaryDataset(train_records, args.max_bytes, lang_dropout=0.0,
                             merge_inflection=args.merge_inflection,
                             use_prior=args.use_prior, wordbanks=wordbanks,
                             prior_soft=args.prior_soft, prior_sidecar=train_side)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=char_collate, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = int(len(loader) * args.epochs)
    if args.smoke:
        total_steps = 60
    print(f"total_steps={total_steps}")

    if args.wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "lexide-parsley"),
                   name="char-tokenizer", config=vars(args))

    step = 0
    t0 = time.time()
    run = 0.0
    best = -1.0
    model.train()
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in loader:
            if step >= total_steps:
                done = True
                break
            byte_ids = batch["byte_ids"].to(device)
            labels = batch["labels"].to(device)
            prior_ids = batch["prior_ids"].to(device) if "prior_ids" in batch else None

            # Curriculum. For the first `--prior-warmup-frac` of training the model gets
            # neither the language token nor the proposal, so it has to learn boundaries
            # from bytes alone; both are then introduced at their configured dropout rates.
            # The point is to build a model that *uses* the prior rather than one that has
            # outsourced the task to it — v11, trained with the prior available from step 0,
            # scores jpn 93.3 with it and 21.2 without.
            warm = step < args.prior_warmup_frac * total_steps
            p_lang = 1.0 if warm else args.lang_dropout
            p_prior = 1.0 if warm else args.prior_dropout
            if p_lang:
                drop = torch.rand(byte_ids.size(0), device=device) < p_lang
                byte_ids[:, 0] = torch.where(drop, byte_ids.new_full((1,), BOS_BYTE)[0],
                                             byte_ids[:, 0])
            if prior_ids is not None and p_prior:
                drop = torch.rand(prior_ids.size(0), device=device) < p_prior
                prior_ids = prior_ids.masked_fill(drop.unsqueeze(1), PRIOR_NONE)

            logits = model(byte_ids, prior_ids)
            loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
            if step % args.log_every == 0:
                sps = step / (time.time() - t0)
                print(f"[tok] step {step}/{total_steps} loss={run/args.log_every:.4f} ({sps:.1f} it/s)", flush=True)
                run = 0.0
            if step % args.eval_every == 0 or (args.smoke and step == total_steps):
                # Mirror the training configuration. During warmup the model has only ever
                # seen a generic BOS and a blank proposal, so scoring it with the language
                # token and the real prior measures a combination it has never been shown —
                # it reads as a catastrophic F1 while the loss falls perfectly normally.
                m = evaluate(model, val_records, device, use_prior=args.use_prior,
                             wordbanks=wordbanks, prior_soft=args.prior_soft,
                             sidecar=val_side, use_lang=not warm, blank_prior=warm)
                tag = " (warmup: no lang, blank prior)" if warm else ""
                print(f"[tok] eval@{step}{tag} {fmt_metrics(m)}", flush=True)
                if args.wandb:
                    import wandb
                    flat = {f"tok/{k}": v for k, v in m.items() if k != "per_lang"}
                    flat.update({f"tok/{lang}_f1": d["token_f1"]
                                 for lang, d in m["per_lang"].items()})
                    wandb.log(flat, step=step)
                if m["token_f1"] > best:
                    best = m["token_f1"]
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
                    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
                        json.dump({"metrics": m, "step": step, "config": vars(args)}, f, indent=2)

    m = evaluate(model, val_records, device, use_prior=args.use_prior, wordbanks=wordbanks,
                     prior_soft=args.prior_soft, sidecar=val_side)
    print(f"[tok] final {fmt_metrics(m)}", flush=True)
    if m["token_f1"] >= best:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "tokenizer.pt"))
    nl = evaluate(model, val_records, device, use_lang=False, use_prior=args.use_prior, wordbanks=wordbanks,
                     prior_soft=args.prior_soft, sidecar=val_side)
    print(f"[tok] final lang-free {fmt_metrics(nl)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
