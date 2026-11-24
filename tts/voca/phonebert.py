"""Phoneme BERT: a text encoder pretrained on phonemes alone (after PL-BERT,
Li et al. 2023, which Kokoro uses).

No audio. Every sentence of a pack dump is tokenized exactly as TTS input is
(voca.text.tokenize), and one transformer is trained on two jobs at once:

* masked phonemes: whole words have their phones hidden (UNK stands in for a
  mask token, so the inventory does not change) and the encoder must restore them;
* phoneme to word: at every phone, name the word it belongs to (where the
  frontend could match the gram's words to its phone groups; elsewhere the
  word is still masked as a whole but not named);
* masked part of speech and rarity: a masked word's part of speech and its
  gram's rarity (on all of the gram's phones, or its other words would give it
  away) are hidden too, and restored from context. Every other word keeps
  them as inputs, as in TTS.

Each job has a single linear head, too small to do any thinking, so the
encoder's per-phone vectors have to carry which words are present and how the
sentence is built: what intonation depends on. The heads are thrown away;
`encoder.pt` holds only the embeddings and the transformer, sized to plug in
where TinyVoca's text encoder is (through a projection if the widths differ).

One per language, like the TTS model: run it with that language's
inventory (``VOCA_LANGUAGE``, see voca.text).

    VOCA_LANGUAGE=jpn python -m voca.phonebert prepare data/jpn_sentences.jsonl features/jpn_phonebert_v1 \
        --exclude data/scripts/jpn_*.jsonl
    VOCA_LANGUAGE=jpn python -m voca.phonebert train features/jpn_phonebert_v1 runs/phonebert-jpn-v1
"""

import argparse
import collections
import dataclasses
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .model import Stack, add_lexical_embeddings, embed_tokens, length_mask
from .text import LEXICAL, SYMBOL_TO_ID, UNK, PHONE_IDS, frontend_fingerprint, tokenize

MASK = SYMBOL_TO_ID[UNK]
IGNORE = -100
POS, RARITY = list(LEXICAL).index("pos"), list(LEXICAL).index("rarity")


def encoder_config(channels=256, heads=4, feedforward=1024, dropout=0.1):
    return dataclasses.replace(Config(), channels=channels, heads=heads, feedforward=feedforward, dropout=dropout)


class PhoneEncoder(nn.Module):
    """Phone + lexical-feature embeddings and a transformer stack: what ships."""

    def __init__(self, c, layers, distinct=0):
        super().__init__()
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        add_lexical_embeddings(self, c.channels)
        self.stack = Stack(c, layers, distinct)

    def forward(self, tokens, lexical, mask):
        return self.stack(embed_tokens(self, tokens, lexical) * mask[..., None], mask)


class PhoneBert(nn.Module):
    def __init__(self, c, layers, words, distinct=0):
        super().__init__()
        self.encoder = PhoneEncoder(c, layers, distinct)
        self.phone_head = nn.Linear(c.channels, c.vocab_size)
        self.word_head = nn.Linear(c.channels, words)
        self.pos_head = nn.Linear(c.channels, LEXICAL["pos"])
        self.rarity_head = nn.Linear(c.channels, LEXICAL["rarity"])

    def forward(self, tokens, lexical, mask):
        hidden = self.encoder(tokens, lexical, mask)
        return self.phone_head(hidden), self.word_head(hidden), self.pos_head(hidden), self.rarity_head(hidden)


def prepare(args):
    excluded = {json.loads(line)["text"] for path in args.exclude for line in open(path, encoding="utf-8")}
    sentences = []
    for line in open(args.dump, encoding="utf-8"):
        row = json.loads(line)
        if row["text"] in excluded:
            continue
        p = tokenize(row["grams"])
        if 3 <= len(p) <= args.max_tokens:
            sentences.append(p)
    counts = collections.Counter(w for p in sentences for w in set(p.words) if w is not None)
    vocabulary = ["<unk>"] + [w for w, _ in counts.most_common(args.words - 1)]
    index = {w: i for i, w in enumerate(vocabulary)}
    random.Random(0).shuffle(sentences)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "words.json").write_text(json.dumps(vocabulary, ensure_ascii=False))
    # Quoted dialogue is 0.5-2% of a pack: repeat it in training so the quote token is learned in context.
    train = [p for p in sentences[args.validation:] for _ in range(args.quoted_repeat if '"' in p.symbols else 1)]
    random.Random(1).shuffle(train)
    for split, rows in (("validation", sentences[:args.validation]), ("train", train)):
        lengths = torch.tensor([len(p) for p in rows])
        torch.save({"frontend": frontend_fingerprint(), "lengths": lengths,
                    "tokens": torch.tensor([i for p in rows for i in p.ids], dtype=torch.int16),
                    "lexical": torch.tensor([f for p in rows for f in p.lexical()], dtype=torch.int8),
                    # Each phone's word occurrence (-1 elsewhere), and that word's vocabulary
                    # index: IGNORE where the frontend could not name the word.
                    "occurrence": torch.tensor([w for p in rows for w in p.word], dtype=torch.int16),
                    "gram": torch.tensor([g for p in rows for g in p.gram], dtype=torch.int16),
                    "words": torch.tensor([IGNORE if w is None else index.get(w, 0) for p in rows for w in p.word_names()],
                                          dtype=torch.int32)}, out / f"{split}.pt")
        print(f"{split}: {len(rows)} sentences, {int(lengths.sum())} tokens")
    covered = sum(c for w, c in counts.items() if w in index) / sum(counts.values())
    unnamed = sum(w is None for p in sentences for w in p.words) / sum(len(p.words) for p in sentences)
    print(f"{len(counts)} distinct words; the top {len(vocabulary) - 1} cover {covered:.1%} of word occurrences; "
          f"{unnamed:.3%} of occurrences have no name (masked from the word target)")


class Corpus:
    FIELDS = ("tokens", "lexical", "occurrence", "words", "gram")

    def __init__(self, path):
        data = torch.load(path)
        if data["frontend"] != frontend_fingerprint():
            raise ValueError(f"{path} was prepared by a different front end; re-run voca.phonebert prepare")
        self.lengths = data["lengths"]
        self.starts = torch.cumsum(self.lengths, 0) - self.lengths
        self.data = {name: data[name].long() for name in self.FIELDS}

    def __len__(self):
        return len(self.lengths)

    def batch(self, rows):
        """tokens, lexical, occurrence, words, gram (padded with 0, 0, -1, IGNORE, -1), lengths."""
        lengths = self.lengths[rows]
        size = int(lengths.max())
        out = []
        for name, fill in zip(self.FIELDS, (0, 0, -1, IGNORE, -1), strict=True):
            source = self.data[name]
            o = source.new_full((len(rows), size, *source.shape[1:]), fill)
            for b, r in enumerate(rows.tolist()):
                o[b, :int(self.lengths[r])] = source[int(self.starts[r]):int(self.starts[r] + self.lengths[r])]
            out.append(o)
        return (*out, lengths)


def mask_words(tokens, lexical, occurrence, gram, rate, generator):
    """Whole-word masking: pick `rate` of the word occurrences; of their phones 80%
    become the mask token, 10% a random phone, 10% stay, their features are
    hidden, and so is the rarity of every gram one of them is in. Returns the
    inputs, their features, and per phone the targets of the masked words: phone,
    part of speech and rarity (IGNORE elsewhere, and where a feature is "none")."""
    phone = torch.tensor(sorted(PHONE_IDS), device=tokens.device)
    # One draw per word occurrence: occurrence k of a row reads that row's k-th draw.
    draw = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    chosen = (occurrence >= 0) & (draw.gather(1, occurrence.clamp_min(0)) < rate)
    action = torch.rand(tokens.shape, generator=generator, device=tokens.device)
    random_phone = phone[torch.randint(len(phone), tokens.shape, generator=generator, device=tokens.device)]
    inputs = torch.where(chosen & (action < 0.8), MASK, tokens)
    inputs = torch.where(chosen & (action >= 0.8) & (action < 0.9), random_phone, inputs)
    slot = gram.clamp_min(0)     # gram k of a row is counted in that row's k-th slot
    hidden_gram = (gram >= 0) & (torch.zeros_like(gram).scatter_add_(1, slot, chosen.long()).gather(1, slot) > 0)
    features = torch.where(chosen[..., None], lexical.new_zeros(()), lexical)
    features[..., RARITY] = torch.where(hidden_gram, 0, features[..., RARITY])

    def target(column):
        return torch.where(chosen & (lexical[..., column] > 0), lexical[..., column], IGNORE)
    return inputs, features, torch.where(chosen, tokens, IGNORE), target(POS), target(RARITY)


def losses(model, batch, generator, rate):
    tokens, lexical, occurrence, words, gram, lengths = batch
    mask = length_mask(lengths, tokens.shape[1])
    inputs, input_lexical, *targets = mask_words(tokens, lexical, occurrence, gram, rate, generator)
    with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=tokens.is_cuda):
        phone_logits, word_logits, pos_logits, rarity_logits = model(inputs, input_lexical, mask)
    stats, total = {}, 0
    for name, logits, target in zip(("phone", "word", "pos", "rarity"), (phone_logits, word_logits, pos_logits, rarity_logits),
                                    (targets[0], words, targets[1], targets[2])):
        loss = F.cross_entropy(logits.float().transpose(1, 2), target, ignore_index=IGNORE)
        total = total + loss
        stats[f"{name}_loss"] = loss.item()
        stats[f"{name}_accuracy"] = (logits.argmax(-1) == target)[target != IGNORE].float().mean().item()
    # The word head names every phone's word; how well it names the masked ones is the harder half.
    stats["masked_word_accuracy"] = (word_logits.argmax(-1) == words)[targets[0] != IGNORE].float().mean().item()
    return total, stats


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.features)
    corpus, validation = Corpus(data / "train.pt"), Corpus(data / "validation.pt")
    vocabulary = json.loads((data / "words.json").read_text())
    c = encoder_config(args.channels, args.heads, args.feedforward)
    model = PhoneBert(c, args.layers, len(vocabulary), args.distinct).to(device)
    print(f"encoder {sum(p.numel() for p in model.encoder.parameters()) / 1e6:.2f}M parameters, "
          f"heads {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M in total")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    schedule = lambda step: min(1, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": dataclasses.asdict(c), "layers": args.layers, "distinct": args.distinct, "args": vars(args)}
    (out / "recipe.json").write_text(json.dumps(meta, indent=1))
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="lexide-tts", name=out.name, id=out.name, resume="allow", config=meta)
    generator = torch.Generator(device).manual_seed(0)
    validation_generator_seed = 1
    order, cursor = torch.randperm(len(corpus)), 0
    log = open(out / "metrics.jsonl", "a")
    started = time.time()
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order, cursor = torch.randperm(len(corpus)), 0
        rows, cursor = order[cursor:cursor + args.batch], cursor + args.batch
        batch = [x.to(device, non_blocking=True) for x in corpus.batch(rows)]
        model.train()
        loss, stats = losses(model, batch, generator, args.mask_rate)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record = None
        if step % args.log_every == 0:
            record = {"step": step, "split": "train", "lr": scheduler.get_last_lr()[0], "elapsed": time.time() - started, **stats}
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            sums, count = collections.Counter(), 0
            fixed = torch.Generator(device).manual_seed(validation_generator_seed)
            with torch.no_grad():
                for start in range(0, len(validation), args.batch):
                    rows = torch.arange(start, min(start + args.batch, len(validation)))
                    _, s = losses(model, [x.to(device) for x in validation.batch(rows)], fixed, args.mask_rate)
                    sums.update({k: v * len(rows) for k, v in s.items()})
                    count += len(rows)
            evaluation = {"step": step, "split": "validation", **{k: v / count for k, v in sums.items()}}
            print(json.dumps(evaluation), flush=True)
            log.write(json.dumps(evaluation) + "\n")
            if run:
                run.log({f"validation/{k}": v for k, v in evaluation.items() if k not in ("step", "split")}, step=step)
            torch.save({"config": dataclasses.asdict(c), "layers": args.layers, "distinct": args.distinct, "frontend": frontend_fingerprint(),
                        "encoder": model.encoder.state_dict()}, out / "encoder.pt")
        if record:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if run:
                run.log({f"train/{k}": v for k, v in record.items() if k not in ("step", "split")}, step=step)
    if run:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("dump", help="yap dump_sentence_phonemes JSONL")
    p.add_argument("output")
    p.add_argument("--exclude", nargs="*", default=[], help="scripts whose sentences must stay unseen (TTS validation)")
    p.add_argument("--words", type=int, default=32768, help="word vocabulary size, including <unk>")
    p.add_argument("--validation", type=int, default=2000)
    p.add_argument("--quoted-repeat", type=int, default=5, help="times each quoted training sentence appears")
    p.add_argument("--max-tokens", type=int, default=256)
    t = commands.add_parser("train")
    t.add_argument("features")
    t.add_argument("output")
    t.add_argument("--channels", type=int, default=256)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--feedforward", type=int, default=1024)
    t.add_argument("--layers", type=int, default=10)
    t.add_argument("--distinct", type=int, default=0, help="loop this many distinct blocks to --layers depth (0: all distinct)")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=256, help="sentences per step")
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--warmup", type=int, default=2000)
    t.add_argument("--mask-rate", type=float, default=0.15)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
