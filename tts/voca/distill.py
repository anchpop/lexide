"""A phonemes-only model from a hinted one, by distilling its text side (Andre's design, Nov 25 2026).

The hinted model reads yap's phonemes plus what only yap knows (part of speech, gram rarity, gram boundaries:
``voca.text.HINTS``). Everything after its two text encoders (``Voca.encode`` and the phoneme encoder ``bert``)
sees only their outputs. The student uses the same encoders without hints, trained on text alone to give, from stock espeak's phonemes of a sentence,
the outputs the teacher gives from yap's phonemes and hints. The rest of the model is the teacher's, unchanged.

    python -m voca.distill --teacher runs/eng-style-voices-v10/inference.pt \
        --pairs data/distill/yap.jsonl data/distill/plain.jsonl --output runs/eng-style-voices-v10-distilled

``--pairs``: two files of the same sentences in the same order, rows {"text", "grams"}: yap's (the teacher's
input) and stock espeak's (``plain/stock_espeak.py``, the student's). The two token sequences are aligned and the
student learns the teacher's outputs on the tokens that line up: the ones they share, and phones they disagree on
one for one, where the student has to hear from the context what yap knew (espeak's "reed" for yap's past-tense
"read" /ɹɛd/, -ed as /st/ or /zd/). espeak joins some words yap keeps apart ("of the", "was a", "husband's"), so a
word break on one side only is left to the student. A sentence where fewer than ``MIN_SHARED`` of the student's
tokens line up is skipped.
"""
import argparse, dataclasses, difflib, json, math, random, re, time
from pathlib import Path
import torch

from .config import Config, check_format
from .model import build_model
from .text import HINTS, PHONE_IDS, frontend_fingerprint, tokenize
from .train import export_model


MIN_SHARED = 0.9


def shared(teacher, student):
    """Positions of the tokens that line up, in order: (teacher positions, student positions). Equal tokens, and
    phones replaced one for one."""
    pairs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, teacher, student, autojunk=False).get_opcodes():
        if op == "equal":
            pairs += zip(range(i1, i2), range(j1, j2))
        elif op == "replace" and i2 - i1 == j2 - j1:
            pairs += [(i, j) for i, j in zip(range(i1, i2), range(j1, j2)) if teacher[i] in PHONE_IDS and student[j] in PHONE_IDS]
    t, s = zip(*pairs) if pairs else ((), ())
    return torch.tensor(t, dtype=torch.long), torch.tensor(s, dtype=torch.long)


def pairs(yap, plain):
    out, skipped = [], 0
    with open(yap, encoding="utf-8") as teachers, open(plain, encoding="utf-8") as students:
        for a, b in zip(teachers, students, strict=True):
            a, b = json.loads(a), json.loads(b)
            if a["text"] != b["text"]:
                raise ValueError("the paired texts differ")
            try:
                t, s = tokenize(a["grams"]), tokenize(b["grams"])
            except KeyError:
                skipped += 1
                continue
            at_teacher, at_student = shared(t.ids, s.ids)
            if len(at_student) < MIN_SHARED * len(s):
                skipped += 1
                continue
            lexical = torch.tensor(s.lexical())
            lexical[:, HINTS] = 0
            out.append((torch.tensor(t.ids), torch.tensor(t.lexical()), torch.tensor(s.ids), lexical, at_teacher, at_student))
    # The words espeak spells the same however yap reads them ("read": /ɹiːd/ for yap's red and reed): every occurrence of
    # them, read either way, is marked, so weighting them (--ambiguous-weight) teaches the context, not one reading.
    spans = lambda ids: [range(m.start(), m.end()) for m in re.finditer(r"1+", "".join("1" if i in PHONE_IDS else "0" for i in ids))]
    ambiguous = set()
    for t, _, s, _, at_t, at_s in out:
        replaced = set(at_s[t[at_t] != s[at_s]].tolist())
        ambiguous |= {tuple(s[w].tolist()) for w in spans(s.tolist()) if replaced.intersection(w)}
    for k, (t, tl, s, sl, at_t, at_s) in enumerate(out):
        marked = {j for w in spans(s.tolist()) if tuple(s[w].tolist()) in ambiguous for j in w}
        out[k] = (t, tl, s, sl, at_t, at_s, torch.tensor([j in marked for j in at_s.tolist()]))
    print(f"{len(ambiguous)} ambiguous espeak words", flush=True)
    return out, skipped


def batch(items, device):
    """Each column of ``pairs``' items padded, then the teacher's and the student's lengths and the shared-token mask."""
    def column(k):
        n = max(len(item[k]) for item in items)
        return torch.stack([torch.cat((item[k], item[k].new_zeros((n - len(item[k]), *item[k].shape[1:])))) for item in items]).to(device)
    lengths = lambda k: torch.tensor([len(item[k]) for item in items], device=device)
    shared_mask = torch.arange(max(len(item[4]) for item in items), device=device)[None] < lengths(4)[:, None]
    return (*(column(k) for k in range(7)), lengths(0), lengths(2), shared_mask)


def text_side(model, tokens, lexical, lengths, speakers):
    hidden, mask = model.encode(tokens, lexical, lengths, speakers)
    return hidden, model.bert(tokens, lexical, mask), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-layers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--ambiguous-weight", type=float, default=1.0, help="loss weight of the words yap and espeak sometimes disagree on")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--held-out", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    state = torch.load(args.teacher, map_location="cpu", weights_only=True)
    check_format(state, args.teacher)
    config = Config.from_dict(state["config"])
    if state["frontend"]["fingerprint"] != frontend_fingerprint():
        parser.error("set VOCA_LANGUAGE to the teacher checkpoint language")
    teacher = build_model(config)
    teacher.load_state_dict(state["model"])
    student = build_model(dataclasses.replace(config, hints=False, front_layers=args.front_layers))
    missing, unexpected = student.load_state_dict(state["model"], strict=False)
    assert not unexpected and all(".front." in k or k.startswith("front.") for k in missing), (missing, unexpected)
    teacher.to(args.device).eval().requires_grad_(False)
    student.to(args.device).eval().requires_grad_(False)
    text = [student.tokens, student.voices, student.encoder, student.front, student.bert] + \
           [getattr(student, name) for name in ("stress", "tone", "accent")]
    params = [p for m in text if m is not None for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    data, skipped = pairs(*args.pairs)
    random.Random(0).shuffle(data)
    if not 0 < args.held_out < len(data):
        parser.error("--held-out must leave at least one training pair and one validation pair")
    held, train = data[:args.held_out], data[args.held_out:]
    print(f"{len(train)} training pairs, {len(held)} held out, {skipped} skipped", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)
    speakers = len(config.speakers)
    rng = random.Random(1)

    def loss_of(items, grad):
        tokens, lexical, s_tokens, s_lexical, at_teacher, at_student, ambiguous, lengths, s_lengths, m = batch(items, args.device)
        who = torch.tensor([rng.randrange(speakers) for _ in items], device=args.device)
        with torch.no_grad():
            h, b, _ = text_side(teacher, tokens, lexical, lengths, who)
        with torch.set_grad_enabled(grad):
            sh, sb, _ = text_side(student, s_tokens, s_lexical, s_lengths, who)
        at = lambda x, index: x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))
        # The ambiguous words are a sliver of the tokens but all of what the student must learn from context.
        m = (m * (1 + (args.ambiguous_weight - 1) * ambiguous) if grad else m)[..., None].float()
        err = lambda x, y: ((x - y) ** 2 * m).sum() / ((y ** 2) * m).sum()    # relative squared error over the shared tokens
        return err(at(sh, at_student), at(h, at_teacher)), err(at(sb, at_student), at(b, at_teacher))

    args.output.mkdir(parents=True, exist_ok=True)
    log = open(args.output / "metrics.jsonl", "a")
    by_length = sorted(range(len(train)), key=lambda i: len(train[i][0]))
    chunks = [[]]    # runs of similar length: --batch sentences, fewer where they are long (--batch-tokens, padded)
    for i in by_length:
        if len(chunks[-1]) == args.batch or (len(chunks[-1]) + 1) * len(train[i][0]) > args.batch_tokens:
            chunks.append([])
        chunks[-1].append(i)
    def held_out():
        with torch.no_grad():
            errs = [loss_of(held[i:i + args.batch], False) for i in range(0, len(held), args.batch)]
        return {"held_encoder": sum(e[0].item() for e in errs) / len(errs), "held_bert": sum(e[1].item() for e in errs) / len(errs)}
    print(json.dumps({"step": 0, **held_out()}), flush=True)     # the teacher's weights reading stock espeak without hints
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        lr = args.learning_rate * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in optimizer.param_groups:
            g["lr"] = lr
        items = [train[i] for i in rng.choice(chunks)]
        a, b = loss_of(items, True)
        optimizer.zero_grad(set_to_none=True)
        (a + b).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            row = {"step": step, "train_encoder": a.item(), "train_bert": b.item(), **held_out(), "lr": lr, "seconds": time.monotonic() - start}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n"); log.flush()
    export_model(student, args.output / "inference.pt", state["step"], state.get("annotation"))
    (args.output / "config.json").write_text(json.dumps(student.config.to_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
