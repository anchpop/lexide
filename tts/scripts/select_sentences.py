"""Pick a language's TTS scripts from its pack dump, one per voice.

    python scripts/select_sentences.py data/deu_sentences.jsonl data/scripts \
        --voices achernar=10 kore=0.5 leda=0.5 puck=0.5 iapetus=0.5 orus=0.5

Writes ``<output>/<language>_<voice>.jsonl`` (the language from the dump's name,
``<language>_sentences.jsonl``). Every voice reads the same ``--holdout``
validation sentences and the same ``--shared`` training sentences, so that
speaker and content can be told apart; then each voice reads sentences of its
own (no two voices share them) for its hours, picked greedily for the phone
bigrams it does not cover yet, each sentence type (``sentence_type``: statement,
question, exclamation) for its share of the dump's sentences. Quoted dialogue is rare in a pack (0.5-2% of
sentences) and the model must still read it, so each voice also reads a
quoted set (``--quoted-share`` of its hours, about 500 sentences for 10 h),
round-robin over how the sentences quote (``quote_kind``: the whole sentence
quoted, a quote then its attribution, the attribution inside the quote,
nested quotes, a quote inside the sentence) and over the language's quotation
marks; every voice also reads ``--quoted-holdout`` quoted validation sentences.
Rows say which of these they are (``role``).

Candidates are plain sentences: in the language's script, without digits,
brackets or proper nouns, every word phonemized, every phone known to the
aligner, of moderate length. Duration is estimated from the phone count
(``--phones-per-second``, Achernar's English rate).

A script already written (and perhaps recorded) is extended, never
rewritten: rerunning keeps every role it has and adds the ones it lacks.
"""
import argparse, hashlib, heapq, json, pathlib, random, sys, unicodedata

from voca.build import FrameMatrixRecognizer
from voca.text import bare_phones, sentence_type

# Letters a sentence may use, by the first word of their Unicode name; Latin otherwise.
SCRIPTS = {"rus": {"CYRILLIC"}, "hin": {"DEVANAGARI"}, "tha": {"THAI"}, "kor": {"HANGUL"}, "zho-hans": {"CJK", "IDEOGRAPHIC"},
           "jpn": {"CJK", "IDEOGRAPHIC", "HIRAGANA", "KATAKANA", "KATAKANA-HIRAGANA"}}
SPOKEN_MARKS = set(",.?!;:'’-—–…、，。？！：；।¿¡")   # punctuation a plain sentence may have
# Quotation marks, by family: a sentence with marks of two families nests one quote in another.
QUOTE_FAMILIES = ('"', "“”„", "«»", "‹›‚", "「」", "『』")
QUOTES = set("".join(QUOTE_FAMILIES))
TYPES = ("statement", "question", "exclamation")    # voca.text.sentence_type; an unpunctuated sentence competes as a statement


def plain(text, language):
    scripts = SCRIPTS.get(language, {"LATIN"})
    for c in text:
        if c.isalpha() or unicodedata.category(c).startswith("M"):
            name = unicodedata.name(c, "")
            if name.split(" ")[0] not in scripts | {"COMBINING"}:
                return False
        elif not (c.isspace() or c in SPOKEN_MARKS or c in QUOTES):   # French writes a narrow space before ? ! : ; »
            return False     # digits, brackets, symbols
    return True


def quote_kind(text):
    """How a sentence quotes, for a varied quoted set: (shape, the family of its first mark) or None.
    Shapes: ``nested`` (marks of two families), ``interrupted`` (two quoted spans: attribution inside
    the quote), ``whole`` (the sentence is the quote), ``then-said`` (a quote, then its attribution),
    ``inside`` (a quote, or a quoted word, after the sentence starts)."""
    marks = [c for c in text if c in QUOTES]
    if not marks:
        return None
    families = {next(i for i, f in enumerate(QUOTE_FAMILIES) if c in f) for c in marks}
    family = QUOTE_FAMILIES[next(i for i, f in enumerate(QUOTE_FAMILIES) if marks[0] in f)]
    body = text.strip().strip(".?!…。？！। ")
    if len(families) > 1:
        shape = "nested"
    elif len(marks) >= 4:
        shape = "interrupted"
    elif body[:1] in QUOTES:
        shape = "whole" if body[-1:] in QUOTES else "then-said"
    else:
        shape = "inside"
    return shape, family


def candidates(dump, language, aligner, limits):
    """The plain sentences, and the dump's mix of ``TYPES`` (shares of its distinct sentences)."""
    rows, seen, mix = [], set(), dict.fromkeys(TYPES, 0)
    for line in open(dump, encoding="utf-8"):
        row = json.loads(line)
        text = row["text"].strip()
        if text in seen:
            continue
        seen.add(text)
        mix[sentence_type(text)] += 1
        if not plain(text, language):
            continue
        # Every word gram phonemized (g2p sometimes leaves one empty), every phone alignable.
        if any(not g["phonemes"].strip() and any(c.isalnum() for w in g["words"] for c in w) for g in row["grams"]):
            continue
        # No proper nouns: g2p reads a foreign name the language's way and Gemini often doesn't, so
        # the clip fails alignment (French: 61% of such clips accepted, 85% of the rest), and the
        # bigram greedy below favours exactly them, for their rare phone sequences.
        if any("PROPN" in g["pos"] for g in row["grams"]):
            continue
        phones = bare_phones(row["grams"])
        if not limits[0] <= len(phones) <= limits[1] or not all(map(aligner, phones)):
            continue
        rows.append({"text": text, "grams": row["grams"], "phones": len(phones), "bigrams": set(zip(["#"] + phones, phones + ["#"])),
                     "quote": quote_kind(text), "type": sentence_type(text)})
    return rows, {t: n / sum(mix.values()) for t, n in mix.items()}


def greedy(pool, taken, covered, budget, cost):
    """Lazy greedy on new bigrams per sqrt(phone): a sentence's gain only falls as
    coverage grows, so a stale heap entry is an upper bound."""
    gain = lambda r: sum(1 / (1 + covered.get(b, 0)) for b in r["bigrams"]) / r["phones"] ** 0.5
    heap = [(-gain(r), i) for i, r in enumerate(pool) if r["text"] not in taken]
    heapq.heapify(heap)
    chosen, used = [], 0.0
    while used < budget and heap:
        _, i = heapq.heappop(heap)
        score = gain(pool[i])
        if heap and score < -heap[0][0]:
            heapq.heappush(heap, (-score, i))
            continue
        for b in pool[i]["bigrams"]:
            covered[b] = covered.get(b, 0) + 1
        chosen.append(pool[i])
        taken.add(pool[i]["text"])
        used += cost(pool[i])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--voices", nargs="+", required=True, help="voice=hours of its own sentences")
    parser.add_argument("--shared", type=int, default=50, help="training sentences every voice reads")
    parser.add_argument("--holdout", type=int, default=60, help="validation sentences every voice reads")
    parser.add_argument("--quoted-share", type=float, default=0.05, help="quoted sentences per voice, as a share of its hours")
    parser.add_argument("--quoted-holdout", type=int, default=12, help="quoted validation sentences every voice reads")
    parser.add_argument("--aligner-labels", type=pathlib.Path, required=True,
                        help="a frame matrix (scripts/fetch_frames.py output) whose labels the aligner knows")
    parser.add_argument("--phones-per-second", type=float, default=7.8, help="Gemini Achernar speaks ~7.8 in English")
    parser.add_argument("--min-phones", type=int, default=12)
    parser.add_argument("--max-phones", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    language = args.dump.name.removesuffix("_sentences.jsonl")
    hours = {v: float(h) for v, h in (x.split("=") for x in args.voices)}
    recognizer = FrameMatrixRecognizer(json.loads(args.aligner_labels.read_text()))

    def alignable(phone):
        try:
            return recognizer.label(phone) is not None
        except KeyError:
            return False
    pool, mix = candidates(args.dump, language, alignable, (args.min_phones, args.max_phones))
    random.Random(args.seed).shuffle(pool)
    print(f"{language}: {len(pool)} candidate sentences; the dump's mix "
          + ", ".join(f"{t} {share:.1%}" for t, share in mix.items()), file=sys.stderr)
    by_text = {r["text"]: r for r in pool}
    # A script already written (and maybe recorded) is extended, never rewritten: each role keeps its sentences.
    scripts = {v: args.output / f"{language}_{v}.jsonl" for v in hours}
    roles = {v: {} for v in hours}
    for voice, path in scripts.items():
        for line in open(path, encoding="utf-8") if path.exists() else []:
            r = json.loads(line)
            roles[voice].setdefault(r["role"], []).append({**by_text.get(r["text"], {}), **r})
    first = roles[next(iter(hours))]
    taken = {r["text"] for rs in roles.values() for rows in rs.values() for r in rows}
    seconds = lambda r: r["phones"] / args.phones_per_second
    everyone = {}    # roles every voice reads
    for role, select in (("validation", lambda: [r for r in pool if r["text"] not in taken and not r["quote"]][:args.holdout]),
                         ("quoted-validation", lambda: varied(pool, taken, args.quoted_holdout, lambda r: 1)),
                         ("shared", lambda: greedy(pool, taken, {}, args.shared, lambda r: 1))):
        everyone[role] = first.get(role) or select()
        taken |= {r["text"] for r in everyone[role]}
    shared_coverage = {}
    for r in everyone["shared"]:
        for b in r.get("bigrams", ()):
            shared_coverage[b] = shared_coverage.get(b, 0) + 1
    for voice in hours:
        if "own" not in roles[voice]:     # each sentence type its share of the hours, as in the dump
            coverage = dict(shared_coverage)
            roles[voice]["own"] = [r for t in TYPES for r in greedy([r for r in pool if r["type"] == t], taken, coverage,
                                                                     mix[t] * hours[voice] * 3600, seconds)]
        if "quoted" not in roles[voice]:
            roles[voice]["quoted"] = varied(pool, taken, args.quoted_share * hours[voice] * 3600, seconds)
        roles[voice].update(everyone)
    for voice, path in scripts.items():
        with open(path.with_suffix(".tmp"), "w", encoding="utf-8") as out:    # renamed into place: generation may be reading it
            for role in ("own", "quoted", "shared", "validation", "quoted-validation"):
                for row in roles[voice][role]:
                    out.write(json.dumps({"id": hashlib.sha256(row["text"].encode()).hexdigest()[:16], "text": row["text"],
                                          "grams": row["grams"], "split": "validation" if "validation" in role else "train",
                                          "role": role}, ensure_ascii=False) + "\n")
        path.with_suffix(".tmp").replace(path)
        print(f"{path.name}: " + ", ".join(f"{len(rows)} {role}" for role, rows in roles[voice].items()), file=sys.stderr)


def varied(pool, taken, budget, cost):
    """Quoted sentences up to ``budget``, round-robin over how they quote (``quote_kind``),
    so every shape and every quotation mark the language uses is read."""
    strata = {}
    for r in pool:
        if r["quote"] and r["text"] not in taken:
            strata.setdefault(r["quote"], []).append(r)
    chosen, used, queues = [], 0.0, [strata[k] for k in sorted(strata)]
    while used < budget and any(queues):
        for queue in queues:
            if queue and used < budget:
                chosen.append(queue.pop(0))
                taken.add(chosen[-1]["text"])
                used += cost(chosen[-1])
    return chosen


if __name__ == "__main__":
    main()
