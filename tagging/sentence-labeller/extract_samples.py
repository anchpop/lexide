#!/usr/bin/env python3
"""Extract deterministic, varied-length passages from every Harry Potter translation."""

import argparse
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import random
import re
import zipfile

BOOKS = {
    "fra": [103, 105, 109, 110, 108, 106, 107],
    "deu": [118, 117, 116, 119, 120, 122, 121],
    "jpn": [126, 127, 128, 123, 125, 124, 129],
}


class TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "h1", "h2", "h3", "h4", "li", "blockquote"}
    SKIP = {"script", "style", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if not self.skip and tag in self.BLOCKS:
            self.parts.append("\n\n")
        elif not self.skip and tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def epub_text(data):
    paragraphs = []
    with zipfile.ZipFile(io.BytesIO(data)) as epub:
        names = [n for n in epub.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in names:
            parser = TextExtractor()
            try:
                parser.feed(epub.read(name).decode("utf-8", errors="replace"))
            except Exception:
                continue
            text = html.unescape("".join(parser.parts)).replace("\r\n", "\n").replace("\r", "\n")
            for p in re.split(r"\n\s*\n", text):
                p = re.sub(r"[ \t\f\v]+", " ", p).strip()
                if len(p) >= 20 and sum(c.isalpha() for c in p) >= 8:
                    paragraphs.append(p)
    return paragraphs


def varied_targets(count, minimum, maximum, rng):
    """Log-uniform lengths give many useful short samples and some long contexts."""
    targets = [round(math.exp(rng.uniform(math.log(minimum), math.log(maximum)))) for _ in range(count)]
    rng.shuffle(targets)
    return targets


def passage_at(paragraphs, start, target):
    parts = []
    length = 0
    for paragraph in paragraphs[start:]:
        separator = "\n\n" if parts else ""
        if parts and length + len(separator) + len(paragraph) > target:
            break
        parts.append(paragraph)
        length += len(separator) + len(paragraph)
        if length >= target:
            break
    if not parts:
        return paragraphs[start][:target]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="../data/jkrowling.zip")
    ap.add_argument("--extra-epub-dir", default="../data/harry-potter-extra")
    ap.add_argument("--output", default="data/samples-large.jsonl")
    ap.add_argument("--samples-per-book", type=int, default=100)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=20260318)
    args = ap.parse_args()

    rows = []
    with zipfile.ZipFile(args.archive) as outer:
        files = outer.infolist()
        for lang, ids in BOOKS.items():
            for volume, book_id in enumerate(ids, 1):
                needle = f"({book_id})/"
                matches = [i for i in files if needle in i.filename and i.filename.lower().endswith(".epub")]
                if len(matches) != 1:
                    raise RuntimeError(f"expected one EPUB for {lang} book {volume} ({book_id}), got {len(matches)}")
                member = matches[0]
                paragraphs = epub_text(outer.read(member))
                seed_material = f"{args.seed}:{lang}:{volume}".encode()
                rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
                targets = varied_targets(args.samples_per_book, args.min_chars, args.max_chars, rng)
                starts = list(range(len(paragraphs)))
                rng.shuffle(starts)
                if len(starts) < args.samples_per_book:
                    starts = [rng.randrange(len(paragraphs)) for _ in range(args.samples_per_book)]
                for sample, (start, target) in enumerate(zip(starts, targets), 1):
                    text = passage_at(paragraphs, start, target)
                    rows.append({
                        "id": f"hpv2-{lang}-{volume}-{sample}",
                        "lang": lang,
                        "book": volume,
                        "sample": sample,
                        "source": member.filename,
                        "text": text,
                    })
                print(f"{lang} book {volume}: {len(paragraphs)} paragraphs, {args.samples_per_book} samples")

    extra_dir = Path(args.extra_epub_dir)
    for epub_path in sorted(extra_dir.glob("*-book-*.epub")):
        match = re.fullmatch(r"([a-z]{3})-book-(\d+)\.epub", epub_path.name)
        if not match:
            continue
        lang, volume_text = match.groups()
        volume = int(volume_text)
        paragraphs = epub_text(epub_path.read_bytes())
        if not paragraphs:
            raise RuntimeError(f"no usable paragraphs in {epub_path}")
        seed_material = f"{args.seed}:{lang}:{volume}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
        targets = varied_targets(args.samples_per_book, args.min_chars, args.max_chars, rng)
        starts = list(range(len(paragraphs)))
        rng.shuffle(starts)
        if len(starts) < args.samples_per_book:
            starts = [rng.randrange(len(paragraphs)) for _ in range(args.samples_per_book)]
        for sample, (start, target) in enumerate(zip(starts, targets), 1):
            text = passage_at(paragraphs, start, target)
            rows.append({
                "id": f"hpv3-{lang}-{volume}-{sample}",
                "lang": lang,
                "book": volume,
                "sample": sample,
                "source": str(epub_path),
                "text": text,
            })
        print(f"{lang} book {volume}: {len(paragraphs)} paragraphs, {args.samples_per_book} samples")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    chars = sum(len(r["text"]) for r in rows)
    print(f"Wrote {len(rows)} samples / {chars:,} characters to {out}")


if __name__ == "__main__":
    main()
