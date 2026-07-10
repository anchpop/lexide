"""Blind lemma judge: ask a fresh claude-fable-5 for each token's lemma, compare to Gemma.

The judge sees only (language, sentence, word, POS) — never Gemma's proposed lemma — so it
answers independently. We then bucket each item as agree / defensible / gemma-disagrees.
"""
import json
import os
import urllib.request
import urllib.error
from collections import defaultdict

KEY = open("/tmp/claude-1000/-data-coding-lexide-tagging/68294016-c301-4f43-9008-4a9d1852a080/scratchpad/akey").read().strip()
ITEMS = json.load(open("/tmp/claude-1000/-data-coding-lexide-tagging/68294016-c301-4f43-9008-4a9d1852a080/scratchpad/judge_items.json"))
MODEL = "claude-fable-5"
LANG_NAME = {"deu":"German","eng":"English","fra":"French","hin":"Hindi","ita":"Italian",
             "jpn":"Japanese","kor":"Korean","por":"Portuguese","rus":"Russian","spa":"Spanish"}

SYSTEM = (
    "You are an expert multilingual linguistic annotator. For each item you get a language, "
    "a sentence, one token (word) from that sentence, and its universal POS tag. Output the "
    "LEMMA (dictionary / citation form) of that token as used in that sentence, following "
    "standard lemmatization conventions (verbs -> infinitive/dictionary form, nouns -> "
    "singular/base, adjectives -> base, etc.). If more than one lemma is genuinely defensible, "
    "list the alternates in also_acceptable. Return ONLY a JSON array, one object per item, in "
    'the same order and with the same i values: '
    '[{"i": <int>, "lemma": "<string>", "also_acceptable": ["<string>", ...]}]. No prose.'
)


def call(items):
    """items: list of dicts with i, lang, sentence, form, pos -> parsed JSON list from Fable."""
    user_payload = [
        {"i": it["i"], "language": LANG_NAME[it["lang"]], "sentence": it["sentence"],
         "word": it["form"], "pos": it["pos"]}
        for it in items
    ]
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM,
        "output_config": {"effort": "low"},  # simple task; keep Fable fast/cheap
        "messages": [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    if resp.get("stop_reason") == "refusal":
        raise RuntimeError("refusal")
    text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip() if "```" in text[3:] else text.strip("`")
    return json.loads(text)


def main():
    # tag items with a stable index and group by language (10 each) -> 10 blind calls
    for idx, it in enumerate(ITEMS):
        it["i"] = idx
    by_lang = defaultdict(list)
    for it in ITEMS:
        by_lang[it["lang"]].append(it)

    verdicts = {}
    for lang, items in by_lang.items():
        try:
            out = call(items)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as e:
            print(f"[{lang}] call failed: {e}")
            continue
        by_i = {o["i"]: o for o in out if "i" in o}
        for it in items:
            j = by_i.get(it["i"])
            if not j:
                continue
            judge_lemma = (j.get("lemma") or "").strip()
            alts = [a.strip() for a in (j.get("also_acceptable") or [])]
            gem = it["gemma_lemma"]
            if gem == judge_lemma:
                bucket = "agree"
            elif gem in alts:
                bucket = "defensible"
            else:
                bucket = "gemma_disagrees"
            verdicts[it["i"]] = {**it, "judge_lemma": judge_lemma, "judge_alts": alts, "bucket": bucket}
        print(f"[{lang}] judged {sum(1 for v in verdicts.values() if v['lang']==lang)}/{len(items)}")

    # summarize
    per = defaultdict(lambda: defaultdict(int))
    for v in verdicts.values():
        per[v["lang"]][v["bucket"]] += 1
        per["ALL"][v["bucket"]] += 1
    print("\n=== blind-judge results (Fable vs Gemma silver) ===")
    print(f"{'lang':4} {'agree':>6} {'defensible':>11} {'GEMMA-WRONG':>12} {'n':>4}")
    for lang in list(LANG_NAME) + ["ALL"]:
        c = per.get(lang)
        if not c:
            continue
        n = sum(c.values())
        print(f"{lang:4} {c['agree']:>6} {c['defensible']:>11} {c['gemma_disagrees']:>12} {n:>4}")

    print("\n=== cases where Fable disagrees with Gemma (potential silver errors) ===")
    for v in verdicts.values():
        if v["bucket"] == "gemma_disagrees":
            print(f"  [{v['lang']}] {v['form']!r} (POS {v['pos']}) in {v['sentence']!r}")
            print(f"       gemma={v['gemma_lemma']!r}  fable={v['judge_lemma']!r}  alts={v['judge_alts']}")

    out_path = "/tmp/claude-1000/-data-coding-lexide-tagging/68294016-c301-4f43-9008-4a9d1852a080/scratchpad/judge_verdicts.json"
    json.dump(list(verdicts.values()), open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
