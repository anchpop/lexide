"""Give English scripts the grams of a newer yap TTS dump (same text), e.g. after yap gained heteronym readings.

    python scripts/regram.py <yap tts rows .jsonl> data/scripts/eng_achernar.jsonl ...   # rewrites them in place
"""
import sys, json
pack = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line); pack[r["text"]] = r["grams"]
for path in sys.argv[2:]:
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    hit = changed = 0
    for r in rows:
        new = pack.get(r["text"])
        if new is not None:
            hit += 1; changed += new != r["grams"]; r["grams"] = new
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{path}: {len(rows)} rows, {hit} found in the dump, {changed} with new grams")
