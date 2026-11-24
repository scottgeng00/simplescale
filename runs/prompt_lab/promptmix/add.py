"""Append catalog entries: python add.py < entries.json  (a JSON list)."""
import json, sys
rows = json.load(sys.stdin)
have = {json.loads(l)["id"] for l in open("catalog.jsonl")}
with open("catalog.jsonl", "a") as f:
    for r in rows:
        r.setdefault("verbatim", True)
        if r["id"] in have:
            print("skip dup", r["id"]); continue
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(sum(1 for _ in open("catalog.jsonl")), "entries")
