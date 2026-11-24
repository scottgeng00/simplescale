"""Simplescale arv2 run -> lab out.json: python from_run.py RUN_DIR OUT.json
Reads results/dataset.json shards if finished, else every attempt file in results/attempts."""
import json, sys, glob, os
run, out = sys.argv[1], sys.argv[2]
ds = f"{run}/results/dataset.json"
files = [s["uri"] for s in json.load(open(ds))["shards"]] if os.path.exists(ds) else sorted(glob.glob(f"{run}/results/attempts/*.jsonl"))
res, seen = [], set()
for f in files:
    for l in open(f):
        r = json.loads(l)["result"]
        src = r["source"]; key = (src.get("source_subset"), src.get("source_row"))
        if key in seen: continue
        seen.add(key)
        ps = []
        for (p, s) in zip(r["prompts"], r["samples"]):
            fr = s["finish_reason"]
            ps.append({"title": p["title"], "prompt": p["prompt"], "output": s["text"], "raw": s["text"], "finish": fr,
                       "reject": fr.split(":", 1)[1] if fr and fr.startswith("rejected:") else None, "tokens": 0})
        res.append({"src_row": src["source_row"], "plan_tokens": r["completion_tokens"], "plans": [{"plan": x} for x in r["strategies"]],
                    "plan_finish": "stop", "completion_tokens": r["completion_tokens"], "prompts": ps})
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print(len(res), "docs", sum(len(d["prompts"]) for d in res), "views")
