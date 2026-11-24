"""Stats of a prompt_pool run (mix16 / mix16plan), optionally its code-checked copy.

    /home/sgeng/scottg/pretrain/lingua-clean/.venv-kd/bin/python mix16_stats.py RUN_DIR [--max-shards N] [--out stats.json]

Per type: views, finish reasons (stop / length / rejected:*), generated tokens per view (reasoning included), kept-view
Qwen3 tokens (+1 EOS, as the export counts them); code checks by language; planner status / score / selection
distributions; views per doc; projected kept tokens for the 1,503,775-doc pool.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

POOL_DOCS = 1_503_775
TOKENIZER = "/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--max-shards", type=int)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out")
a = ap.parse_args()
ds = json.load(open(f"{a.run}/results/dataset.json"))
shards = ds["shards"]
if a.max_shards and len(shards) > a.max_shards:
    shards = random.Random(a.seed).sample(shards, a.max_shards)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOKENIZER)
finish = defaultdict(Counter)
gen_tokens, kept_tokens = defaultdict(list), defaultdict(list)
code = Counter()
planner_status, scores, chosen = Counter(), defaultdict(Counter), Counter()
views_per_doc, kept_per_doc = Counter(), Counter()
docs = 0
kept_texts = []
for s in shards:
    for line in open(s["uri"]):
        r = json.loads(line)["result"]
        docs += 1
        views_per_doc[len(r["samples"])] += 1
        chosen.update(r.get("types", []))
        if "planner" in r:
            p = r["planner"]
            planner_status[p["status"]] += 1
            for k, v in (p["scores"] or {}).items():
                scores[k][v] += 1
        k_doc = 0
        for x in r["samples"]:
            kind = x["kind"]
            fr = x["finish_reason"]
            if fr == "stop" and not x["text"]:
                fr = "empty"
            finish[kind][fr] += 1
            gen_tokens[kind].append(x.get("tokens", 0))
            if "code_check" in x:
                c = x["code_check"]
                key = (kind, x["slots"]["language"])
                code[key + (c["status"],)] += 1
                if kind == "code_problem":
                    code[key + ("io_extracted",)] += bool(c.get("has_input") and c.get("has_output"))
                    if "match" in c:
                        code[key + (f"match_{c['match']}",)] += 1
            if fr == "stop":
                k_doc += 1
                kept_texts.append((kind, x["text"]))
        kept_per_doc[k_doc] += 1
ids = tok([t for _, t in kept_texts], add_special_tokens=False)["input_ids"] if kept_texts else []
for (kind, _), x in zip(kept_texts, ids):
    kept_tokens[kind].append(len(x) + 1)

types = sorted(finish, key=lambda k: -sum(finish[k].values()))
total_views = sum(sum(c.values()) for c in finish.values())
kept_total = sum(sum(v) for v in kept_tokens.values())
out = {"run": a.run, "docs": docs, "views": total_views, "views_per_doc": dict(views_per_doc),
       "kept_views_per_doc": dict(sorted(kept_per_doc.items())),
       "kept_tokens_per_doc": kept_total / max(1, docs),
       "projected_pool_kept_tokens": kept_total / max(1, docs) * POOL_DOCS,
       "types": {}, "planner_status": dict(planner_status),
       "planner_scores": {k: dict(sorted(scores[k].items())) for k in scores},
       "chosen": dict(chosen.most_common()),
       "code": {"|".join(k): v for k, v in sorted(code.items())}}
for k in types:
    n = sum(finish[k].values())
    out["types"][k] = {"views": n, "finish": dict(finish[k].most_common()),
                       "kept_frac": finish[k]["stop"] / n,
                       "gen_tokens_mean": sum(gen_tokens[k]) / n,
                       "kept_tokens_mean": sum(kept_tokens[k]) / max(1, len(kept_tokens[k]))}
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
print(f"{a.run}: {docs} docs, {total_views} views, kept tokens/doc {out['kept_tokens_per_doc']:.0f}, "
      f"projected pool kept tokens {out['projected_pool_kept_tokens'] / 1e9:.2f}B")
print(f"{'type':15s} {'views':>6s} {'kept':>6s} {'gen_tok':>8s} {'kept_tok':>8s}  finish")
for k, v in out["types"].items():
    print(f"{k:15s} {v['views']:6d} {v['kept_frac']:6.1%} {v['gen_tokens_mean']:8.0f} {v['kept_tokens_mean']:8.0f}  "
          + ", ".join(f"{r}:{c}" for r, c in v["finish"].items()))
if planner_status:
    print("planner:", dict(planner_status))
    print(f"{'type':15s} {'s0':>5s} {'s1':>5s} {'s2':>5s} {'chosen':>7s}")
    for k in out["types"]:
        c = scores[k]
        print(f"{k:15s} {c[0]:5d} {c[1]:5d} {c[2]:5d} {chosen[k]:7d}")
if code:
    print("code checks:")
    for k, v in sorted(code.items()):
        print("  ", "|".join(k), v)
