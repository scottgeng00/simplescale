"""Manifests for later rounds of the prompt_pool arms (mix16 / mix16plan), from the round-1 pool manifest.

  round 2, mix16:      every source, the 8 types round 1 did not draw (complement_seed 0; no round-1 output needed)
  round 2, mix16plan:  every source, the types with planner score > 0 that round 1 did not draw (round-1 scores and
                       types read from --runs; no new planner call; sources with none are skipped)
  top-up (round 3):    --pairs fresh (source, type) draws: source uniform over the pool, type uniform (mix16) or from
                       the source's planner weights 0/1/3 (mix16plan, with replacement across rounds); grouped per
                       source as explicit ``kinds``, sources in manifest order

Every task gets ``round`` and ``seed`` = round - 1 (fresh slots / samples per round).

    .venv/bin/python scripts/build_mix16_rounds.py --arm mix16 --round 2
    .venv/bin/python scripts/build_mix16_rounds.py --arm mix16plan --round 2 --runs runs/pool025-qwen35-9b-mix16plan
    .venv/bin/python scripts/build_mix16_rounds.py --arm mix16 --round 3 --pairs 1000000 --runs ...
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from simplescale.harnesses import mix16

MANIFESTS = "/home/sgeng/scottg/pretrain/simplescale/runs/manifests"


def round1(runs):
    """{(subset, row): (types used so far, planner scores or None)} from the given runs' results."""
    out = {}
    for run in runs:
        for s in json.load(open(f"{run}/results/dataset.json"))["shards"]:
            for line in open(s["uri"]):
                r = json.loads(line)["result"]
                k = (r["source"]["source_subset"], r["source"]["source_row"])
                used, scores = out.get(k, ([], None))
                out[k] = (used + r["types"], (r.get("planner") or {}).get("scores") or scores)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["mix16", "mix16plan"], required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="earlier rounds' run dirs (mix16plan, top-ups)")
    ap.add_argument("--pairs", type=int, help="top-up: number of (source, type) draws")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = f"{MANIFESTS}/dclm.pool025.{a.arm}.manifest.jsonl"
    tag = "pool025_r2" if a.round == 2 else f"pool025_topup{a.round}"
    out = f"{MANIFESTS}/dclm.{tag}.{a.arm}.manifest.jsonl"
    prev = round1(a.runs) if a.runs else None
    lines = open(src).readlines()
    rows = [json.loads(x) for x in lines]
    kinds = {}
    if a.round == 2 and a.arm == "mix16":
        pass
    elif a.round == 2:
        for i, t in enumerate(rows):
            used, scores = prev[(t["source_subset"], t["source_row"])]
            elig = [k for k in mix16.TYPES if (scores is None or scores[k] > 0) and k not in used]
            if elig:
                kinds[i] = elig
    else:
        rng = random.Random(f"topup-{a.round}-{a.seed}")
        picked = defaultdict(list)
        for _ in range(a.pairs):
            i = rng.randrange(len(rows))
            if a.arm == "mix16":
                picked[i].append(rng.choice(mix16.TYPES))
            else:
                scores = prev[(rows[i]["source_subset"], rows[i]["source_row"])][1]
                w = [1 if scores is None else mix16.SCORE_WEIGHT[scores[k]] for k in mix16.TYPES]
                picked[i].append(rng.choices(mix16.TYPES, weights=w)[0])
        kinds = dict(picked)
    n_views = Counter()
    with open(out + ".tmp", "w") as f:
        for i, t in enumerate(rows):
            t = {**t, "round": a.round, "seed": a.round - 1}
            if a.round == 2 and a.arm == "mix16":
                t["complement_seed"] = 0
                n_views.update(["(complement)"] * 8)
            elif i in kinds:
                t["kinds"] = kinds[i]
                n_views.update(kinds[i])
            else:
                continue
            f.write(json.dumps(t, separators=(",", ":")) + "\n")
    import os
    os.replace(out + ".tmp", out)
    tasks = len(rows) if a.round == 2 and a.arm == "mix16" else len(kinds)
    print(f"{out}: {tasks} tasks, {sum(n_views.values())} views; by type {dict(n_views.most_common())}")


if __name__ == "__main__":
    main()
