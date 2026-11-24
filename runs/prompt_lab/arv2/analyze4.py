import json, random, collections, re, sys
import numpy as np
exec(open("analyze_emb.py").read().split('print("real"')[0])
src_e = {r: E[by["real"][r][0]] for r in rows}
vs_cos = {j: float(E[j] @ src_e[M[j]["src_row"]]) for r in rows for j in by["arv2"][r]}
for kind in ["direct_nothink", "mix"]:
    for r in rows:
        for j in by[kind][r]: vs_cos[j] = float(E[j] @ src_e[r])
def bsel(kind, rule, B=256, reps=20, seed=0):
    rng = random.Random(seed); vs = []; cs = []
    for _ in range(reps):
        rs = rng.sample(rows, B); pick = [rule(by[kind][r], rng) for r in rs]
        vs.append(vendi(np.stack([E[j] for j in pick]))); cs.append(np.mean([vs_cos[j] for j in pick]))
    return round(float(np.mean(vs)), 1), round(float(np.mean(cs)), 3)
q = lambda lo, hi: (lambda c, rng: rng.choice(sorted(c, key=lambda j: vs_cos[j])[int(lo * len(c)):max(int(lo * len(c)) + 1, int(hi * len(c)))]))
for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.0)]:
    print(f"arv2 view-source-cos quartile {lo}-{hi}: (batch Vendi, mean view-src cos) =", bsel("arv2", q(lo, hi)))
print("arv2 random", bsel("arv2", lambda c, rng: rng.choice(c)))
for kind in ["direct_nothink", "mix"]:
    print(kind, "random", bsel(kind, lambda c, rng: rng.choice(c)), "lowest-quartile", bsel(kind, q(0, .25)))
