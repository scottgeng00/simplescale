import json, random, collections, re, sys
import numpy as np
exec(open("analyze_emb.py").read().split('print("real"')[0])
def bvw(kind, B=256, reps=20, seed=0, power=1.0, filt=lambda m: True):
    rng = random.Random(seed); vs = []
    for _ in range(reps):
        rs = rng.sample(rows, B); pick = []
        for r in rs:
            c = [j for j in by[kind][r] if filt(M[j])] or by[kind][r]
            w = [max(1, M[j].get("words", 1)) ** power for j in c]
            pick.append(rng.choices(c, weights=w)[0])
        vs.append(vendi(np.stack([E[j] for j in pick])))
    return round(float(np.mean(vs)), 1)
print("token-weighted batch Vendi: arv2", bvw("arv2"), "direct", bvw("direct_nothink"), "mix", bvw("mix"))
print("B=96: real", bv("real", B=96)[0], "arv2 all", bv("arv2", B=96)[0], "arv2 i<24", bv("arv2", lambda m: m["i"] < 24, B=96)[0], "arv2 i>=24", bv("arv2", lambda m: m["i"] >= 24, B=96)[0])
print("B=128: real", bv("real", B=128)[0], "arv2 >=300w", bv("arv2", lambda m: m["words"] >= 300, B=128)[0], "arv2 600+", bv("arv2", lambda m: m["words"] >= 600, B=128)[0], "direct", bv("direct_nothink", B=128)[0])
idx = [j for r in rows for j in by["arv2"][r]]; X = E[idx]; mu = X.mean(0)
U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); sc = (X - mu) @ Vt[0]
wds = np.array([M[j]["words"] for j in idx]); print("corr(PC0, log words)", round(float(np.corrcoef(sc, np.log(wds))[0, 1]), 3), "corr(PC1, log words)", round(float(np.corrcoef((X-mu)@Vt[1], np.log(wds))[0, 1]), 3))
# real docs projected on arv2 PC0
R = E[[by["real"][r][0] for r in rows]]; print("real mean proj on PC0", round(float(((R - mu) @ Vt[0]).mean()), 3), "arv2 PC0 std", round(float(sc.std()), 3))
