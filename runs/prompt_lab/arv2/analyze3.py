import json, random, collections, re, sys
import numpy as np
exec(open("analyze_emb.py").read().split('print("real"')[0])
CAS = re.compile(r"forum|thread|reply|comment|social|post\b|question posted|mailing", re.I)
def bvf(weight, B=256, reps=20, seed=0):
    rng = random.Random(seed); vs = []
    for _ in range(reps):
        rs = rng.sample(rows, B); pick = []
        for r in rs:
            c = by["arv2"][r]; w = [weight(M[j]) for j in c]
            if sum(w) == 0: w = [1] * len(c)
            pick.append(rng.choices(c, weights=w)[0])
        vs.append(vendi(np.stack([E[j] for j in pick])))
    return round(float(np.mean(vs)), 1)
print("uniform", bvf(lambda m: 1))
print("no <150w", bvf(lambda m: m["words"] >= 150))
print("no <250w", bvf(lambda m: m["words"] >= 250))
print("casual x0.3", bvf(lambda m: 0.3 if CAS.search(m["title"]) else 1))
print("casual x0", bvf(lambda m: 0 if CAS.search(m["title"]) else 1))
print("no <150w & casual x0.3", bvf(lambda m: (m["words"] >= 150) * (0.3 if CAS.search(m["title"]) else 1)))
print("token-weighted & casual x0.3", bvf(lambda m: m["words"] * (0.3 if CAS.search(m["title"]) else 1)))
print("share casual titles", round(sum(1 for r in rows for j in by['arv2'][r] if CAS.search(M[j]['title'])) / sum(len(by['arv2'][r]) for r in rows), 3))
