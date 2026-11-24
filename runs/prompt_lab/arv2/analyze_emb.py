import json, random, collections, re, sys
import numpy as np
d = np.load(sys.argv[1]); E = d["E"].astype(np.float32); M = [json.loads(x) for x in d["meta"]]
def vendi(X):
    K = X @ X.T / len(X); w = np.linalg.eigvalsh(K); w = w[w > 1e-12]; return float(np.exp(-(w * np.log(w)).sum()))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for j, m in enumerate(M): by[m["kind"]][m["src_row"]].append(j)
rows = sorted(by["real"])
def bv(kind, filt=lambda m: True, B=256, reps=20, seed=0):
    rng = random.Random(seed); vs = []
    cand = {r: [j for j in by[kind][r] if filt(M[j])] for r in rows}
    ok = [r for r in rows if cand[r]]
    if len(ok) < B: return None, len(ok)
    for _ in range(reps):
        rs = rng.sample(ok, B); vs.append(vendi(np.stack([E[rng.choice(cand[r])] for r in rs])))
    return round(float(np.mean(vs)), 1), len(ok)
print("real", bv("real")); print("direct", bv("direct_nothink")); print("mix", bv("mix")); print("arv2 all", bv("arv2"))
for lo, hi in [(0, 150), (150, 300), (300, 600), (600, 5000)]:
    print(f"arv2 words {lo}-{hi}", bv("arv2", lambda m: lo <= m["words"] < hi))
print("arv2 first 8 views", bv("arv2", lambda m: m["i"] < 8))
print("arv2 views >= 24 (later call)", bv("arv2", lambda m: m["i"] >= 24, B=96))
for kw in ["forum", "blog", "news", "faq|q&a|question", "guide|how-to", "encyclopedia|entry|glossary", "review", "listicle|list"]:
    print("arv2 title~", kw, bv("arv2", lambda m: re.search(kw, m["title"].lower()) is not None, B=128))
print("arv2 excluding forum/Q&A/comment/social", bv("arv2", lambda m: not re.search(r"forum|thread|reply|q&a|comment|social|post", m["title"].lower())))
# top principal directions of arv2 batch: which titles load most
X = np.stack([E[j] for r in rows for j in by["arv2"][r]]); idx = [j for r in rows for j in by["arv2"][r]]
mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
R = np.stack([E[by["real"][r][0]] for r in rows]); Ur, Sr, _ = np.linalg.svd(R - R.mean(0), full_matrices=False)
print("top-10 sing. values^2 share arv2", np.round((S[:10]**2 / (S**2).sum()), 3)); print("real", np.round((Sr[:10]**2 / (Sr**2).sum()), 3))
for k in range(4):
    sc = (X - mu) @ Vt[k]; top = np.argsort(-sc)[:6]; bot = np.argsort(sc)[:6]
    print(f"PC{k} +:", [M[idx[t]]["title"][:50] for t in top]); print(f"PC{k} -:", [M[idx[t]]["title"][:50] for t in bot])
