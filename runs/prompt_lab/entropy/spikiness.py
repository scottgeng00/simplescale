"""How surprise is distributed along human vs generated continuations (tokens_9b.json from run.py with DUMP)."""
import json, numpy as np
res = json.load(open("tokens_9b.json"))
conts = ["human", "gen9b_T1", "gen9b_T1p95", "gen9b_prod"]
arr = lambda d, c, k: np.array(d["scores"][f"9b|{c}"][k], float)
out = {}
for c in conts:
    S = {}
    add = lambda k, v: S.setdefault(k, []).append(v)
    for d in res:
        L = min(len(arr(d, "human", "nll")), len(arr(d, c, "nll")))
        if L < 100: continue
        x = arr(d, c, "nll")[:L]; h = arr(d, c, "hlo")[:L]
        s = np.sort(x)[::-1]; tot = s.sum()
        add("mean", x.mean()); add("p50", np.median(x)); add("p90", np.quantile(x, .9)); add("p99", np.quantile(x, .99))
        add("std", x.std()); add("cv", x.std() / x.mean())
        cs = np.cumsum(np.sort(x)); add("gini", 1 - 2 * cs.sum() / (L * cs[-1]) + 1 / L)
        add("top10share", s[:L // 10].sum() / tot); add("frac_lt0.1", (x < 0.1).mean()); add("frac_gt5", (x > 5).mean())
        xc = x - x.mean(); add("ac1", (xc[:-1] * xc[1:]).mean() / xc.var())
        hi = x > 5; add("burst", (hi[1:] & hi[:-1]).mean() / max(1e-9, hi.mean() ** 2))
        w = np.convolve(x, np.ones(32) / 32, "valid"); add("win32_cv", w.std() / w.mean())
        add("entropy", h.mean()); add("excess", (x - h).mean())
        q = L // 4; add("first_q", x[:q].mean()); add("last_q", x[-q:].mean())
    out[c] = {k: float(np.mean(v)) for k, v in S.items()} | {"n": len(S["mean"])}
keys = ["n", "mean", "p50", "p90", "p99", "frac_lt0.1", "frac_gt5", "std", "cv", "gini", "top10share", "ac1", "burst",
        "win32_cv", "entropy", "excess", "first_q", "last_q"]
print(f"{'metric':12s}" + "".join(f"{c:>13s}" for c in conts))
for k in keys: print(f"{k:12s}" + "".join(f"{out[c][k]:13.3f}" for c in conts))
json.dump(out, open("spikiness_9b.json", "w"), indent=1)
