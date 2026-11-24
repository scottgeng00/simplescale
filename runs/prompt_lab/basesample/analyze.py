import json, re, sys
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "sweep.json"))
def punct(t, n=300):
    w = t.split()[-n:]; return sum(bool(re.search(r"[.!?;:]$", x)) for x in w) / max(1, len(w))
def loop(t):
    w = t.split()[-400:]; g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return bool(g) and 1 - len(set(g)) / len(g) > 0.3
thr = 0.02
print(f"{'config':24s} {'ended':>6s} {'loop':>5s} {'salad':>6s} {'clean':>6s} {'clean&ended':>11s} {'len p50':>7s}")
for c in dict.fromkeys(r["cfg"] for r in res):
    L = [r for r in res if r["cfg"] == c]; n = len(L)
    lp = [loop(r["text"]) for r in L]; sl = [len(r["text"].split()) > 150 and punct(r["text"]) < thr for r in L]
    clean = [not a and not b for a, b in zip(lp, sl)]
    t = sorted(r["tokens"] for r in L)
    print(f"{c:24s} {sum(r['end'] != 'length' for r in L)/n:6.0%} {sum(lp)/n:5.0%} {sum(sl)/n:6.0%} {sum(clean)/n:6.0%} "
          f"{sum(cl and r['end'] != 'length' for cl, r in zip(clean, L))/n:11.0%} {t[n // 2]:7d}")
