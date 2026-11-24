"""Embedding diversity for prompt-lab rounds vs the existing arms on the same docs.
python eval_emb.py OUT.json [OUT2.json ...] --save X.json
Per arm: within-doc Vendi of 8 views (first 8), view-source cosine, cross-doc cosine (views of different docs),
batch Vendi: one random view per doc, batches of min(256, ndocs) docs, mean over 20 batches (real = sources)."""
import argparse, json, sys, random
import numpy as np, torch
sys.path.insert(0, "/home/sgeng/scottg/pretrain/lingua-clean/analysis")
import multisample_vendi as mv
from transformers import AutoModel, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("outs", nargs="+"); ap.add_argument("--save", required=True)
ap.add_argument("--arms", default="direct_nothink,direct_think,mix,ideas"); ap.add_argument("--max-tokens", type=int, default=4096)
a = ap.parse_args()
runs = {f: {d["src_row"]: [p["output"] for p in d["prompts"] if not p.get("reject")] for d in json.load(open(f))} for f in a.outs}
rows = sorted(set.intersection(*[set(r) for r in runs.values()]))
rows = [r for r in rows if all(len(v[r]) >= 8 for v in runs.values())]
want = set(rows)
src = {}
for l in open("data/audit/sources.jsonl"):
    x = json.loads(l)
    if x["src_row"] in want: src[x["src_row"]] = x["text"]
arms = {f: {r: runs[f][r] for r in rows} for f in a.outs}  # within: first 8 views; batch: any view
for arm in [x for x in a.arms.split(",") if x]:
    arms[arm] = {r: [] for r in rows}
    for l in open(f"data/audit/{arm}_all.jsonl"):
        x = json.loads(l)
        if x["src_row"] in want: arms[arm][x["src_row"]].append(x["text"])
keys = [("real", r, 0) for r in rows] + [(k, r, i) for k in arms for r in rows for i in range(len(arms[k][r]))]
texts = [src[r] if k == "real" else arms[k][r][i] for k, r, i in keys]
tok = AutoTokenizer.from_pretrained(mv.MODEL)
model = AutoModel.from_pretrained(mv.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
E, _ = mv.embed(texts, tok, model, a.max_tokens)
at = {k: i for i, k in enumerate(keys)}
B = min(256, len(rows)); out = {"ndocs": len(rows), "batch": B}
def batch_vendi(pick):
    rng = random.Random(0); vs = []
    for _ in range(20):
        rs = rng.sample(rows, B); vs.append(mv.vendi(np.stack([E[at[pick(r, rng)]] for r in rs])))
    return float(np.mean(vs))
R = np.stack([E[at["real", r, 0]] for r in rows]); K = R @ R.T
out["real"] = {"cross_doc_cos": float((K.sum() - np.trace(K)) / (len(R) * (len(R) - 1))), "batch_vendi": batch_vendi(lambda r, g: ("real", r, 0))}
print("real", out["real"], flush=True)
for k in arms:
    rk = [r for r in rows if arms[k][r]]
    W = [np.stack([E[at[k, r, i]] for i in range(min(8, len(arms[k][r])))]) for r in rk]
    vs = [float(E[at[k, r, i]] @ E[at["real", r, 0]]) for r in rk for i in range(len(arms[k][r]))]
    allv = np.concatenate(W); lab = np.concatenate([[j] * len(x) for j, x in enumerate(W)])
    Kk = allv @ allv.T; m = lab[:, None] != lab[None, :]
    out[k] = {"within_vendi": float(np.mean([mv.vendi(x) for x in W])), "view_source_cos": float(np.mean(vs)),
              "cross_doc_cos": float(Kk[m].mean()), "batch_vendi": batch_vendi(lambda r, g: (k, r, g.randrange(len(arms[k][r])))) if len(rk) == len(rows) else None, "n": int(len(allv))}
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open(a.save, "w"), indent=1)
