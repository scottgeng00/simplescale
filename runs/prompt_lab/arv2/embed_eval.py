"""Within-doc Vendi, view-source cosine, cross-doc cosine for arv2 vs existing arms on the 16 audit docs."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/sgeng/scottg/pretrain/lingua-clean/analysis")
import multisample_vendi as mv
from transformers import AutoModel, AutoTokenizer
res = json.load(open("out.json")); rows = [d["src_row"] for d in res]
src = {json.loads(l)["src_row"]: json.loads(l)["text"] for l in open("data/sources.jsonl") if json.loads(l)["src_row"] in rows}
arms = {"arv2": {d["src_row"]: [p["output"] for p in d["prompts"]][:8] for d in res}}
for a in ["direct_nothink", "direct_think", "mix", "ideas"]:
    arms[a] = {r: [] for r in rows}
    for l in open(f"data/{a}_all.jsonl"):
        x = json.loads(l)
        if x["src_row"] in arms[a]: arms[a][x["src_row"]].append(x["text"])
keys = [("real", r, 0) for r in rows] + [(a, r, i) for a in arms for r in rows for i in range(len(arms[a][r]))]
texts = [src[r] if a == "real" else arms[a][r][i] for a, r, i in keys]
tok = AutoTokenizer.from_pretrained(mv.MODEL)
model = AutoModel.from_pretrained(mv.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
E, _ = mv.embed(texts, tok, model, 4096)
at = {k: i for i, k in enumerate(keys)}
out = {}
for a in arms:
    W = [np.stack([E[at[a, r, i]] for i in range(len(arms[a][r]))]) for r in rows]
    within = [mv.vendi(x) for x in W]
    vs = [float(E[at[a, r, i]] @ E[at["real", r, 0]]) for r in rows for i in range(len(arms[a][r]))]
    allv = np.concatenate(W); lab = np.concatenate([[j] * len(x) for j, x in enumerate(W)])
    K = allv @ allv.T; m = lab[:, None] != lab[None, :]
    out[a] = {"within_vendi": float(np.mean(within)), "view_source_cos": float(np.mean(vs)), "cross_doc_cos": float(K[m].mean()),
              "pooled_vendi_all_views": mv.vendi(allv), "n": int(len(allv))}
    print(a, json.dumps(out[a]), flush=True)
R = np.stack([E[at["real", r, 0]] for r in rows]); K = R @ R.T
print("real cross_doc_cos", float((K.sum() - np.trace(K)) / (len(R) * (len(R) - 1))))
json.dump(out, open("embed_eval.json", "w"), indent=1)
