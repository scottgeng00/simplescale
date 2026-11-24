"""arv2 on the reference documents of lingua-clean/analysis/multisample_vendi.py (seed 0: 48 chunks, 256 groups,
5 batches of 256). Reuses that script's saved embeddings for real/other arms; embeds arv2 (first 8 kept views per
group doc; batch docs: kept view index pick % n_kept)."""
import json, random, sys
import numpy as np, torch
sys.path.insert(0, "/home/sgeng/scottg/pretrain/lingua-clean/analysis")
import multisample_vendi as mv
from transformers import AutoModel, AutoTokenizer
RUN = "/home/sgeng/scottg/pretrain/simplescale/runs/pool025-qwen35-9b-arv2"
rng = random.Random(0)
chunks = rng.sample(range(mv.POOL_DOCS // mv.CHUNK), 48)
views = {arm: mv.load_chunks(arm, chunks) for arm in mv.ARMS}
docs = sorted(set.intersection(*[{r for r, v in views[arm].items() if len(v) == 8 and all(v)} for arm in mv.ARMS]))
groups = rng.sample(docs, 256)
batches = [rng.sample(docs, 256) for _ in range(5)]
pick = {(b, r): rng.randrange(8) for b, rows in enumerate(batches) for r in rows}
shards = {s["chunk"]: s["uri"] for s in json.load(open(f"{RUN}/results/dataset.json"))["shards"]}
arv2 = {}
for c in chunks:
    for cc in (2 * c, 2 * c + 1):
        for l in open(shards[cc]):
            r = json.loads(l)["result"]
            arv2[r["source"]["source_row"]] = [s["text"] for s in r["samples"] if s["finish_reason"] == "stop" and s["text"]]
missing = [r for r in docs if r not in arv2]
print(len(docs), "docs;", len(missing), "missing from arv2", flush=True)
ref = np.load("/home/sgeng/scottg/pretrain/lingua-clean/analysis/multisample_vendi.npz")
RE = ref["E"].astype(np.float32); at = {tuple(k.split("|")): i for i, k in enumerate(ref["keys"])}
def refv(arm, r, v): return RE[at[(arm, str(r), str(v))]]
keys = sorted({(r, i) for r in groups for i in range(min(8, len(arv2[r])))} | {(r, pick[b, r] % len(arv2[r])) for b, rows in enumerate(batches) for r in rows})
tok = AutoTokenizer.from_pretrained(mv.MODEL)
model = AutoModel.from_pretrained(mv.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
E, lens = mv.embed([arv2[r][i] for r, i in keys], tok, model, 4096)
ka = {k: j for j, k in enumerate(keys)}
w = [np.stack([E[ka[r, i]] for i in range(min(8, len(arv2[r])))]) for r in groups]
acr = [np.stack([E[ka[r, pick[b, r] % len(arv2[r])]] for r in rows]) for b, rows in enumerate(batches)]
res = {"arv2": {"within_vendi": float(np.mean([mv.vendi(x) for x in w])), "across_vendi": float(np.mean([mv.vendi(x) for x in acr])),
                "across_vendi_sem": float(np.std([mv.vendi(x) for x in acr]) / np.sqrt(5)), "across_cos": float(np.mean([mv.mean_cos(x) for x in acr])),
                "mean_tokens": float(np.mean(lens)),
                "view_source_cos": float(np.mean([E[ka[r, pick[b, r] % len(arv2[r])]] @ refv("real", r, -1) for b, rows in enumerate(batches) for r in rows]))}}
for arm in mv.ARMS + ["real"]:
    acr2 = [np.stack([refv(arm, r, -1 if arm == "real" else pick[b, r]) for r in rows]) for b, rows in enumerate(batches)]
    res[arm] = {"across_vendi_recomputed": float(np.mean([mv.vendi(x) for x in acr2])), "across_cos": float(np.mean([mv.mean_cos(x) for x in acr2]))}
print(json.dumps(res, indent=1))
json.dump(res, open("arv2_vendi_ref.json", "w"), indent=1)
