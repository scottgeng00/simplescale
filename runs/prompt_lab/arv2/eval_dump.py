"""Embed every accepted view of a round + its sources (+ optional arm) and save for offline analysis.
python eval_dump.py rounds/r5/out.json rounds/r5/emb_all.npz"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/sgeng/scottg/pretrain/lingua-clean/analysis")
import multisample_vendi as mv
from transformers import AutoModel, AutoTokenizer
res = json.load(open(sys.argv[1]))
rows = [d["src_row"] for d in res]; want = set(rows)
src = {}
for l in open("data/audit/sources.jsonl"):
    x = json.loads(l)
    if x["src_row"] in want: src[x["src_row"]] = x["text"]
meta, texts = [], []
for d in res:
    meta.append({"kind": "real", "src_row": d["src_row"]}); texts.append(src[d["src_row"]])
    for i, p in enumerate(d["prompts"]):
        if p.get("reject"): continue
        meta.append({"kind": "arv2", "src_row": d["src_row"], "i": i, "title": p["title"], "words": len(p["output"].split())}); texts.append(p["output"])
for arm in ["direct_nothink", "mix"]:
    for l in open(f"data/audit/{arm}_all.jsonl"):
        x = json.loads(l)
        if x["src_row"] in want:
            meta.append({"kind": arm, "src_row": x["src_row"], "i": x.get("view", 0), "title": x.get("kind", ""), "words": len(x["text"].split())}); texts.append(x["text"])
tok = AutoTokenizer.from_pretrained(mv.MODEL)
model = AutoModel.from_pretrained(mv.MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
E, _ = mv.embed(texts, tok, model, 4096)
np.savez(sys.argv[2], E=E.astype(np.float16), meta=np.array([json.dumps(m) for m in meta]))
print("saved", len(texts))
