"""Pilot Vendi (Qwen3-Embedding-4B, lingua-clean/analysis/multisample_vendi.py's embed / vendi) of mix16 / mix16plan vs
the mix arm (runs/f200w-qwen35-9b-mix) and the real documents, on the 256 pilot documents.

  within: per document, Vendi of its kept views (export-eligible: finish "stop", non-empty); mix: its 8 views.
  across: Vendi of 256 documents (one random kept view each), mean over 10 draws; real = the sources.

    lingua-clean/.venv-kd/bin/python pilot_vendi.py --out pilot_vendi.json
"""
import argparse
import json
import random
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/home/sgeng/scottg/pretrain/lingua-clean/analysis")
from multisample_vendi import MODEL, embed, mean_cos, vendi  # noqa: E402

RUNS = "/home/sgeng/scottg/pretrain/simplescale/runs"
ROWS = [int(i * 1503775 / 256) for i in range(256)]  # pilot rows of the pool manifest (= f200w manifest rows)


def run_views(run):
    out = {}
    for s in json.load(open(f"{RUNS}/{run}/results/dataset.json"))["shards"]:
        for line in open(s["uri"]):
            r = json.loads(line)["result"]
            out[(r["source"]["source_subset"], r["source"]["source_row"])] = [
                x["text"].strip() for x in r["samples"] if x["finish_reason"] == "stop" and x["text"].strip()]
    return out


def mix_views(keys):
    ds = json.load(open(f"{RUNS}/f200w-qwen35-9b-mix/results/dataset.json"))
    shards = {s["chunk"]: s["uri"] for s in ds["shards"]}
    want, out = set(keys), {}
    for c in sorted({r // 64 for r in ROWS}):
        for line in open(shards[c]):
            r = json.loads(line)["result"]
            k = (r["source"]["source_subset"], r["source"]["source_row"])
            if k in want:
                out[k] = [x["text"].strip() for x in r["samples"] if x["finish_reason"] == "stop" and x["text"].strip()]
    return out


def sources(keys):
    want, out = set(keys), {}
    for i, line in enumerate(open(f"{RUNS}/manifests/dclm.pilot256.mix16.manifest.jsonl")):
        t = json.loads(line)
        ref = t["_ref"]
        with open(ref["path"], "rb") as f:
            f.seek(ref["offset"])
            out[(t["source_subset"], t["source_row"])] = json.loads(f.read(ref["length"]))["text"]
    return {k: v for k, v in out.items() if k in want}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {"mix16": run_views("pilot256-qwen35-9b-mix16"), "mix16plan": run_views("pilot256-qwen35-9b-mix16plan")}
    keys = sorted(set(arms["mix16"]) & set(arms["mix16plan"]))
    arms["mix"] = mix_views(keys)
    keys = [k for k in keys if k in arms["mix"] and all(len(arms[x][k]) >= 2 for x in arms)]
    src = sources(keys)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    res = {"docs": len(keys)}
    rng = random.Random(0)
    S, _ = embed([src[k] for k in keys], tok, model, a.max_tokens)
    res["real"] = {"across": vendi(S), "across_cos": mean_cos(S)}
    for arm, views in arms.items():
        flat = [(i, t) for i, k in enumerate(keys) for t in views[k]]
        E, _ = embed([t for _, t in flat], tok, model, a.max_tokens)
        idx = np.asarray([i for i, _ in flat])
        within = [vendi(E[idx == i]) for i in range(len(keys))]
        across = []
        for _ in range(a.draws):
            pick = [rng.choice(np.flatnonzero(idx == i).tolist()) for i in range(len(keys))]
            across.append(vendi(E[pick]))
        vs = [float(E[idx == i].mean(0) @ S[i] / np.linalg.norm(E[idx == i].mean(0))) for i in range(len(keys))]
        res[arm] = {"within": float(np.mean(within)), "views_per_doc": len(flat) / len(keys),
                    "across": float(np.mean(across)), "across_sd": float(np.std(across)),
                    "view_source_cos": float(np.mean([E[j] @ S[i] for j, i in enumerate(idx)]))}
        print(arm, res[arm], flush=True)
    print(res)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
