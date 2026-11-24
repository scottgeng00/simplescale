"""Human vs base-model text under the base models themselves (Qwen3.5-9B-Base / 4B-Base).

32 random pool docs (>= 288 tokens); prefix = first 32 tokens; human continuation = next min(len-32, 1024) tokens.
Each generator model continues the prefix for the same number of tokens under three settings:
  T1      T=1.0 top_p=1.0
  T1p95   T=1.0 top_p=0.95
  prod    T=1.0 top_p=0.95 presence_penalty=0.5 (basegen production)
Every continuation (and the human one) is scored teacher-forced by every scorer model: per-token NLL of the realized
token (exact), next-token entropy from top-100 (lower bound: top-100 part + -t log t of tail mass t; upper bound:
tail spread uniformly over the remaining V-100 tokens), argmax match, share of NLL > 5 nats. Compared on the same
positions (first min(len_human, len_gen) continuation tokens).

Env: GEN=9b,4b  SCORE=9b,4b  OUT=result.json  NDOCS=32  SEED=0  CONC=4 (per-server concurrent requests)
Server address files are re-read on every request (servers auto-requeue)."""
import asyncio, json, math, os, random
import aiohttp, numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ADDR = {m: f"/home/sgeng/scottg/pretrain/simplescale/runs/interactive/sglang_qwen35_{m}_base.addr" for m in ["9b", "4b"]}
ADDR.update({k[5:].lower(): v for k, v in os.environ.items() if k.startswith("ADDR_")})  # e.g. ADDR_9B=/path/file
GEN = os.environ.get("GEN", "9b,4b").split(",")
SCORE = os.environ.get("SCORE", "9b,4b").split(",")
OUT = os.environ.get("OUT", os.path.join(HERE, "result.json"))
NDOCS, SEED, CONC = int(os.environ.get("NDOCS", 32)), int(os.environ.get("SEED", 0)), int(os.environ.get("CONC", 4))
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

tok = AutoTokenizer.from_pretrained("/home/sgeng/scottg/pretrain/assets/models/Qwen3.5-9B-Base")
V, K, P, CAP, MIN = len(tok), 100, 32, 1024, 288
SETTINGS = {"T1": {"temperature": 1.0, "top_p": 1.0},
            "T1p95": {"temperature": 1.0, "top_p": 0.95},
            "prod": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.5}}
MAN = "/home/sgeng/scottg/pretrain/simplescale/runs/manifests/dclm.f200w.ideas.manifest.jsonl"


def sample_docs(k, seed):
    """Random manifest lines (byte-offset seek + resync to next line), keep docs with >= MIN tokens."""
    rng, size, out, seen = random.Random(seed), os.path.getsize(MAN), [], set()
    with open(MAN, "rb") as f:
        while len(out) < k:
            f.seek(rng.randrange(size)); f.readline(); line = f.readline()
            if not line: continue
            ref = json.loads(line)["_ref"]
            key = (ref["path"], ref["offset"])
            if key in seen: continue
            seen.add(key)
            fd = os.open(ref["path"], os.O_RDONLY)
            try: text = json.loads(os.pread(fd, ref["length"], ref["offset"]))["text"]
            finally: os.close(fd)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= MIN: out.append({"ref": ref, "ids": ids})
    return out


SEMS = {}


async def post(s, model, body):
    """POST /generate with retries; re-reads the address file each attempt (server may restart elsewhere)."""
    async with SEMS[model]:
        for attempt in range(200):
            try:
                url = open(ADDR[model]).read().strip()
                async with s.post(f"{url}/generate", json=body) as r:
                    if r.status == 200: return await r.json()
                    msg = (await r.text())[:200]
            except Exception as e:
                msg = repr(e)[:200]
            print(f"[{model}] retry {attempt}: {msg}", flush=True)
            await asyncio.sleep(min(60, 5 * (attempt + 1)))
        raise RuntimeError(f"{model} server unavailable")


async def gen(s, model, prefix, n, sp):
    o = await post(s, model, {"input_ids": prefix, "sampling_params": {**sp, "max_new_tokens": n}})
    ids = o["output_ids"][:n]
    eos = o["meta_info"]["finish_reason"]["type"] != "length"
    if ids and ids[-1] in (tok.eos_token_id, tok.pad_token_id) and eos: ids = ids[:-1]  # drop EOS token if included
    return ids, o["meta_info"]["finish_reason"]["type"]


async def score(s, model, seq):
    # logprob_start_len=P-1: sglang returns null for the token at index start_len, so starting one earlier gives
    # real logprobs for every continuation token (index P onward).
    body = {"input_ids": seq, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": P - 1, "top_logprobs_num": K}
    m = (await post(s, model, body))["meta_info"]
    lps, tops = m["input_token_logprobs"][1:], m["input_top_logprobs"][1:]
    assert len(lps) == len(seq) - P, (len(lps), len(seq))
    nll, hlo, hhi, am = [], [], [], []
    for (lp, tid, _), top in zip(lps, tops):
        tl = np.array([t[0] for t in top], dtype=np.float64); ps = np.exp(tl)
        tail = min(1.0, max(0.0, 1 - ps.sum()))
        h = float(-(ps * tl).sum())
        nll.append(-lp)
        hlo.append(h + (-tail * math.log(tail) if tail > 0 else 0.0))
        hhi.append(h + (tail * math.log((V - K) / tail) if tail > 0 else 0.0))
        am.append(float(lp >= tl.max() - 1e-6))
    return {"nll": nll, "hlo": hlo, "hhi": hhi, "am": am}


async def do_doc(s, d):
    x = d["ids"]; prefix, human = x[:P], x[P:P + CAP]
    conts = {"human": (human, "human")}
    jobs = [(g, st) for g in GEN for st in SETTINGS]
    outs = await asyncio.gather(*[gen(s, g, prefix, len(human), SETTINGS[st]) for g, st in jobs])
    for (g, st), o in zip(jobs, outs): conts[f"gen{g}_{st}"] = o
    keys = [(sc, c) for sc in SCORE for c in conts]
    scs = await asyncio.gather(*[score(s, sc, prefix + conts[c][0]) for sc, c in keys])
    return {"ref": d["ref"], "len_human": len(human),
            "conts": {c: {"len": len(v[0]), "finish": v[1], "ids": v[0]} for c, v in conts.items()},
            "scores": {f"{sc}|{c}": r for (sc, c), r in zip(keys, scs)}}


def summarize(res):
    rows = {}
    for sc in SCORE:
        for g in GEN:
            for st in SETTINGS:
                c = f"gen{g}_{st}"
                a = {k: [] for k in ["h_nll", "g_nll", "h_hlo", "g_hlo", "h_hhi", "g_hhi", "h_am", "g_am", "h_hi5", "g_hi5", "L"]}
                gl, early = [], 0
                for r in res:
                    L = min(r["len_human"], r["conts"][c]["len"])
                    gl.append(r["conts"][c]["len"]); early += r["conts"][c]["len"] < r["len_human"]
                    if L == 0: continue
                    H, G = r["scores"][f"{sc}|human"], r["scores"][f"{sc}|{c}"]
                    a["L"].append(L)
                    for p, S in [("h", H), ("g", G)]:
                        nll = np.array(S["nll"][:L])
                        a[f"{p}_nll"].append(nll.mean()); a[f"{p}_hi5"].append((nll > 5).mean())
                        a[f"{p}_hlo"].append(np.mean(S["hlo"][:L])); a[f"{p}_hhi"].append(np.mean(S["hhi"][:L]))
                        a[f"{p}_am"].append(np.mean(S["am"][:L]))
                row = {k: float(np.mean(v)) for k, v in a.items()}  # doc-averaged (each doc's per-token mean)
                row["frac_docs_gen_nll_lower"] = float(np.mean(np.array(a["g_nll"]) < np.array(a["h_nll"])))
                row["n_docs_early_stop"] = int(early); row["gen_len_mean"] = float(np.mean(gl))
                row["human_len_mean"] = float(np.mean([r["len_human"] for r in res]))
                rows[f"score{sc}|gen{g}_{st}"] = row
    return rows


async def main():
    docs = sample_docs(NDOCS, SEED)
    print(f"{len(docs)} docs, human cont lens mean {np.mean([min(len(d['ids']) - P, CAP) for d in docs]):.0f}", flush=True)
    for m in set(GEN) | set(SCORE): SEMS[m] = asyncio.Semaphore(CONC)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), trust_env=False) as s:
        res = []
        for i, d in enumerate(docs):  # doc at a time; requests within a doc run concurrently (bounded per server)
            res.append(await do_doc(s, d)); print(f"doc {i + 1}/{len(docs)} done", flush=True)
            json.dump({"partial": True, "per_doc": res}, open(OUT + ".partial", "w"))
    rows = summarize(res)
    if os.environ.get("DUMP"):  # per-token arrays (nll, entropy bounds, argmax) for every continuation
        json.dump(res, open(os.environ["DUMP"], "w"))
    print(json.dumps(rows, indent=1))
    per_doc = [{"ref": r["ref"], "len_human": r["len_human"],
                "conts": {c: {k: v[k] for k in ["len", "finish"]} for c, v in r["conts"].items()},
                "means": {k: {m: float(np.mean(v[m])) if v[m] else None for m in v} for k, v in r["scores"].items()}}
               for r in res]
    json.dump({"config": {"GEN": GEN, "SCORE": SCORE, "NDOCS": NDOCS, "SEED": SEED, "P": P, "CAP": CAP, "K": K, "V": V,
                          "settings": SETTINGS}, "rows": rows, "per_doc": per_doc}, open(OUT, "w"), indent=1)
    os.remove(OUT + ".partial")


asyncio.run(main())
