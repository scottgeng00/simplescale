"""Base-model sampling regime: '## Document 1\n\n{doc}\n\n## Document 2\n\n' -> sample. Sweep decoding params and
measure whether generations end on their own (EOS or the next '## Document' header) before max_new_tokens,
their length, and degeneration (share of repeated 4-grams)."""
import asyncio, json, random, re, sys
import aiohttp
url = open("../../interactive/sglang_qwen35_9b_base.addr").read().strip()
docs = [json.loads(l)["text"] for l in open("/tmp/sgeng-setup/audit/sources.jsonl")]
R = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
picked = [" ".join(d.split()[:600]) for d in R.sample([d for d in docs if len(d.split()) >= 150], 48)]
PROMPT = "## Document 1\n\n{doc}\n\n## Document 2\n\n"
STOP = ["\n## Document 3", "## Document 3", "\n## Document"]
CONFIGS = {
    "T1.0_p0.95":             dict(temperature=1.0, top_p=0.95),
    "T1.0_p0.95_pp0.3":       dict(temperature=1.0, top_p=0.95, presence_penalty=0.3),
    "T1.0_p0.95_pp0.5":       dict(temperature=1.0, top_p=0.95, presence_penalty=0.5),
    "T1.0_p0.95_pp0.8":       dict(temperature=1.0, top_p=0.95, presence_penalty=0.8),
    "T1.0_p0.95_rep1.05":     dict(temperature=1.0, top_p=0.95, repetition_penalty=1.05),
    "T1.0_p0.95_pp0.3_rep1.03": dict(temperature=1.0, top_p=0.95, presence_penalty=0.3, repetition_penalty=1.03),
}
MAX = 4096
SEM = asyncio.Semaphore(48)
def rep4(t):
    w = t.split(); g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0
async def one(s, cfg, doc, seed):
    body = {"text": PROMPT.format(doc=doc), "sampling_params": {**CONFIGS[cfg], "max_new_tokens": MAX, "stop": STOP, "sampling_seed": seed}}
    async with SEM:
        async with s.post(f"{url}/generate", json=body) as r: o = await r.json()
    fr = o["meta_info"]["finish_reason"]; n = o["meta_info"]["completion_tokens"]
    how = "length" if fr.get("type") == "length" else ("eos" if fr.get("matched") in (None, 248044) or isinstance(fr.get("matched"), int) else "header")
    return {"cfg": cfg, "text": o["text"], "tokens": n, "end": how, "rep4": rep4(o["text"])}
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        res = await asyncio.gather(*[one(s, c, d, i) for c in CONFIGS for i, d in enumerate(picked)])
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "sweep.json", "w"), indent=1, ensure_ascii=False)
    print(f"{'config':22s} {'ended':>6s} {'eos':>5s} {'hdr':>5s} {'len p50':>8s} {'p90':>6s} {'rep4 mean':>9s} {'rep4>0.3':>8s}")
    for c in CONFIGS:
        L = [r for r in res if r["cfg"] == c]; n = len(L)
        t = sorted(r["tokens"] for r in L)
        print(f"{c:22s} {sum(r['end'] != 'length' for r in L)/n:6.0%} {sum(r['end'] == 'eos' for r in L)/n:5.0%} {sum(r['end'] == 'header' for r in L)/n:5.0%} "
              f"{t[n // 2]:8d} {t[int(n * .9)]:6d} {sum(r['rep4'] for r in L)/n:9.3f} {sum(r['rep4'] > 0.3 for r in L)/n:8.0%}")
asyncio.run(main())
