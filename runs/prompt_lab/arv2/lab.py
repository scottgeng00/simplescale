"""Prompt-lab round: python lab.py --tag r1 --prompt plan_prompt_r1.txt --examples examples_r1.txt --docs 48 --n 8
Docs: the 16 pilot docs + random audit docs (seed 0). Writes rounds/<tag>/out.json."""
import argparse, asyncio, json, os, random, re, time
import aiohttp
import arv2_plan as P
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--prompt", required=True); ap.add_argument("--examples", required=True)
ap.add_argument("--docs", type=int, default=48); ap.add_argument("--n", type=int, default=8); ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--skip", type=int, default=0, help="skip first k docs of the doc order (fresh docs)")
ap.add_argument("--calls", type=int, default=1)
ap.add_argument("--exec", default=None, help="exec wrapper file with {text} {prompt}")
a = ap.parse_args()
P.load(a.prompt)
EXAMPLES = [e.strip() for e in open(a.examples).read().split("=====") if e.strip()]
EXEC = open(a.exec).read() if a.exec else P.EXEC
pilot = [d["src_row"] for d in json.load(open("out.json"))]
allsrc = [json.loads(l) for l in open("data/audit/sources.jsonl")]
rng0 = random.Random(0); others = [d["src_row"] for d in allsrc if d["src_row"] not in pilot]; rng0.shuffle(others)
order = pilot + others
rows = order[a.skip:a.skip + a.docs]
src = {d["src_row"]: " ".join(d["text"].split(" ")[:3000]) for d in allsrc if d["src_row"] in set(rows)}
HEAD = re.compile(r"^##(?!#)\s*(.*)$", re.M)
sem = asyncio.Semaphore(a.conc)
import glob, itertools
_rr = itertools.count()
def url():
    files = sorted(glob.glob("sglang_*.addr")) or ["../../interactive/sglang_qwen35_9b.addr"]
    return open(files[next(_rr) % len(files)]).read().strip()
async def chat(s, msg, think, max_tokens):
    body = {"messages": [{"role": "user", "content": msg}], "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    for attempt in range(1000):
        try:
            async with sem:
                async with s.post(f"{url()}/v1/chat/completions", json=body) as r: o = await r.json()
            break
        except Exception as e:
            print("retry", e, flush=True); await asyncio.sleep(30)
    c = o["choices"][0]["message"].get("content") or ""
    if think:
        m = re.search(r"</think>", c); c = c[m.end():] if m else ""
    return c.strip(), o["choices"][0].get("finish_reason"), o.get("usage", {}).get("completion_tokens", 0)
def split(reply):
    hs = list(HEAD.finditer(reply))
    return [(h.group(1).strip(), reply[h.end():(hs[i + 1].start() if i + 1 < len(hs) else len(reply))].strip()) for i, h in enumerate(hs)]
async def doc(s, r):
    rng = random.Random(f"{a.tag}-{r}")
    plans, prompts, tasks, calls_of = [], [], [], []
    if a.n > 0:
        sizes = [a.n] * a.calls
    else:  # source-scaled view count, split into calls of at most P.PER_CALL
        total = P.n_views(src[r]); k = -(-total // P.PER_CALL)
        sizes = [total // k + (i < total % k) for i in range(k)]
    for c, n in enumerate(sizes):
        msg = P.build_plan(src[r], n, rng, EXAMPLES, prev=prompts or None)
        plan, fin, ptok = await chat(s, msg, True, 16384)
        new = [x for x in split(plan) if not P.bad_prompt(x[1], src[r])][:n] if hasattr(P, "bad_prompt") else split(plan)[:n]
        plans.append({"msg": msg, "finish": fin, "tokens": ptok, "plan": plan, "n": len(new)})
        calls_of += [c] * len(new)
        tasks += [asyncio.ensure_future(chat(s, P.exec_message(src[r], p) if hasattr(P, "exec_message") else EXEC.replace("{text}", src[r]).replace("{prompt}", p), False, 4096)) for _, p in new]
        prompts += new
    outs = await asyncio.gather(*tasks)
    return {"src_row": r, "plan_msg": plans[0]["msg"], "plan_finish": plans[0]["finish"], "plan_tokens": sum(x["tokens"] for x in plans), "plans": plans,
            "prompts": [{"title": t, "prompt": p, "raw": o, "output": P.finish_view(o, src[r])[0], "finish": f, "tokens": k, "call": c, "reject": P.finish_view(o, src[r])[1]} for (t, p), (o, f, k), c in zip(prompts, outs, calls_of)]}
async def main():
    t = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        res = await asyncio.gather(*[doc(s, r) for r in rows])
    os.makedirs(f"rounds/{a.tag}", exist_ok=True)
    json.dump(res, open(f"rounds/{a.tag}/out.json", "w"), indent=1, ensure_ascii=False)
    print("done", len(res), "docs", round(time.time() - t), "s")
asyncio.run(main())
