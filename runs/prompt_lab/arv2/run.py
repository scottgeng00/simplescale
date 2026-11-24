"""Active-reading v2 pilot: stage 1 plan_prompt.txt (instruct, thinking on) writes n document-specific prompts;
stage 2 executes each with the source (instruct, thinking off). Docs = the 16 'within' docs of the data audit."""
import asyncio, json, re, sys
import aiohttp
url = open("../../interactive/sglang_qwen35_9b.addr").read().strip()
PLAN = open("plan_prompt.txt").read()
rows = [int(x) for x in re.findall(r"======== DOC \d+ \(src_row (\d+)\)", open("/tmp/sgeng-setup/audit/direct_nothink_within.md").read())]
src = {}
for l in open("/tmp/sgeng-setup/audit/sources.jsonl"):
    d = json.loads(l)
    if d["src_row"] in rows: src[d["src_row"]] = " ".join(d["text"].split(" ")[:3000])
EXEC = "Source document:\n<document>\n{text}\n</document>\n\n{prompt}"
HEAD = re.compile(r"^##(?!#)\s*(.*)$", re.M)
async def chat(s, msg, think, max_tokens):
    body = {"messages": [{"role": "user", "content": msg}], "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    async with s.post(f"{url}/v1/chat/completions", json=body) as r: o = await r.json()
    c = o["choices"][0]["message"].get("content") or ""
    if think:
        m = re.search(r"</think>", c); c = c[m.end():] if m else ""
    return c.strip(), o["choices"][0].get("finish_reason")
def split(reply):
    hs = list(HEAD.finditer(reply))
    return [(h.group(1).strip(), reply[h.end():(hs[i + 1].start() if i + 1 < len(hs) else len(reply))].strip()) for i, h in enumerate(hs)]
async def doc(s, r):
    plan, fin = await chat(s, PLAN.replace("{n}", "8").replace("{text}", src[r]), True, 16384)
    prompts = split(plan)
    outs = await asyncio.gather(*[chat(s, EXEC.replace("{text}", src[r]).replace("{prompt}", p), False, 4096) for _, p in prompts])
    return {"src_row": r, "plan_finish": fin, "prompts": [{"title": t, "prompt": p, "output": o, "finish": f} for (t, p), (o, f) in zip(prompts, outs)]}
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5400)) as s:
        res = await asyncio.gather(*[doc(s, r) for r in rows])
    json.dump(res, open("out.json", "w"), indent=1, ensure_ascii=False)
    for d in res: print(d["src_row"], d["plan_finish"], len(d["prompts"]), [p["title"][:40] for p in d["prompts"]])
asyncio.run(main())
