"""Exec-only A/B of execution wrappers on a fixed sample of r4 prompts."""
import asyncio, json, random, re, sys, glob, itertools, collections
import aiohttp
import arv2_plan as P
res = json.load(open("rounds/r4/out.json"))
srcs = {json.loads(l)["src_row"]: " ".join(json.loads(l)["text"].split(" ")[:3000]) for l in open("data/audit/sources.jsonl")}
items = [(d["src_row"], p["prompt"]) for d in res for p in d["prompts"]]
random.Random(0).shuffle(items); items = items[:int(sys.argv[1])]
RULES = """- Include only what this page needs for its focus; leave out the rest.
- Keep every fact, number, date, name, attribution, and hedge ("may", "probably", "some") as given. Add no facts, numbers, dates, names, quotations, causes, or attributions, and do not turn relative times into dates.
- Plain text as it appears when copied from a web page: no Markdown (#, **, *, pipe tables), no HTML, no genre label as a heading, no sign-offs or placeholders.
- Use your own words, quoting only short phrases; avoid stock phrases such as "Ultimately", "a testament to", and "In conclusion"; stay close to the requested length."""
SAN = [(re.compile(r",?\s*(?:as )?(?:stated|noted|mentioned|described|given|found|presented|listed|quoted|outlined|explained) (?:in|by) (?:the|this) (?:provided |original |given |source )?(?:text|source text|source material|source document|source|document|passage|material|article)\b", re.I), ""),
       (re.compile(r"\b(?:according to|within) (?:the|this) (?:provided |original |given |source )?(?:text|source text|source material|source document|source|document|passage|material)\b,?\s*", re.I), ""),
       (re.compile(r"\b(?:the|this) (?:provided |original |given )?(?:source text|source material|source document|source|text|document|passage)('s)?\b", re.I), lambda m: "its" if m.group(1) else "it")]
def sanitize(p):
    for r, s in SAN: p = r.sub(s, p)
    return p
V = {
 "A_r4": lambda t, p: P.EXEC.replace("{text}", t).replace("{prompt}", p),
 "B_r1style": lambda t, p: f"Source document:\n<document>\n{t}\n</document>\n\n{p}\n\nHow to write it:\n{RULES}\n\nWrite the document as its author would publish it on the web. Do not mention a source document, these instructions, or that the text is a rewrite.",
 "C_r1style_sanitized": lambda t, p: f"Source document:\n<document>\n{t}\n</document>\n\n{sanitize(p)}\n\nHow to write it:\n{RULES}\n\nWrite the document as its author would publish it on the web. Do not mention a source document, these instructions, or that the text is a rewrite.",
 "D_author_notes": lambda t, p: f"You are writing a web page. Everything you know about the topic is in these notes:\n<notes>\n{t}\n</notes>\n\n{sanitize(p)}\n\nHow to write it:\n{RULES}\n- Write as the page's author, stating the information directly or attributing it to the people and organizations involved.",
}
_rr = itertools.count()
def url():
    fs = sorted(glob.glob("sglang_*.addr")); return open(fs[next(_rr) % len(fs)]).read().strip()
sem = asyncio.Semaphore(96)
async def chat(s, msg):
    body = {"messages": [{"role": "user", "content": msg}], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
            "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(100):
        try:
            async with sem:
                async with s.post(f"{url()}/v1/chat/completions", json=body) as r: o = await r.json()
            return o["choices"][0]["message"]["content"] or ""
        except Exception as e:
            await asyncio.sleep(20)
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        out = {}
        for k, f in V.items():
            outs = await asyncio.gather(*[chat(s, f(srcs[r], p)) for r, p in items])
            rej = collections.Counter(P.reject_reason(P.clean_markdown(o), srcs[r]) for (r, _), o in zip(items, outs))
            wc = sorted(len(o.split()) for o in outs)
            print(k, dict(rej), "median words", wc[len(wc)//2], flush=True)
            out[k] = outs
    json.dump({"items": items, "outs": out}, open("rounds/r4/wrapper_ab.json", "w"))
asyncio.run(main())
