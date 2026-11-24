"""Base-model 'challenging problem seeded by a real document' prompt iteration.
    python run.py <version> [n_docs] [areas_per_doc]   (reads template_<version>.txt, examples_<version>.json)"""
import sys, json, asyncio, aiohttp, random, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
url = open(f"{HERE}/../../interactive/sglang_qwen35_9b_base.addr").read().strip()
ver = sys.argv[1]; n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 6; per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TEMPLATE = open(f"{HERE}/template_{ver}.txt").read()
EX = json.load(open(f"{HERE}/examples_{ver}.json" if os.path.exists(f"{HERE}/examples_{ver}.json") else f"{HERE}/examples_v1.json"))
AREAS = ["combinatorics", "number theory", "probability", "geometry", "algebra / sequences", "optimization",
         "expected value", "graph theory", "game theory", "recurrences", "inequalities", "counting with symmetry"]
docs = [json.loads(l) for l in open("/tmp/sgeng-setup/audit/sources.jsonl")]
R = random.Random(int(os.environ.get("SEED", 11)))
picked = R.sample([d for d in docs if 120 <= len(d["text"].split()) <= 1500], n_docs)
def excerpt(t, n=220): w = t.split(); return " ".join(w[:n]) + (" ..." if len(w) > n else "")
def item(k, seed, area, problem=None, e=None):
    t = TEMPLATE.split("===ITEM===")[1].format(k=k, seed=seed, area=area).rstrip("\n")
    if e is None and "classical mathematics" not in TEMPLATE:
        return t
    e = e or {}
    if "The Riddler" in TEMPLATE:
        return t if problem is None else f"{t}\n\n{problem}\n"
    if "Problems from Everywhere" in TEMPLATE and "classical mathematics" in TEMPLATE:  # v3: candidates, classic, idea
        cands = e.get("cands")
        c = (f"\n*Candidate classics:* {cands}" if cands else "")
        if problem is None:
            return t + c + "\n*Classic used:*"
        return f"{t}{c}\n*Classic used:* {e['classic']}\n*Idea:* {e['idea']}\n*Difficulty:* {e['difficulty']}\n\n{problem}\n"
    if "*Idea:*" in t:  # v2+: idea line, difficulty, then the problem
        return f"{t} {e['idea']}\n*Difficulty:* {e['difficulty']}\n\n{problem}\n"
    return t + "\n\n" + problem + "\n"
def prompt(doc, area, rng):
    head = TEMPLATE.split("===ITEM===")[0]
    ex = [dict(e) for e in rng.sample(EX, 3)]; k0 = rng.randrange(5, 40)
    CL = json.load(open(f"{HERE}/classics.json")); tgt = None
    if os.environ.get("CANDS"):
        for e in ex:
            o = rng.sample([c for c in CL if c != e["classic"]], 2) + [e["classic"]]; rng.shuffle(o); e["cands"] = "; ".join(o)
        tgt = {"cands": "; ".join(rng.sample(CL, 3))}
    if "The Riddler" in TEMPLATE:
        NAMES = ["Tomás from Austin", "Aiko from Osaka", "Grace from Dublin", "Sam from Nairobi", "Ines from Lisbon", "Owen from Halifax", "Mei from Singapore", "Jonas from Oslo"]
        parts = [head] + [item(e["name"], e["seed"], "", e["problem"], e) for e in ex] + [item(rng.choice(NAMES), excerpt(doc), "")]
        return "\n".join(parts)
    parts = [head] + [item(k0 + i, e["seed"], e.get("area",""), e["problem"], e) for i, e in enumerate(ex)]
    parts.append(item(k0 + 3, excerpt(doc), area, None, tgt) if "classical mathematics" in TEMPLATE else item(k0 + 3, excerpt(doc), area))
    return "\n".join(parts)
async def one(s, p):
    body = {"prompt": p, "temperature": float(os.environ.get("T", 0.9)), "top_p": 0.95, "max_tokens": 450, "frequency_penalty": float(os.environ.get("FP", 0.3)),
            "stop": ["\n## Riddler", "\nFrom ", "\n### ", "\n**Problem", "<|endoftext|>", "\nSolution", "\n*Seed", "\nSeed", "\n*Solution"]}
    async with s.post(f"{url}/v1/completions", json=body) as r: o = await r.json()
    return o["choices"][0]["text"].strip(), o["choices"][0].get("finish_reason")
async def main():
    jobs = [(d, a, prompt(d["text"], a, random.Random(d["src_row"] * 7 + j))) for d in picked for j, a in enumerate(R.sample(AREAS, per))]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        outs = await asyncio.gather(*[one(s, p) for _, _, p in jobs])
    res = [{"src_row": d["src_row"], "area": a, "prompt": p, "problem": o, "finish": f} for (d, a, p), (o, f) in zip(jobs, outs)]
    json.dump(res, open(f"{HERE}/out_{ver}.json", "w"), indent=1, ensure_ascii=False)
    if os.environ.get("SHOW_PROMPT"): print(res[0]["prompt"], "\n=====")
    last = None
    for r in res:
        if r["src_row"] != last:
            d = next(x for x in picked if x["src_row"] == r["src_row"]); print(f"\n######## DOC {r['src_row']}: {excerpt(d['text'], 60)}")
            last = r["src_row"]
        print(f"--- [{r['area']}] ({r['finish']})\n{r['problem']}")
asyncio.run(main())
