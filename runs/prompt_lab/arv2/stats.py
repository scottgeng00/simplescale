"""Quick stats for a round: python stats.py rounds/r1/out.json [more...]"""
import json, re, sys, collections, statistics
EXLEAK = ["Dovetail", "QuickCache", "Crumbline", "Harwell", "Lune Brook", "Amos Pell", "Brennan", "Orchard Street",
          "tool-lending", "operating costs", "early blight", "concentric rings", "hydration", "origin reads", "successful write",
          "invalidates", "public subscription", "412", "Porter"]
SRCREF = re.compile(r"\b(the|this) (provided|given|original|source|above|supplied) (text|document|material|source|article|passage)|source (document|text|material)|provided (text|document|article|material)", re.I)
PLANSRC = re.compile(r"\bthe source\b|provided (text|document|source)|\bthe document\b", re.I)
META = re.compile(r"\b(DATE|FROM|TO|SUBJECT|CC|RE):|\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, (19|20)\d\d|\b\d{1,2}:\d\d ?(AM|PM|am|pm)\b|\[(Your|Name|Date|Insert)[^\]]*\]")
INVENT_PLAN = re.compile(r"\b(recently (visited|examined|tried|attended|discovered|read)|my (own )?experience|personal (experience|anecdote|story)|announces? new|interview(ed|s)? with)\b", re.I)
srcs = {}
for l in open("data/audit/sources.jsonl"):
    d = json.loads(l); srcs[d["src_row"]] = d["text"]
for f in sys.argv[1:]:
    res = json.load(open(f))
    outs = [(d["src_row"], p) for d in res for p in d["prompts"]]
    n = len(outs); N = lambda c: f"{c}/{n} ({100*c/max(n,1):.0f}%)"
    raw = [p.get("raw", p["output"]) for _, p in outs]; out = [p["output"] for _, p in outs]
    wc = [len(o.split()) for o in out]
    q = statistics.quantiles(wc, n=10)
    print(f"== {f}: {len(res)} docs, {n} views, prompts/doc {collections.Counter(len(d['prompts']) for d in res)}")
    print("  plan finish", collections.Counter(d.get("plan_finish") for d in res), "plan tokens median", statistics.median([d.get("plan_tokens", 0) for d in res]))
    print("  exec finish", collections.Counter(p["finish"] for _, p in outs), "exec tokens mean", round(statistics.mean([p.get("tokens", 0) for _, p in outs])))
    print(f"  words: median {statistics.median(wc):.0f} p10 {q[0]:.0f} p90 {q[8]:.0f} mean {statistics.mean(wc):.0f}; <150w {N(sum(w<150 for w in wc))}; >900w {N(sum(w>900 for w in wc))}")
    print("  raw md title (first line #/**)", N(sum(bool(re.match(r"\s*(#|\*\*)", r)) for r in raw)), "| cleaned has # or **", N(sum(bool(re.search(r"^#|\*\*", o, re.M)) for o in out)),
          "| md table", N(sum("|---" in o or "| ---" in o for o in out)), "| html", N(sum(bool(re.search(r"<(html|div|p|body|h1)\b", o)) for o in out)))
    print("  'Based on' openings", N(sum(o.lstrip().lower().startswith("based on") for o in out)), "| source refs in output", N(sum(bool(SRCREF.search(o)) for o in out)),
          "| metadata patterns", N(sum(bool(META.search(o)) for o in out)))
    leaks = [(r, k) for r, p in outs for k in EXLEAK if k in p["prompt"] + p["output"] and k not in srcs[r]]
    print("  example leaks (prompt or output, absent from source)", len(leaks), collections.Counter(k for _, k in leaks).most_common(8))
    print("  plans mentioning 'the source/document/provided'", N(sum(bool(PLANSRC.search(p["prompt"])) for _, p in outs)), "| plan invented-experience cues", N(sum(bool(INVENT_PLAN.search(p["prompt"])) for _, p in outs)))
    op = collections.Counter(" ".join(p["prompt"].split()[:2]) for _, p in outs)
    print("  plan openers top", op.most_common(6), "distinct", len(op))
    fl = collections.Counter(re.sub(r"\W+", " ", o.split("\n")[0]).strip().split(" ")[0].lower() if o else "" for o in out)
    print("  output first words top", fl.most_common(8))
