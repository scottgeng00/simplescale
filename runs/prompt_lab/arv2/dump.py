"""Reader packets: python dump.py rounds/r1/out.json rounds/r1/read 4"""
import json, sys, random, os
res = json.load(open(sys.argv[1])); outd = sys.argv[2]; k = int(sys.argv[3]); per = int(sys.argv[4]) if len(sys.argv) > 4 else 99; os.makedirs(outd, exist_ok=True)
srcs = {}
for l in open("data/audit/sources.jsonl"):
    d = json.loads(l); srcs[d["src_row"]] = d["text"]
for j in range(k):
    with open(f"{outd}/within_{j}.md", "w") as f:
        for d in res[j::k][:per]:
            s = " ".join(srcs[d["src_row"]].split(" ")[:3000])
            f.write(f"\n\n######## DOC src_row {d['src_row']}\n\n### SOURCE (full, up to 3000 words)\n{s}\n")
            for i, p in enumerate(d["prompts"]):
                f.write(f"\n### VIEW {i+1} (plan call {p.get('call', 0)+1}{', REJECTED: ' + p['reject'] if p.get('reject') else ''}): {p['title']}\n**Prompt:** {p['prompt']}\n\n**Output ({len(p['output'].split())} words):**\n{p['output']}\n")
rng = random.Random(1)
with open(f"{outd}/across.md", "w") as f:
    f.write("One randomly chosen view per source document (first 700 characters), documents in random order.\n")
    order = list(res); rng.shuffle(order)
    for d in order:
        p = rng.choice(d["prompts"])
        f.write(f"\n---\n[src_row {d['src_row']}] ({len(p['output'].split())} words)\n{p['output'][:700]}\n")
    f.write("\n\n=== All plan titles per doc ===\n")
    for d in res:
        f.write(f"\n[src_row {d['src_row']}] " + " | ".join(p["title"] for p in d["prompts"]) + "\n")
