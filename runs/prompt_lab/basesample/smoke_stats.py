"""Stats of a base_continue run: end types, filter drop rates, Qwen3 tokens per kept view (the export's tokenizer).
    .venv/bin/python runs/prompt_lab/basesample/smoke_stats.py RUN_DIR [--pairs OUT.jsonl]"""
import argparse, collections, glob, json, os
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--pairs", help="write (source text, samples) per document for reading")
a = ap.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer")
rows = [json.loads(l) for f in sorted(glob.glob(f"{a.run}/results/attempts/*.jsonl")) for l in open(f)]
seen, uniq = set(), []
for r in rows:
    k = (r["result"]["source"]["source_subset"], r["result"]["source"]["source_row"])
    if k not in seen:
        seen.add(k); uniq.append(r)
rows = uniq
S = [s for r in rows for s in r["result"]["samples"]]
n = len(S)
end = collections.Counter(s["end"] for s in S)
fr = collections.Counter(s["finish_reason"] for s in S)
kept = [s for s in S if s["finish_reason"] == "stop"]
lens = [len(x) + 1 for x in tok([s["text"] for s in kept], add_special_tokens=False)["input_ids"]]
lens.sort()
print(f"docs {len(rows)}  samples {n}  views/doc {n / len(rows):.2f}")
print("end:", {k: f"{v / n:.1%}" for k, v in end.items()})
print("finish:", {k: f"{v / n:.1%}" for k, v in fr.items()})
print("filtered among length-capped:", f"{sum(s['finish_reason'] != 'stop' for s in S if s['end'] == 'length') / max(1, end['length']):.1%}")
print(f"kept {len(kept)} ({len(kept) / n:.1%}); Qwen3 tokens/kept view mean {sum(lens) / len(lens):.0f} "
      f"p50 {lens[len(lens) // 2]} p90 {lens[int(len(lens) * .9)]}; per doc {sum(lens) / len(rows):.0f}")
print(f"source truncated: {sum(r['result']['source_truncated'] for r in rows) / len(rows):.1%}")
print(f"generated tokens (model tokenizer) per doc: {sum(r['result']['completion_tokens'] for r in rows) / len(rows):.0f}")
if a.pairs:
    man = json.load(open(f"{a.run}/results/dataset.json"))["manifest"] if os.path.exists(f"{a.run}/results/dataset.json") else None
    man = man or json.load(open(f"{a.run}/state/run.json"))["manifest"]
    refs = {}
    for l in open(man):
        t = json.loads(l); refs[(t["source_subset"], t["source_row"])] = t["_ref"]
    with open(a.pairs, "w") as out:
        for r in rows:
            src = r["result"]["source"]; ref = refs[(src["source_subset"], src["source_row"])]
            with open(ref["path"], "rb") as f:
                f.seek(ref["offset"]); text = json.loads(f.read(ref["length"]))["text"]
            out.write(json.dumps({"source": src, "source_text": text, "source_truncated": r["result"]["source_truncated"],
                                  "samples": r["result"]["samples"]}, ensure_ascii=False) + "\n")
