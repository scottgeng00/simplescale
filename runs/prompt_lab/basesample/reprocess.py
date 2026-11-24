"""Re-apply base_continue post-processing (current harness) to a run's samples offline; stats on Qwen3 tokens.
    .venv/bin/python runs/prompt_lab/basesample/reprocess.py PAIRS.jsonl [length_keep]"""
import collections, json, os, sys
sys.path.insert(0, "src")
from simplescale.harnesses.multisample import _base_sample, clean_cut
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/sgeng/scottg/pretrain/assets/models/Qwen3-0.6B-Base-tokenizer")
keep = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
rows = [json.loads(l) for l in open(sys.argv[1])]
fr, kept, n = collections.Counter(), [], 0
for r in rows:
    src = r["source_text"].strip()
    for s in r["samples"]:
        reason = {"type": "length"} if s["end"] == "length" else {"type": "stop", "matched": "x" if s["end"] == "header" else 248044}
        o = _base_sample({"text": s["text"], "meta_info": {"finish_reason": reason}}, src, keep)
        n += 1
        fr[(s["end"], o["finish_reason"])] += 1
        if o["finish_reason"] == "stop":
            kept.append(o["text"])
lens = sorted(len(x) + 1 for x in tok(kept, add_special_tokens=False)["input_ids"])
tot = collections.Counter()
for (e, f), c in fr.items():
    tot[f] += c
print(f"length_keep {keep}: samples {n}, kept {len(kept)} ({len(kept)/n:.1%})")
print("finish:", {k: f"{v/n:.1%}" for k, v in sorted(tot.items())})
print("by end:", {f"{e}/{f}": c for (e, f), c in sorted(fr.items())})
print(f"Qwen3 tokens per kept view mean {sum(lens)/len(lens):.0f} p50 {lens[len(lens)//2]}; per generated view {sum(lens)/n:.0f}; per doc {sum(lens)/len(rows):.0f}")
