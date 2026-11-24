"""3C comparison of synthetic-data methods x generator size -> 3c_methods.html (dot-plot small multiples)."""
import json
G = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/multisample_dc_w1152"
S = 24588
K7 = ["arc_easy:rc::olmes:full", "arc_challenge:rc::olmes:full", "hellaswag:rc::olmes:full", "piqa:rc::olmes:full",
      "winogrande:rc::olmes:full", "sciq:rc::olmo3", "mmlu:rc::olmes"]
def metrics(run):
    v = json.load(open(f"{G}/{run}/evals/{S:010d}/validation.json"))
    d = [json.loads(l) for l in open(f"{G}/{run}/metrics.olmes_eval.jsonl") if json.loads(l).get("global_step") == S][-1]
    g = lambda k: d[f"olmes_eval/{k}/primary_score"]
    return {"avg7": sum(g(k) for k in K7) / 7, "gsm8k": g("gsm8k::olmo3:n32:pass32"), "mathbpb": g("olmo3:base_easy:math_bpb"),
            "dclm": v["dclm_heldout"]["bpb"], "paloma": v["paloma_qwen3/mean"]["bpb"]}
lex = json.load(open("/home/sgeng/scottg/pretrain/lingua-clean/analysis/lexical_diversity.json"))
lexpct = lambda k: 100 * (lex[k]["batch"]["types"] / lex["real (sources)"]["batch"]["types"] - 1)
VENDI = {"direct_nothink_q2b": 146.6, "direct_nothink": 147.0, "direct_nothink_q27b": 146.0, "mix_q2b": 136.9, "mix": 142.0,
         "mix_q27b": 142.8, "arv2": 125.9}
ROWS = [  # (group, label, size, run, vendi key, lexical key)
    ("direct (wiki rewrite)", "direct · 2B", "2B", "r4s_direct_nothink_q2b_3x", "direct_nothink_q2b", "direct_nothink (2B)"),
    ("direct (wiki rewrite)", "direct · 9B", "9B", "r4s_direct_nothink_3x", "direct_nothink", "direct_nothink (9B)"),
    ("direct (wiki rewrite)", "direct · 27B", "27B", "r4s_direct_nothink_q27b_3x", "direct_nothink_q27b", "direct_nothink (27B)"),
    ("mix (8 formats)", "mix · 2B", "2B", "r4s_mix_q2b_3x", "mix_q2b", "mix (2B)"),
    ("mix (8 formats)", "mix · 9B", "9B", "r4s_mix_3x", "mix", "mix (9B)"),
    ("mix (8 formats)", "mix · 27B", "27B", "r4s_mix_q27b_3x", "mix_q27b", "mix (27B)"),
    ("active reading v2", "arv2 · 9B", "9B", "r4s_arv2_3x", "arv2", "arv2 (9B)"),
    ("base-model sampling", "basegen · 9B base", "9B", "r4s_basegen_3x", None, "basegen (9B base)"),
]
rows = []
for grp, lab, size, run, vk, lk in ROWS:
    m = metrics(run); m["vendi"] = VENDI.get(vk) if vk else None; m["lex"] = lexpct(lk)
    rows.append({"group": grp, "label": lab, "size": size, "run": run, **m})
ref = {"real": metrics("real_p12") | {"vendi": 143.9, "lex": 0.0}, "fresh": metrics("fresh_3x") | {"vendi": None, "lex": None}}
COLS = [("avg7", "avg7 (7 OLMES tasks)", "higher", 3), ("gsm8k", "GSM8K pass@32", "higher", 3),
        ("mathbpb", "math bpb", "lower", 3), ("dclm", "DCLM held-out bpb", "lower", 3), ("paloma", "Paloma bpb (mean)", "lower", 3),
        ("vendi", "batch Vendi (/256)", "higher", 1), ("lex", "word types vs real (%)", "higher", 0)]
data = {"rows": rows, "ref": ref, "cols": COLS}
html = open("template.html").read().replace("/*DATA*/null", json.dumps(data))
open("3c_methods.html", "w").write(html)
print(json.dumps(data, indent=1)[:1500])
