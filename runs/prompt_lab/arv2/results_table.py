"""Results table for the arv2 report: python3 results_table.py"""
import json, os
E = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/multisample_dc_w1152"
MAN = json.load(open("/home/sgeng/scottg/pretrain/lingua-clean/apps/main/configs/experiments/multisample_dc_w1152/manifest.json"))
RUNS = ["r4s_arv2", "r4s_ideas", "r4s_mix", "r4s_direct_nothink", "r4s_direct_think", "real_p8",
        "r4s_arv2_3x", "r4s_mix_3x", "r4s_direct_nothink_3x", "real_p12"]
T7 = ["arc_easy:rc::olmes:full", "arc_challenge:rc::olmes:full", "hellaswag:rc::olmes:full", "piqa:rc::olmes:full",
      "winogrande:rc::olmes:full", "sciq:rc::olmo3", "mmlu:rc::olmes"]
EX = ["mmlu:rc:bpb::olmes", "gsm8k::olmo3:n32:pass32", "olmo3:base_easy:math_bpb"]
def bpb(v, k):
    x = v.get(k, {}); return x.get("bpb") if isinstance(x, dict) else None
hdr = ["run", "steps", "dclm_bpb", "paloma_bpb"] + [t.split(":")[0] for t in T7] + ["avg7", "mmlu_bpb", "gsm8k_p32", "math_bpb"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in RUNS:
    steps = MAN.get(r, {}).get("steps")
    if steps is None: print(f"| {r} | (no config) |"); continue
    vp = f"{E}/{r}/evals/{steps:010d}/validation.json"
    v = json.load(open(vp)) if os.path.exists(vp) else {}
    row = None
    mp = f"{E}/{r}/metrics.olmes_eval.jsonl"
    if os.path.exists(mp):
        for l in open(mp):
            x = json.loads(l)
            if x.get("global_step") == steps: row = x
    g = lambda t: (row or {}).get(f"olmes_eval/{t}/primary_score")
    s7 = [g(t) for t in T7]; avg = sum(s7) / 7 if all(x is not None for x in s7) else None
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    print("| " + " | ".join([r, str(steps), f(bpb(v, "dclm_heldout")), f(bpb(v, "paloma_qwen3/mean"))] + [f(x, 3) for x in s7] + [f(avg, 3)] + [f(g(t), 3) for t in EX]) + " |")
