"""Results table for the basegen arm vs existing multisample_dc_w1152 arms at the final step."""
import json, os
D = "/home/sgeng/scottg/pretrain/lingua-runs/runs/midtrain/experiments/multisample_dc_w1152"
TASKS = ["arc_easy:rc::olmes:full", "arc_challenge:rc::olmes:full", "hellaswag:rc::olmes:full", "piqa:rc::olmes:full",
         "winogrande:rc::olmes:full", "sciq:rc::olmo3", "mmlu:rc::olmes"]
EXTRA = ["mmlu:rc:bpb::olmes", "gsm8k::olmo3:n32:pass32", "olmo3:base_easy:math_bpb"]
TABLES = {16392: ["r4s_basegen", "r4s_direct_nothink", "r4s_mix", "r4s_direct_think", "r4s_ideas", "real_p8"],
          24588: ["r4s_basegen_3x", "r4s_direct_nothink_3x", "r4s_mix_3x", "real_p12"]}


def row(run, steps):
    out = {"run": run}
    v = f"{D}/{run}/evals/{steps:010d}/validation.json"
    if os.path.exists(v):
        d = json.load(open(v))
        out["dclm"] = d.get("dclm_heldout", {}).get("bpb")
        out["paloma"] = d.get("paloma_qwen3/mean", {}).get("bpb")
    m = f"{D}/{run}/metrics.olmes_eval.jsonl"
    if os.path.exists(m):
        rows = [json.loads(l) for l in open(m) if l.strip()]
        rows = [r for r in rows if r.get("global_step") == steps]
        if rows:
            r = {k: v for x in rows for k, v in x.items()}
            for t in TASKS + EXTRA:
                out[t] = r.get(f"olmes_eval/{t}/primary_score")
            vals = [out[t] for t in TASKS]
            out["avg7"] = sum(vals) / 7 if all(x is not None for x in vals) else None
    return out


def f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


cols = ["dclm", "paloma"] + TASKS + ["avg7"] + EXTRA
short = ["DCLM bpb", "Paloma bpb", "arc_e", "arc_c", "hellaswag", "piqa", "winogr", "sciq", "mmlu_rc", "avg7",
         "mmlu_bpb", "gsm8k p@32", "math_bpb"]
for steps, runs in TABLES.items():
    print(f"\n### {'2C' if steps == 16392 else '3C'} ({steps} steps)\n")
    print("| run | " + " | ".join(short) + " |")
    print("|---" * (len(short) + 1) + "|")
    for run in runs:
        r = row(run, steps)
        print(f"| {run} | " + " | ".join(f(r.get(c)) for c in cols) + " |")
