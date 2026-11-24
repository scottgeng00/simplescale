"""Pilot the 16-prompt pool: the same N random pool documents (seed) through every prompt (the 8 mix prompts from
the harness + the 8 new ones in pool16.json, {slot}s drawn from its slots), Qwen3.5-9B instruct, mix-arm sampling,
thinking off. Writes pool_pilot_<seed>.json: [{doc, outputs: {kind: {text, finish, fill}}}].

    .venv/bin/python runs/prompt_lab/promptmix/pilot/run_pool.py N SEED
"""
import json, os, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/sgeng/scottg/pretrain/simplescale/src")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)
from app import sample_doc  # noqa: E402
from simplescale.harnesses.multisample import MIX, _mix_messages, DEFAULT_PARAMS  # noqa: E402
URL = open("/home/sgeng/scottg/pretrain/simplescale/runs/interactive/sglang_qwen35_9b.addr").read().strip()
POOL = json.load(open(os.path.join(HERE, "pool16.json")))
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(seed)
random.seed(seed)
docs = [sample_doc()["text"][:12000] for _ in range(n)]


def messages(kind, text):
    if kind in MIX:
        return _mix_messages(kind, text), {}
    tpl = POOL["new_prompts"][kind]
    fill = {k: rng.choice(v) for k, v in POOL["slots"].items() if "{" + k + "}" in tpl}
    for k, v in fill.items():
        tpl = tpl.replace("{" + k + "}", v)
    return [{"role": "user", "content": f"{tpl}\nDocument:\n{text}"}], fill


def call(job):
    i, kind, msgs, fill = job
    p = {k: v for k, v in DEFAULT_PARAMS.items() if k != "max_new_tokens"}
    body = {"messages": msgs, **p, "max_tokens": DEFAULT_PARAMS["max_new_tokens"], "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            return i, kind, {"text": r["choices"][0]["message"]["content"], "finish": r["choices"][0]["finish_reason"], "fill": fill}
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    return i, kind, {"text": "", "finish": f"error {err}", "fill": fill}


jobs = [(i, kind, *messages(kind, d)) for i, d in enumerate(docs) for kind in MIX + list(POOL["new_prompts"])]
out = [{"doc": d, "outputs": {}} for d in docs]
with ThreadPoolExecutor(32) as ex:
    for k, (i, kind, o) in enumerate(ex.map(call, jobs)):
        out[i]["outputs"][kind] = o
        if k % 50 == 0:
            print(f"{k}/{len(jobs)}", flush=True)
json.dump(out, open(os.path.join(HERE, f"pool_pilot_{seed}{os.environ.get("TAG", "")}.json"), "w"), indent=1)
print("done")
