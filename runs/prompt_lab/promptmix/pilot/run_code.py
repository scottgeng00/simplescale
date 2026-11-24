"""Pilot a prompt template: python run_code.py N SEED TEMPLATE.txt; {slot}s filled at random from slots.json; N random pool docs, Qwen3.5-9B instruct (thinking off,
mix-arm sampling). Writes code_pilot.json."""
import json, os, random, sys, urllib.request
sys.path.insert(0, "/home/sgeng/scottg/pretrain/simplescale/runs/prompt_lab")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)
from app import sample_doc  # noqa: E402
LANGS = ["Python", "C", "Java"]
URL = open("/home/sgeng/scottg/pretrain/simplescale/runs/interactive/sglang_qwen35_9b.addr").read().strip()
TPL = open(sys.argv[3]).read()
SLOTS = json.load(open("slots.json"))
n, seed = int(sys.argv[1]), int(sys.argv[2])
random.seed(seed)
out = []
for i in range(n):
    doc = sample_doc()
    fill = {k: random.choice(v) for k, v in SLOTS.items() if "{" + k + "}" in TPL}
    prompt = TPL
    for k, v in fill.items():
        prompt = prompt.replace("{" + k + "}", v)
    lang = fill
    body = {"messages": [{"role": "user", "content": prompt.replace("{document}", doc["text"][:12000])}],
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    out.append({"lang": lang, "doc": doc["text"], "code": r["choices"][0]["message"]["content"],
                "finish": r["choices"][0]["finish_reason"]})
json.dump(out, open(sys.argv[3].replace(".txt", f"_pilot{seed}.json"), "w"), indent=1)
for o in out:
    print("=" * 100, "\nSOURCE:", " ".join(o["doc"].split())[:500], f"\n--- {o['lang']} ({o['finish']}, {len(o['code'].split())} words):\n{o['code'][:2500]}")
