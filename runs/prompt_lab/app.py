"""Prompt lab for Active Reading: sample a pool document, edit the strategy / application prompts, run each stage on
the Qwen3.5-9B instruct or base sglang server, and read the strategies and their applications side by side.
Second tab ("Generate"): one free-form prompt with a {document} placeholder, n samples from instruct or base.
Third tab ("Surprisal"): teacher-forced logprob of every token of a real document, coloured, with top-k alternatives.

    cd ~/scottg/pretrain/simplescale && .venv/bin/python runs/prompt_lab/app.py --port 8765
Servers: runs/interactive/sglang_qwen35_9b.addr (instruct) and sglang_qwen35_9b_base.addr (base), read per request.
"""
import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
ADDR = {"instruct": HERE.parent / "interactive/sglang_qwen35_9b.addr",
        "base": HERE.parent / "interactive/sglang_qwen35_9b_base.addr",
        "base4b": HERE.parent / "interactive/sglang_qwen35_4b_base.addr"}  # Qwen3.5-4B-Base
MANIFEST = Path("/home/sgeng/scottg/pretrain/simplescale/runs/manifests/dclm.f200w.ideas.manifest.jsonl")
THINK = re.compile(r"^(.*?)</think>", re.DOTALL)
HEADING = re.compile(r"^##(?!#)", re.MULTILINE)
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)


def sample_doc():
    """A uniformly random line of the source manifest (filtered, deduplicated DCLM docs)."""
    size = MANIFEST.stat().st_size
    with open(MANIFEST, "rb") as f:
        f.seek(random.randrange(size))
        f.readline()
        line = f.readline() or (f.seek(0) or f.readline())
    task = json.loads(line)
    ref = task["_ref"]
    with open(ref["path"], "rb") as f:
        rec = json.loads(os.pread(f.fileno(), ref["length"], ref["offset"]))
    return {"id": rec.get("id"), "text": rec["text"], "source_subset": task.get("source_subset"),
            "source_row": task.get("source_row")}


def split_strategies(reply):
    starts = [m.start() for m in HEADING.finditer(reply)]
    return [reply[a:b].strip() for a, b in zip(starts, starts[1:] + [len(reply)]) if reply[a:b].strip("# \n")]


async def generate(session, model, prompt, opts):
    """One generation. instruct: chat with enable_thinking, reasoning split off. base: raw completion of
    prompt + prefix, the prefix put back in front of the output."""
    url = ADDR[model].read_text().strip()
    common = {"temperature": float(opts.get("temperature", 1.0)), "top_p": float(opts.get("top_p", 0.95)),
              "max_tokens": int(opts.get("max_tokens", 4096))}
    if opts.get("stop"):
        common["stop"] = opts["stop"]
    if model == "instruct":
        body = {**common, "top_k": 20, "presence_penalty": float(opts.get("presence_penalty", 1.5)),
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": bool(opts.get("thinking"))}}
        async with session.post(f"{url}/v1/chat/completions", json=body) as r:
            out = await r.json()
        if "choices" not in out:
            raise RuntimeError(json.dumps(out)[:500])
        content = out["choices"][0]["message"].get("content") or ""
        reasoning = ""
        if opts.get("thinking"):
            m = THINK.match(content)
            reasoning, content = (m.group(1), content[m.end():]) if m else (content, "")
        return {"text": content.strip(), "reasoning": reasoning.strip(), "usage": out.get("usage", {}),
                "finish_reason": out["choices"][0].get("finish_reason")}
    prefix = opts.get("prefix", "")
    body = {**common, "prompt": prompt + prefix}
    for k in ("presence_penalty", "frequency_penalty"):  # base: only when set (instruct always sends presence_penalty)
        if opts.get(k) not in (None, "", 0, "0"):
            body[k] = float(opts[k])
    async with session.post(f"{url}/v1/completions", json=body) as r:
        out = await r.json()
    if "choices" not in out:
        raise RuntimeError(json.dumps(out)[:500])
    return {"text": (prefix + out["choices"][0]["text"]).strip(), "reasoning": "", "usage": out.get("usage", {}),
            "finish_reason": out["choices"][0].get("finish_reason")}


def fill(template, **values):
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


async def api_sample(request):
    return web.json_response(sample_doc())


async def api_strategies(request):
    p = await request.json()
    prompt = fill(p["template"], text=p["text"], n=p.get("n", 8), examples=p.get("examples", ""))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
            out = await generate(s, p["model"], prompt, p.get("opts", {}))
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": repr(e)}, status=500)
    return web.json_response({**out, "prompt": prompt, "strategies": split_strategies(out["text"])})


async def api_apply(request):
    p = await request.json()
    prompt = fill(p["template"], text=p["text"], strategy=p["strategy"])
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
            out = await generate(s, p["model"], prompt, p.get("opts", {}))
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": repr(e)}, status=500)
    return web.json_response({**out, "prompt": prompt})


async def api_generate(request):
    """Free-form tab: fill {document} in the template, draw n samples from the chosen model."""
    p = await request.json()
    prompt = p["template"].replace("{document}", p.get("text", ""))
    n = max(1, min(int(p.get("n", 1)), 16))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
            outs = await asyncio.gather(*[generate(s, p["model"], prompt, p.get("opts", {})) for _ in range(n)])
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": repr(e)}, status=500)
    return web.json_response({"prompt": prompt, "outputs": outs})


async def api_surprisal(request):
    """Teacher-forced logprobs of every token of prefix + text under the chosen server (sglang /generate with
    return_logprob): per token its logprob and the top-k alternatives. Prefix tokens are flagged."""
    p = await request.json()
    prefix, text = p.get("prefix", ""), p["text"]
    url = ADDR[p.get("model", "base")].read_text().strip()
    body = {"text": prefix + text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": 0, "top_logprobs_num": int(p.get("top_k", 5)), "return_text_in_logprobs": True}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
            async with s.post(f"{url}/generate", json=body) as r:
                out = await r.json()
        m = out["meta_info"]
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": repr(e)}, status=500)
    toks, pos = [], 0
    for (lp, _, t), top in zip(m["input_token_logprobs"], m["input_top_logprobs"]):
        toks.append({"t": t, "lp": lp, "top": [[tt, tl] for tl, _, tt in (top or [])], "prefix": pos < len(prefix)})
        pos += len(t)
    return web.json_response({"tokens": toks})


async def api_presets(request):
    return web.json_response(json.loads((HERE / "presets.json").read_text()))


async def api_servers(request):
    status = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
        for name, path in ADDR.items():
            try:
                url = path.read_text().strip()
                async with s.get(f"{url}/v1/models") as r:
                    status[name] = "up" if r.status == 200 else f"HTTP {r.status}"
            except Exception as e:  # noqa: BLE001
                status[name] = f"down ({type(e).__name__})"
    return web.json_response(status)


async def index(request):
    return web.Response(text=(HERE / "index.html").read_text(), content_type="text/html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    app = web.Application(client_max_size=64 * 2**20)
    app.add_routes([web.get("/", index), web.get("/api/sample", api_sample), web.get("/api/servers", api_servers),
                    web.post("/api/strategies", api_strategies), web.post("/api/apply", api_apply),
                    web.post("/api/generate", api_generate), web.get("/api/presets", api_presets),
                    web.post("/api/surprisal", api_surprisal)])
    app.router.add_static("/viz", HERE / "viz")  # static charts (viz/*.html)
    web.run_app(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
