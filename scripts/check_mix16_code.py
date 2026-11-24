"""Execution check of the code / code_problem views of a finished prompt_pool run (mix16 / mix16plan), on a CPU node.

Writes a derived run dir <out> whose results/dataset.json lists checked copies of the shards (same chunks), so
lingua-clean's setup/export_multisample.py can export it with --run <out>. Every code / code_problem view that finished
gets "code_check" (simplescale.harnesses.mix16.check_code); failures get finish_reason "rejected:code_<status>", so the
export drops them. Java views cannot be compiled (no javac on the cluster): "unverified", kept unless --java drop.
--require-match also rejects code_problem views whose output on the extracted example input differs from the stated
output ("rejected:code_mismatch"; views without extractable example I/O only need to run). Counts -> <out>/code_check.json.

    .venv/bin/python scripts/check_mix16_code.py --run runs/pool025-qwen35-9b-mix16 --out runs/pool025-qwen35-9b-mix16-checked
"""
import argparse
import json
import os
import tempfile
from collections import Counter
from multiprocessing import Pool

from simplescale.harnesses import mix16

_SHIM = None
_ARGS = None


def _init(args):
    global _SHIM, _ARGS
    _ARGS = args
    _SHIM = mix16.build_shim(tempfile.mkdtemp(prefix="mix16shim-"))
    assert _SHIM, "could not build the connect() shim"


def check_shard(job):
    src, dst = job
    counts = Counter()
    if os.path.exists(dst) and os.path.exists(dst + ".counts.json"):
        return Counter(json.load(open(dst + ".counts.json")))
    lines = []
    for line in open(src):
        row = json.loads(line)
        for s in row["result"]["samples"]:
            if s["kind"] not in ("code", "code_problem") or s["finish_reason"] != "stop" or not s["text"]:
                continue
            lang = s["slots"]["language"]
            c = mix16.check_code(s["kind"], s["text"], lang, _SHIM, timeout=_ARGS.timeout)
            s["code_check"] = c
            key = f"{s['kind']}|{lang}"
            counts[f"{key}|{c['status']}"] += 1
            if s["kind"] == "code_problem":
                counts[f"{key}|io_extracted"] += bool(c.get("has_input") and c.get("has_output"))
                if "match" in c:
                    counts[f"{key}|match_{c['match']}"] += 1
            bad = c["status"] not in ("pass", "unverified")
            if c["status"] == "unverified" and _ARGS.java == "drop":
                bad = True
            if _ARGS.require_match and c.get("match") is False:
                s["finish_reason"] = "rejected:code_mismatch"
            elif bad:
                s["finish_reason"] = f"rejected:code_{c['status']}"
        lines.append(json.dumps(row, ensure_ascii=False))
    with open(dst + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(dst + ".tmp", dst)
    json.dump(counts, open(dst + ".counts.json", "w"))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--java", choices=["keep", "drop"], default="keep")
    ap.add_argument("--require-match", action="store_true")
    ap.add_argument("--max-chunk", type=int, help="only chunks below this index (e.g. the pool's)")
    a = ap.parse_args()
    ds = json.load(open(f"{a.run}/results/dataset.json"))
    os.makedirs(f"{a.out}/results/attempts", exist_ok=True)
    shards = [s for s in ds["shards"] if a.max_chunk is None or s["chunk"] < a.max_chunk]
    jobs = [(s["uri"], f"{a.out}/results/attempts/{os.path.basename(s['uri'])}") for s in shards]
    total = Counter()
    with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
        for i, c in enumerate(pool.imap_unordered(check_shard, jobs, chunksize=4)):
            total.update(c)
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} shards", flush=True)
    out = {**ds, "shards": [{**s, "uri": dst} for s, (_, dst) in zip(shards, jobs)], "checked_from": a.run,
           "code_check": {"java": a.java, "require_match": a.require_match, "timeout": a.timeout}}
    json.dump(out, open(f"{a.out}/results/dataset.json", "w"), indent=1)
    json.dump(dict(sorted(total.items())), open(f"{a.out}/code_check.json", "w"), indent=1)
    print(json.dumps(dict(sorted(total.items())), indent=1))


if __name__ == "__main__":
    main()
