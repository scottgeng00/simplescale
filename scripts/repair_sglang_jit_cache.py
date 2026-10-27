"""Make sglang's shared JIT cache (~/.cache/sglang/jit) hit again.

sglang 0.5.21 records the build's own staging copy of cuda.cu (<scope>/.staging-<uuid>/cuda.cu,
deleted right after publishing) as an absolute dependency of every leaf, so no leaf ever validates:
every process misses, takes the per-module flock on NFS and rebuilds, one at a time across all
workers (10k+ leaves per module, workers stuck for >20 min in CUDA-graph capture). For each
module, this restores the staging cuda.cu of the newest leaf (copied from the leaf, digest
checked) and bumps the leaf's mtime, so find_prebuilt's newest-first scan hits it.
"""
import glob, hashlib, json, os, shutil, sys

root = os.path.expanduser("~/.cache/sglang/jit/sm90a")
for scope in sorted(glob.glob(f"{root}/*/build-*")):
    leaves = [e for e in os.scandir(scope) if e.name.startswith("deps-")]
    if len(leaves) < 2:
        continue
    leaves.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    fixed = False
    for leaf in leaves[:20]:
        try:
            deps = json.load(open(os.path.join(leaf.path, "sgl_deps.json")))
        except Exception:
            continue
        absd = [d for d in deps if d[0] == "abs"]
        if len(absd) != 1 or not absd[0][1].endswith("/cuda.cu") or ".staging-" not in absd[0][1]:
            continue
        src = os.path.join(leaf.path, "cuda.cu")
        if not os.path.exists(src) or hashlib.sha256(open(src, "rb").read()).hexdigest() != absd[0][2]:
            continue
        os.makedirs(os.path.dirname(absd[0][1]), exist_ok=True)
        shutil.copyfile(src, absd[0][1])
        os.utime(leaf.path, None)
        print(f"{os.path.relpath(scope, root)}: pinned {leaf.name} ({len(leaves)} leaves)")
        fixed = True
        break
    if not fixed:
        print(f"{os.path.relpath(scope, root)}: no repairable leaf among newest 20 ({len(leaves)} leaves)", file=sys.stderr)
