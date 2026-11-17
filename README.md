# SimpleScale

SimpleScale runs a finite JSONL generation workload across elastic Slurm GPU
jobs. A small CPU manager leases chunks; each GPU worker runs the user harness
against its own local SGLang server and writes results directly to shared disk.
Inference data never passes through the manager.

## Handler API

Each JSONL line is one task. A handler may make any number of local generation
calls and returns a JSON-serializable result:

```python
async def generate(task, llm):
    first = await llm.generate(prompt=task["prompt"])
    second = await llm.generate(prompt=f"Critique: {first['text']}")
    return {"draft": first, "critique": second}
```

Pass it as `package.module:generate`. A complete output row contains the input
task ID and the returned value.

`llm.generate(prompt=...)` performs raw text continuation. Use
`llm.chat(messages=...)` to apply the model's chat template through SGLang's
OpenAI-compatible chat endpoint.

## Megadocs baselines

The paper's exact prompts and sampling defaults are available as handlers:

```text
simplescale.harnesses.megadocs:rephrase
simplescale.harnesses.megadocs:latent_thoughts
```

Both consume DCLM records directly (`text` and, normally, `id`), avoiding a
second copied input manifest. Set `generations` per task (default 1);
`sampling_params` may override temperature, output length, or other SGLang
options. Rephrasing returns the independent articles. Latent thoughts
splits the document into `G+1` equal-token pieces with SGLang's own tokenizer
and necessarily embeds the source in the assembled `<think>...</think>`
megadoc. Set `max_document_tokens` to truncate an overlong source before
prompting or splitting; each result records this as `source_truncated`.
For one setting across an existing manifest, use a tiny wrapper rather than
rewriting the data:

```python
from simplescale.harnesses import rephrase

async def generate(task, llm):
    return await rephrase(task | {"generations": 8}, llm)
```

A task may instead reference one record in a raw JSONL file. Other manifest
fields are overlaid on the loaded record:

```json
{"_ref":{"path":"/data/raw.jsonl","offset":0,"length":1234},"sampling_params":{"temperature":1.0}}
```

Build one with `python scripts/build_manifest.py RAW OUT --rows N`; generation
count defaults to `--generations 1`, and sampling parameters default to
`--sampling-params '{"temperature":1.0}'`. For a 65,536-token worker context
and up to 2,048 output tokens, add `--max-document-tokens 62000`.

## Export aligned pretraining splits

Convert the DCLM source and its 2B and 27B rephrases into pretraining JSONL
shards with:

```bash
.venv/bin/python scripts/export_rewrite_splits.py --num-chunks 16
```

By default this writes `dclm_15m`, `qwen35_2b_rephrase`, and
`qwen35_27b_rephrase` under `../assets/data/rewrite_splits`. The original DCLM
order was already shuffled, so the exporter does not shuffle it again. Instead,
it deals every aligned triplet round-robin into the output shards.

**Alignment is a hard invariant.** For global input row `k`, all three variants
are written to shard `k % num_chunks` at position `k // num_chunks`. Before a
row is written, the exporter requires the original `id` and both rewrite
`task_id` values to match. Any mismatch, missing row, extra row, or non-single
rewrite aborts the export and removes its temporary output. The final directory
is published atomically only after every row passes, so a completed export
guarantees that document `i` in shard `j` has its source and both rewrites at
document `i` in shard `j`.

## Setup

```bash
uv sync --extra test --extra sglang --python /usr/bin/python3
.venv/bin/pytest -q
```

## Run

Submit one manager and any mix of worker pools:

```bash
.venv/bin/simplescale submit /checkpoint/run/work.jsonl \
  --model /checkpoint/model \
  --handler mypackage.harness:generate \
  --run-dir /checkpoint/run \
  --pool h200_dream_high=8 \
  --pool h200_lowest=32 \
  --worker-time 2-00:00:00 \
  --wandb-project simplescale \
  --tp 8
```

TP may be 1, 2, 4, or 8. Resources scale with TP from the partition's
8-GPU/192-CPU/1,992,294-MB node shape, so TP8 consumes the complete node.
Workers process prefetched chunks through one shared concurrency limit, keeping
SGLang fed across chunk tails. `h200_dream_high` workers prefetch eight chunks
and other QoS pools prefetch six; override this with `--prefetch`. On Slurm's
preemption signal, a worker stops
claiming work, drains its leased chunks, and then requeues. Lease expiry returns
work from dead workers to the queue; stale attempts cannot be accepted.

Accepted shards are listed in `results/dataset.json`. `_SUCCESS` appears after
every chunk is complete. The manager persists only an append-only completion
journal, so a manager restart requeues unfinished work.

## Setting up on a new cluster

### Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is missing
uv sync --extra test --extra sglang --python /usr/bin/python3   # -> .venv, sglang==0.5.21 (pinned in uv.lock)
.venv/bin/pytest -q
```

Run the CLI from the repo root (`--repo` defaults to the current directory). Workers import handlers
from the repo `src/`, so harnesses in `src/simplescale/harnesses/` need no install step.

### Cluster profiles

`simplescale submit` picks a profile from `src/simplescale/clusters.py`: by Slurm's `ClusterName`
(`$SLURM_CLUSTER_NAME`), else by the GPU type in the partitions' GRES. `--cluster NAME` overrides
detection; it prints the profile it uses.

| Profile | Cluster | Node | Default pool | Manager | Default server args |
|---|---|---|---|---|---|
| `h200` | original H200 cluster (detected by GRES `h200`) | 8x H200, 192 CPUs, x86_64 | `h200_dream_high` (partition inferred from QoS) | `--partition=cpu --qos=cpu_lowest` | none |
| `gb300-dalton` | `fair-cw-use2-3` (FAIR CW USE2-3, Dalton) | 4x GB300, 144 CPUs, aarch64 | `g3_comm_shared`, account `mnm`, partition `g3` | GPU-less job on `g3` / `g3_comm_shared` | `--sampling-backend pytorch` |
| `h100-dm1` | `fair-cw-use1-1` (FAIR CW USE1-1, DM1) | 8x H100, 128 CPUs, x86_64 | `h100_mdf_lowest`, account `dream`, partition `h100_mdf` | GPU-less job on `h100_mdf` / `h100_mdf_scale` | none |

A profile gives the account, partition, GRES string, per-node CPUs / memory (a TP-n worker gets n /
GPUs-per-node of the node), default pool, manager placement and default extra `sglang serve` flags.
`--server-arg` (repeatable) replaces the profile's server flags; `--server-arg ""` clears them.
To add a cluster, add a `Cluster(...)` with its `ClusterName` and GPU type, and a case in
`tests/test_clusters.py`.

Toolchain differences are handled on the node by `slurm/worker.sbatch`, not by the profile:
- if the venv's Python has no `Python.h` (venv on a system Python without dev headers), it uses the
  headers of the matching uv-managed CPython (`~/.local/share/uv/python/cpython-<ver>.*-<arch>-*`);
- if no `nvcc` is on `PATH` (no system CUDA toolkit), it uses nvcc / headers / libcudart from the
  venv's `nvidia/cu1x` wheel, with a per-job `libcudart.so` link for `-lcudart`, and ninja from the venv.

Per-job local caches (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`,
`SGLANG_JIT_CACHE_DIR`) live under `$SLURM_TMPDIR` (or `/tmp`). The venv is per architecture: build
one with `uv sync` on each cluster (aarch64 on Dalton, x86_64 on DM1 / H200).

Pools are passed as `--pool QOS=COUNT` (repeatable). Each pool becomes one worker job array of
COUNT single-node workers on that QoS. Both sbatch scripts unset proxy variables. If you submit
from a shell that has `http_proxy` set, also strip it from the `simplescale submit` environment.

### Models

`--model` takes a local model directory (it is `resolve()`d, so a symlink into the HF cache works).
Our generators are `Qwen/Qwen3.5-{2B,9B,27B}`, downloaded to `assets/models/Qwen3.5-<SIZE>`. TP1 is
enough for 2B/9B; we used TP2 for 27B. `--context-length` defaults to 16384; the Gacha harness needs
32768.

### Manifest format and handler contract

- A manifest is JSONL with one task per line. A line is either the task itself, or a pointer
  `{"_ref": {"path": ..., "offset": <byte offset>, "length": <bytes>}, ...overrides}`. The worker
  `pread`s the referenced line, parses it, and overlays the other manifest fields on top
  (`record | overrides`). `scripts/build_manifest.py` and lingua-clean's
  `setup/build_dclm_rewrite_manifest.py` build pointer manifests over 16-chunk round-robin
  subsets in row order, so output row j is the rewrite of source row j.
- A handler is `async def fn(task: dict, llm) -> dict`. The returned value must be JSON-serializable.
  Each output row is `{"task_id": task.get("id", <row index>), "result": <returned value>}`.
- `llm` is a client for the worker's local SGLang server:
  - `llm.generate(prompt=... | input_ids=..., sampling_params=...)` for raw `/generate`;
  - `llm.chat(messages=..., sampling_params=...)` for OpenAI chat completions. Extra keys in
    `sampling_params` (e.g. `reasoning_effort`, `response_format`) are passed through to the request;
    `max_new_tokens` is mapped to `max_completion_tokens`;
  - `llm.tokenize(text)` / `llm.detokenize([ids])`.
- Results go to `<run-dir>/results/attempts/chunk-*.jsonl`. `results/dataset.json` lists accepted
  shards, and `results/_SUCCESS` marks completion.

### Rewrite harnesses

- `simplescale.harnesses.rewrite:wiki`: the Megadocs/WRAP Wikipedia-style rephrase.
- `simplescale.harnesses.rewrite:guided`: the FinePhrase guided rewrite.
- `simplescale.harnesses.rewrite:nemotron`: Nemotron-CC prompts `diverse_qa` / `distill` /
  `knowledge_list`, one rewrite per entry of `task["views"]`.
- `simplescale.harnesses.rewrite:wofmt`: the wiki prompt retargeted to WebOrganizer formats; also one
  rewrite per `task["views"]` entry.
- `simplescale.harnesses.gacha:rewrite_wiki`: Gacha decoding on the wiki prompt, with JSON-schema
  constrained planning turns and dice seeded by the doc id.

All of them take DCLM records (`text`, `id`) plus task fields:
- `generations` (wiki / guided / gacha);
- `views` (nemotron / wofmt);
- `sampling_params` (e.g. `{"temperature": 1.0, "max_new_tokens": 4096}`);
- `max_document_tokens` (truncates the source first; recorded as `source_truncated`).

The submission and export scripts live in lingua-clean: `setup/submit_dclm_rewrites.sh`,
`setup/submit_dclm_multirewrites.sh`, `setup/prepare_dclm_rewrites.py` and
`setup/prepare_dclm_multirewrites.py`.

### Known issue: the sglang 0.5.21 JIT cache never hits

sglang compiles small CUDA kernels at startup into a JIT cache, `~/.cache/sglang/jit` by default
(`SGLANG_JIT_CACHE_DIR`). In 0.5.21 every cached build records its own deleted staging `cuda.cu`
as a dependency, so no cached build ever validates. Every worker then rebuilds under one
per-module `flock`. On a shared filesystem with hundreds of workers, this serializes startup:
workers sit at "Capturing prefill shape ..." for 20+ minutes, hit `--startup-timeout`, and requeue.

`slurm/worker.sbatch` now sets `SGLANG_JIT_CACHE_DIR` to a per-job directory, so each worker
compiles once locally (a few seconds per kernel). Alternatively, repair the shared cache in place with
`scripts/repair_sglang_jit_cache.py`. Check the
result on a GPU node, because login nodes without the CUDA toolkit always miss.
