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
options. Rephrasing returns only the independent articles. Latent thoughts
splits the document into `G+1` equal-token pieces with SGLang's own tokenizer
and necessarily embeds the source in the assembled `<think>...</think>`
megadoc. Use a worker context length large enough for the untruncated source.
For one setting across an existing manifest, use a tiny wrapper rather than
rewriting the data:

```python
from simplescale.harnesses import rephrase

async def generate(task, llm):
    return await rephrase(task | {"generations": 8}, llm)
```

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
  --tp 8
```

TP may be 1, 2, 4, or 8. Resources scale with TP from the partition's
8-GPU/192-CPU/1,992,294-MB node shape, so TP8 consumes the complete node.
Workers process prefetched chunks through one shared concurrency limit, keeping
SGLang fed across chunk tails. `h200_dream_high` workers prefetch three chunks
and other QoS pools prefetch two. On Slurm's preemption signal, a worker stops
claiming work, drains its leased chunks, and then requeues. Lease expiry returns
work from dead workers to the queue; stale attempts cannot be accepted.

Accepted shards are listed in `results/dataset.json`. `_SUCCESS` appears after
every chunk is complete. The manager persists only an append-only completion
journal, so a manager restart requeues unfinished work.
