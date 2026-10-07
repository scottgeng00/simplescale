#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from simplescale.client import LocalSGLangClient
from simplescale.harnesses.megadocs import REPHRASE, SYSTEM
from simplescale.worker import _resolve, _start_server, _stop_process


def load_tasks(path: Path, count: int) -> list[dict]:
    tasks = []
    with path.open() as source:
        for line in source:
            tasks.append(_resolve(json.loads(line)))
            if len(tasks) == count:
                return tasks
    raise ValueError(f"manifest has fewer than {count} rows")


async def run(args: argparse.Namespace) -> None:
    levels = [int(value) for value in args.concurrency.split(",")]
    tasks = load_tasks(args.manifest, args.requests * len(levels))
    command = [
        str(Path(args.python).with_name("sglang")),
        "serve",
        "--model-path", args.model,
        "--tp-size", str(args.tp_size),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--context-length", str(args.context_length),
        "--mem-fraction-static", str(args.mem_fraction_static),
    ]
    server_args = argparse.Namespace(
        startup_attempts=2,
        startup_timeout=args.startup_timeout,
        port=args.port,
    )
    stopping, draining = asyncio.Event(), asyncio.Event()
    process = await _start_server(command, server_args, stopping, draining)
    if process is None:
        raise RuntimeError("server stopped during startup")
    try:
        async with LocalSGLangClient(
            f"http://127.0.0.1:{args.port}", max(levels)
        ) as llm:
            for phase, concurrency in enumerate(levels):
                batch = tasks[phase * args.requests : (phase + 1) * args.requests]
                semaphore = asyncio.Semaphore(concurrency)

                async def generate(task: dict) -> tuple[int, int]:
                    async with semaphore:
                        response = await llm.chat(
                            messages=[
                                {"role": "system", "content": SYSTEM},
                                {
                                    "role": "user",
                                    "content": REPHRASE.format(text=task["text"]),
                                },
                            ],
                            sampling_params={
                                "temperature": 1,
                                "max_new_tokens": 1024,
                                "reasoning_effort": "none",
                                "n": 1,
                            },
                        )
                    usage = response["usage"]
                    return usage["prompt_tokens"], usage["completion_tokens"]

                print(f"BENCHMARK_START concurrency={concurrency}", flush=True)
                start = time.monotonic()
                results = await asyncio.gather(
                    *(generate(task) for task in batch), return_exceptions=True
                )
                elapsed = time.monotonic() - start
                completed = [
                    result for result in results if not isinstance(result, Exception)
                ]
                prompt_tokens = sum(result[0] for result in completed)
                output_tokens = sum(result[1] for result in completed)
                print(
                    "BENCHMARK_RESULT "
                    + json.dumps(
                        {
                            "concurrency": concurrency,
                            "requests": len(completed),
                            "errors": len(results) - len(completed),
                            "seconds": elapsed,
                            "prompt_tps": prompt_tokens / elapsed,
                            "output_tps": output_tokens / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        await _stop_process(process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--python", default=str(Path(__file__).parents[1] / ".venv/bin/python")
    )
    parser.add_argument("--concurrency", default="16,32,64,128,256,384,512,628")
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=262144)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
