from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import aiohttp

from .client import LeaseClient, LocalSGLangClient, StaleLease


def _load_handler(spec: str) -> Callable:
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("handler must be module:function")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), name)


def _read_tasks(lease: dict) -> list[dict]:
    chunk = lease["chunk"]
    tasks = []
    with Path(lease["manifest"]).open("rb") as source:
        source.seek(chunk["start"])
        while source.tell() < chunk["end"]:
            tasks.append(json.loads(source.readline()))
    return tasks


def _write_results(lease: dict, rows: list[dict]) -> dict[str, Any]:
    chunk, attempt = lease["chunk"]["id"], lease["attempt"]
    directory = Path(lease["output_dir"]) / "attempts"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"chunk-{chunk:08d}-{attempt}.jsonl"
    temporary = directory / f".{final.name}.{uuid.uuid4().hex}.tmp"
    data = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, final)
    return {
        "attempt": attempt,
        "uri": str(final),
        "rows": len(rows),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


async def _wait_ready(endpoint: str, process: asyncio.subprocess.Process) -> None:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while process.returncode is None:
            try:
                async with session.get(endpoint + "/health") as response:
                    if response.status < 400:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    raise RuntimeError(f"SGLang exited before becoming ready: {process.returncode}")


async def _run_lease(
    lease: dict,
    manager: LeaseClient,
    llm: LocalSGLangClient,
    handler: Callable,
    semaphore: asyncio.Semaphore,
    heartbeat_seconds: float,
    stopping: asyncio.Event,
) -> None:
    chunk, attempt = lease["chunk"]["id"], lease["attempt"]
    lost = asyncio.Event()

    async def heartbeat() -> None:
        while not stopping.is_set() and not lost.is_set():
            await asyncio.sleep(heartbeat_seconds)
            try:
                await manager.heartbeat(chunk, attempt)
            except StaleLease:
                lost.set()
            except (OSError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError):
                pass

    heartbeat_task = asyncio.create_task(heartbeat())
    result_path: Path | None = None
    try:
        tasks = await asyncio.to_thread(_read_tasks, lease)

        async def run_one(index: int, task: dict) -> dict:
            async with semaphore:
                value = handler(task, llm)
                if inspect.isawaitable(value):
                    value = await value
                task_id = task.get("id", lease["chunk"]["first_row"] + index)
                return {"task_id": task_id, "result": value}

        rows = await asyncio.gather(*(run_one(i, task) for i, task in enumerate(tasks)))
        if lost.is_set() or stopping.is_set():
            return
        metadata = await asyncio.to_thread(_write_results, lease, rows)
        result_path = Path(metadata["uri"])
        while not stopping.is_set() and not lost.is_set():
            try:
                await manager.complete(chunk, metadata)
                result_path = None
                return
            except StaleLease:
                lost.set()
            except (OSError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(2)
    except Exception as error:
        print(f"chunk {chunk} failed: {error!r}", flush=True)
        with contextlib.suppress(Exception):
            await manager.release(chunk, attempt, repr(error))
        await asyncio.sleep(2)
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        if result_path:
            result_path.unlink(missing_ok=True)
        if stopping.is_set() and not lost.is_set():
            with contextlib.suppress(Exception):
                await manager.release(chunk, attempt, "worker stopping")


async def supervise(args: argparse.Namespace) -> int:
    handler = _load_handler(args.handler)
    command = args.command or [
        str(Path(sys.executable).with_name("sglang")),
        "serve",
        "--model-path",
        args.model,
        "--tp-size",
        str(args.tp_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if not args.command and args.load_format:
        command.extend(["--load-format", args.load_format])
    if not args.command and args.skip_tokenizer_init:
        command.append("--skip-tokenizer-init")
    process = await asyncio.create_subprocess_exec(*command)
    stopping = asyncio.Event()
    draining = asyncio.Event()

    def stop() -> None:
        stopping.set()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop)
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        loop.add_signal_handler(sig, draining.set)

    endpoint = f"http://127.0.0.1:{args.port}"
    await _wait_ready(endpoint, process)
    worker_id = args.worker_id or "-".join(
        filter(None, [os.getenv("SLURM_JOB_ID"), os.getenv("SLURM_ARRAY_TASK_ID")])
    ) or f"worker-{os.getpid()}"
    prefetch = 3 if os.getenv("SLURM_JOB_QOS") == "h200_dream_high" else 2
    semaphore = asyncio.Semaphore(args.task_concurrency)
    active: set[asyncio.Task] = set()
    finished = False
    async with LeaseClient(args.discovery_file) as manager, LocalSGLangClient(
        endpoint, args.task_concurrency
    ) as llm:
        while process.returncode is None and not stopping.is_set():
            while (
                not stopping.is_set()
                and not draining.is_set()
                and not finished
                and len(active) < prefetch
            ):
                try:
                    lease = await manager.claim(worker_id)
                except (OSError, ValueError, RuntimeError, aiohttp.ClientError):
                    await asyncio.sleep(2)
                    break
                if stopping.is_set() or draining.is_set():
                    if lease["status"] == "lease":
                        with contextlib.suppress(Exception):
                            await manager.release(
                                lease["chunk"]["id"], lease["attempt"], "worker draining"
                            )
                    break
                if lease["status"] == "done":
                    finished = True
                    break
                if lease["status"] == "wait":
                    break
                active.add(
                    asyncio.create_task(
                        _run_lease(
                            lease,
                            manager,
                            llm,
                            handler,
                            semaphore,
                            args.heartbeat_seconds,
                            stopping,
                        )
                    )
                )
            if active:
                done, _ = await asyncio.wait(
                    active, timeout=2, return_when=asyncio.FIRST_COMPLETED
                )
                active.difference_update(done)
                await asyncio.gather(*done)
            elif draining.is_set() or finished:
                break
            else:
                await asyncio.sleep(2)

        if active:
            if not draining.is_set():
                stopping.set()
                for task in active:
                    task.cancel()
            await asyncio.gather(*active, return_exceptions=True)

    clean_shutdown = stopping.is_set() or draining.is_set() or process.returncode is None
    if process.returncode is None:
        process.terminate()
    try:
        code = await asyncio.wait_for(process.wait(), timeout=30)
    except asyncio.TimeoutError:
        process.kill()
        code = await process.wait()
    return 0 if clean_shutdown else code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SimpleScale generation worker")
    parser.add_argument("--discovery-file", required=True)
    parser.add_argument("--handler", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--port",
        type=int,
        default=30000 + int(os.getenv("SLURM_ARRAY_TASK_ID", "0")) % 10000,
    )
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--task-concurrency", type=int, default=512)
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--load-format")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    raise SystemExit(asyncio.run(supervise(args)))


if __name__ == "__main__":
    main()
