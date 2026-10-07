from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import socket
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from aiohttp import web


class StaleLease(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    id: int
    start: int
    end: int
    first_row: int
    rows: int


@dataclass
class Lease:
    attempt: str
    worker_id: str
    expires_at: float


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _scan_manifest(path: Path, chunk_size: int) -> tuple[list[Chunk], str, int]:
    chunks: list[Chunk] = []
    digest = hashlib.sha256()
    start = first_row = rows = total = 0
    with path.open("rb") as source:
        while line := source.readline():
            json.loads(line)
            digest.update(line)
            rows += 1
            total += 1
            if rows == chunk_size:
                chunks.append(Chunk(len(chunks), start, source.tell(), first_row, rows))
                start, first_row, rows = source.tell(), total, 0
        if rows:
            chunks.append(Chunk(len(chunks), start, source.tell(), first_row, rows))
    return chunks, digest.hexdigest(), total


class WorkQueue:
    def __init__(
        self,
        manifest: str | Path,
        output_dir: str | Path,
        state_dir: str | Path,
        *,
        chunk_size: int = 16,
        lease_seconds: float = 180,
        start_workers: int = 1,
        now: Callable[[], float] = time.time,
    ) -> None:
        if chunk_size < 1 or lease_seconds <= 0 or start_workers < 1:
            raise ValueError("chunk size, lease seconds, and start workers must be positive")
        self.manifest = Path(manifest).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.lease_seconds = lease_seconds
        self.start_workers = start_workers
        self.now = now
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.chunks, digest, rows = _scan_manifest(self.manifest, chunk_size)
        metadata = {
            "manifest": str(self.manifest),
            "sha256": digest,
            "rows": rows,
            "chunk_size": chunk_size,
            "chunks": len(self.chunks),
        }
        meta_path = self.state_dir / "run.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
            raise ValueError(f"state directory belongs to a different run: {meta_path}")
        _atomic_json(meta_path, metadata)
        self.metadata = metadata
        self.journal_path = self.state_dir / "completions.jsonl"
        self.completed: dict[int, dict] = {}
        if self.journal_path.exists():
            for line in self.journal_path.read_text().splitlines():
                record = json.loads(line)
                self.completed[int(record["chunk"])] = record
        self.journal = self.journal_path.open("a")
        self.pending = deque(c.id for c in self.chunks if c.id not in self.completed)
        self.leases: dict[int, Lease] = {}
        self.waiting: dict[str, float] = {}
        self.started_at: float | None = None
        self.finished_at: float | None = None
        if not self.pending:
            dataset_path = self.output_dir / "dataset.json"
            if dataset_path.exists():
                dataset = json.loads(dataset_path.read_text())
                self.started_at = dataset["started_at"]
                self.finished_at = dataset["finished_at"]
            else:
                self.finished_at = self.now()
                self._finalize()

    def close(self) -> None:
        self.journal.close()

    def _expire(self) -> None:
        now = self.now()
        for chunk, lease in list(self.leases.items()):
            if lease.expires_at <= now:
                del self.leases[chunk]
                self.pending.append(chunk)
        self.waiting = {
            worker: seen
            for worker, seen in self.waiting.items()
            if seen + self.lease_seconds > now
        }

    def claim(self, worker_id: str) -> dict:
        self._expire()
        if len(self.completed) == len(self.chunks):
            return {"status": "done"}
        now = self.now()
        self.waiting[worker_id] = now
        if self.started_at is None:
            if len(self.waiting) < self.start_workers:
                return {"status": "wait", "workers_waiting": len(self.waiting)}
            self.started_at = now
        while self.pending:
            chunk = self.chunks[self.pending.popleft()]
            if chunk.id in self.completed or chunk.id in self.leases:
                continue
            attempt = uuid.uuid4().hex
            lease = Lease(attempt, worker_id, now + self.lease_seconds)
            self.leases[chunk.id] = lease
            return {
                "status": "lease",
                "chunk": asdict(chunk),
                "attempt": attempt,
                "expires_at": lease.expires_at,
                "manifest": str(self.manifest),
                "output_dir": str(self.output_dir),
            }
        return {"status": "wait" if self.leases else "done"}

    def _lease(self, chunk: int, attempt: str) -> Lease:
        self._expire()
        lease = self.leases.get(chunk)
        if not lease or lease.attempt != attempt:
            raise StaleLease(f"stale lease for chunk {chunk}")
        return lease

    def heartbeat(self, chunk: int, attempt: str) -> float:
        lease = self._lease(chunk, attempt)
        lease.expires_at = self.now() + self.lease_seconds
        return lease.expires_at

    def release(self, chunk: int, attempt: str) -> None:
        self._lease(chunk, attempt)
        del self.leases[chunk]
        self.pending.append(chunk)

    def complete(self, chunk: int, payload: dict) -> dict:
        existing = self.completed.get(chunk)
        if existing:
            if existing["uri"] == payload["uri"]:
                return existing
            raise StaleLease(f"chunk {chunk} is already complete")
        self._lease(chunk, str(payload["attempt"]))
        record = {
            "chunk": chunk,
            "attempt": str(payload["attempt"]),
            "uri": str(payload["uri"]),
            "rows": int(payload["rows"]),
            "bytes": int(payload["bytes"]),
            "sha256": str(payload["sha256"]),
        }
        self.journal.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.completed[chunk] = record
        del self.leases[chunk]
        if len(self.completed) == len(self.chunks):
            self.finished_at = self.now()
            self._finalize()
        return record

    def status(self) -> dict:
        self._expire()
        return {
            "chunks": len(self.chunks),
            "pending": len(self.pending),
            "running": len(self.leases),
            "completed": len(self.completed),
            "workers_waiting": len(self.waiting),
            "done": len(self.completed) == len(self.chunks),
        }

    def _finalize(self) -> None:
        value = {
            **self.metadata,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": (
                self.finished_at - self.started_at
                if self.started_at is not None and self.finished_at is not None
                else 0
            ),
            "shards": [self.completed[i] for i in sorted(self.completed)],
        }
        _atomic_json(self.output_dir / "dataset.json", value)
        (self.output_dir / "_SUCCESS").write_text("")


def create_app(queue: WorkQueue, token: str = "") -> web.Application:
    @web.middleware
    async def authenticate(request: web.Request, handler):
        if request.path == "/health" or not token:
            return await handler(request)
        if request.headers.get("Authorization") != f"Bearer {token}":
            raise web.HTTPUnauthorized()
        return await handler(request)

    app = web.Application(middlewares=[authenticate])

    def conflict(error: StaleLease) -> web.HTTPConflict:
        return web.HTTPConflict(text=str(error))

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, **queue.status()})

    async def status(_: web.Request) -> web.Response:
        return web.json_response(queue.status())

    async def claim(request: web.Request) -> web.Response:
        payload = await request.json()
        return web.json_response(queue.claim(str(payload["worker_id"])))

    async def heartbeat(request: web.Request) -> web.Response:
        payload = await request.json()
        try:
            expires = queue.heartbeat(
                int(request.match_info["chunk"]), str(payload["attempt"])
            )
        except StaleLease as error:
            raise conflict(error)
        return web.json_response({"expires_at": expires})

    async def complete(request: web.Request) -> web.Response:
        payload = await request.json()
        try:
            record = queue.complete(int(request.match_info["chunk"]), payload)
        except StaleLease as error:
            raise conflict(error)
        return web.json_response(record)

    async def release(request: web.Request) -> web.Response:
        payload = await request.json()
        try:
            queue.release(int(request.match_info["chunk"]), str(payload["attempt"]))
        except StaleLease as error:
            raise conflict(error)
        return web.json_response({"released": True})

    app.router.add_get("/health", health)
    app.router.add_get("/v1/status", status)
    app.router.add_post("/v1/claim", claim)
    app.router.add_post("/v1/leases/{chunk}/heartbeat", heartbeat)
    app.router.add_post("/v1/leases/{chunk}/complete", complete)
    app.router.add_post("/v1/leases/{chunk}/release", release)
    return app


async def serve(args: argparse.Namespace) -> None:
    state_dir = Path(args.state_dir).resolve()
    token_path = state_dir / "token"
    state_dir.mkdir(parents=True, exist_ok=True)
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(32) + "\n")
        token_path.chmod(0o600)
    token = token_path.read_text().strip()
    queue = WorkQueue(
        args.manifest,
        args.output_dir,
        state_dir,
        chunk_size=args.chunk_size,
        lease_seconds=args.lease_seconds,
        start_workers=args.start_workers,
    )
    runner = web.AppRunner(create_app(queue, token))
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    host = args.advertise_host or socket.gethostbyname(socket.gethostname())
    endpoint = f"http://{host}:{int(runner.addresses[0][1])}"
    _atomic_json(
        state_dir / "discovery.json",
        {
            "endpoint": endpoint,
            "epoch": uuid.uuid4().hex,
            "token_file": str(token_path),
        },
    )
    print(f"manager ready at {endpoint}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        queue.close()
        await runner.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SimpleScale lease manager")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lease-seconds", type=float, default=180)
    parser.add_argument("--start-workers", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host")
    parser.add_argument("--port", type=int, default=0)
    return parser


def main() -> None:
    asyncio.run(serve(build_parser().parse_args()))


if __name__ == "__main__":
    main()
