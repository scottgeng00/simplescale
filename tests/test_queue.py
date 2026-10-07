from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from simplescale import LeaseClient, StaleLease
from simplescale.manager import StaleLease as QueueStaleLease
from simplescale.manager import WorkQueue, create_app


def result(attempt: str, uri: str, rows: int = 2) -> dict:
    return {
        "attempt": attempt,
        "uri": uri,
        "rows": rows,
        "bytes": 1,
        "sha256": "x",
    }


def test_expiry_fencing_and_restart(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(3)))
    now = [100.0]
    queue = WorkQueue(
        manifest,
        tmp_path / "out",
        tmp_path / "state",
        chunk_size=2,
        lease_seconds=10,
        now=lambda: now[0],
    )
    first = queue.claim("a")
    now[0] = 111
    other = queue.claim("b")
    replacement = queue.claim("c")
    assert replacement["chunk"]["id"] == first["chunk"]["id"]
    with pytest.raises(QueueStaleLease):
        queue.complete(0, result(first["attempt"], "old"))
    queue.complete(0, result(replacement["attempt"], "accepted"))
    queue.complete(1, result(other["attempt"], "last", rows=1))
    assert queue.status()["done"]
    queue.close()

    restored = WorkQueue(manifest, tmp_path / "out", tmp_path / "state", chunk_size=2)
    assert restored.status()["completed"] == 2
    dataset = json.loads((tmp_path / "out/dataset.json").read_text())
    assert [shard["uri"] for shard in dataset["shards"]] == ["accepted", "last"]
    restored.close()


def test_max_documents(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(5)))
    queue = WorkQueue(
        manifest, tmp_path / "out", tmp_path / "state", max_documents=3
    )
    assert queue.metadata["rows"] == 3
    queue.close()


async def test_http_claim_complete(tmp_path):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text('{"id":1}\n')
    queue = WorkQueue(manifest, tmp_path / "out", tmp_path / "state")
    server = TestClient(TestServer(create_app(queue, "secret")))
    await server.start_server()
    token = tmp_path / "token"
    token.write_text("secret")
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps({"endpoint": str(server.make_url("/")).rstrip("/"), "token_file": str(token)})
    )
    try:
        async with LeaseClient(discovery) as client:
            lease = await client.claim("worker")
            await client.heartbeat(0, lease["attempt"])
            await client.complete(0, result(lease["attempt"], "shard", rows=1))
            assert (await client.status())["done"]
            with pytest.raises(StaleLease):
                await client.release(0, lease["attempt"])
    finally:
        await server.close()
        queue.close()
