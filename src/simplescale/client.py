from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiohttp


class StaleLease(RuntimeError):
    pass


class LeaseClient:
    def __init__(self, discovery_file: str | Path, timeout: float = 30) -> None:
        self.discovery_file = Path(discovery_file)
        self.timeout = timeout
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "LeaseClient":
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=16),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session:
            await self.session.close()

    def _manager(self) -> tuple[str, str]:
        discovery = json.loads(self.discovery_file.read_text())
        token = Path(discovery["token_file"]).read_text().strip()
        return discovery["endpoint"].rstrip("/"), token

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.session:
            raise RuntimeError("client is not started")
        endpoint, token = self._manager()
        async with self.session.request(
            method,
            endpoint + path,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            body = await response.text()
            if response.status == 409:
                raise StaleLease(body)
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
            return json.loads(body) if body else {}

    async def claim(self, worker_id: str) -> dict[str, Any]:
        return await self._request("POST", "/v1/claim", {"worker_id": worker_id})

    async def heartbeat(self, chunk: int, attempt: str) -> None:
        await self._request(
            "POST", f"/v1/leases/{chunk}/heartbeat", {"attempt": attempt}
        )

    async def complete(self, chunk: int, payload: dict[str, Any]) -> None:
        await self._request("POST", f"/v1/leases/{chunk}/complete", payload)

    async def release(self, chunk: int, attempt: str, error: str = "") -> None:
        await self._request(
            "POST",
            f"/v1/leases/{chunk}/release",
            {"attempt": attempt, "error": error},
        )

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/status")


class LocalSGLangClient:
    def __init__(self, endpoint: str, concurrency: int, timeout: float = 3600) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.concurrency = concurrency
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "LocalSGLangClient":
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=self.concurrency),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session:
            await self.session.close()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.session:
            raise RuntimeError("client is not started")
        async with self.session.post(self.endpoint + path, json=payload) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
            return json.loads(body)

    async def generate(
        self,
        *,
        prompt: str | None = None,
        input_ids: list[int] | None = None,
        sampling_params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if (prompt is None) == (input_ids is None):
            raise ValueError("provide exactly one of prompt or input_ids")
        payload: dict[str, Any] = {"sampling_params": sampling_params or {}}
        payload["text" if prompt is not None else "input_ids"] = (
            prompt if prompt is not None else input_ids
        )
        if request_id:
            payload["rid"] = request_id
        return await self._post("/generate", payload)

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        params = dict(sampling_params or {})
        if "max_new_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_new_tokens")
        payload = {**params, "messages": messages}
        if request_id:
            payload["rid"] = request_id
        return await self._post("/v1/chat/completions", payload)

    async def tokenize(self, text: str) -> list[int]:
        response = await self._post(
            "/v1/tokenize", {"prompt": text, "add_special_tokens": False}
        )
        return response["tokens"]

    async def detokenize(self, tokens: list[list[int]]) -> list[str]:
        response = await self._post("/v1/detokenize", {"tokens": tokens})
        return response["text"]
