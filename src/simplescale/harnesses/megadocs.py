"""Baselines from arXiv:2603.18534, Data-efficient pre-training by scaling synthetic megadocs."""

from __future__ import annotations

import asyncio
from typing import Any

SYSTEM = "Provide a direct response to the instructions without adding additional notes."
REPHRASE = """For the following document, regardless of its original content or formatting, write a full article of the same content in high quality English language as in texts on Wikipedia:

Document:
{text}

Rephrased article:"""
LATENT = """You are provided with a pair of web document prefix and suffix. Your task is to insert latent thoughts between them underlying the creation of the suffix conditioned on the prefix. The latent thoughts should include any missing background knowledge and any reasoning traces underlying each claim (especially, step-by-step derivations or logical reasoning).

Prefix: {prefix}

Suffix: {suffix}

Now provide the latent thoughts. Use concise, simple, and declarative language. Do not give any supporting remarks or references to the terms 'prefix' and 'suffix', as this output will go directly into a computer program. Do not apply any markdown formatting or text embellishments. Optimize the content to ensure every word is informative, avoid vague language like 'xxx is essential'. Emphasize on the suffix without repeating the content in the prefix. Focus on implicit reasoning and background knowledge that is not explicitly stated in the suffix, and use concrete logical reasoning or mathematical derivations when applicable."""


def _params(task: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 1,
        "max_new_tokens": max_new_tokens,
        "reasoning_effort": "none",
        **task.get("sampling_params", {}),
    }


def _messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]


def _outputs(response: dict[str, Any]) -> list[str]:
    return [choice["message"]["content"] for choice in response["choices"]]


def _token_limit(task: dict[str, Any]) -> int | None:
    limit = task.get("max_document_tokens")
    if limit is None:
        return None
    limit = int(limit)
    if limit < 1:
        raise ValueError("max_document_tokens must be positive")
    return limit


async def _truncate(task: dict[str, Any], llm: Any) -> tuple[str, bool]:
    text = task["text"]
    limit = _token_limit(task)
    if limit is None or len(text) * 4 <= limit or len(text.encode()) <= limit:
        return text, False
    tokens = await llm.tokenize(text)
    if len(tokens) <= limit:
        return text, False
    return (await llm.detokenize([tokens[:limit]]))[0], True


async def rephrase(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Generate G independent Wikipedia-style rephrases of ``task['text']``."""
    text, truncated = await _truncate(task, llm)
    params = _params(task, 1024)
    params["n"] = int(task.get("generations", 1))
    if params["n"] < 1:
        raise ValueError("generations must be positive")
    response = await llm.chat(
        messages=_messages(REPHRASE.format(text=text)),
        sampling_params=params,
    )
    return {"rephrases": _outputs(response), "source_truncated": truncated}


def _split(tokens: list[int], count: int) -> list[list[int]]:
    width, extra = divmod(len(tokens), count)
    sizes = [width + (i < extra) for i in range(count)]
    ends = []
    for size in sizes:
        ends.append((ends[-1] if ends else 0) + size)
    return [tokens[start:end] for start, end in zip([0, *ends], ends)]


async def latent_thoughts(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Insert G generated thoughts at equal-token split points in ``task['text']``."""
    requested = int(task.get("generations", 1))
    if requested < 1:
        raise ValueError("generations must be positive")
    tokens = await llm.tokenize(task["text"])
    limit = _token_limit(task)
    truncated = limit is not None and len(tokens) > limit
    if truncated:
        tokens = tokens[:limit]
    generations = min(requested, max(0, len(tokens) - 1))
    if not generations:
        text = (await llm.detokenize([tokens]))[0] if truncated else task["text"]
        return {
            "megadoc": text,
            "generations": 0,
            "source_truncated": truncated,
        }

    pieces = await llm.detokenize(_split(tokens, generations + 1))
    params = _params(task, 512)
    params["n"] = 1

    async def generate(split: int) -> str:
        response = await llm.chat(
            messages=_messages(
                LATENT.format(
                    prefix="".join(pieces[:split]),
                    suffix="".join(pieces[split:]),
                )
            ),
            sampling_params=params,
        )
        return _outputs(response)[0]

    thoughts = await asyncio.gather(
        *(generate(split) for split in range(1, generations + 1))
    )
    megadoc = pieces[0] + "".join(
        f"<think>{thought}</think>{piece}"
        for thought, piece in zip(thoughts, pieces[1:])
    )
    return {
        "megadoc": megadoc,
        "generations": generations,
        "source_truncated": truncated,
    }
