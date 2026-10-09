import json

from simplescale.harnesses.megadocs import latent_thoughts, rephrase
from simplescale.worker import _read_tasks


class LLM:
    async def tokenize(self, text):
        return list(range(len(text)))

    async def detokenize(self, pieces):
        return ["abcde"[piece[0] : piece[-1] + 1] for piece in pieces]

    async def chat(self, *, messages, sampling_params):
        n = sampling_params.get("n", 1)
        return {"choices": [{"message": {"content": "thought"}}] * n}


async def test_megadocs_harnesses():
    assert await rephrase({"text": "abcde", "generations": 2}, LLM()) == {
        "rephrases": ["thought", "thought"],
        "source_truncated": False,
    }
    result = await latent_thoughts({"text": "abcde", "generations": 2}, LLM())
    assert result == {
        "megadoc": "ab<think>thought</think>cd<think>thought</think>e",
        "generations": 2,
        "source_truncated": False,
    }
    assert await rephrase({"text": "abcde", "max_document_tokens": 3}, LLM()) == {
        "rephrases": ["thought"],
        "source_truncated": True,
    }
    result = await latent_thoughts(
        {"text": "abcde", "generations": 2, "max_document_tokens": 3}, LLM()
    )
    assert result == {
        "megadoc": "a<think>thought</think>b<think>thought</think>c",
        "generations": 2,
        "source_truncated": True,
    }


def test_pointer_task(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"id":"source-id","text":"hello"}\n')
    task = {
        "_ref": {"path": str(raw), "offset": 0, "length": raw.stat().st_size},
        "sampling_params": {"temperature": 1.0},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(task) + "\n")
    lease = {
        "manifest": str(manifest),
        "chunk": {"start": 0, "end": manifest.stat().st_size},
    }
    assert _read_tasks(lease) == [
        {
            "id": "source-id",
            "text": "hello",
            "sampling_params": {"temperature": 1.0},
        }
    ]
