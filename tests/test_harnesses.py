from simplescale.harnesses.megadocs import latent_thoughts, rephrase


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
        "rephrases": ["thought", "thought"]
    }
    result = await latent_thoughts({"text": "abcde", "generations": 2}, LLM())
    assert result == {
        "megadoc": "ab<think>thought</think>cd<think>thought</think>e",
        "generations": 2,
    }
