import json

from simplescale.harnesses.multisample import MIX, active_reading, prompt_mix, split_strategies, wiki


class LLM:
    """Echoes a canned reply; records every request."""

    def __init__(self, reply="out", strategies="## a\nx\n## b\ny"):
        self.reply, self.strategies, self.calls = reply, strategies, []

    async def tokenize(self, text):
        return list(range(len(text)))

    async def detokenize(self, pieces):
        return ["".join("d" for _ in p) for p in pieces]

    async def chat(self, *, messages, sampling_params):
        self.calls.append((messages, sampling_params))
        prompt = messages[-1]["content"]
        content = self.strategies if prompt.startswith("Consider the following document") else self.reply
        n = sampling_params["n"]
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}] * n,
                "usage": {"completion_tokens": 10 * n}}


async def test_wiki_n_samples_thinking_off_by_default():
    llm = LLM()
    out = await wiki({"text": "doc", "generations": 8, "id": "http://u", "src_row": 5, "source_subset": "dclm_8m_0",
                      "source_row": 3}, llm)
    assert out["source"] == {"id": "http://u", "src_row": 5, "source_subset": "dclm_8m_0", "source_row": 3}
    assert [s["text"] for s in out["samples"]] == ["out"] * 8 and out["completion_tokens"] == 80
    params = llm.calls[0][1]
    assert params["chat_template_kwargs"] == {"enable_thinking": False} and params["temperature"] == 1.0


async def test_thinking_strips_reasoning_and_drops_unfinished():
    out = await wiki({"text": "doc", "generations": 1, "thinking": True}, LLM(reply="plan...\n</think>\n\nanswer"))
    assert out["samples"][0]["text"] == "answer"
    out = await wiki({"text": "doc", "generations": 1, "thinking": True}, LLM(reply="still planning"))
    assert out["samples"][0]["text"] == ""


async def test_prompt_mix_one_sample_per_prompt():
    llm = LLM()
    out = await prompt_mix({"text": "doc"}, llm)
    assert [s["kind"] for s in out["samples"]] == MIX and len(llm.calls) == 8
    faq = next(m for m, _ in llm.calls if m[-1]["content"].startswith("Rewrite the document as a comprehensive FAQ"))
    assert faq[-1]["content"].endswith("\nDocument:\ndoc") and len(faq) == 1  # FinePhrase prompts: no system turn


async def test_active_reading_thinks_while_planning_only():
    strategies = "<think>hmm</think>Intro\n" + "".join(f"## S{i}\nbody {i}\n### sub\n" for i in range(9))
    llm = LLM(strategies=strategies)
    out = await active_reading({"text": "doc", "generations": 8}, llm)
    assert len(out["samples"]) == 8 and len(llm.calls) == 9
    plan, apply = llm.calls[0][1], llm.calls[1][1]
    assert plan["chat_template_kwargs"] == {"enable_thinking": True} and plan["max_new_tokens"] == 16384
    assert apply["chat_template_kwargs"] == {"enable_thinking": False} and apply["max_new_tokens"] == 4096
    assert "## S0\nbody 0\n### sub" in llm.calls[1][0][-1]["content"]


async def test_active_reading_retries_once_when_short():
    llm = LLM(strategies="</think>## only one")
    out = await active_reading({"text": "doc", "generations": 8}, llm)
    assert len(llm.calls) == 2 + 1 and len(out["samples"]) == 1


def test_split_strategies():
    assert split_strategies("x\n## A\n1\n### a\n## B\n2") == ["## A\n1\n### a", "## B\n2"]


async def test_prompt_pool_uniform_and_planned():
    from simplescale.harnesses import mix16
    from simplescale.harnesses.multisample import prompt_pool

    class PoolLLM(LLM):
        async def chat(self, *, messages, sampling_params):
            self.calls.append((messages, sampling_params))
            prompt = messages[-1]["content"]
            if prompt.startswith("Below is a document, followed by a menu"):
                content = json.dumps({k: 2 if k in ("code", "math") else 0 for k in mix16.TYPES})
            elif sampling_params["chat_template_kwargs"]["enable_thinking"]:
                content = "reasoning</think>\n```python\nprint(1)\n```"
            else:
                content = "According to the text, water boils at 100 C."
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}

    task = {"text": "Water boils at 100 C.", "source_subset": "s", "source_row": 1, "id": "u"}
    llm = PoolLLM()
    out = await prompt_pool(task, llm)
    kinds = [s["kind"] for s in out["samples"]]
    assert len(kinds) == len(set(kinds)) == 8 and kinds == out["types"] and "planner" not in out
    assert (await prompt_pool(task, PoolLLM()))["types"] == kinds  # seeded by the source
    for s, (m, p) in zip(out["samples"], llm.calls):
        think = s["kind"] in mix16.THINKING
        assert p["chat_template_kwargs"] == {"enable_thinking": think} and p["max_new_tokens"] == (12288 if think else 4096)
        if s["kind"] in mix16.NEW_PROMPTS and not think:
            assert s["text"] == "Water boils at 100 C." and s["finish_reason"] == "stop"
        if s["kind"] in MIX:
            assert s["text"].startswith("According to the text") and s["slots"] == {}
    llm = PoolLLM()
    out = await prompt_pool({**task, "planned": True}, llm)
    assert out["planner"]["status"] == "ok" and set(out["types"]) >= {"code", "math"} and len(out["types"]) == 8
    assert set(out["types"]) - {"code", "math"} <= set(mix16.UNIVERSAL)
    assert llm.calls[0][1]["temperature"] == 0.7 and llm.calls[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    code = next(s for s in out["samples"] if s["kind"] == "code")
    assert code["text"] == "```python\nprint(1)\n```" and code["slots"]["language"] in ("Python", "C", "Java")


def test_mix16_helpers():
    from simplescale.harnesses import mix16
    assert mix16.parse_scores("x") is None
    reply = "```json\n{" + ", ".join(f'"{k}": {i % 3}' for i, k in enumerate(mix16.TYPES)) + "}\n```"
    assert mix16.parse_scores(reply)["math"] == 0 and mix16.parse_scores(reply)["discussion"] == 1
    assert mix16.repair_references("Based on the document, the bridge opened in 1932.") == "The bridge opened in 1932."
    assert mix16.repair_references("It opened in 1932, as mentioned in the text.") == "It opened in 1932."
    assert mix16.source_reference("The provided text gives no date.", "x")
    assert mix16.source_reference("The document was signed in 1215 and the document states rights.", "x")
    assert not mix16.source_reference("The document states rights.", "Magna Carta: the document states ...")


async def test_prompt_pool_later_rounds():
    from simplescale.harnesses import mix16
    from simplescale.harnesses.multisample import prompt_pool
    task = {"text": "Water boils at 100 C.", "source_subset": "s", "source_row": 1, "id": "u"}
    r1 = await prompt_pool(task, LLM())
    r2 = await prompt_pool({**task, "seed": 1, "round": 2, "complement_seed": 0}, LLM())
    assert sorted(r1["types"] + r2["types"]) == sorted(mix16.TYPES) and r2["samples"][0]["round"] == 2
    llm = LLM()
    r3 = await prompt_pool({**task, "planned": True, "kinds": ["code", "code"], "round": 3}, llm)
    assert r3["types"] == ["code", "code"] and len(llm.calls) == 2 and "planner" not in r3
