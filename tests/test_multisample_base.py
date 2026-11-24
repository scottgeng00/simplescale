from simplescale.harnesses.multisample import BASE_STOP, base_continue, clean_cut, degenerate

GOOD = " ".join(f"Sentence {i} talks about topic{i} with detail{i * 7} and note{i * 3}." for i in range(30))
GOOD2 = " ".join(f"Line {i} covers matter{i} plus item{i * 5} and fact{i * 2}." for i in range(30))


class BaseLLM:
    """Raw /generate stub: replays canned (text, finish_reason) replies in order; records every request."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def tokenize(self, text):
        return text.split(" ")

    async def detokenize(self, pieces):
        return [" ".join(p) for p in pieces]

    async def generate(self, *, prompt, sampling_params):
        self.calls.append((prompt, sampling_params))
        text, reason = self.replies[len(self.calls) - 1]
        return {"text": text, "meta_info": {"finish_reason": reason, "completion_tokens": 7}}


async def test_base_continue_prompt_params_and_postprocessing():
    replies = [
        (GOOD + "\n\n#", {"type": "stop", "matched": "\n## Document"}),
        (GOOD, {"type": "stop", "matched": 248044}),
        (GOOD + "\n\n" + GOOD2 + "\n\n" + GOOD2[:50], {"type": "length", "length": 8192}),
        ("too short.", {"type": "stop", "matched": 248044}),
        ("the same four words " * 100, {"type": "length", "length": 8192}),
        (" ".join(f"word{i}" for i in range(200)), {"type": "stop", "matched": 248044}),
    ]
    llm = BaseLLM(replies)
    out = await base_continue({"text": " doc\n", "generations": 6, "id": "u", "source_subset": "s", "source_row": 2,
                               "sampling_params": {"max_new_tokens": 100}}, llm)
    prompt, params = llm.calls[0]
    assert prompt == "## Document 1\n\ndoc\n\n## Document 2\n\n"
    assert params["stop"] == BASE_STOP and params["temperature"] == 1.0 and params["presence_penalty"] == 0.5
    assert params["max_new_tokens"] == 100 and "chat_template_kwargs" not in params
    s = out["samples"]
    assert out["source"] == {"id": "u", "source_subset": "s", "source_row": 2} and out["completion_tokens"] == 42
    assert s[0]["text"] == GOOD and s[0]["end"] == "header" and s[0]["finish_reason"] == "stop"
    assert s[1]["end"] == "eos" and s[2]["end"] == "length" and s[2]["finish_reason"] == "stop"
    assert [x["finish_reason"] for x in s[3:]] == ["filtered_short", "filtered_loop", "filtered_salad"]
    assert all(x["kind"] == "basegen" for x in s)


async def test_base_continue_truncates_at_a_boundary():
    doc = "First paragraph is the longer one here.\n\nSecond goes on and on and on"
    llm = BaseLLM([(GOOD, {"type": "stop", "matched": 248044})])
    out = await base_continue({"text": doc, "generations": 1, "max_document_tokens": 8}, llm)
    assert out["source_truncated"] and llm.calls[0][0] == "## Document 1\n\nFirst paragraph is the longer one here.\n\n## Document 2\n\n"


def test_clean_cut_and_degenerate():
    assert clean_cut("A b c. D e f. G h") == "A b c. D e f."
    assert clean_cut("x" * 10 + "\n" + "y" * 50) == "x" * 10 + "\n" + "y" * 50  # break too early: unchanged
    assert degenerate(GOOD) is None


async def test_base_continue_cuts_leaks_trims_length_and_drops_copies():
    leak = GOOD + "\n\n# Document 3\n\nmore stuff here"
    src = " ".join(f"Source sentence {i} has words w{i} and more words v{i}." for i in range(40))
    llm = BaseLLM([(leak, {"type": "stop", "matched": 248044}),
                   (GOOD + "\n\n" + GOOD.upper(), {"type": "length", "length": 8192}),
                   (src, {"type": "stop", "matched": 248044})])
    out = await base_continue({"text": src, "generations": 3}, llm)
    s = out["samples"]
    assert s[0]["text"] == GOOD and s[0]["raw_chars"] == len(leak)
    assert s[1]["text"] == GOOD and s[1]["finish_reason"] == "stop"  # length: first half, cut at the paragraph break
    assert s[2]["finish_reason"] == "filtered_copy"
    assert llm.calls[0][1]["stop_token_ids"] == [248045, 248046]
