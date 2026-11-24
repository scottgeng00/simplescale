"""Several synthetic samples per source document, for DCLM records (``text``, ``id``).

Three ways to get ``generations`` (default 8) samples per document:

- ``wiki``: direct sampling, the Megadocs full-document version of WRAP's Wikipedia-style rephrase, n samples
  from one request.
- ``prompt_mix``: one sample per prompt from ``MIX``: seven FinePhrase format prompts (article, discussion, faq,
  math, narrative, table, tutorial; commentary and explanation left out) plus the wiki rephrase.
- ``active_reading``: Active Reading. One call asks for ``generations`` document-specific learning strategies
  (markdown, each prefixed with ``##``), then each strategy is applied to the document in its own call.

Task fields: ``thinking`` (Qwen3.5 chat-template ``enable_thinking``, default off; reasoning is stripped from the
output and only the final answer kept), ``sampling_params`` (merged over ``DEFAULT_PARAMS``),
``max_document_tokens`` (truncates the source first; recorded as ``source_truncated``). ``active_reading`` uses
``thinking`` for the applications only; the strategies call has its own ``strategy_thinking`` (default on) and
``strategy_max_new_tokens`` (default 16384, room for the reasoning).

Output: ``{"source", "samples": [{"kind", "text", "finish_reason"}, ...], "completion_tokens", "source_truncated"}``;
``source`` copies the task's ``id`` (the DCLM URL), ``src_row`` and the manifest's ``source_subset`` / ``source_row``
(row within the subset) so every sample traces back to its document;
``completion_tokens`` counts every generated token for the document, reasoning and planning included.
``active_reading`` also returns the raw ``strategies`` reply. A sample cut off by ``max_new_tokens``
(``finish_reason == "length"``) is kept but should be dropped at export.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from .megadocs import REPHRASE, SYSTEM, _truncate

# Qwen3.5's recommended sampling for general tasks (thinking mode; also its non-thinking setting for reasoning
# tasks): T=1 for diversity across the samples of one document.
DEFAULT_PARAMS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_new_tokens": 4096}

# FinePhrase (HuggingFaceFW/finephrase) "Format" prompts, verbatim; [TEXT] -> the document.
FINEPHRASE = {
    "article": "Transform the document into a magazine-style feature article. Open with an engaging lead, then blend narrative storytelling with factual explanation. Maintain an accessible yet polished tone suitable for a general but informed readership. Output only the feature article, nothing else.",
    "discussion": "Reformulate the document as a dialogue between a teacher and a student. The teacher should guide the student toward understanding the key points while clarifying complex concepts. Keep the exchange natural, informative, and faithful to the original content. Output only the dialogue, nothing else.",
    "faq": "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions a reader would have about this topic, then provide clear, direct answers. Order questions logically—from foundational to advanced, or by topic area. Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing else.",
    "math": "Rewrite the document to create a mathematical word problem based on the numerical data or relationships in the text. Provide a step-by-step solution that shows the calculation process clearly. Create a problem that requires multi-step reasoning and basic arithmetic operations. It should include the question followed by a detailed solution showing each calculation step. Output only the problem and solution, nothing else.",
    "narrative": "Rewrite the document as a clear narrative that emphasizes the temporal sequence and causal relationships between events or steps. Reorganize the content to show how actions, events, or situations naturally flow from one to the next, making cause-and-effect relationships explicit. If describing a process or activity, show the logical progression of steps and explain why each step follows from the previous one. Output only the narrative, nothing else.",
    "table": "Rewrite the document as a structured table that organizes the key information, then generate one question-answer pair based on the table. First extract the main data points and organize them into a clear table format with appropriate headers using markdown table syntax with proper alignment. After the table, generate one insightful question that can be answered using the table data. Provide a clear, concise answer to the question based on the information in the table. Output only the table followed by the question-answer pair, nothing else.",
    "tutorial": "Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet points where appropriate to enhance clarity. Preserve all essential information while ensuring the style feels didactic and easy to follow. Output only the tutorial, nothing else.",
}
MIX = [*FINEPHRASE, "wiki"]

# Active Reading (Lin et al. 2025): stage 1 asks for document-specific learning strategies, stage 2 applies one.
# Few-shot examples of good strategies (written for other documents; no documents shown).
STRATEGY_EXAMPLES = """## Build a cause-and-effect timeline
List every dated event in the order it happened. For each one, write a sentence on what led to it and what it led to, so the whole sequence can be recalled as a single chain rather than as isolated dates.

## Teach the mechanism to a beginner
Explain how the process works step by step in plain language, as if to someone with no background, defining each technical term the first time it appears and saying why each step is needed.

## Quiz yourself on the exact details
Write a question for every specific number, name, date and definition, answer each one from memory, then check the answers against the text and rewrite the ones you got wrong."""
STRATEGIES = """Consider the following document. What are some strategies specific to this document that I can use to help me learn and remember all of the information contained? Give exactly {n} strategies. Use markdown and prefix each strategy with ##

Here are examples of good strategies written for other documents. Yours should be specific to this document, not copies of these:

{examples}

<document>
{text}
</document>"""
APPLY = """Here's a learning strategy:
{strategy}

Apply this strategy to the following document:
<document>
{text}
</document>"""
_THINK = re.compile(r"^.*?</think>", re.DOTALL)
_HEADING = re.compile(r"^##(?!#)", re.MULTILINE)


def _thinking(task: dict[str, Any]) -> bool:
    return bool(task.get("thinking", False))


def _params(task: dict[str, Any], n: int = 1, thinking: bool | None = None, **override: Any) -> dict[str, Any]:
    return {
        **DEFAULT_PARAMS,
        **task.get("sampling_params", {}),
        **override,
        "n": n,
        "chat_template_kwargs": {"enable_thinking": _thinking(task) if thinking is None else thinking},
    }


def _answer(choice: dict[str, Any], thinking: bool) -> str:
    """The final answer. No SGLang reasoning parser runs, so with thinking on the content is the reasoning, then
    </think>, then the answer; no </think> means the limit was hit while reasoning, and there is no answer."""
    content = choice["message"].get("content") or ""
    if not thinking:
        return content.strip()
    return _THINK.sub("", content, count=1).strip() if "</think>" in content else ""


def _tokens(response: dict[str, Any]) -> int:
    return (response.get("usage") or {}).get("completion_tokens", 0)


def _samples(response: dict[str, Any], kind: str, thinking: bool) -> list[dict[str, Any]]:
    return [{"kind": kind, "text": _answer(c, thinking), "finish_reason": c.get("finish_reason")}
            for c in response["choices"]]


def _source(task: dict[str, Any]) -> dict[str, Any]:
    return {k: task[k] for k in ("id", "src_row", "source_subset", "source_row") if k in task}


def _generations(task: dict[str, Any]) -> int:
    n = int(task.get("generations", 8))
    if n < 1:
        raise ValueError("generations must be positive")
    return n


def split_strategies(reply: str) -> list[str]:
    """The ``##`` sections of a strategies reply, heading line included; text before the first one is dropped."""
    starts = [m.start() for m in _HEADING.finditer(reply)]
    return [reply[a:b].strip() for a, b in zip(starts, starts[1:] + [len(reply)]) if reply[a:b].strip("# \n")]


async def wiki(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    text, truncated = await _truncate(task, llm)
    response = await llm.chat(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": REPHRASE.format(text=text)}],
        sampling_params=_params(task, _generations(task)),
    )
    return {"source": _source(task), "samples": _samples(response, "wiki", _thinking(task)), "completion_tokens": _tokens(response),
            "source_truncated": truncated}


def _mix_messages(kind: str, text: str) -> list[dict[str, str]]:
    if kind == "wiki":
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": REPHRASE.format(text=text)}]
    return [{"role": "user", "content": f"{FINEPHRASE[kind]}\nDocument:\n{text}"}]


async def prompt_mix(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    kinds = task.get("prompts", MIX)
    unknown = set(kinds) - set(MIX)
    if unknown:
        raise ValueError(f"unknown prompts {sorted(unknown)}")
    text, truncated = await _truncate(task, llm)
    responses = await asyncio.gather(*(
        llm.chat(messages=_mix_messages(kind, text), sampling_params=_params(task)) for kind in kinds))
    return {"source": _source(task), "samples": [s for kind, r in zip(kinds, responses) for s in _samples(r, kind, _thinking(task))],
            "completion_tokens": sum(map(_tokens, responses)), "source_truncated": truncated}


async def active_reading(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Strategies (retried once if the reply does not split into ``generations`` sections), then one call per
    strategy; keeps at most ``generations`` strategies."""
    n = _generations(task)
    text, truncated = await _truncate(task, llm)
    think = bool(task.get("strategy_thinking", True))
    planning = _params(task, thinking=think, max_new_tokens=int(task.get("strategy_max_new_tokens", 16384)))
    tokens, reply, strategies = 0, "", []
    for _ in range(2):
        response = await llm.chat(
            messages=[{"role": "user", "content": STRATEGIES.format(n=n, examples=STRATEGY_EXAMPLES, text=text)}], sampling_params=planning)
        tokens += _tokens(response)
        reply = _answer(response["choices"][0], think)
        strategies = split_strategies(reply)
        if len(strategies) >= n:
            break
    strategies = strategies[:n]
    responses = await asyncio.gather(*(
        llm.chat(messages=[{"role": "user", "content": APPLY.format(strategy=s, text=text)}],
                 sampling_params=_params(task)) for s in strategies))
    return {"source": _source(task), "samples": [s for r in responses for s in _samples(r, "active_reading", _thinking(task))],
            "completion_tokens": tokens + sum(map(_tokens, responses)), "strategies": reply,
            "source_truncated": truncated}


# --- base_continue: sampling from a BASE model (raw completion, no chat template) ---------------------------------
# The source is shown as "Document 1" and the model writes "Document 2" (``generations`` independent requests, which
# share the prompt's prefix cache). Decoding tuned on Qwen3.5-9B-Base (runs/prompt_lab/basesample): lower
# temperature loops, frequency/repetition penalties give punctuation-free word salad, presence penalty avoids both.
# A source cut by ``max_document_tokens`` is cut back to a paragraph / sentence boundary (mid-sentence cuts make the
# model continue Document 1 under the Document 2 header).
# Stops: a following document header (any of BASE_STOP), EOS, or a chat-template token (<|im_start|>/<|im_end|>:
# the base model has seen chat data and sometimes switches to a user/assistant exchange).
# Post-processing of every sample (smoke review: runs/prompt_lab/basesample/smoke_review.md):
#   1. cut at a leaked document marker the stops missed ("# Document 3", "Document 4", ...) or a "<think>" tag, and
#      strip a trailing "#"/whitespace;
#   2. a sample cut at max_new_tokens ("length") keeps only its first ``length_keep`` (task field, default 0.5) of
#      characters, cut back to a paragraph / sentence boundary: such samples degenerate late (drift, loops, salad);
#   3. degenerate samples (``degenerate``: short, loop, repeat, salad, gibberish, copy, script) get finish_reason
#      "filtered_<reason>" so export_multisample.py drops them (counted under its "length" drops).
# All other samples get finish_reason "stop", INCLUDING the trimmed "length" ones; ``end`` records how the sample ended
# ("eos", "header" = a following document header, "length"), ``tokens`` its generated tokens and ``raw_chars`` its
# length before post-processing.
BASE_PROMPT = "## Document 1\n\n{document}\n\n## Document 2\n\n"
BASE_STOP = ["\n## Document", "\n# Document", "\n### Document", "\n**Document", "\nDocument 3"]
BASE_STOP_TOKEN_IDS = [248045, 248046]  # Qwen3.5 <|im_start|>, <|im_end|>
BASE_PARAMS = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.5, "max_new_tokens": 8192}
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")
_WORD_END_PUNCT = re.compile(r"[.!?;:。！？；：][\"')\]]*$")
_LEAK = re.compile(r"^[ \t]*#{0,6}[ \t]*\**[ \t]*Doc(?:ument)?[ \t]*\d+\b|<think>|</think>|<\|[a-z_]+\|>", re.MULTILINE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_ALPHA4, _ALPHA5 = re.compile(r"[A-Za-z]{4,}"), re.compile(r"[A-Za-z]{5,}")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")


def clean_cut(text: str, keep: float = 0.5) -> str:
    """``text`` cut back to its last paragraph break, else line break, else sentence end, if that keeps at least
    ``keep`` of it; else unchanged."""
    floor = int(len(text) * keep)
    for sep in ("\n\n", "\n"):
        i = text.rfind(sep)
        if i >= floor:
            return text[:i].rstrip()
    ends = [m.end() for m in _SENTENCE_END.finditer(text, floor)]
    return text[:ends[-1]].rstrip() if ends else text


def loop_share(text: str, words: int = 400) -> float:
    """Share of repeated word 4-grams among the last ``words`` words."""
    w = text.split()[-words:]
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def punct_rate(text: str, words: int = 300) -> float:
    """Words ending in sentence punctuation (Latin or CJK) per word over the last ``words`` words."""
    w = text.split()[-words:]
    return sum(bool(_WORD_END_PUNCT.search(x)) for x in w) / max(1, len(w))


def copy_share(text: str, source: str, n: int = 8) -> float:
    """Share of the sample's word n-grams that occur in the source."""
    w, s = text.split(), source.split()
    g = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return len(g & {" ".join(s[i:i + n]) for i in range(len(s) - n + 1)}) / len(g) if g else 0.0


def foreign_share(text: str) -> float:
    """Share of letters outside the Latin scripts (Latin-1 + Latin Extended)."""
    letters = [c for c in text if c.isalpha()]
    return sum(ord(c) > 0x24F for c in letters) / max(1, len(letters))


def max_repeat(text: str) -> int:
    """Occurrences of the most frequent sentence / line of at least 8 words."""
    c: dict[str, int] = {}
    for s in _SENTENCES.split(text):
        if len(s.split()) >= 8:
            c[s.strip()] = c.get(s.strip(), 0) + 1
    return max(c.values(), default=0)


def gibberish(text: str) -> bool:
    """OCR-like junk: many all-caps or vowel-less words (thresholds above the maximum / 99.9th percentile of 3.9k
    real DCLM documents)."""
    w4, w5 = _ALPHA4.findall(text), _ALPHA5.findall(text)
    caps = sum(x.isupper() for x in w4) / max(1, len(w4))
    vowelless = sum(not _VOWEL.search(x) for x in w5) / max(1, len(w5))
    return caps > 0.25 or vowelless > 0.03


def degenerate(text: str, source: str | None = None, min_words: int = 50) -> str | None:
    """Why a sample is unusable ("short" < ``min_words`` words, "loop", "repeat" = a sentence 4+ times, "salad",
    "gibberish", "copy" = mostly the source's text, "script" = mostly non-Latin letters while the source is Latin),
    or None."""
    n = len(text.split())
    if n < min_words:
        return "short"
    if loop_share(text) > 0.3:
        return "loop"
    if max_repeat(text) >= 4:
        return "repeat"
    if n > 150 and punct_rate(text) < 0.02:
        return "salad"
    if gibberish(text):
        return "gibberish"
    if source is not None:
        if copy_share(text, source) > 0.5:
            return "copy"
        if foreign_share(text) > 0.3 and foreign_share(source) < 0.1:
            return "script"
    return None


def _base_sample(response: dict[str, Any], source: str, length_keep: float) -> dict[str, Any]:
    meta = response.get("meta_info") or {}
    reason = meta.get("finish_reason") or {}
    if reason.get("type") == "length":
        end = "length"
    else:
        end = "header" if isinstance(reason.get("matched"), str) else "eos"
    raw = response.get("text") or ""
    text = raw
    leak = _LEAK.search(text, 1)
    if leak:
        text = text[:leak.start()]
    if end == "length" and length_keep < 1:
        text = clean_cut(text[:int(len(text) * length_keep)], keep=0.7)
    text = text.rstrip("# \t\n").strip("\n")
    bad = degenerate(text, source, min_words=150 if end == "length" else 50)
    return {"kind": "basegen", "text": text, "finish_reason": f"filtered_{bad}" if bad else "stop", "end": end,
            "tokens": meta.get("completion_tokens", 0), "raw_chars": len(raw)}


async def base_continue(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    text, truncated = await _truncate(task, llm)
    if truncated:
        text = clean_cut(text)
    text = text.strip()
    prompt = BASE_PROMPT.format(document=text)
    params = {**BASE_PARAMS, "stop_token_ids": BASE_STOP_TOKEN_IDS, **task.get("sampling_params", {}),
              "stop": list(task.get("stop", BASE_STOP))}
    responses = await asyncio.gather(*(llm.generate(prompt=prompt, sampling_params=params)
                                       for _ in range(_generations(task))))
    keep = float(task.get("length_keep", 0.5))
    samples = [_base_sample(r, text, keep) for r in responses]
    return {"source": _source(task), "samples": samples, "completion_tokens": sum(s["tokens"] for s in samples),
            "source_truncated": truncated}


# Active Reading v2 (prompts and helpers in .arv2, from simplescale/runs/prompt_lab/arv2).
import random  # noqa: E402

from . import arv2 as _arv2  # noqa: E402


async def active_reading_v2(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Active Reading v2: planned alternative web documents. The source gets ``views`` views (default
    ``arv2.n_views``: one per ``words_per_view`` = 23 source words, clamped to [``min_views``, ``max_views``] =
    [8, 48]), planned (``strategy_thinking``, default on; ``strategy_max_new_tokens``, default 16384) in calls of at
    most ``prompts_per_call`` (24) prompts; each later call sees the earlier calls' prompts and must choose other
    content and forms. A plan call is retried once if it yields fewer usable prompts than asked (draft-like sections
    are dropped, ``arv2.bad_prompt``). Every prompt is written (``thinking``, default off) as soon as its plan returns.
    Each prompt is executed with ``arv2.exec_message`` (source framed as the writer's notes; references to "the
    text" in the prompt rewritten). Views go through ``arv2.finish_view`` (markdown strip, "The text notes that X" ->
    "X" repair); a finished view failing ``arv2.reject_reason`` keeps its text but gets ``finish_reason`` "rejected:<reason>", so export drops it. Slots, forms and example are drawn from a
    random.Random seeded by ``seed`` (default 0) and the source's subset / row / id. Output as ``active_reading``
    with kind "arv2", plus ``strategies`` (the plan replies) and ``prompts`` ([{title, prompt}])."""
    text, truncated = await _truncate(task, llm)
    rng = random.Random(f"{task.get('seed', 0)}-{task.get('source_subset', '')}-{task.get('source_row', '')}-{task.get('id', '')}")
    total = int(task["views"]) if "views" in task else _arv2.n_views(
        text, int(task.get("words_per_view", _arv2.WORDS_PER_VIEW)), int(task.get("min_views", _arv2.MIN_VIEWS)),
        int(task.get("max_views", _arv2.MAX_VIEWS)))
    calls = -(-total // int(task.get("prompts_per_call", _arv2.PER_CALL)))
    think = bool(task.get("strategy_thinking", True))
    planning = _params(task, thinking=think, max_new_tokens=int(task.get("strategy_max_new_tokens", 16384)))
    tokens, plans, prompts, jobs = 0, [], [], []
    for n in [total // calls + (i < total % calls) for i in range(calls)]:
        new: list[tuple[str, str]] = []
        for _ in range(2):
            message = _arv2.plan_message(text, n, rng, prev=prompts or None)
            response = await llm.chat(messages=[{"role": "user", "content": message}], sampling_params=planning)
            tokens += _tokens(response)
            reply = _answer(response["choices"][0], think)
            new = [p for p in _arv2.split_prompts(reply) if not _arv2.bad_prompt(p[1], text)][:n]
            if len(new) >= n:
                break
        plans.append(reply)
        jobs += [asyncio.ensure_future(llm.chat(
            messages=[{"role": "user", "content": _arv2.exec_message(text, p)}],
            sampling_params=_params(task))) for _, p in new]
        prompts += new
    responses = await asyncio.gather(*jobs)
    samples = []
    for r in responses:
        for s in _samples(r, "arv2", _thinking(task)):
            s["text"], why = _arv2.finish_view(s["text"], text)
            why = why if s["finish_reason"] == "stop" and s["text"] else None
            if why:
                s["finish_reason"] = f"rejected:{why}"
            samples.append(s)
    return {"source": _source(task), "samples": samples, "completion_tokens": tokens + sum(map(_tokens, responses)),
            "strategies": plans, "prompts": [{"title": t, "prompt": p} for t, p in prompts],
            "source_truncated": truncated}


# --- prompt_pool: 8 of a 16-prompt pool per document (mix16 / mix16plan; assets and helpers in .mix16) -------------
from . import mix16 as _mix16  # noqa: E402


def _pool_messages(kind: str, text: str, rng: random.Random) -> tuple[list[dict[str, str]], dict[str, str]]:
    if kind in MIX:
        return _mix_messages(kind, text), {}
    prompt, fill = _mix16.fill_template(kind, rng)
    return [{"role": "user", "content": f"{prompt}\nDocument:\n{text}"}], fill


async def prompt_pool(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """``generations`` (default 8) views per source from the 16-type pool of ``mix16.TYPES`` (the 8 ``MIX`` prompts,
    built as in ``prompt_mix``, plus the 8 ``mix16.NEW_PROMPTS`` with their {persona} / {audience} / {language} slots
    drawn at random). Types are drawn without replacement by ``mix16.choose_types``: uniformly, or with ``planned``
    weighted by a planner call (``mix16.PLANNER``, thinking off, ``mix16.PLANNER_PARAMS``) scoring every type 0/1/2;
    a reply that does not parse is retried once, then the draw falls back to uniform. Each view is one call, thinking
    off except for ``mix16.THINKING`` types (code, code_problem: thinking on, ``thinking_max_new_tokens``, default
    12288, reasoning stripped). Views of the NEW_PROMPTS types go through ``mix16.finish_view`` (source-reference
    repair, then "rejected:source_reference"); mix-type views are kept as ``prompt_mix`` returns them. Draws use a
    random.Random seeded by ``seed`` (default 0) and the source's subset / row / id. Later rounds: ``kinds`` (an explicit
    type list, no planner call) or ``complement_seed`` (the types a uniform draw with that seed did not pick, e.g. round
    2 of mix16 = the other 8 types); ``round`` (default 1) is recorded in every sample.

    Output as ``prompt_mix``, each sample with ``slots`` (the filled slots) and ``tokens`` (its call's completion
    tokens, reasoning included), plus ``types`` (the drawn types) and, with ``planned``, ``planner`` = {"scores"
    (None on fallback), "status" ("ok", "retry_ok", "fallback"), "replies"}."""
    n = _generations(task)
    text, truncated = await _truncate(task, llm)
    rng = random.Random(f"{task.get('seed', 0)}-{task.get('source_subset', '')}-{task.get('source_row', '')}-{task.get('id', '')}")
    tokens, planner, scores, kinds = 0, None, None, None
    key = f"{task.get('source_subset', '')}-{task.get('source_row', '')}-{task.get('id', '')}"
    if "kinds" in task:  # later rounds: types fixed by the manifest
        kinds = list(task["kinds"])
    elif "complement_seed" in task:  # the types a uniform draw with that seed did NOT pick
        used = _mix16.choose_types(random.Random(f"{task['complement_seed']}-{key}"), n)
        kinds = [k for k in _mix16.TYPES if k not in used]
    elif task.get("planned", False):
        params = _params(task, thinking=False, **_mix16.PLANNER_PARAMS, **task.get("planner_params", {}))
        replies, status = [], "fallback"
        for attempt in range(2):
            response = await llm.chat(messages=[{"role": "user", "content": _mix16.planner_message(text)}],
                                      sampling_params=params)
            tokens += _tokens(response)
            replies.append(_answer(response["choices"][0], False))
            scores = _mix16.parse_scores(replies[-1])
            if scores is not None:
                status = "ok" if attempt == 0 else "retry_ok"
                break
        planner = {"scores": scores, "status": status, "replies": replies}
    if kinds is None:
        kinds = _mix16.choose_types(rng, n, scores)
    jobs = []
    for kind in kinds:
        messages, fill = _pool_messages(kind, text, rng)
        think = kind in _mix16.THINKING
        extra = {"max_new_tokens": int(task.get("thinking_max_new_tokens", _mix16.THINKING_MAX_NEW_TOKENS))} if think else {}
        jobs.append((kind, fill, think, llm.chat(messages=messages, sampling_params=_params(task, thinking=think, **extra))))
    responses = await asyncio.gather(*(j[3] for j in jobs))
    samples = []
    for (kind, fill, think, _), r in zip(jobs, responses):
        s = _samples(r, kind, think)[0]
        if s["finish_reason"] == "stop" and s["text"]:
            s["text"], why = _mix16.finish_view(kind, s["text"], text)
            if why:
                s["finish_reason"] = f"rejected:{why}"
        samples.append({**s, "slots": fill, "tokens": _tokens(r), "round": int(task.get("round", 1))})
    out = {"source": _source(task), "samples": samples, "completion_tokens": tokens + sum(map(_tokens, responses)),
           "types": kinds, "source_truncated": truncated}
    if planner is not None:
        out["planner"] = planner
    return out
