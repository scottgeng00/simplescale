"""Gacha Decoding from arXiv:2610.01382, Eliciting Diverse Generations Through Instruction Following.

Each generation plans T decision axes, then walks them in one multi-turn
conversation: the model enumerates options for the next axis and external RNG
picks one uniformly. A final turn enumerates complete approaches (also picked
by RNG), and a fresh call writes the response from all sampled choices.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

PLAN = """Design a schema that partitions the space of possible outputs for the task below into sequential decision points, so that distinct paths through the schema yield {goal}. Use AT MOST {T} questions total. Each decision point is phrased as a question, one per turn; its options are enumerated later, conditioned on all earlier answers.
{rules}
Task: {task}
Output exactly: {{"questions": [{{"name": "...", "ask": "..."}}]}}"""
RULES = """
Rules:
- Only partition what the task leaves free; never re-decide what it fixes.
- First brainstorm the set of questions, and then commit to the ordering. A good ordering should place questions with the fewest dependencies first, and leave the largest room for variation in later questions.
- Prefer simple, enumerable questions at the start, deferring finer-grained, abstract, or interpretive questions for later.
- Each question must introduce new high-level information that meaningfully steers the output.
- Give each question a two-or-three-word name for labeling its answer.
"""
DECISION = """{ask} List at least {m} distinct options. Make sure that each option is consistent with all previously made choices.
Output exactly:
{{"options": ["...", "...", ...]}}"""
IDEA = """Given every choice above, list {k} distinct creative angles, themes, or approaches for the task. Be specific and varied.
Task: {task}
Output exactly:
{{"options": ["...", "...", ...]}}"""
EXECUTE = """Your task: {task}
Planned decisions:
{decisions}
{idea}Write the single best {artifact} that satisfies the request while (a) satisfying the task and all its constraints and (b) realizing every design choice above. If a choice cannot be honored within the request's exact constraints, sacrifice the choice, not the request. Give the answer itself without preamble or meta-commentary."""
DOCUMENT_AXES = "topic, subject domain, genre and format, intended audience, purpose, author voice and register, level of expertise, structure"
REPHRASE_AXES = "format and genre, intended audience, purpose, voice and register, structure, level of detail"
GUIDANCE = "When generating diverse {artifact}s, give particular weight to varying them along these axes: {axes}. Do not override choices already fixed by the task."
STORY_AXES = "genre, setting, narrative tone, narrator and point of view, premise, emotional resolution, dominant imagery, main character"

# Appendix B hyperparameters. The image planner drops the rules block. The
# document and rephrase presets are ours, for pretraining text; they mirror the story preset.
PRESETS: dict[str, dict[str, Any]] = {
    "rephrase": {
        "goal": f"rewrites of the source document that vary widely in: {REPHRASE_AXES}, while preserving its facts",
        "artifact": "rewrite",
        "decisions": 4,
        "options": 32,
    },
    "document": {
        "goal": f"documents that vary widely in: {DOCUMENT_AXES}",
        "artifact": "document",
        "decisions": 4,
        "options": 32,
    },
    "chat": {
        "goal": "diverse high-quality responses",
        "artifact": "response",
        "decisions": 4,
        "options": 24,
    },
    "story": {
        "goal": f"stories that vary widely in: {STORY_AXES}",
        "artifact": "story",
        "decisions": 4,
        "options": 32,
    },
    "image": {
        "goal": "diverse, high-quality generated images that would appear perceptually distinct even at a glance",
        "artifact": "caption",
        "decisions": 6,
        "options": 32,
        "rules": False,
    },
}
# Prompt C.7 analogue: baselines are told which axes diversity is judged on.
BASELINE_SYSTEM = {
    "story": GUIDANCE.format(artifact="story", axes=STORY_AXES),
    "document": GUIDANCE.format(artifact="document", axes=DOCUMENT_AXES),
    "rephrase": GUIDANCE.format(artifact="rewrite", axes=REPHRASE_AXES),
}
# Appended to the task in both Gacha and the baseline. Without it, Gacha's
# decisions drift toward simulated artifacts (mock logs, fictional registries).
REQUIREMENT = {
    "rephrase": "The rewrite must be a self-contained document that preserves the source's facts and core content. Any additional facts, figures, names, or claims must be supported by the claims in the source. Do not mention the source or that this is a rewrite.",
    "document": "The document must read as real published text: something an actual person or organization would write for real readers. Keep facts accurate. Do not produce simulated artifacts, mock-ups, placeholders, or invented products, institutions, or technologies presented as real. Fiction is fine only when it is clearly a work of fiction.",
}
_THINK = re.compile(r"^.*</think>", re.DOTALL)


def _content(response: dict[str, Any]) -> str:
    """Return the final answer, dropping reasoning if no SGLang parser removed it."""
    return _THINK.sub("", response["choices"][0]["message"]["content"] or "").strip()


def _json(text: str, key: str) -> list[Any]:
    """Return ``key`` from the last JSON object in ``text`` that has a non-empty list."""
    decoder = json.JSONDecoder()
    found: list[Any] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get(key), list) and value[key]:
            found = value[key]
    if not found:
        raise ValueError(f"no JSON list {key!r} in reply")
    return found


def _prompt(task: dict[str, Any]) -> str:
    """The task prompt plus the domain requirement; ``requirement: ""`` disables it."""
    requirement = task.get("requirement", REQUIREMENT.get(task.get("domain", "chat")))
    return f"{task['prompt']}\n\n{requirement}" if requirement else task["prompt"]


def _option(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


async def _sample(
    task: dict[str, Any], llm: Any, config: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    params = {**task.get("sampling_params", {}), "n": 1}
    # Plan, decision, and idea turns may use their own settings (e.g. thinking off);
    # the final document always uses sampling_params.
    planning = {**params, **task.get("planning_sampling_params", {}), "n": 1}
    prompt = _prompt(task)
    used = {"tokens": 0}

    async def ask(messages: list[dict[str, str]], final: bool = False) -> str:
        response = await llm.chat(messages=messages, sampling_params=params if final else planning)
        used["tokens"] += (response.get("usage") or {}).get("completion_tokens", 0)
        return _content(response)

    try:
        reply = await ask([{
            "role": "user",
            "content": PLAN.format(
                goal=config["goal"],
                T=config["decisions"],
                rules=RULES if config.get("rules", True) else "",
                task=prompt,
            ),
        }])
        axes = [
            q for q in _json(reply, "questions")
            if isinstance(q, dict) and q.get("name") and q.get("ask")
        ][: config["decisions"]]
    except Exception:
        axes = []

    # Failed turns are skipped, carrying the last committed choice forward.
    messages: list[dict[str, str]] = []
    prefix = f"Task: {prompt}\n\n"
    decisions = []
    for axis in axes:
        turn = prefix + DECISION.format(ask=axis["ask"], m=config["options"])
        try:
            reply = await ask([*messages, {"role": "user", "content": turn}])
            options = [_option(o) for o in _json(reply, "options")]
        except Exception:
            continue
        choice = rng.choice(options)
        messages += [
            {"role": "user", "content": turn},
            {"role": "assistant", "content": reply},
        ]
        prefix = f"Chosen {axis['name']}: {choice}\n"
        decisions.append({**axis, "options": options, "choice": choice})

    idea = None
    turn = prefix + IDEA.format(k=config["ideas"], task=prompt)
    try:
        reply = await ask([*messages, {"role": "user", "content": turn}])
        idea = rng.choice([_option(o) for o in _json(reply, "options")])
    except Exception:
        pass

    response = await ask(final=True, messages=[{
        "role": "user",
        "content": EXECUTE.format(
            task=prompt,
            decisions="".join(f"{d['name']}: {d['choice']}\n" for d in decisions),
            idea=(
                f"Overall approach (the guiding take — write the whole response as this): {idea}\n"
                if idea is not None else ""
            ),
            artifact=config["artifact"],
        ),
    }])
    return {"response": response, "decisions": decisions, "idea": idea, "generated_tokens": used["tokens"]}


async def generate(task: dict[str, Any], llm: Any) -> dict[str, list[dict[str, Any]]]:
    """Generate ``task['generations']`` Gacha samples for ``task['prompt']``.

    ``domain`` selects a preset (chat, story, image); ``goal``, ``artifact``,
    ``decisions`` (T), ``options`` (m), ``ideas`` (k), and ``rules`` override it.
    An optional ``seed`` makes the dice reproducible; LM sampling still is not.
    """
    config = {"ideas": 8, **PRESETS[task.get("domain", "chat")]}
    config |= {key: task[key] for key in (*config, "rules") if key in task}
    generations = int(task.get("generations", 1))
    if generations < 1:
        raise ValueError("generations must be positive")
    seed = task.get("seed")
    rngs = [
        random.Random(f"{seed}:{i}") if seed is not None else random.SystemRandom()
        for i in range(generations)
    ]
    samples = await asyncio.gather(
        *(_sample(task, llm, config, rng) for rng in rngs)
    )
    return {"samples": list(samples)}


async def direct(task: dict[str, Any], llm: Any) -> dict[str, list[dict[str, Any]]]:
    """Direct-sampling baseline with the same task fields and output shape.

    The bare prompt is the user turn. For story and document domains the
    system turn names the diversity axes, as the paper does for baselines.
    """
    generations = int(task.get("generations", 1))
    if generations < 1:
        raise ValueError("generations must be positive")
    system = BASELINE_SYSTEM.get(task.get("domain", "chat"))
    messages = [{"role": "user", "content": _prompt(task)}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    response = await llm.chat(
        messages=messages,
        sampling_params={**task.get("sampling_params", {}), "n": generations},
    )
    # SGLang reports usage for the whole request; split it evenly across samples.
    tokens = (response.get("usage") or {}).get("completion_tokens", 0) // generations
    return {
        "samples": [
            {"response": _content({"choices": [choice]}), "generated_tokens": tokens}
            for choice in response["choices"]
        ]
    }


# Either turn shape: the plan's {"questions": [{name, ask}]} or a decision / idea turn's {"options": [str]}.
PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "minItems": 1, "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "ask": {"type": "string"}},
            "required": ["name", "ask"], "additionalProperties": False}},
        "options": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


async def rewrite_wiki(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Gacha on the megadocs/WRAP wiki rephrase prompt, for DCLM records (``text``, ``id``).

    Truncates the source like ``rewrite.wiki``, then runs ``generate`` with the rephrase preset.
    Each of ``generations`` samples plans its own decision axes; the dice are seeded by the
    document id so reruns roll the same choices. Planning turns use the rewrite's sampling
    settings capped at ``planning_max_new_tokens``, constrained to JSON with a ``questions`` or
    ``options`` list (at T=1 the 9B otherwise breaks the JSON in ~5% of planning turns and ~25% of
    idea turns, and those choices are silently skipped).
    """
    from .megadocs import REPHRASE, _params, _truncate

    text, truncated = await _truncate(task, llm)
    params = _params(task, 1024)
    planning = {**params, "max_new_tokens": int(task.get("planning_max_new_tokens", 2048)),
                "response_format": {"type": "json_schema", "json_schema": {"name": "gacha", "schema": PLANNING_SCHEMA}}}
    out = await generate({
        "prompt": REPHRASE.format(text=text),
        "domain": "rephrase",
        "generations": int(task.get("generations", 1)),
        "seed": task.get("id"),
        "sampling_params": params,
        "planning_sampling_params": planning,
    }, llm)
    samples = out["samples"]
    return {
        "rephrases": [s["response"] for s in samples],
        "gacha": [{"decisions": [{"name": d["name"], "choice": d["choice"]} for d in s["decisions"]],
                   "idea": s["idea"], "generated_tokens": s["generated_tokens"]} for s in samples],
        "source_truncated": truncated,
    }
