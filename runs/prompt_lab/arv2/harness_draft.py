

# Active Reading v2 (``active_reading_v2``): stage 1 (thinking on) plans document-specific prompts for alternative
# web documents, stage 2 (thinking off) writes one document per prompt. Developed in
# simplescale/runs/prompt_lab/arv2 (iterations.md); the plan prompt, slot lists, example pool, execution wrapper
# and filters below are that lab's final version (arv2_plan.py, plan_prompt_<final>.txt, examples_<final>.txt).
ARV2_PLAN = """__PLAN__"""
ARV2_PREVIOUS = """__PREVIOUS__"""
ARV2_EXAMPLES = __EXAMPLES__
ARV2_EXEC = """__EXEC__"""
ARV2_LENGTHS = __LENGTHS__
ARV2_QUALITY = __QUALITY__
ARV2_FORMS = __FORMS__
ARV2_LENGTH_FRAC, ARV2_ROUGH_MAX_WORDS = __LENGTH_FRAC__, __ROUGH_MAX_WORDS__
_ARV2_HEAD = re.compile(r"^##(?!#)[ \t]*(.*)$", re.MULTILINE)
_MD_HEAD = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t#]*$", re.M)
_MD_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_MD_ITAL = re.compile(r"(?<![\*\w])\*(?=[^\s\*])([^\*\n]+?)(?<=[^\s\*])\*(?![\*\w])")
_MD_RULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$\n?", re.M)
_MD_BULLET = re.compile(r"^([ \t]*)\*[ \t]+", re.M)
_PLACEHOLDER = __PLACEHOLDER__
_SRCREF = __SRCREF__
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def clean_markdown(text: str) -> str:
    """Deterministic markdown strip that keeps the content: heading markers (#), bold/italic markers (** __ *),
    horizontal rules; '* ' bullets become '- '. Tables, code, and '- ' / '1.' lists are left alone."""
    text = _MD_HEAD.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\2", text)
    text = _MD_ITAL.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = _MD_BULLET.sub(r"\1- ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ngrams(text: str, k: int) -> list[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]


def arv2_reject(text: str, source: str) -> str | None:
    """Why a finished view is dropped (None = keep): placeholder, source_reference, script_switch, repetition (10-gram
    repeats > 20%), copy (> 50% of its 8-grams in the source)."""
    if _PLACEHOLDER.search(text):
        return "placeholder"
    if _SRCREF.search(text):
        return "source_reference"
    if _CJK.search(text) and not _CJK.search(source):
        return "script_switch"
    g = _ngrams(text, 10)
    if g and 1 - len(set(g)) / len(g) > 0.2:
        return "repetition"
    g = _ngrams(text, 8)
    if g:
        s = set(_ngrams(source, 8))
        if sum(x in s for x in g) / len(g) > 0.5:
            return "copy"
    return None


def _arv2_pick(rng: random.Random, items: list) -> Any:
    return rng.choices(items, weights=[x[1] for x in items])[0]


def arv2_slots(text: str, n: int, rng: random.Random) -> str:
    """One assignment per prompt: a length (only lengths whose typical size <= ARV2_LENGTH_FRAC x the source's words),
    a quality (rough only up to ARV2_ROUGH_MAX_WORDS), a form and an alternative (rough: rough-plausible forms)."""
    words = len(text.split())
    lengths = [x for x in ARV2_LENGTHS if x[2] <= ARV2_LENGTH_FRAC * words] or ARV2_LENGTHS[:1]
    out = []
    for i in range(n):
        length = _arv2_pick(rng, lengths)
        quality = _arv2_pick(rng, ARV2_QUALITY if length[2] <= ARV2_ROUGH_MAX_WORDS else ARV2_QUALITY[:2])[0]
        pool = [f for f in ARV2_FORMS if f[2]] if quality == "rough" else ARV2_FORMS
        form = _arv2_pick(rng, pool)[0]
        alt = form
        while alt == form:
            alt = _arv2_pick(rng, pool)[0]
        out.append(f"- Prompt {i + 1}: {length[0]}; {quality}; form: {form} (alternative: {alt}).")
    return "\n".join(out)


def arv2_plan_message(text: str, n: int, rng: random.Random, previous: list[tuple[str, str]] | None = None) -> str:
    prev = ""
    if previous:
        items = "\n".join(f"- {t}: {' '.join(p.split()[:30])}..." for t, p in previous)
        prev = ARV2_PREVIOUS.replace("{items}", items).replace("{n}", str(n))
    example = rng.choice(ARV2_EXAMPLES)
    return (ARV2_PLAN.replace("{previous}", prev).replace("{n}", str(n)).replace("{slots}", arv2_slots(text, n, rng))
            .replace("{examples}", example).replace("{text}", text))


def split_prompts(reply: str) -> list[tuple[str, str]]:
    """(title, prompt) for each ``##`` section of a plan reply; empty prompts dropped."""
    heads = list(_ARV2_HEAD.finditer(reply))
    out = []
    for i, h in enumerate(heads):
        body = reply[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(reply)].strip()
        if body:
            out.append((h.group(1).strip(), body))
    return out


def arv2_views(words: int, task: dict[str, Any]) -> int:
    """Views for a source of ``words`` words: one per ``words_per_view`` (default 30), clamped to
    [``min_views``, ``max_views``] (default 12, 48)."""
    per, lo, hi = int(task.get("words_per_view", 30)), int(task.get("min_views", 12)), int(task.get("max_views", 48))
    return max(lo, min(hi, -(-words // per)))


async def active_reading_v2(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Active Reading v2. The source gets ``arv2_views`` views, planned in calls of at most ``prompts_per_call``
    (default 12) prompts; each later call sees the earlier calls' prompts (titles + first words) and must pick other
    content and forms. A plan call is retried once if it yields fewer prompts than asked. Each prompt is executed as
    soon as its plan returns (thinking off). Views are markdown-stripped (``clean_markdown``); a view failing
    ``arv2_reject`` keeps its text but gets ``finish_reason`` "rejected:<reason>", so export drops it.
    Randomness (slots, forms, example) is seeded by the source id / row and ``seed`` (default 0)."""
    text, truncated = await _truncate(task, llm)
    rng = random.Random(f"{task.get('seed', 0)}-{task.get('id', '')}-{task.get('source_subset', '')}-{task.get('source_row', task.get('src_row', ''))}")
    total = int(task["views"]) if "views" in task else arv2_views(len(text.split()), task)
    per_call = int(task.get("prompts_per_call", 12))
    k = -(-total // per_call)
    sizes = [total // k + (i < total % k) for i in range(k)]
    think = bool(task.get("strategy_thinking", True))
    planning = _params(task, thinking=think, max_new_tokens=int(task.get("strategy_max_new_tokens", 16384)))
    tokens, plans, prompts, jobs = 0, [], [], []
    for n in sizes:
        new: list[tuple[str, str]] = []
        for _ in range(2):
            response = await llm.chat(messages=[{"role": "user", "content": arv2_plan_message(text, n, rng, prompts or None)}],
                                      sampling_params=planning)
            tokens += _tokens(response)
            reply = _answer(response["choices"][0], think)
            new = split_prompts(reply)[:n]
            if len(new) >= n:
                break
        plans.append(reply)
        jobs += [asyncio.ensure_future(llm.chat(
            messages=[{"role": "user", "content": ARV2_EXEC.replace("{text}", text).replace("{prompt}", p)}],
            sampling_params=_params(task))) for _, p in new]
        prompts += new
    responses = await asyncio.gather(*jobs)
    samples = []
    for r in responses:
        for s in _samples(r, "arv2", _thinking(task)):
            s["text"] = clean_markdown(s["text"])
            why = arv2_reject(s["text"], text) if s["finish_reason"] == "stop" and s["text"] else None
            if why:
                s["finish_reason"] = f"rejected:{why}"
            samples.append(s)
    return {"source": _source(task), "samples": samples, "completion_tokens": tokens + sum(map(_tokens, responses)),
            "strategies": plans, "prompts": [{"title": t, "prompt": p} for t, p in prompts], "source_truncated": truncated}
