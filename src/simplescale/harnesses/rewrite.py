"""Single-document rewrite harnesses for the generator-scaling experiments.

``wiki`` is the Megadocs/WRAP Wikipedia-style rephrase (same prompt and system prompt as
``megadocs.rephrase``); ``guided`` is FinePhrase's guided rewrite prompt
(https://huggingface.co/spaces/HuggingFaceFW/finephrase#nemotron), sent with the same
system prompt so the two differ only in the user prompt. Both return
``{"rephrases", "finish_reasons", "source_truncated"}``; a ``finish_reasons`` entry of
``"length"`` means the output hit ``max_new_tokens``.

``nemotron`` and ``wofmt`` produce one rewrite per entry of ``task["views"]`` (same return
shape, plus ``views``): ``nemotron`` views are Nemotron-CC prompts (``diverse_qa``, ``distill``,
``knowledge_list``; NVIDIA-NeMo/Curator nemotron_cc/prompts.py, each with its published system
prompt and its leading answer header stripped); ``wofmt`` views are WebOrganizer format labels
(NoURL classifier names), each filled into the wiki prompt in place of "article ... as in texts
on Wikipedia".
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from .megadocs import REPHRASE, SYSTEM, _messages, _params, _truncate

GUIDED = """Below is a draft from an AI Assistant when trying to accomplish a task or solve a problem. Analyze and understand the task and problem(s) to be solved. Then pretend to be the expert who is most skillful to accomplish this task, and use detailed thinking and internal reasoning to identify a strategy and develop a plan about how to solve this problem. Experts usually apply meta-reasoning and planning to reason about how to best accomplish the task before jumping to a solution.
Deliberate meta-reasoning also involves reflection which can help identify issues and take a step back to explore other paths. Below are some generic examples of starting questions experts could ask themselves during the meta-reasoning process. The expert will come up with the most relevant questions that can help with their thinking process, which are also very specific to the task.
Consider these questions during your internal reasoning process:
- What is the core issue or problem that needs to be addressed? What are the key assumptions underlying this problem?
- How can I break down this problem into smaller, more manageable parts? How can I simplify the problem so that it is easier to solve?
- What kinds of solutions are typically produced for this kind of problem specification? Given the problem specification and the current best solution, what other possible solutions exist? If the current best solution is totally wrong, what other ways are there to think about the problem specifically?
- What is the best way to modify this current best solution, given what you know about these kinds of problem specifications?
- Am I on the right track? Check your progress so far.
- Develop a step by step plan internally.

Finally, rewrite the original content from the author's perspective, maintaining their voice and intent while making substantial improvements. Take information and details from the original draft whenever they are useful. The rewritten content should not be shorter than the original response. The improved version should have significantly better formatting and readability, with more coherent and in-depth reasoning, enhanced clarity, stronger structure, and removal of any noise or digression. Write as if you are the original author meaningfully improving their own work - not just making minor edits.
IMPORTANT: Your output must be ONLY the actual rewritten content itself - nothing else. Do NOT include any analysis, commentary, description, summary, or explanation about the improvements made. Do NOT add any meta-commentary like "This version improves..." or similar statements. Do NOT reference "the original draft" or "the draft" in your output. Output ONLY the content as if it were the final published piece that readers would see, with absolutely no additional text before or after it.
Original Draft:
{text}"""

assert SYSTEM  # both prompts are sent with the Megadocs system prompt

NEMOTRON_SYSTEM = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions."
NEMOTRON_DISTILL_SYSTEM = "You are an artificial intelligence assistant. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning."
DIVERSE_QA = """Task:
Read the text, ask questions and answer them.

Follow these instructions:
1. Ask diverse questions that require different cognitive skills or cover different aspects of the text.
2. Ask questions in various forms such as:
  - Yes/No questions that require determining whether a statement is true or false.
  - Open-ended questions that begin with words like what, how, when, where, why and who.
  - Multi-choice questions that offers two or more options to choose from. Include the options in the question.
  - Comparison questions that compare two quantities or objects and determine the relationship between them.
  - Reading comprehension questions that test the ability to understand and analyze the text.
  - Problem-solving questions that test the ability to solve mathematical, physical, or logical problems.
3. Focus on asking questions about factual information, important knowledge, or concrete details in the text.
4. Write questions and answers using clear and concise language.
5. Use plain text. Do not use Markdown.
6. Each question and answer pair should be on a separate line. Tag the question with "Question:" and the answer with "Answer:".

Text:
{text}

Task:
After reading the above text, ask up to 8 questions and provide the correct answers following the instructions. Give your response in this format:

Here are the questions and answers based on the provided text:
- Question: [first question] Answer: [first answer]
- Question: [second question] Answer: [second answer]
...."""
DISTILL = """Your task is to read and paraphrase the provided text following these instructions:
- Aim to create a condensed but accurate and informative version of the original text, not a simplistic summary.
- Capture and preserve the crucial information, key concepts, important values, factual details in the original text, while making it more readable and accessible.
- Retain technical terms, specialized vocabulary, and complex concepts.
- Retain examples, explanations of reasoning processes, and supporting evidence to maintain the text's depth and context.
- Only include information that is present in the original text. Do not adding new or unsubstantiated claims.
- Write the text in plain text without formatting.

Here is the text:
{text}

Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions. Begin your response with "Paraphrased Text:"."""
KNOWLEDGE_LIST = """Review the text and extract the key information. Follow these instructions:
- Carefully read the above text and provide a concise and organized list of factual information, concrete details, key concepts, and important numbers and statistics extracted from the text.
- Ensure each point is clear, specific, and supported by the original text.
- Ensure the extract text is information-dense and easier to learn from.
- Do not add titles or headings.

Text:
{text}

Task:
Extract the factual information, concrete details, and key concepts from the above text following the instructions."""
# view -> (system prompt, user template, leading header to strip)
NEMOTRON = {
    "diverse_qa": (NEMOTRON_SYSTEM, DIVERSE_QA, "Here are the questions and answers based on the provided text:"),
    "distill": (NEMOTRON_DISTILL_SYSTEM, DISTILL, "Paraphrased Text:"),
    "knowledge_list": (NEMOTRON_SYSTEM, KNOWLEDGE_LIST, None),
}

WOFMT = """For the following document, regardless of its original content or formatting, write a full {format} of the same content in high quality English language:

Document:
{text}

Rewritten document:"""
# WebOrganizer format label (FormatClassifier-NoURL) -> target phrase. Truncated, Spam / Ads and
# Legal Notices are not sensible targets; Knowledge Article is what the wiki prompt already writes.
FORMATS = {
    "Nonfiction Writing": "long-form nonfiction piece (e.g. an essay, profile, or editorial)",
    "Personal Blog": "personal blog post",
    "Tutorial": "tutorial or how-to guide",
    "News Article": "news article",
    "Academic Writing": "academic paper (e.g. a research paper, abstract, or literature review)",
    "Q&A Forum": "Q&A forum thread (e.g. on Stack Exchange or Quora)",
    "Product Page": "product or service page",
    "Listicle": "listicle (an article presented as a list)",
    "Structured Data": "structured data document (e.g. a table, glossary, or datasheet)",
    "Comment Section": "discussion forum thread with multiple posts or comments",
    "Documentation": "technical documentation (e.g. API documentation or a README)",
    "News (Org.)": "organizational announcement (e.g. a press release)",
    "Audio Transcript": "transcript of spoken language (e.g. an interview, lecture, or podcast)",
    "FAQ": "FAQ page",
    "Content Listing": "content listing page (e.g. a catalog or directory with short entries)",
    "Customer Support": "customer support page (e.g. a troubleshooting guide)",
    "Creative Writing": "piece of creative writing (e.g. a short story or poem)",
    "About (Org.)": "organization's About page",
    "About (Pers.)": "personal About page",
    "User Review": "user review",
}


async def _rewrite(template: str, task: dict[str, Any], llm: Any) -> dict[str, Any]:
    text, truncated = await _truncate(task, llm)
    params = _params(task, 1024)
    params["n"] = int(task.get("generations", 1))
    if params["n"] < 1:
        raise ValueError("generations must be positive")
    response = await llm.chat(
        messages=_messages(template.replace("{text}", text)),
        sampling_params=params,
    )
    choices = response["choices"]
    return {
        "rephrases": [c["message"]["content"] for c in choices],
        "finish_reasons": [c.get("finish_reason") for c in choices],
        "source_truncated": truncated,
    }


async def wiki(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Wikipedia-style rephrase (Megadocs/WRAP prompt)."""
    return await _rewrite(REPHRASE, task, llm)


async def guided(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """FinePhrase guided rewrite."""
    return await _rewrite(GUIDED, task, llm)


def _strip_header(text: str, header: str | None) -> str:
    if text and header:
        text = re.sub(r"^\s*" + re.escape(header) + r"[ \t]*\n?", "", text, count=1, flags=re.IGNORECASE)
    return text


async def _views(task: dict[str, Any], llm: Any, build) -> dict[str, Any]:
    """One rewrite per ``task['views']`` entry; ``build(view, text)`` -> (messages, header)."""
    text, truncated = await _truncate(task, llm)
    params = {**_params(task, 1024), "n": 1}
    views = list(task["views"])
    built = [build(v, text) for v in views]
    responses = await asyncio.gather(*(llm.chat(messages=m, sampling_params=params) for m, _ in built))
    choices = [r["choices"][0] for r in responses]
    return {
        "rephrases": [_strip_header(c["message"]["content"], h) for c, (_, h) in zip(choices, built)],
        "finish_reasons": [c.get("finish_reason") for c in choices],
        "views": views,
        "source_truncated": truncated,
    }


async def nemotron(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Nemotron-CC prompts, one rewrite per view in ``task['views']``."""
    def build(view, text):
        system, template, header = NEMOTRON[view]
        return [{"role": "system", "content": system},
                {"role": "user", "content": template.replace("{text}", text)}], header
    return await _views(task, llm, build)


async def wofmt(task: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Wiki prompt retargeted to a WebOrganizer format, one rewrite per label in ``task['views']``."""
    def build(view, text):
        return _messages(WOFMT.replace("{format}", FORMATS[view]).replace("{text}", text)), None
    return await _views(task, llm, build)
