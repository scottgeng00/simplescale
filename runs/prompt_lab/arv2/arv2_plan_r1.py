"""Active reading v2 plan-prompt builder (prompt-lab copy; the harness version lives in
simplescale/harnesses/multisample.py). build_plan(text, n, rng) -> stage-1 user message.

Per call the builder samples (a) one length target and one writing-quality level per prompt slot, (b) a random
menu of web forms and organizing principles, (c) one fictional worked example (of EXAMPLES), so that the
collection-level choices differ from call to call instead of collapsing onto the same handful of genres."""
import random
import re

VERSION = "r1"

TEMPLATE = None  # set by load()

LENGTHS = [  # (label, weight, min source words needed)
    ("very short: about 40-120 words", 0.14, 0),
    ("short: about 120-250 words", 0.20, 0),
    ("medium: about 250-500 words", 0.30, 0),
    ("long: about 500-900 words", 0.24, 250),
    ("very long: about 900-1500 words", 0.12, 600),
]
QUALITY = [
    ("carefully edited", 0.30),
    ("competent everyday writing", 0.40),
    ("rough, low-effort writing", 0.30),
]
FORMS = [
    "news report", "local news brief", "press release", "encyclopedia-style entry", "wiki talk-page discussion",
    "personal blog post", "opinion column", "letter to the editor", "forum question with several replies",
    "single forum reply", "Q&A-site answer", "blog comment", "social media thread", "how-to guide", "FAQ page",
    "glossary entries", "timeline page", "student study notes", "lecture notes", "classroom handout",
    "quiz with answer key", "exam review sheet", "newsletter item", "club or community bulletin", "book or media review",
    "product or service review", "comparison article", "listicle", "myth-versus-fact page", "podcast show notes",
    "reading-group discussion guide", "executive summary", "policy brief", "explainer for children",
    "explainer for specialists", "case study write-up", "checklist", "troubleshooting page", "course syllabus excerpt",
    "museum or exhibit label text", "tour or travel guide entry", "debate-style pro and con piece",
    "worked example with explanation", "short summary for a link-sharing site", "fan or hobbyist page",
    "trivia list", "informal notes to self", "email to a colleague", "reference page with sections",
    "speech or talk transcript", "annotated bibliography-style entry", "school assignment answer",
    "beginner's introduction", "advanced deep dive", "historical background piece", "practical tips page",
    "editorial", "community announcement", "parent or caregiver guide", "frequently confused terms page",
]
PRINCIPLES = [
    "mechanism (how it works)", "chronology", "comparison", "practical application", "misconception and correction",
    "exception or edge case", "consequence (what follows from it)", "decision (what someone should do)",
    "cause and effect", "problem and solution", "question and answer", "classification into kinds",
    "definition and examples", "argument and counterargument", "step-by-step procedure", "single detail in depth",
    "big-picture overview", "people and roles", "numbers and what they mean", "local or personal relevance",
]


def _pick(rng, items):
    return rng.choices([x[0] for x in items], weights=[x[1] for x in items])[0]


def slots(text, n, rng):
    words = len(text.split())
    lengths = [x for x in LENGTHS if words >= x[2]]
    out = []
    for i in range(n):
        out.append(f"- Prompt {i + 1}: length {_pick(rng, lengths)}; {_pick(rng, QUALITY)}.")
    return "\n".join(out)


def menu(n, rng):
    forms = rng.sample(FORMS, min(len(FORMS), 2 * n + 4))
    princ = rng.sample(PRINCIPLES, min(len(PRINCIPLES), n + 2))
    return "Forms: " + "; ".join(forms) + ".\nOrganizing principles: " + "; ".join(princ) + "."


def load(path):
    global TEMPLATE
    TEMPLATE = open(path).read()


def build_plan(text, n, rng, examples):
    ex = rng.choice(examples)
    return (TEMPLATE.replace("{n}", str(n)).replace("{slots}", slots(text, n, rng)).replace("{menu}", menu(n, rng))
            .replace("{examples}", ex).replace("{text}", text))


EXEC = ("Source document:\n<document>\n{text}\n</document>\n\n{prompt}\n\n"
        "Write the document as its author would publish it on the web. Use plain text as it would appear when copied "
        "from a web page: no Markdown symbols (#, **, *, tables with pipes) and no HTML. Do not mention a source "
        "document, these instructions, or that the text is a rewrite.")


_MD_HEAD = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t#]*$", re.M)
_MD_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_MD_ITAL = re.compile(r"(?<![\*\w])\*(?=[^\s\*])([^\*\n]+?)(?<=[^\s\*])\*(?![\*\w])")
_MD_RULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$\n?", re.M)
_MD_BULLET = re.compile(r"^([ \t]*)\*[ \t]+", re.M)


def clean_markdown(text):
    """Deterministic markdown strip that keeps the content: heading markers (#), bold/italic markers (** __ *),
    horizontal rules; '* ' bullets become '- '. Tables, code, and '- ' / '1.' lists are left alone."""
    text = _MD_HEAD.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\2", text)
    text = _MD_ITAL.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = _MD_BULLET.sub(r"\1- ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
