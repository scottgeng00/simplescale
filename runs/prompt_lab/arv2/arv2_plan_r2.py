"""Active reading v2 plan-prompt builder (prompt-lab copy; the harness version lives in
simplescale/harnesses/multisample.py). build_plan(text, n, rng) -> stage-1 user message.

Per call the builder samples (a) one length target and one writing-quality level per prompt slot, (b) a random
menu of web forms and organizing principles, (c) one fictional worked example (of EXAMPLES), so that the
collection-level choices differ from call to call instead of collapsing onto the same handful of genres."""
import random
import re

VERSION = "r2"

TEMPLATE = None  # set by load()

LENGTHS = [  # (label, weight, min source words needed)
    ("very short (about 40-120 words)", 0.14, 0),
    ("short (about 120-250 words)", 0.20, 0),
    ("medium (about 250-500 words)", 0.30, 0),
    ("long (about 500-900 words)", 0.24, 450),
    ("very long (about 900-1500 words)", 0.12, 900),
]
QUALITY = [("polished", 0.30), ("everyday", 0.40), ("rough", 0.30)]
ROUGH_MAX_LENGTH = 2  # rough only for very short / short / medium (index into LENGTHS)
FORMS = [  # public web page types, weighted roughly toward what a web crawl contains
    ("news article", 6), ("local news brief", 2), ("press release", 2), ("blog post", 8), ("personal blog post", 5),
    ("opinion column", 3), ("editorial", 1), ("letter to the editor", 1), ("forum thread with replies", 5),
    ("single forum reply", 3), ("Q&A-site answer", 5), ("comment under an article", 3), ("social media post", 2),
    ("encyclopedia-style entry", 4), ("glossary entry", 1), ("how-to guide", 4), ("FAQ page", 3),
    ("educational web page", 4), ("study notes posted online", 2), ("classroom handout", 2), ("quiz with answers", 1),
    ("newsletter item", 2), ("community bulletin item", 1), ("book or media review", 2), ("product or service page", 2),
    ("customer-style review", 1), ("comparison article", 2), ("listicle", 3), ("myth-versus-fact page", 1),
    ("podcast or video description", 1), ("reading or discussion guide", 1), ("short summary for a link-sharing site", 2),
    ("case study", 1), ("checklist", 1), ("troubleshooting page", 1), ("organization's about or program page", 1),
    ("event announcement", 1), ("travel or visitor guide entry", 1), ("timeline page", 1), ("explainer for children", 1),
    ("in-depth article for specialists", 2), ("beginner's introduction", 2), ("pros and cons piece", 1),
    ("wiki talk-page discussion", 1), ("transcript of a short talk", 1), ("hobbyist or fan page", 1),
]


def _pick(rng, items):
    return rng.choices([x[0] for x in items], weights=[x[1] for x in items])[0]


def slots(text, n, rng):
    words = len(text.split())
    lengths = [x for x in LENGTHS if words >= x[2]]
    out = []
    for i in range(n):
        length = _pick(rng, lengths)
        quality = _pick(rng, QUALITY if LENGTHS.index(next(x for x in LENGTHS if x[0] == length)) <= ROUGH_MAX_LENGTH else QUALITY[:2])
        forms = []
        while len(forms) < 3:
            f = _pick(rng, FORMS)
            if f not in forms:
                forms.append(f)
        out.append(f"- Prompt {i + 1}: {length}; {quality}; suggested forms: {forms[0]}, {forms[1]}, or {forms[2]}.")
    return "\n".join(out)


def load(path):
    global TEMPLATE
    TEMPLATE = open(path).read()


PREVIOUS = """### Prompts already written for this source

These prompts were written for this source in an earlier request. Each of your {n} new prompts must have a primary focus, purpose, and form different from all of them; cover content units they leave out or use, and do not write another broad treatment.

{items}

"""


def previous_block(prev, n):
    items = "\n".join(f"- {t}: {' '.join(p.split()[:30])}..." for t, p in prev)
    return PREVIOUS.replace("{items}", items).replace("{n}", str(n))


def build_plan(text, n, rng, examples, prev=None):
    ex = rng.choice(examples)
    return (TEMPLATE.replace("{previous}", previous_block(prev, n) if prev else "").replace("{n}", str(n)).replace("{slots}", slots(text, n, rng))
            .replace("{examples}", ex).replace("{text}", text))


EXEC = """Source document:
<document>
{text}
</document>

{prompt}

Rules for the finished page:
- Plain text as it appears when copied from a web page: no Markdown (#, **, *, pipe tables) and no HTML.
- Write it as the page itself. Never mention a source, "the text", a prompt, instructions, or that it is a rewrite.
- Keep every fact, number, name, attribution, and hedge as the material states it. Add no facts, quotations, dates, names, headers, sign-offs, or placeholders that it does not contain.
- Use your own words; quote only short phrases.
- Stay close to the requested length."""


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


_PLACEHOLDER = re.compile(r"\[(?:Your|Insert|Date|Name|Today|Colleague|Company|Recipient|Current|Author|Link|Phone|Email)[^\]\n]{0,40}\]", re.I)
_SRCREF = re.compile(r"\b(?:the|this) (?:provided|given|original|source|above|supplied) (?:text|document|material|source|article|passage)\b|"
                     r"\bsource (?:document|text|material)\b|\bprovided (?:text|document|article|material|passage)\b|"
                     r"\b(?:the|this) prompt\b|\baccording to the text\b|\bthe text (?:says|states|notes|mentions|explains|describes|identifies)\b", re.I)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _ngrams(text, k=8):
    w = re.findall(r"\w+", text.lower())
    return [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]


def copy_rate(text, source, k=8):
    """Share of the text's word 8-grams that also occur in the source."""
    g = _ngrams(text, k)
    if not g:
        return 0.0
    s = set(_ngrams(source, k))
    return sum(x in s for x in g) / len(g)


def repetition(text, k=10):
    """Share of word 10-grams that are repeats of an earlier 10-gram (degenerate loops)."""
    g = _ngrams(text, k)
    return 1 - len(set(g)) / len(g) if g else 0.0


def reject_reason(text, source):
    """Why a finished view should be dropped (None = keep)."""
    if _PLACEHOLDER.search(text):
        return "placeholder"
    if _SRCREF.search(text):
        return "source_reference"
    if _CJK.search(text) and not _CJK.search(source):
        return "script_switch"
    if repetition(text) > 0.2:
        return "repetition"
    if copy_rate(text, source) > 0.5:
        return "copy"
    return None
