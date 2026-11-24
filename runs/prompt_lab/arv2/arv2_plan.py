"""Active reading v2 plan-prompt builder (prompt-lab copy; the harness version lives in
simplescale/harnesses/multisample.py). build_plan(text, n, rng) -> stage-1 user message.

Per call the builder samples (a) one length target and one writing-quality level per prompt slot, (b) a random
menu of web forms and organizing principles, (c) one fictional worked example (of EXAMPLES), so that the
collection-level choices differ from call to call instead of collapsing onto the same handful of genres."""
import random
import re

VERSION = "r5"

TEMPLATE = None  # set by load()

LENGTHS = [  # (label, weight, typical words: allowed if typical <= LENGTH_FRAC x source words)
    ("very short (about 40-120 words)", 0.14, 80),
    ("short (about 120-250 words)", 0.20, 185),
    ("medium (about 250-500 words)", 0.30, 375),
    ("long (about 500-900 words)", 0.24, 700),
    ("very long (about 900-1500 words)", 0.12, 1200),
]
LENGTH_FRAC = 1.2
QUALITY = [("polished", 0.30), ("everyday", 0.40), ("rough", 0.30)]
ROUGH_MAX_WORDS = 500  # rough only for lengths up to medium
FORMS = [  # (public web page type, weight, rough writing plausible)
    ("news article", 6, 0), ("local news brief", 2, 0), ("press release", 1, 0), ("blog post", 8, 1),
    ("personal blog post", 5, 1), ("opinion column", 3, 0), ("editorial", 1, 0), ("letter to the editor", 1, 1),
    ("forum thread with replies", 5, 1), ("single forum reply", 3, 1), ("Q&A-site answer", 5, 1),
    ("comment under an article", 3, 1), ("social media post", 2, 1), ("encyclopedia-style entry", 3, 0),
    ("glossary entry", 1, 0), ("how-to guide", 4, 1), ("FAQ page", 3, 0), ("educational web page", 3, 0),
    ("study notes posted online", 1, 1), ("classroom handout", 1, 0), ("quiz with answers", 1, 0),
    ("newsletter item", 2, 0), ("community bulletin item", 1, 1), ("book or media review", 2, 1),
    ("product or service page", 1, 0), ("comparison article", 2, 0),
    ("listicle", 3, 1), ("myth-versus-fact page", 1, 0), ("podcast or video description", 1, 1),
    ("short summary for a link-sharing site", 2, 1), ("case study", 1, 0), ("checklist", 1, 1),
    ("troubleshooting page", 1, 1), ("organization's about or program page", 1, 0), ("timeline page", 1, 0),
    ("explainer for children", 1, 0), ("in-depth article for specialists", 2, 0), ("beginner's guide", 2, 0),
    ("pros and cons piece", 1, 1), ("hobbyist or fan page", 2, 1), ("question posted to a forum", 2, 1),
    ("mailing-list post", 1, 1), ("local-interest feature", 1, 0), ("local business or practice website page", 1, 0),
    ("magazine feature", 2, 0), ("trade or industry newsletter item", 1, 0), ("advice column", 1, 0),
    ("personal essay", 1, 1), ("resource page with short annotated entries", 1, 0),
]
WORDS_PER_VIEW, MIN_VIEWS, MAX_VIEWS, PER_CALL = 23, 8, 48, 24


def n_views(text, words_per_view=WORDS_PER_VIEW, lo=MIN_VIEWS, hi=MAX_VIEWS):
    """Views for a source: one per words_per_view source words, clamped to [lo, hi]."""
    return max(lo, min(hi, -(-len(text.split()) // words_per_view)))


def _pick(rng, items):
    return rng.choices([x[0] for x in items], weights=[x[1] for x in items])[0]


def slots(text, n, rng):
    words = len(text.split())
    lengths = [x for x in LENGTHS if x[2] <= LENGTH_FRAC * words] or LENGTHS[:1]
    out = []
    for i in range(n):
        length = rng.choices(lengths, weights=[x[1] for x in lengths])[0]
        quality = _pick(rng, QUALITY if length[2] <= ROUGH_MAX_WORDS else QUALITY[:2])
        pool = [f for f in FORMS if f[2]] if quality == "rough" else FORMS
        form = _pick(rng, pool)
        alt = form
        while alt == form:
            alt = _pick(rng, pool)
        out.append(f"- Prompt {i + 1}: {length[0]}; {quality}; form: {form} (alternative: {alt}).")
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


EXEC = """You are writing a web page. Everything you know about the topic is in these notes:
<notes>
{text}
</notes>

{prompt}

How to write it:
- Include only what this page needs for its focus; leave out the rest.
- Keep every fact, number, date, name, attribution, and hedge ("may", "probably", "some") as given. Add no facts, numbers, dates, names, quotations, causes, or attributions, and do not turn relative times into dates.
- Plain text as it appears when copied from a web page: no Markdown (#, **, *, pipe tables), no HTML, no genre label as a heading, no sign-offs or placeholders.
- Use your own words, quoting only short phrases; avoid stock phrases such as "Ultimately", "a testament to", and "In conclusion"; stay close to the requested length.
- Write as the page's author, stating the information directly or attributing it to the people and organizations involved."""

# References to "the text" / "the source" in a plan prompt make the writer cite them; rewrite them before execution.
_SAN = [(re.compile(r",?\s*(?:as )?(?:stated|noted|mentioned|described|given|found|presented|listed|quoted|outlined|explained) (?:in|by) (?:the|this) (?:provided |original |given |source )?(?:text|source text|source material|source document|source|document|passage|material|article)\b", re.I), ""),
        (re.compile(r"\b(?:according to|within) (?:the|this) (?:provided |original |given |source )?(?:text|source text|source material|source document|source|document|passage|material)\b,?\s*", re.I), ""),
        (re.compile(r"\b(?:the|this) (?:provided |original |given )?(?:source text|source material|source document|source|text|document|passage)('s)?\b", re.I), lambda m: "its" if m.group(1) else "it")]


def sanitize_prompt(prompt):
    for r, sub in _SAN:
        prompt = r.sub(sub, prompt)
    return prompt


def exec_message(text, prompt):
    return EXEC.replace("{text}", text).replace("{prompt}", sanitize_prompt(prompt))


# References to the source ("The text notes that X", "According to the notes, X", "as the notes point out",
# "mentioned in the notes") are rewritten to plain statements; only neutral reporting verbs are repaired, so no hedge
# such as "suggests" is dropped. Views that still refer to their source are rejected.
_N = (r"(?:the|these|this|my|our|your) (?:provided |original |given |above |source |accompanying )?"
      r"(?:notes|text|source text|source material|source document|source|material|document|passage|information provided|provided information)")
_VC = (r"(?:says?|said|states?|stated|notes?|noted|points? out|pointed out|adds?|added|confirms?|confirmed|tells? us|told us|"
       r"reminds? us|reminded us|puts? it)")  # take a clause: "The notes say there were ..." -> "There were ..."
_VT = (r"(?:mentions?|mentioned|explains?|explained|describes?|described|clarif(?:y|ies|ied)|specif(?:y|ies|ied)|reports?|"
       r"reported|lists?|listed|observes?|observed|records?|recorded|shows?|showed|highlights?|highlighted|emphasi[sz]es?|"
       r"emphasi[sz]ed|stress(?:es|ed)?|makes? clear|made clear)")  # take an object: repaired only before "that"
_V = rf"(?:{_VC}|{_VT})"
_START = r"(^|(?<=[.!?]\s)|(?<=[.!?]\"\s)|(?<=\n))"
_REPAIR = [
    (re.compile(rf"{_START}(?:Also, |However, |But |And |So |Plus, )?{_N} (?:also |further |explicitly |clearly |specifically |simply |even )?(?:{_VC},? (?:that )?|{_VT} that )(?!(?:again|also|too|then)\b)(\w)", re.I | re.M), lambda m: m.group(1) + m.group(2).upper()),
    (re.compile(rf"{_START}As {_N} (?:also |further |explicitly |clearly |specifically |simply |even )?{_V},\s*(\w)", re.I | re.M), lambda m: m.group(1) + m.group(2).upper()),
    (re.compile(rf"{_START}(?:According to|Per|Based on|From|As (?:noted|stated|described|mentioned|explained|listed|recorded) in) {_N},\s*(\w)", re.I | re.M), lambda m: m.group(1) + m.group(2).upper()),
    (re.compile(rf",? (?:as|according to|per) {_N}(?: (?:also )?{_V})?(?=[,.;:!?)])", re.I), ""),
    (re.compile(rf"(?<!\bAs)(?<!\bas)(?<!\bis)(?<!\bare)(?<!\bwas)(?<!\bwere)(?<!\bbe) (?:mentioned|listed|described|noted|given|cited|stated|recorded|found|shown|outlined|explained|quoted) (?:in|by) {_N}(?=[,.;:!?)])", re.I), ""),
]


def repair_references(text):
    for r, sub in _REPAIR:
        text = r.sub(sub, text)
    return text


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
_SRCREF = re.compile(r"\b(?:the|this) (?:provided|given|original|source|above|supplied|accompanying) (?:text|document|material|source|article|passage|information|notes|data)\b|"
                     r"\bsource (?:document|text|material)\b|\bprovided (?:text|document|article|material|passage|information|notes|data)\b|"
                     r"\binformation provided\b|\bthis source\b|"
                     r"\b(?:the|this) prompt\b|\b(?:according to|based on|per) the (?:text|material|document|source|passage|notes)\b|"
                     r"\b(?:in|from) the (?:text|material|passage|notes) (?:above|below)\b|\bthis page covers\b|"
                     r"\b(?:the|these|my|our|your) notes (?:\w+ly )?(?:say|said|says|state|states|mention|mentions|mentioned|note|record|records|recorded|point|points|call|calls|list|lists|describe|describes|suggest|suggests|indicate|indicates|show|shows|do not|don't|didn't|confirm|give|gives|provide|provides|cover|covers|remind|explain|explains)\b|"
                     r"\b(?:in|from|per|according to) (?:the|these|my|our|your) notes\b|"
                     r"\bthe (?:text|material|document|passage|source) (?:\w+ly )?(?:says|said|states|stated|notes|noted|mentions|mentioned|explains|describes|identifies|suggests|indicates|shows|does not|doesn't|did not|didn't|points out|claims|adds|lists|clarifies|emphasizes|warns|specifies|call|calls|records|gives|provides|offers|covers|focuses|implies|argues|classifies|reminds)\b", re.I)
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


_INSTRUCTION = re.compile(r"\bwords\b|return only|\b(?:write|compose|create|draft|produce|prepare|explain|describe|generate|summari[sz]e|answer|put together|turn|build|make|give|post|imagine)\b|\byou(?:r)?\b", re.I)


# Names and phrases specific to the fictional examples: a prompt or view containing one that its source lacks copied
# the example and is dropped.
EXAMPLE_MARKERS = ["Dovetail", "tool-lending", "QuickCache", "origin read", "successful write", "written key",
                   "Crumbline", "whole-wheat swap", "Harwell", "Lune Brook", "Amos Pell", "early blight", "concentric rings",
                   "Brennan", "Orchard Street"]


def example_leak(text, source):
    return any(m.lower() in text.lower() and m.lower() not in source.lower() for m in EXAMPLE_MARKERS)


def bad_prompt(prompt, source):
    """Plan sections that are drafts rather than instructions (pasted source text, no instruction words), absurdly
    long, or copied from a fictional example."""
    return (copy_rate(prompt, source) > 0.35 or len(prompt.split()) > 300 or not _INSTRUCTION.search(prompt)
            or example_leak(prompt, source))


def finish_view(text, source):
    """(final text, reject reason or None): markdown strip, reference repair, then the reject checks."""
    text = repair_references(clean_markdown(text))
    return text, reject_reason(text, source)


def reject_reason(text, source):
    """Why a finished view should be dropped (None = keep)."""
    if _PLACEHOLDER.search(text):
        return "placeholder"
    if example_leak(text, source):
        return "example_leak"
    if _SRCREF.search(text):
        return "source_reference"
    if _CJK.search(text) and not _CJK.search(source):
        return "script_switch"
    if repetition(text) > 0.2:
        return "repetition"
    if copy_rate(text, source) > 0.5:
        return "copy"
    return None
