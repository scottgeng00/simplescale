import re, zlib, json
WORD = re.compile(r"\S+")
END_P = re.compile(r"([.!?;:]|[。！？；：])[\"')\]]*$")
CJK_P = re.compile(r"[。！？；]")
def windows(text, size=200, step=100):
    ws = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text)]
    for i in range(0, max(1, len(ws) - size + 1), step):
        yield ws[i][0] if ws else 0, ws[i:i + size]
def wstats(win, text):
    w = [x[2] for x in win]
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    loop = 1 - len(set(g)) / len(g) if g else 0
    punct = (sum(bool(END_P.search(x)) for x in w) + sum(len(CJK_P.findall(x)) for x in w)) / max(1, len(w))
    seg = text[win[0][0]:win[-1][1]] if win else ""
    letters = [c for c in seg if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(1, len(letters))
    z = len(zlib.compress(seg.encode(), 9)) / max(1, len(seg.encode()))
    lines = [l.strip() for l in seg.split("\n") if l.strip()]
    dup = 1 - len(set(lines)) / len(lines) if len(lines) >= 8 else 0
    return dict(loop=loop, punct=punct, caps=caps, z=z, dup=dup)
def bad_window(s):
    if s["loop"] > 0.3: return "loop"
    if s["dup"] > 0.35: return "loop"
    if s["z"] < 0.30: return "loop"
    if s["punct"] < 0.01: return "salad"
    if s["caps"] > 0.4: return "caps"
    return None
def trim(text):
    for start, win in windows(text):
        r = bad_window(wstats(win, text))
        if r:
            return text[:start], r
    return text, None
