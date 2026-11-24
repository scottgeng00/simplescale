"""Static PNG of the 3C method x generator-size dot plot (same data as 3c_methods.html)."""
import json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
s = open("3c_methods.html").read()
D = json.loads(re.search(r"const D = (\{.*?\});\n", s, re.S).group(1))
C = {"2B": "#86b6ef", "9B": "#2a78d6", "27B": "#104281"}
rows = D["rows"]; cols = D["cols"]
ypos, labels, groups, y = [], [], [], 0
for i, r in enumerate(rows):
    if i == 0 or rows[i - 1]["group"] != r["group"]:
        groups.append((r["group"], y - 0.25)); y += 0.9
    ypos.append(y); labels.append(r["label"]); y += 1
fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols) + 1.6, 0.42 * y + 1.6), sharey=True)
for ax, (key, title, better, dp) in zip(axes, cols):
    vals = [r[key] for r in rows if r[key] is not None] + [D["ref"][k][key] for k in ("real", "fresh") if D["ref"][k][key] is not None]
    lo, hi = min(vals), max(vals); pad = (hi - lo) * .1 or 1; ax.set_xlim(lo - pad, hi + pad)
    for k, ls in (("real", (0, (5, 4))), ("fresh", (0, (1.5, 3)))):
        v = D["ref"][k][key]
        if v is not None: ax.axvline(v, color="#52514e", lw=1.4, ls=ls, zorder=1)
    for yy, r in zip(ypos, rows):
        if r[key] is None: ax.text(lo, yy, " n/a", va="center", fontsize=8, color="#898781"); continue
        ax.scatter(r[key], yy, s=60, color=C[r["size"]], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    ax.set_title(f"{title}\n{'→ better' if better == 'higher' else '← better'}", fontsize=9, loc="left", color="#0b0b0b")
    ax.grid(axis="x", color="#e1e0d9", lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="x", labelsize=8, colors="#52514e"); ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels, fontsize=9); axes[0].invert_yaxis()
for g, gy in groups: axes[0].text(-0.02, gy, g.upper(), transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color="#898781", fontweight="bold")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=C[k], markersize=7, label=f"{k} generator") for k in C] + \
    [Line2D([], [], color="#52514e", ls=(0, (5, 4)), label="real repeats (12×)"), Line2D([], [], color="#52514e", ls=(0, (1.5, 3)), label="fresh real")]
fig.legend(handles=h, loc="upper left", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.12, 1.0))
fig.suptitle("3× Chinchilla: synthetic-data method × generator size (R4 + synthetic; single seeds)", x=0.12, ha="left", y=1.05, fontsize=11)
fig.savefig("3c_methods.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
