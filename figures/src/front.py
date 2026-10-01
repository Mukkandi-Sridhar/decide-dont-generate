"""Figures for the front matter: the prologue's night shift, reading paths, and the book's boxes."""

import numpy as np
from matplotlib.patches import Rectangle

from jevkit import soc
from jevkit.figs import figure, draw, C, ZONE, subplots, clean, results, synthetic_tag
from jevkit.figs.bookmap import PARTS

CH = "front"
PER_HOUR = 5        # one analyst, 12 minutes an alert


def night():
    a = soc.load()
    t = a.timestamp.astype("datetime64[ns]")
    n = a[(t >= np.datetime64("2026-09-23T00:00")) & (t < np.datetime64("2026-09-23T07:00"))].copy()
    tt = n.timestamp.astype("datetime64[ns]").to_numpy()
    n["hour"] = (tt - np.datetime64("2026-09-23T00:00")).astype("timedelta64[m]").astype(float) / 60
    n = n.sort_values("hour").reset_index(drop=True)
    opened = np.zeros(len(n), bool)
    clock = 0.0
    for i, h in enumerate(n.hour):                 # first come, first served, 12 minutes each
        start = max(clock, h)
        if start + 0.2 <= 7:
            opened[i] = True
            clock = start + 0.2
    n["opened"] = opened
    return n


def record():
    n = night()
    y = n.malicious.to_numpy().astype(bool)
    results(CH, night_alerts=len(n), night_threats=int(y.sum()), night_opened=int(n.opened.sum()),
            night_threats_opened=int((y & n.opened.to_numpy()).sum()))


@figure(CH, "night")
def night_fig():
    record()
    n = night()
    f, ax = subplots(width="text", height=1.6)
    clean(ax, "none")
    y = n.malicious.to_numpy().astype(bool)
    op = n.opened.to_numpy()
    # real threats are taller, thicker and carry a mark on top, so they stand out without colour
    ax.vlines(n.hour[~y], 0.2, 0.8, color=C["rule"], lw=0.6)
    ax.vlines(n.hour[y], 0.0, 1.0, color=C["fail"], lw=1.4)
    ax.scatter(n.hour[y], np.full(y.sum(), 1.07), s=7, color=C["fail"], marker="v", lw=0)
    ax.scatter(n.hour[op], np.full(op.sum(), -0.25), s=6, color=C["ink"], marker="|")
    ax.text(7.05, 0.5, f"{len(n)} alerts,\n{y.sum()} real (tall red\nticks, marked on top)", fontsize=6.0, va="center")
    ax.text(7.05, -0.25, f"you open {op.sum()},\n{(y & op).sum()} of them real", fontsize=6.0, va="center")
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{h}:00" for h in range(8)], fontsize=6)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    f.subplots_adjust(right=0.78)
    synthetic_tag(f, "SYNTHETIC ALERTS · Kestrel Logistics is fictional")
    return f


@figure(CH, "paths")
def paths():
    f, ax = draw.canvas("text", 2.4)
    parts = [p[0] for p in PARTS]
    names = ["Probability\n& decisions", "Networks,\nLLMs, agents", "Jev in\ndepth", "Decision\nlayer", "Building", "What\nnow"]
    w = 0.72
    for i, (r, nm) in enumerate(zip(parts, names)):
        x = i * (w + 0.06)
        draw.box(ax, x, 1.7, w, 0.55, f"{r}\n{nm}", kind="neutral", size=5.4)
    routes = [("New to machine learning", [0, 1, 2, 3, 4, 5], C["data"], 1.35),
              ("Engineer who knows LLMs", [0, 2, 3, 4, 5], C["jev"], 0.95),
              ("Leader or product manager", [0, 2, 3, 5], C["llm"], 0.55)]
    for lab, stops, col, y in routes:
        draw.text(ax, 0.0, y + 0.17, lab, size=6.0, weight="semibold", color=col)
        xs = [s * (w + 0.06) + w / 2 for s in stops]
        ax.plot(xs, [y] * len(xs), color=col, lw=1.4)
        for x in xs:
            draw.dot(ax, x, y, r=0.045, color=col)
    draw.text(ax, 0.0, 0.12, "Part I is the foundation for every route: probabilities, calibration and costs. Part II is optional if you know it.",
              size=5.9, color=C["ink2"], style="italic")
    return f


@figure(CH, "boxes")
def boxes():
    f, ax = draw.canvas("text", 3.29)
    items = [("KEY IDEA", "key", "One sentence worth remembering a year from now. The thick bar marks it."),
             ("KEEP THESE", "keep", "At the end of each chapter: that chapter’s key ideas, in one list."),
             ("TRY IT", "jev", "Code you can run. The chapter’s lab notebook has the full version."),
             ("SET THE THRESHOLD", "review", "A small decision to make yourself, with costs. The answer follows in brackets."),
             ("WHERE THIS BREAKS", "fail", "The honest limits of what the chapter just showed."),
             ("GOING DEEPER", "neutral", "Optional maths. Skip it and nothing later depends on it."),
             ("WHY NOT …?", "neutral", "A side box that answers the obvious objection in a few lines."),
             ("SYNTHETIC", "plain", "Every Jev number comes from a mock model, and says so.")]
    key_bar, key_tint = "#1E5E42", "#C9E0D2"      # as in print: assets/latex/preamble.tex (keybar, keytint)
    for i, (t, k, d) in enumerate(items):
        y = 3.09 - i * 0.36
        if k in ("key", "keep"):
            # the book's key idea style: a light tint with a thick dark bar on the left (a frame too for "keep these")
            ax.add_patch(Rectangle((0.0, y - 0.14), 1.25, 0.28, fc=key_tint, ec=key_bar if k == "keep" else "none",
                                   lw=0.8, zorder=2))
            ax.add_patch(Rectangle((0.0, y - 0.14), 0.07, 0.28, fc=key_bar, ec="none", zorder=3))
            draw.text(ax, 0.66, y, t, size=5.6, weight="bold", color=key_bar, ha="center")
        else:
            draw.box(ax, 0.0, y - 0.14, 1.25, 0.28, t, kind=k, size=5.6, weight="bold",
                     textcolor="white" if k == "review" else None, fill=draw.KIND["review"][0] if k == "review" else None)
        draw.text(ax, 1.4, y, d, size=6.1)
    return f
