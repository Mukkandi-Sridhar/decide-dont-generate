"""Recurring book graphics: the 'you are here' strip, and the one-page visual summary."""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyBboxPatch

from .bookmap import PARTS, CHAPTERS
from .style import C, ROOT, setup, TEXT_W, WIDE_W, save
from . import draw

SITE_BASE = "https://mukkandi-sridhar.github.io/decide-dont-generate"

PART_KIND = {"I": "data", "II": "data", "III": "llm", "IV": "jev", "V": "jev", "VI": "jev"}


def you_are_here(chapter: str):
    """A thin strip showing all 29 chapters grouped by part, with this chapter highlighted."""
    n_here = CHAPTERS[chapter][0]
    f, ax = draw.canvas(TEXT_W, 0.5)
    total = sum(len(p[2]) for p in PARTS)
    gap = 0.1
    x0, x1 = 0.02, TEXT_W - 0.02
    step = (x1 - x0 - gap * (len(PARTS) - 1)) / total
    x = x0
    for roman, name, chs in PARTS:
        w = step * len(chs)
        here_part = any(c[0] == n_here for c in chs)
        col = C["ink"] if here_part else C["muted"]
        ax.text(x, 0.44, f"PART {roman}", fontsize=5.4, fontweight="bold" if here_part else "semibold", color=col,
                va="center", ha="left")
        for i, (n, slug, title) in enumerate(chs):
            cx = x + step * (i + 0.5)
            here = n == n_here
            done = n < n_here
            fc = C["ink"] if here else ("#BFC5CE" if done else "white")
            ec = C["ink"] if here else ("#BFC5CE" if done else C["rule"])
            r = 0.075 if here else 0.05
            draw.dot(ax, cx, 0.2, r=r, color=fc, ec=ec, lw=0.8)
            if here:
                ax.text(cx, 0.2, str(n), ha="center", va="center", fontsize=5.2, color="white", fontweight="bold", zorder=6)
        ax.plot([x, x + w], [0.2, 0.2], color=C["grid"], lw=0.8, zorder=1)
        x += w + gap
    return f




def summary_page(chapter: str, title: str, panels: list[dict], footer: str | None = None, height: float = 7.75):
    """The one-page visual summary that closes every chapter.

    Each panel is dict(title=..., body=..., draw=callable(ax, x, y, w, h) or None, kind=...).
    Panels flow in two columns; a panel with span=2 takes a whole row.
    """
    W = WIDE_W
    f, ax = draw.canvas(W, height)
    n, t, roman, pname = CHAPTERS[chapter]
    ax.text(0.0, height - 0.08, f"CHAPTER {n} · ON ONE PAGE", fontsize=6.4, fontweight="bold", color=C["jev"], va="top")
    ax.text(0.0, height - 0.26, title, fontsize=13, fontweight="bold", color=C["ink"], va="top", family="Inter")
    ax.plot([0, W], [height - 0.62, height - 0.62], color=C["ink"], lw=1.2)
    top = height - 0.78
    gap = 0.14
    colw = (W - gap) / 2
    y = top
    col = 0
    row_h = 0
    for p in panels:
        span = p.get("span", 1)
        h = p.get("h", 1.5)
        if span == 2 and col == 1:
            y -= row_h + gap
            col, row_h = 0, 0
        x = 0 if col == 0 else colw + gap
        w = W if span == 2 else colw
        kind = p.get("kind", "neutral")
        edge, face = draw.KIND[kind]
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc="white", ec=C["grid"], lw=0.8))
        ax.add_patch(FancyBboxPatch((x, y - 0.05), w, 0.05, boxstyle="square,pad=0", fc=edge, ec="none"))
        if p.get("num") is not None:
            draw.number_badge(ax, x + 0.17, y - 0.22, p["num"], color=edge)
            tx = x + 0.32
        else:
            tx = x + 0.12
        ax.text(tx, y - 0.22, p["title"], fontsize=8.2, fontweight="bold", va="center", color=C["ink"])
        body_top = y - 0.4
        if p.get("body"):
            ax.text(x + 0.12, body_top, draw.wrap(p["body"], p.get("wrapw", int(w * 18.5))), fontsize=6.9,
                    color=C["ink2"], va="top", linespacing=1.35, family="Source Serif 4")
        if p.get("draw"):
            dh = p.get("draw_h", h * 0.5)
            p["draw"](ax, x + 0.12, y - h + 0.1, w - 0.24, dh)
        row_h = max(row_h, h)
        if span == 2 or col == 1:
            y -= row_h + gap
            col, row_h = 0, 0
        else:
            col = 1
    if footer:
        ax.plot([0, W], [0.32, 0.32], color=C["grid"], lw=0.8)
        ax.text(0, 0.16, footer, fontsize=7.2, color=C["ink"], fontweight="semibold", va="center", family="Inter")
    return f
