"""Diagram primitives drawn in Matplotlib so diagrams and charts share one look.

A `canvas` is an axes whose units are inches on the printed page, so a
0.9-inch box is 0.9 inches wide in the book. That makes layouts predictable.
"""

from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Polygon

from .style import C, KIND, ZONE, ZONE_TEXT, ZONE_HATCH, setup, fig as _fig, TEXT_W, WIDE_W


def canvas(width="text", height=2.0):
    setup()
    w = {"text": TEXT_W, "wide": WIDE_W}.get(width, width)
    f = _fig(w, height)
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return f, ax


def wrap(text: str, width: int) -> str:
    return "\n".join("\n".join(textwrap.wrap(line, width)) if line else "" for line in text.split("\n"))


def box(ax, x, y, w, h, text="", kind="neutral", size=7.6, weight="medium", color=None, fill=None,
        radius=0.06, lw=0.9, align="center", wrapw=None, sub=None, subsize=6.4, dashed=False, textcolor=None,
        family=None, zorder=2):
    """A rounded box with centred text. (x, y) is the lower-left corner, in inches."""
    edge, face = KIND.get(kind, KIND["neutral"])
    edge = color or edge
    face = fill or face
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=face, ec=edge, lw=lw,
                       ls=(0, (3, 2)) if dashed else "-", zorder=zorder)
    ax.add_patch(p)
    if wrapw:
        text = wrap(text, wrapw)
    tx = x + w / 2 if align == "center" else x + 0.08
    ha = "center" if align == "center" else "left"
    ty = y + h / 2 + (0.07 if sub else 0)
    kw = dict(family=family) if family else {}
    if text:
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=textcolor or C["ink"],
                linespacing=1.25, zorder=zorder + 1, **kw)
    if sub:
        ax.text(tx, y + h / 2 - 0.1, sub, ha=ha, va="center", fontsize=subsize, color=C["ink2"], linespacing=1.2,
                zorder=zorder + 1)
    return p


def arrow(ax, a, b, color=None, lw=1.0, style="-|>", rad=0.0, label=None, labelpos=0.5, labelsize=6.4,
          dashed=False, shrink=2, head=5, labeloffset=(0, 0.07), zorder=1, labelcolor=None):
    color = color or C["ink2"]
    p = FancyArrowPatch(a, b, arrowstyle=f"{style},head_length={head * 0.9},head_width={head * 0.55}" if style != "-" else "-",
                        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, shrinkA=shrink, shrinkB=shrink,
                        ls=(0, (3, 2)) if dashed else "-", zorder=zorder)
    ax.add_patch(p)
    if label:
        mx = a[0] + (b[0] - a[0]) * labelpos + labeloffset[0]
        my = a[1] + (b[1] - a[1]) * labelpos + labeloffset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=labelsize, color=labelcolor or C["ink2"], zorder=zorder + 2)
    return p


def text(ax, x, y, s, size=7.4, color=None, weight="normal", ha="left", va="center", wrapw=None, style="normal",
         family=None, **kw):
    if wrapw:
        s = wrap(s, wrapw)
    extra = dict(family=family) if family else {}
    return ax.text(x, y, s, fontsize=size, color=color or C["ink"], fontweight=weight, ha=ha, va=va,
                   linespacing=1.3, fontstyle=style, **extra, **kw)


def pill(ax, x, y, s, kind="neutral", size=6.4, pad=0.05, weight="semibold", textcolor=None, ha="center"):
    edge, face = KIND.get(kind, KIND["neutral"])
    t = ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight, color=textcolor or C["ink"],
                bbox=dict(boxstyle=f"round,pad={pad * 4},rounding_size=0.12", fc=face, ec=edge, lw=0.7), zorder=5)
    return t


def dot(ax, x, y, r=0.05, color=None, ec="white", lw=1.2, zorder=4):
    c = Circle((x, y), r, fc=color or C["ink"], ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(c)
    return c


def number_badge(ax, x, y, n, color=None, r=0.085, size=6.6):
    dot(ax, x, y, r=r, color=color or C["ink"], ec="white", lw=1)
    ax.text(x, y - 0.004, str(n), ha="center", va="center", fontsize=size, color="white", fontweight="bold", zorder=6)


def zone_bar(ax, x, y, w, h, low, high, labels=True, size=6.8, ticks=True, names=("act", "review", "escalate"),
             sublabels=None, tick_size=6.2):
    """The book's signature three-zone bar: 0 ........ low ..... high ........ 1."""
    xs = [x, x + w * low, x + w * high, x + w]
    for i, z in enumerate(("act", "review", "escalate")):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 <= 0:
            continue
        # grey level and pattern both change from zone to zone, so the bar reads in black and white
        r = Rectangle((x0, y), x1 - x0, h, fc=ZONE[z], ec="white", lw=1.5, hatch=ZONE_HATCH[z], zorder=2)
        ax.add_patch(r)
        if not labels:
            continue
        lab = names[i].upper()
        if x1 - x0 > 0.35:
            ax.text((x0 + x1) / 2, y + h / 2 + (0.05 if sublabels else 0), lab, ha="center", va="center", fontsize=size,
                    fontweight="bold", color=ZONE_TEXT[z], zorder=3,
                    bbox=dict(boxstyle="square,pad=0.12", fc=ZONE[z], ec="none") if ZONE_HATCH[z] else None)
            if sublabels:
                ax.text((x0 + x1) / 2, y + h / 2 - 0.09, sublabels[i], ha="center", va="center", fontsize=size - 1.2,
                        color=ZONE_TEXT[z], zorder=3,
                        bbox=dict(boxstyle="square,pad=0.08", fc=ZONE[z], ec="none") if ZONE_HATCH[z] else None)
        else:
            # too narrow for its name: label it just above the bar, with a short leader
            ax.plot([(x0 + x1) / 2] * 2, [y + h, y + h + 0.06], color=C["ink2"], lw=0.5, zorder=3)
            ax.text((x0 + x1) / 2, y + h + 0.07, lab, ha="left" if i == 0 else "center", va="bottom",
                    fontsize=size - 0.8, fontweight="bold", color=C["ink"], zorder=3)
    if ticks:
        for v, xx in ((0, xs[0]), (low, xs[1]), (high, xs[2]), (1, xs[3])):
            ax.plot([xx, xx], [y - 0.05, y], color=C["ink2"], lw=0.6)
            ax.text(xx, y - 0.08, f"{v:g}", ha="center", va="top", fontsize=tick_size, color=C["ink2"])
    return xs


def bracket(ax, x0, x1, y, text_="", size=6.4, color=None, up=True):
    color = color or C["ink2"]
    d = 0.05 if up else -0.05
    ax.plot([x0, x0, x1, x1], [y, y + d, y + d, y], color=color, lw=0.7)
    if text_:
        ax.text((x0 + x1) / 2, y + d + (0.04 if up else -0.04), text_, ha="center", va="bottom" if up else "top",
                fontsize=size, color=color)


def cylinder(ax, x, y, w, h, kind="data", label="", size=6.8):
    edge, face = KIND[kind]
    e = 0.08
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=face, ec="none", zorder=2))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=face, ec=edge, lw=0.9, zorder=2))
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=face, ec=edge, lw=0.9, zorder=3))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=edge, lw=0.9, zorder=3)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=edge, lw=0.9, zorder=3)
    if label:
        ax.text(x + w / 2, y + h / 2 - 0.02, label, ha="center", va="center", fontsize=size, fontweight="medium", zorder=4)


def doc_icon(ax, x, y, w=0.28, h=0.36, kind="data", lines=4):
    edge, face = KIND[kind]
    f = 0.08
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)], closed=True,
                         fc=face, ec=edge, lw=0.8, zorder=3))
    for i in range(lines):
        yy = y + h - 0.1 - i * (h - 0.16) / max(1, lines)
        ax.plot([x + 0.05, x + w - 0.06 - (0.05 if i == lines - 1 else 0)], [yy, yy], color=edge, lw=0.6, zorder=4)


def gauge(ax, x, y, w, p, kind="jev", h=0.09, label=True, size=6.2):
    """A small horizontal probability meter with a lighter track of the same hue."""
    edge, face = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", fc=face, ec="none", zorder=2))
    if p > 0:
        ax.add_patch(FancyBboxPatch((x, y), max(0.02, w * p), h, boxstyle="round,pad=0,rounding_size=0.03", fc=edge,
                                    ec="none", zorder=3))
    if label:
        ax.text(x + w + 0.05, y + h / 2, f"{p:.2f}", ha="left", va="center", fontsize=size, color=C["ink2"],
                family="JetBrains Mono")
