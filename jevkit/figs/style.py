"""The book's visual system for Matplotlib.

One colour code for the whole book (validated for colour-vision deficiency
with the dataviz palette checker; see DECISIONS.md):

    data     blue    anything that is input, evidence, features
    llm      orange  large language models, System 2, generation
    jev      green   Jev / System One decisions
    fail     red     failure modes, errors, missed attacks
    zones    purple  act / review / escalate (a single-hue ordinal ramp)

Black-and-white safe (DECISIONS.md D-88): hue is never the only cue. Series differ by line style and marker as well
as colour (SERIES), bars and areas that touch carry hatching (HATCH), lines get direct labels, and every fill is at
least 15% darker than white, and 15% apart from its neighbours, once printed in greyscale.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "assets" / "fonts"

C = dict(
    data="#2F6DB5", llm="#E07A1F", jev="#3B9C6E", fail="#B8323A",
    ink="#000000", ink2="#4B5563", muted="#6B7280", grid="#E4E7EB", rule="#9AA1AB",
    surface="#FFFFFF", paper="#FBFAF7",
    # tints for filled boxes (text sits on these in ink)
    data_t="#CBDAEF", llm_t="#F3D2B6", jev_t="#C4E1D0", fail_t="#F0CACA", neutral_t="#D5D8DD",
    # extra categorical slots for the rare chart that needs them (fixed order)
    slate="#5B6B82", gold="#C9971C",
)
ZONE = dict(act="#B7A6E0", review="#8468C9", escalate="#4B2C8F")
ZONE_T = dict(act="#DCD3F0", review="#C3B5E3", escalate="#A896D4")
# Zones also differ by pattern, so they read without colour: act plain, review hatched, escalate cross-hatched
ZONE_HATCH = dict(act="", review="////", escalate="xxxx")
ZONE_TEXT = dict(act=C["ink"], review="#FFFFFF", escalate="#FFFFFF")
KIND = dict(data=(C["data"], C["data_t"]), llm=(C["llm"], C["llm_t"]), jev=(C["jev"], C["jev_t"]),
            fail=(C["fail"], C["fail_t"]), neutral=(C["ink2"], C["neutral_t"]),
            act=(ZONE["act"], ZONE_T["act"]), review=(ZONE["review"], ZONE_T["review"]),
            escalate=(ZONE["escalate"], ZONE_T["escalate"]), plain=(C["rule"], "#FFFFFF"))

# Page geometry (inches) - must match assets/latex/geometry in _quarto.yml
# Series styles in fixed order: colour, line style and marker all change together, so series stay apart in greyscale
LINESTYLES = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1, 1.5)), (0, (8, 2)), (0, (2, 2))]
MARKERS = ["o", "s", "^", "X", "D", "v"]
HATCH = ["", "////", "....", "xxxx", "\\\\", "++"]
SERIES = [dict(color=c, ls=ls, marker=m) for c, ls, m in zip(
    ["#2F6DB5", "#E07A1F", "#3B9C6E", "#B8323A", "#5B6B82", "#C9971C"], LINESTYLES, MARKERS)]

TEXT_W = 4.7
WIDE_W = 5.95
MARGIN_W = 1.08

SANS = "Inter"
SERIF = "Source Serif 4"
MONO = "JetBrains Mono"

_ready = False


def setup():
    global _ready
    if _ready:
        return
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": SANS,
        "font.size": 7.8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 7.8,
        "axes.labelcolor": C["ink2"],
        "axes.edgecolor": C["rule"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": C["ink2"],
        "ytick.color": C["ink2"],
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.handlelength": 1.4,
        "text.color": C["ink"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,           # raster parts (heatmaps) print at 300 dpi or more
        "axes.prop_cycle": mpl.cycler(color=[C["data"], C["llm"], C["jev"], C["fail"], C["slate"], C["gold"]])
                           + mpl.cycler(linestyle=LINESTYLES),
        "hatch.linewidth": 0.6,
    })
    _ready = True


def fig(width: str | float = "text", height: float = 2.4, **kw):
    """New figure at print size. width: 'text' (4.7in), 'wide' (5.95in), 'margin' or inches."""
    setup()
    w = {"text": TEXT_W, "wide": WIDE_W, "margin": MARGIN_W}.get(width, width)
    return plt.figure(figsize=(w, height), **kw)


def subplots(nrows=1, ncols=1, width: str | float = "text", height: float = 2.4, **kw):
    setup()
    w = {"text": TEXT_W, "wide": WIDE_W, "margin": MARGIN_W}.get(width, width)
    return plt.subplots(nrows, ncols, figsize=(w, height), **kw)


def save(f, chapter: str, name: str, outdir: str | os.PathLike | None = None, tight=True):
    """Write <name>.pdf (print) and <name>.svg (web) under figures/<chapter>/."""
    out = Path(outdir) if outdir else ROOT / "figures" / chapter
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.04) if tight else {}
    f.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}, **kw)
    f.savefig(out / f"{name}.svg", metadata={"Date": None, "Creator": None}, **kw)
    plt.close(f)
    return out / f"{name}.pdf"


def synthetic_tag(f_or_ax, text="SYNTHETIC · not measured on real Jev", loc="tr"):
    """The small label every Jev number carries. Default: just above the figure's top-right corner."""
    f = f_or_ax.figure if hasattr(f_or_ax, "figure") and not isinstance(f_or_ax, plt.Figure) else f_or_ax
    x, ha = (0.995, "right") if loc.endswith("r") else (0.005, "left")
    y, va = (0.0, "top") if loc.startswith("b") else (1.0, "bottom")
    # sit clear of anything already drawn past the figure's edge (panel titles, tick labels)
    try:
        r = f.canvas.get_renderer()
        bb = f.get_tightbbox(r)
        h = f.get_figheight()
        y = min(0.0, bb.y0 / h - 0.01) if loc.startswith("b") else max(1.0, bb.y1 / h + 0.01)
    except Exception:
        pass
    f.text(x, y, text, ha=ha, va=va, fontsize=5.6, color=C["muted"], fontweight="medium",
           bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc="white", ec=C["grid"], lw=0.5))


def luma(color) -> float:
    """Lightness of a colour as a black-and-white printer sees it (0 black, 1 white)."""
    r, g, b = mpl.colors.to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def ink_on(color) -> str:
    """Text colour that stays readable on a fill: white on dark fills, black on light ones."""
    return "white" if luma(color) < 0.5 else C["ink"]


def hatch_kw(i: int, color) -> dict:
    """Fill style for the i-th of several bars or areas that touch: its colour plus a pattern (none for the first),
    drawn in white on dark fills and in dark grey on light ones, so neighbours differ without colour."""
    return dict(color=color, hatch=HATCH[i % len(HATCH)] or None,
                hatchcolor="white" if luma(color) < 0.55 else C["ink2"], edgecolor="white", linewidth=0.6)


def clean(ax, grid="y"):
    """Recessive axes: hairline grid on one axis only."""
    ax.grid(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True)
    if grid in ("x", "both"):
        ax.xaxis.grid(True)
    ax.spines["left"].set_visible(grid != "y")
    return ax


def pct(x, pos=None):
    return f"{x:.0%}"
