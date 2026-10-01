"""Figures for Chapter 12: The Jevons paradox of decisions."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import econ
from jevkit.figs import hatch_kw, figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch19"
CAPACITY = 240          # reviews a day Kestrel's team can clear (Chapter 14)
ILLUS = "ILLUSTRATIVE MODEL · assumptions, not measurements"


def money(v, _=None):
    if v >= 1:
        return f"${v:,.0f}"
    if v >= 0.01:
        return f"${v:.2f}"
    return f"${v:.{int(round(-np.log10(v)))}f}"


def count(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:.0f}M" if v >= 1e7 else f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


@lru_cache(None)
def worlds():
    llm = econ.day(econ.LLM_PRICE, econ.LLM_LATENCY)
    jev = econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[0])
    jev_slow = econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[1])
    fl = econ.flags(econ.LLM_PRICE, econ.LLM_LATENCY)
    fj = econ.flags(econ.JEV_PRICE, econ.JEV_LATENCY[0])
    return llm, jev, jev_slow, fl, fj


def record():
    llm, jev, jev_slow, fl, fj = worlds()
    results(CH, llm_price=econ.LLM_PRICE, jev_price=econ.JEV_PRICE, llm_latency=econ.LLM_LATENCY,
            llm_decisions=llm["decisions"], jev_decisions=jev["decisions"], jev_slow_decisions=jev_slow["decisions"],
            decisions_ratio=jev["decisions"] / llm["decisions"], llm_spend=llm["spend"], jev_spend=jev["spend"],
            llm_value=llm["value"], jev_value=jev["value"], value_gain=jev["value"] - llm["value"],
            llm_by_pool=llm["by_pool"], jev_by_pool=jev["by_pool"],
            pool_logins=next(p.per_day for p in econ.POOLS if p.name == "logins"),
            pool_raw=next(p.per_day for p in econ.POOLS if p.name == "raw log events"),
            flags_top_llm=fl["top_share"], flags_top_jev=fj["top_share"],
            flags_cost_llm=fl["cost_line"], flags_cost_jev=fj["cost_line"], capacity=CAPACITY,
            reviewable_llm=fl["decided"], reviewable_jev=fj["decided"],
            new_job=econ.NEW_JOB.per_day, new_job_value=econ.NEW_JOB.median_value,
            llm_new_spend=econ.day(econ.LLM_PRICE, econ.LLM_LATENCY, extra=(econ.NEW_JOB,))["spend"],
            jev_new_spend=econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[0], extra=(econ.NEW_JOB,))["spend"],
            llm_new_taken=econ.day(econ.LLM_PRICE, econ.LLM_LATENCY, extra=(econ.NEW_JOB,))["by_pool"][econ.NEW_JOB.name],
            jev_new_taken=econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[0], extra=(econ.NEW_JOB,))["by_pool"][econ.NEW_JOB.name],
            spend_e05=float(econ.elastic_spend(0.01, 0.5)), spend_e15=float(econ.elastic_spend(0.01, 1.5)))


@figure(CH, "loop")
def loop():
    f, ax = draw.canvas("text", 2.55)
    rows = [("1865 · COAL", "neutral", ["Watt’s engine gets\nmore work from\neach ton of coal", "work done by\ncoal gets cheaper",
                                        "engines spread to\nmills, mines,\nrailways, ships", "Britain burns\nmore coal,\nnot less"]),
            ("NOW · DECISIONS", "jev", ["a model answers\nin one pass,\nnot token by token", "each decision\ncosts a tiny\nfraction of a cent",
                                        "decisions nobody\ncould afford\nget made", "far more decisions\nin total, and\nnew uses for them"])]
    for r, (label, kind, steps) in enumerate(rows):
        y = 1.35 - r * 1.2
        draw.text(ax, 0.0, y + 0.86, label, size=6.6, weight="bold", color=C["jev"] if kind == "jev" else C["ink2"])
        for i, s in enumerate(steps):
            x = i * 1.2
            last = i == 3
            draw.box(ax, x, y, 1.02, 0.7, s, kind=("llm" if last and kind == "neutral" else kind), size=6.2,
                     weight="semibold" if last else "medium")
            if i < 3:
                draw.arrow(ax, (x + 1.02, y + 0.35), (x + 1.2, y + 0.35), head=3.5)
    return f


@figure(CH, "elasticity")
def elasticity():
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    cheaper = np.logspace(0, 2, 200)
    lines = [(0.5, "demand barely grows:\nspend falls", C["data"]),
             (1.0, "demand grows exactly as fast\nas price falls: spend flat", C["muted"]),
             (1.5, "demand grows faster than\nprice falls: spend rises", C["llm"])]
    for e, lab, col in lines:
        s = econ.elastic_spend(1 / cheaper, e)
        ax.plot(cheaper, s, color=col, lw=2)
        ax.text(105, s[-1], f"elasticity {e:g}: {lab}", fontsize=6.0, va="center", color=C["ink"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.08, 13)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["today", "10× cheaper", "100× cheaper"])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["a tenth", "same as\ntoday", "ten times"])
    ax.set_ylabel("total spend")
    ax.set_xlabel("price per unit")
    f.subplots_adjust(right=0.6)
    return f


@figure(CH, "pools")
def pools():
    rec = worlds()
    f, ax = subplots(width="text", height=2.5)
    clean(ax, "x")
    vals = econ.values()
    for i, p in enumerate(econ.POOLS):
        y = len(econ.POOLS) - 1 - i
        v = vals[p.name]
        lo, med, hi = np.percentile(v, [5, 50, 95])
        # the range is a mid-grey bar with dark end ticks, so it survives a photocopy
        ax.plot([lo, hi], [y, y], color=C["data"], lw=5, ls="-", solid_capstyle="butt", alpha=0.5)
        ax.plot([lo, lo], [y - 0.18, y + 0.18], color=C["data"], lw=1.0, ls="-")
        ax.plot([hi, hi], [y - 0.18, y + 0.18], color=C["data"], lw=1.0, ls="-")
        ax.scatter([med], [y], color=C["data"], s=22, zorder=3)
        ax.text(3e3, y, f"{count(p.per_day)} a day", va="center", fontsize=6.2, color=C["ink2"])
    for price, lab, col in ((econ.LLM_PRICE, "LLM price", C["llm"]), (econ.JEV_PRICE, "Jev price", C["jev"])):
        ax.axvline(price, color=col, lw=1.3)
        ax.text(price * 1.15, len(econ.POOLS) - 0.35, lab, color=col, fontsize=6.2, fontweight="semibold", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e3)
    ax.set_ylim(-0.6, len(econ.POOLS) - 0.1)
    ax.set_yticks(range(len(econ.POOLS))[::-1])
    ax.set_yticklabels([p.name for p in econ.POOLS], fontsize=6.6)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 100])
    ax.set_xlabel("value of one decision: the expected loss it avoids (log scale)")
    synthetic_tag(f, ILLUS)
    return f


@figure(CH, "gates")
def price_latency_map():
    f, ax = subplots(width="text", height=2.7)
    clean(ax, "none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xl, yl = (1e-6, 300), (0.03, 200)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    # feasible regions: cheap enough (value above price) and fast enough (budget above latency)
    ax.fill_between([econ.LLM_PRICE, xl[1]], econ.LLM_LATENCY, yl[1], color=C["llm_t"], zorder=0)
    ax.fill_between([econ.JEV_PRICE, xl[1]], econ.JEV_LATENCY[1], yl[1], color=C["jev"], alpha=0.10, zorder=0, lw=0)
    ax.fill_between([econ.JEV_PRICE, xl[1]], econ.JEV_LATENCY[0], econ.JEV_LATENCY[1], color=C["jev"], alpha=0.05,
                    zorder=0, hatch="////", edgecolor=C["jev"], lw=0)
    ax.plot([econ.LLM_PRICE, econ.LLM_PRICE, xl[1]], [yl[1], econ.LLM_LATENCY, econ.LLM_LATENCY], color=C["llm"], lw=1.2)
    ax.plot([econ.JEV_PRICE, econ.JEV_PRICE, xl[1]], [yl[1], econ.JEV_LATENCY[1], econ.JEV_LATENCY[1]], color=C["jev"], lw=1.2)
    ax.plot([econ.JEV_PRICE, xl[1]], [econ.JEV_LATENCY[0]] * 2, color=C["jev"], lw=0.8, ls=(0, (3, 2)))
    ax.text(econ.LLM_PRICE * 1.3, 120, "an LLM can make\nthese decisions", color=C["llm"], fontsize=6.2, fontweight="semibold", va="top")
    ax.text(econ.JEV_PRICE * 1.3, 120, "Jev can make\nthese", color=C["jev"], fontsize=6.2, fontweight="semibold", va="top")
    ax.text(40, (econ.JEV_LATENCY[0] * econ.JEV_LATENCY[1]) ** 0.5, "only at the fast end\nof Jev’s range",
            color=C["jev"], fontsize=5.6, va="center", ha="right")
    vals = econ.values()
    for p in econ.POOLS:
        lo, hi = np.percentile(vals[p.name], [10, 90])
        ax.plot([lo, hi], [p.budget_s] * 2, color=C["data"], lw=1, alpha=0.6)
        ax.scatter([p.median_value], [p.budget_s], color=C["data"], s=24, zorder=3, edgecolor="white", lw=0.8)
        dx, dy, ha = {"raw log events": (1, 1.3, "center"), "inbound emails": (0.7, 1.3, "right"),
                      "checking LLM outputs": (0.7, 0.72, "right")}.get(p.name, (1, 1.28, "center"))
        ax.text(p.median_value * dx, p.budget_s * dy, p.name, fontsize=6.0, ha=ha,
                va="bottom" if dy > 1 else "top", color=C["ink"])
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xticks([1e-5, 1e-3, 1e-1, 10])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1 s", "1 s", "10 s", "100 s"])
    ax.set_xlabel("value of a typical decision (log scale; lines show the middle 80%)")
    ax.set_ylabel("time budget per decision")
    synthetic_tag(f, ILLUS)
    return f


@figure(CH, "curve")
def curve():
    rec = worlds()
    prices = np.logspace(-6.3, -1, 40)
    days = [econ.day(p) for p in prices]
    n = np.array([d["decisions"] for d in days])
    s = np.array([d["spend"] for d in days])
    f, (a1, a2) = subplots(1, 2, width="text", height=2.2)
    for ax in (a1, a2):
        clean(ax, "both")
        ax.set_xscale("log")
        ax.set_xlim(prices[-1] * 1.5, prices[0] / 1.5)     # price falls left to right
        ax.xaxis.set_major_formatter(FuncFormatter(money))
        ax.set_xticks([1e-2, 1e-4, 1e-6])
        ax.set_xlabel("price per decision (falling →)")
    a1.plot(prices, n, color=C["data"], lw=2)
    a2.plot(prices, s, color=C["data"], lw=2)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(count))
    a1.set_title("decisions worth making per day", fontsize=7, loc="left")
    a2.set_title("total spent on decisions per day", fontsize=7, loc="left")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    a2.set_ylim(0, s.max() * 1.15)
    for price, lab, col in ((econ.LLM_PRICE, "LLM", C["llm"]), (econ.JEV_PRICE, "Jev", C["jev"])):
        d = econ.day(price)
        a1.scatter([price], [d["decisions"]], color=col, s=30, zorder=3)
        a2.scatter([price], [d["spend"]], color=col, s=30, zorder=3)
        a1.text(price, d["decisions"] * 1.6, lab, color=col, fontsize=6.4, ha="center", fontweight="semibold")
        a2.text(price, d["spend"] + s.max() * 0.06, lab, color=col, fontsize=6.4, ha="center", fontweight="semibold")
    f.subplots_adjust(wspace=0.4)
    synthetic_tag(f, ILLUS)
    return f


@figure(CH, "flags")
def flags():
    llm, jev, jev_slow, fl, fj = worlds()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    groups = [("LLM prices\n" + f"({count(fl['decided'])} reviewable decisions)", fl),
              ("Jev prices\n" + f"({count(fj['decided'])} reviewable decisions)", fj)]
    w = 0.34
    for i, (lab, d) in enumerate(groups):
        for j, (key, name, col) in enumerate((("top_share", "flag the top 0.1%", C["fail"]),
                                              ("cost_line", "flag when expected loss > review cost", C["jev"]))):
            x = i + (j - 0.5) * (w + 0.03)
            ax.bar(x, d[key], width=w, zorder=2, **hatch_kw(j, col))
            ax.text(x, d[key] + 12, f"{d[key]:,}", ha="center", fontsize=6.2, va="bottom")
            if i == 0:
                ax.bar(0, 0, label=name + (" (striped)" if j else " (solid)"), **hatch_kw(j, col))
    ax.axhline(CAPACITY, color=C["ink"], lw=1, ls=(0, (4, 2)), zorder=3)
    ax.text(1.98, CAPACITY + 20, f"team capacity:\n{CAPACITY} a day", fontsize=6.0, va="bottom", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.4)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(0, max(fj["top_share"], fj["cost_line"]) * 1.45)
    ax.set_ylabel("sent to a person per day")
    ax.legend(loc="upper left", fontsize=6.2, frameon=False)
    synthetic_tag(f, ILLUS)
    return f


def mini_bars(items, fmt):
    """A tiny horizontal bar chart for a summary panel. items: (label, value, colour)."""
    def paint(ax, x, y, w, h):
        top = max(v for _, v, _ in items)
        rowh = h / len(items)
        for k, (lab, v, col) in enumerate(items):
            yy = y + h - (k + 1) * rowh + rowh * 0.2
            bw = (w - 1.25) * v / top
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (x + 0.75, yy), max(bw, 0.02), rowh * 0.55, fc=col, ec="none"))
            ax.text(x + 0.7, yy + rowh * 0.27, lab, ha="right", va="center", fontsize=6.3, color=C["ink2"])
            ax.text(x + 0.8 + max(bw, 0.02), yy + rowh * 0.27, fmt(v), ha="left", va="center", fontsize=6.3, color=C["ink"])
    return paint


def mini_loop(ax, x, y, w, h):
    steps = ["cheaper\nunit", "new uses\nworth it", "more\ntotal use"]
    bw = (w - 0.4) / 3
    for k, st in enumerate(steps):
        draw.box(ax, x + k * (bw + 0.2), y + h * 0.25, bw, h * 0.6, st, kind="neutral" if k < 2 else "llm", size=6.2)
        if k < 2:
            draw.arrow(ax, (x + k * (bw + 0.2) + bw, y + h * 0.55), (x + (k + 1) * (bw + 0.2), y + h * 0.55), head=3)


def mini_curves(ax, x, y, w, h):
    for e, col, lab in ((0.5, C["data"], "spend falls"), (1.0, C["muted"], "flat"), (1.5, C["llm"], "spend rises")):
        xs = np.linspace(0, 1, 30)
        ys = econ.elastic_spend(10 ** (-2 * xs), e)
        yy = y + h * 0.5 + np.log10(ys) / 2 * h * 0.45
        ax.plot(x + xs * (w - 0.9), yy, color=col, lw=1.6)
        ax.text(x + w - 0.85, yy[-1], lab, fontsize=6.1, va="center", color=C["ink"])


@figure(CH, "summary")
def summary():
    record()
    llm, jev, jev_slow, fl, fj = worlds()
    k = lambda v: count(v) + " a day"
    panels = [
        dict(num=1, title="Cheaper often means more", kind="neutral", h=2.0, draw=mini_loop, draw_h=0.8,
             body="Jevons saw coal use rise as engines got more efficient. Cheaper units open uses that weren’t worth it before."),
        dict(num=2, title="It depends on elasticity", kind="neutral", h=2.0, draw=mini_curves, draw_h=0.85,
             body="If demand grows faster than price falls, total spend rises. If not, you pay less and still do more."),
        dict(num=3, title="Most decisions go unmade", kind="data", h=2.0, draw_h=0.8,
             draw=mini_bars([("LLM", llm["decisions"], C["llm"]), ("Jev", jev["decisions"], C["jev"])], k),
             body="Each login, email or log line is worth a sliver of a cent to check. At LLM prices, most aren’t worth asking about."),
        dict(num=4, title="Price and time both gate", kind="jev", h=2.0,
             body=("A decision needs to be cheaper than its value and faster than its deadline. Logins wait for no one: "
                   "they need an answer inside the few hundred milliseconds a person will sit and wait.")),
        dict(num=5, title="Kestrel’s illustrative day", kind="jev", h=2.0, draw_h=0.8,
             draw=mini_bars([("LLM", llm["spend"], C["llm"]), ("Jev", jev["spend"], C["jev"])], lambda v: f"${v:,.0f} a day"),
             body=(f"{jev['decisions'] / llm['decisions']:.0f}× more decisions, a quarter of the spend. "
                   "Within one company’s jobs the bill falls. New jobs are where the paradox bites.")),
        dict(num=6, title="The load moves to people", kind="fail", h=2.0, draw_h=0.8,
             draw=mini_bars([("top 0.1%", fj["top_share"], C["fail"]), ("cost line", fj["cost_line"], C["jev"])],
                            lambda v: f"{v:,} a day"),
             body=(f"Flag a fixed share and reviews grow from {fl['top_share']} to {fj['top_share']:,} a day. "
                   "Flag by cost and they stay put.")),
    ]
    return summary_page(CH, "The Jevons paradox of decisions", panels,
                        footer="Next: six ways to make the same decision, in one fair contest.")
