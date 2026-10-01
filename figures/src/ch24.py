"""Figures for Chapter 17: A hybrid agent: Jev decides, the LLM reasons."""

import json
from collections import Counter
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, agent, calibration as cal, policy as pol
from jevkit.figs import figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch24"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
N = 700
KIND_COL = dict(observe=C["data"], decide=C["jev"], generate=C["llm"], act=C["ink2"])
# each kind of step also has its own pattern, so the bars read in black and white
KIND_HATCH = dict(observe="", decide="////", generate="....", act="xxxx")
KIND_NAME = dict(observe="observe (plain)", decide="decide (striped)", generate="generate (dotted)", act="act (crossed)")
AGENTS = {"v1": dict(), "v2": dict(always=("threat_intel",)),
          "llm": dict(always=("threat_intel",), decider="llm")}


@lru_cache(None)
def runs():
    alerts = soc.load()
    _, live = soc.history_and_live(alerts)
    sub = live.sample(N, random_state=24).reset_index(drop=True)
    per_day = len(live) / 7
    out = {}
    for name, kw in AGENTS.items():
        ag = agent.SOCAgent(**kw)
        tr = [ag.run(r) for r in sub.itertuples()]
        out[name] = tr
    return sub, per_day, out


def stats(name):
    sub, per_day, out = runs()
    tr, y = out[name], sub.malicious.to_numpy()
    acts = np.array([t.action for t in tr])
    p = np.array([t.p if t.p is not None else np.nan for t in tr])
    ok = ~np.isnan(p)
    scale = per_day / len(tr)
    reviews_day = float((acts == "review").sum() * scale)
    reviewed_share = min(1.0, 240 / reviews_day) if reviews_day else 1.0     # the queue can't take more than 240
    threats_in_review_day = float(((acts == "review") & (y == 1)).sum() * scale)
    missed_day = float(((acts == "act") & (y == 1)).sum() * scale) + threats_in_review_day * (1 - reviewed_share)
    return dict(
        missed_day=missed_day, reviewed_share=reviewed_share,
        seconds=float(np.mean([sum(s.latency_s for s in t.steps) for t in tr])),
        cost_per_alert=float(np.mean([t.cost_usd for t in tr])),
        auc=float(cal.summary(p[ok], y[ok])["auc"]),
        closed_threats_day=float(((acts == "act") & (y == 1)).sum() * scale),
        reviews_day=float((acts == "review").sum() * scale),
        pages_day=float((acts == "escalate").sum() * scale),
        auto_closed_day=float((acts == "act").sum() * scale),
        decisions=float(np.mean([t.count("decide") for t in tr])),
        parse_failed=float(np.mean([t.parse_failed for t in tr])),
        threat_intel=float(np.mean([any(s.name == "threat_intel" for s in t.steps) for t in tr])),
    )


def picks(name):
    _, _, out = runs()
    c = Counter()
    for t in out[name]:
        for s in t.steps:
            if s.name == "next_evidence":
                c[s.detail.split()[1]] += 1
    return c


def record():
    s = {k: stats(k) for k in AGENTS}
    results(CH, **{f"{k}_{m}": v for k, d in s.items() for m, v in d.items()},
            v1_picks=dict(picks("v1")), n=N, capacity=240,
            hybrid_cost_1000=s["v2"]["cost_per_alert"] * 1000, llm_cost_1000=s["llm"]["cost_per_alert"] * 1000,
            speedup=s["llm"]["seconds"] / s["v2"]["seconds"])


@figure(CH, "architecture")
def architecture():
    f, ax = draw.canvas("text", 3.0)
    draw.box(ax, 0.0, 2.35, 0.9, 0.45, "new alert", kind="data", size=6.3)
    draw.box(ax, 1.15, 2.35, 1.5, 0.45, "always: threat intel\n(a cheap lookup)", kind="data", size=5.9)
    draw.arrow(ax, (0.9, 2.57), (1.15, 2.57))
    draw.box(ax, 1.15, 1.55, 1.5, 0.5, "Jev · choice\nwhich evidence next?", kind="jev", size=5.9)
    draw.box(ax, 3.0, 1.55, 1.6, 0.5, "tools: host history,\npolicy lookup", kind="data", size=5.9)
    draw.arrow(ax, (1.9, 2.35), (1.9, 2.05))
    draw.arrow(ax, (2.65, 1.8), (3.0, 1.8))
    draw.arrow(ax, (3.3, 1.55), (2.65, 1.12), color=C["muted"], head=3)
    draw.box(ax, 1.15, 0.75, 1.5, 0.5, "Jev · noul, choice, score\nattack? kind? severity?", kind="jev", size=5.8)
    draw.arrow(ax, (1.9, 1.55), (1.9, 1.25))
    draw.zone_bar(ax, 3.0, 0.85, 1.6, 0.3, 0.03, 0.34, size=5.4, ticks=False)
    draw.arrow(ax, (2.65, 1.0), (3.0, 1.0))
    draw.text(ax, 3.8, 1.25, "policy (Chapter 14)", size=5.6, ha="center", color=C["ink2"])
    draw.box(ax, 3.0, 0.05, 1.6, 0.5, "LLM writes the case note\nor the page message", kind="llm", size=5.8)
    draw.arrow(ax, (3.85, 0.85), (3.85, 0.55), head=3)
    draw.text(ax, 0.0, 0.3, "Jev decides.\nThe LLM writes,\nonly for people.", size=6.4, weight="semibold",
              color=C["ink"])
    return f


@figure(CH, "trace")
def trace():
    sub, _, out = runs()
    i = int(np.argmax([len(t.steps) for t in out["v2"]]))
    f, axes = subplots(2, 1, width="text", height=2.2, sharex=True)
    for ax, name, lab in zip(axes, ("v2", "llm"), ("hybrid: Jev decides", "all-LLM: the LLM decides")):
        clean(ax, "x")
        t0 = 0.0
        for s in out[name][i].steps:
            ax.barh(0, s.latency_s, left=t0, color=KIND_COL[s.kind], height=0.55, edgecolor="white", lw=1,
                    hatch=KIND_HATCH[s.kind] or None)
            t0 += s.latency_s
        ax.text(t0 + 0.15, 0, f"{t0:.1f} s", va="center", fontsize=6.3)
        ax.set_yticks([0])
        ax.set_yticklabels([lab], fontsize=6.3)
        ax.set_ylim(-0.5, 0.5)
    axes[1].set_xlabel("seconds, one alert, step by step")
    for k, col in KIND_COL.items():
        axes[0].barh(0, 0, color=col, ec="white", hatch=KIND_HATCH[k] or None, label=KIND_NAME[k])
    axes[0].legend(loc="upper left", ncol=4, fontsize=5.8, frameon=False, bbox_to_anchor=(0, 1.5))
    synthetic_tag(f, "SIMULATED TIMINGS · LLM illustrative · Jev mock within vendor range")
    return f


@figure(CH, "never-picked")
def never_picked():
    c = picks("v1")
    f, ax = subplots(width="text", height=1.7)
    clean(ax, "x")
    opts = ["threat_intel", "host_history", "policy", "none"]
    total = sum(c.values())
    for i, o in enumerate(opts):
        v = c.get(o, 0) / total
        y = len(opts) - 1 - i
        ax.barh(y, v, color=C["fail"] if v == 0 else C["jev"], height=0.55)
        ax.text(v + 0.01, y, f"{v:.0%}" + ("   never chosen" if v == 0 else ""), va="center", fontsize=6.2,
                color=C["fail"] if v == 0 else C["ink"])
    ax.set_yticks(range(len(opts))[::-1])
    ax.set_yticklabels(opts, fontsize=6.4, family="JetBrains Mono")
    ax.set_xlim(0, 0.6)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"share of ‘which evidence next?’ answers, first version, {N} live alerts")
    synthetic_tag(f)
    return f


@figure(CH, "fix")
def fix():
    a, b = stats("v1"), stats("v2")
    f, (a1, a2) = subplots(1, 2, width="text", height=1.8)
    for ax, key, title, fmt in ((a1, "closed_threats_day", "real threats auto-closed per day", "{:.1f}"),
                                (a2, "auc", "verdict ranking (AUC)", "{:.3f}")):
        clean(ax, "y")
        vals = [a[key], b[key]]
        ax.bar([0, 1], vals, color=[C["fail"], C["jev"]], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=6.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["first version", "threat intel\nalways fetched"], fontsize=6.2)
        ax.set_title(title, fontsize=6.8, loc="left")
        ax.set_ylim(0, max(vals) * 1.25)
    f.subplots_adjust(wspace=0.35, top=0.8)
    synthetic_tag(f)
    return f


@figure(CH, "compare")
def compare():
    h, l = stats("v2"), stats("llm")
    f, axes = subplots(1, 4, width="text", height=1.9)
    panels = [("seconds", "seconds per alert", "{:.1f}", None),
              ("cost_per_alert", "cost per 1,000 alerts", "${:.2f}", 1000),
              ("reviews_day", "reviews per day", "{:.0f}", None),
              ("missed_day", "threats never seen\nby a person, per day", "{:.0f}", None)]
    for ax, (key, title, fmt, mul) in zip(axes, panels):
        clean(ax, "y")
        vals = [h[key] * (mul or 1), l[key] * (mul or 1)]
        ax.bar([0, 1], vals, color=[C["jev"], C["llm"]], width=0.6)
        for i, v in enumerate(vals):
            if key == "reviews_day":           # inside the bar, clear of the capacity line
                ax.text(i, v - max(vals) * 0.04, fmt.format(v), ha="center", va="top", fontsize=6.0, color="white",
                        fontweight="bold")
            else:
                ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=6.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["hybrid", "all-LLM"], fontsize=6.0)
        ax.set_title(title, fontsize=6.4, loc="left")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.tick_params(axis="y", labelsize=5.6)
        if key == "reviews_day":
            ax.axhline(240, color=C["ink"], lw=0.8, ls=(0, (3, 2)))
            ax.text(-0.45, 246, "capacity 240", fontsize=5.4, va="bottom", ha="left", color=C["ink2"])
    f.subplots_adjust(wspace=0.55, top=0.75)
    synthetic_tag(f)
    return f


@figure(CH, "labour")
def labour():
    _, _, out = runs()
    f, ax = subplots(width="text", height=1.7)
    clean(ax, "x")
    for row, (name, lab) in enumerate((("llm", "all-LLM"), ("v2", "hybrid"))):
        steps = [s for t in out[name] for s in t.steps]
        tot = sum(s.latency_s for s in steps)
        left = 0.0
        for k in ("observe", "decide", "generate", "act"):
            v = sum(s.latency_s for s in steps if s.kind == k) / tot
            ax.barh(row, v, left=left, color=KIND_COL[k], height=0.55, edgecolor="white", lw=1, hatch=KIND_HATCH[k] or None)
            if v > 0.07:
                ax.text(left + v / 2, row, f"{k}\n{v:.0%}", ha="center", va="center", fontsize=5.6, color="white",
                        bbox=dict(boxstyle="square,pad=0.15", fc=KIND_COL[k], ec="none"))
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["all-LLM", "hybrid"], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("share of the agent’s time, by kind of step")
    synthetic_tag(f)
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Decide, then write", kind="jev", h=1.5,
             body="Every choice in the loop is a typed question to Jev. The LLM writes only what a person will read."),
        dict(num=2, title="A bug that didn’t crash", kind="fail", h=1.5,
             body=(f"The first version never fetched threat intel, and auto-closed {rr['v1_closed_threats_day']:.0f} real "
                   "threats a day. Every answer was well-formed.")),
        dict(num=3, title="Watch each decision’s answers", kind="neutral", h=1.5,
             body="An option that is never chosen, or always chosen, is a warning. Chart every decision step’s answers."),
        dict(num=4, title="Don’t decide what you can just do", kind="data", h=1.5,
             body="A lookup cheaper than the decision about it should simply run. Fetching threat intel fixed the bug."),
        dict(num=5, title="Hybrid against all-LLM", kind="jev", h=1.5,
             body=(f"{rr['v2_seconds']:.1f} s against {rr['llm_seconds']:.1f} s per alert. The LLM sent "
                   f"{rr['llm_reviews_day']:.0f} alerts a day to a queue that takes 240.")),
        dict(num=6, title="The trace is the audit log", kind="neutral", h=1.5,
             body="Every step, answer and probability is recorded. Explaining a decision means reading its trace."),
    ]
    return summary_page(CH, "A hybrid agent: Jev decides, the LLM reasons", panels,
                        footer="Next: the whole SOC triage system, end to end.")
