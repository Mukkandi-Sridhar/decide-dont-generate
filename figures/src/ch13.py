"""Figures for Chapter 7: Agents: Observe, Decide, Act."""

from collections import Counter
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, llm
from jevkit.agent import SOCAgent
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch13"
KCOL = {"observe": C["data"], "decide": C["jev"], "generate": C["llm"], "act": C["slate"]}


@lru_cache(None)
def traces():
    df = soc.load()
    ag = SOCAgent()
    return [ag.run(r) for r in df.head(300).itertuples()]


def record():
    tr = traces()
    c = Counter(s.kind for t in tr for s in t.steps)
    n = len(tr)
    llm_json = llm.MockLLM.simulated_latency_s(40)
    per_fast = [sum(s.latency_s for s in t.steps) for t in tr]
    per_llm = [sum((llm_json if s.kind == "decide" else s.latency_s) for s in t.steps) for t in tr]
    results(CH, per_alert={k: c[k] / n for k in ("observe", "decide", "generate", "act")},
            share_decide=c["decide"] / sum(c.values()), share_generate=c["generate"] / sum(c.values()),
            n_traces=n, t_fast=float(np.mean(per_fast)), t_llm=float(np.mean(per_llm)),
            llm_json_latency=llm_json, compound_95_10=0.95 ** 10, compound_99_10=0.99 ** 10,
            compound_95_20=0.95 ** 20, actions=dict(Counter(t.action for t in tr)))


@figure(CH, "loop")
def loop():
    f, ax = draw.canvas("text", 2.5)
    cx, cy, r = 2.35, 1.25, 0.85
    nodes = [("Observe", "read the alert, call\ntools, fetch memory", 90, "data"),
             ("Decide", "what next? enough?\nwhich door?", -30, "jev"),
             ("Act", "close, queue, page,\nisolate a host", 210, "neutral")]
    pos = {}
    for name, sub, ang, kind in nodes:
        x = cx + r * np.cos(np.radians(ang)) * 1.45
        y = cy + r * np.sin(np.radians(ang))
        pos[name] = (x, y)
        draw.box(ax, x - 0.5, y - 0.2, 1.0, 0.4, name, kind=kind, size=7.4, weight="bold")
        draw.text(ax, x, y - 0.28, sub, size=5.9, ha="center", va="top", color=C["ink2"])
    order = ["Observe", "Decide", "Act", "Observe"]
    for a, b in zip(order, order[1:]):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        draw.arrow(ax, (x1 + (0.5 if x2 > x1 else -0.5) * 0.9, y1), (x2 - (0.5 if x2 > x1 else -0.5) * 0.9, y2 + 0.1),
                   rad=-0.3, color=C["ink2"], lw=0.9)
    draw.text(ax, cx, cy - 0.05, "the world\nchanges", size=6.2, ha="center", color=C["muted"], style="italic")
    return f


@figure(CH, "trace")
def trace():
    t = traces()[3]
    f, ax = draw.canvas("text", 0.35 + 0.23 * len(t.steps))
    H = 0.35 + 0.23 * len(t.steps)
    for i, s in enumerate(t.steps):
        y = H - 0.25 - i * 0.23
        draw.pill(ax, 0.35, y, s.kind, kind={"observe": "data", "decide": "jev", "generate": "llm", "act": "neutral"}[s.kind],
                  size=5.8)
        draw.text(ax, 0.8, y, s.name, size=6.4, family="JetBrains Mono")
        draw.text(ax, 2.05, y, s.detail[:52], size=6.0, color=C["ink2"])
        draw.text(ax, 4.7, y, f"{s.latency_s:.2f} s", size=5.8, ha="right", color=C["muted"], family="JetBrains Mono")
    return f


@figure(CH, "census")
def census():
    record()
    tr = traces()
    c = Counter(s.kind for t in tr for s in t.steps)
    n = len(tr)
    kinds = ["decide", "observe", "act", "generate"]
    labels = {"decide": "small typed decisions", "observe": "tool reads", "act": "actions", "generate": "free text written"}
    f, ax = subplots(width="text", height=1.8)
    clean(ax, "x")
    vals = [c[k] / n for k in kinds]
    ax.barh(range(len(kinds))[::-1], vals, color=[KCOL[k] for k in kinds], height=0.55)
    for i, (k, v) in enumerate(zip(kinds, vals)):
        ax.text(v + 0.05, len(kinds) - 1 - i, f"{v:.1f}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(kinds))[::-1])
    ax.set_yticklabels([labels[k] for k in kinds], fontsize=6.8)
    ax.set_xlabel(f"steps per alert (average over {n} alerts)")
    synthetic_tag(f, "SYNTHETIC · the book’s teaching agent")
    return f


@figure(CH, "compound")
def compound():
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    n = np.arange(1, 31)
    for p, col in ((0.99, C["jev"]), (0.97, C["data"]), (0.95, C["llm"]), (0.90, C["fail"])):
        ax.plot(n, p ** n, color=col, lw=1.7)
        ax.text(30.4, p ** 30, f"{p:.0%} per step", fontsize=6.3, va="center", color=C["ink"])
    ax.set_xlabel("steps in the task")
    ax.set_ylabel("chance every step is right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(1, 36)
    ax.set_ylim(0, 1.02)
    return f


@figure(CH, "latency")
def latency():
    record()
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    tr = traces()
    llm_json = rr["llm_json_latency"]
    f, ax = subplots(width="text", height=1.6)
    clean(ax, "x")
    for yi, (lab, fn) in enumerate((("decisions by an LLM\n(small JSON each)", lambda s: llm_json if s.kind == "decide" else s.latency_s),
                                    ("decisions by a fast\ndecision model", lambda s: s.latency_s))):
        left = 0
        for k in ("observe", "decide", "generate", "act"):
            v = np.mean([sum(fn(s) for s in t.steps if s.kind == k) for t in tr])
            ax.barh(1 - yi, v, left=left, color=KCOL[k], height=0.5, edgecolor="white", lw=1)
            left += v
        ax.text(left + 0.1, 1 - yi, f"{left:.1f} s", va="center", fontsize=6.6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["decisions by an LLM\n(small JSON each)", "decisions by a fast\ndecision model"], fontsize=6.4)
    ax.set_xlabel("seconds per alert (illustrative latency assumptions)")
    from matplotlib.patches import Rectangle
    ax.legend([Rectangle((0, 0), 1, 1, color=KCOL[k]) for k in ("observe", "decide", "generate", "act")],
              ["observe", "decide", "generate", "act"], ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.62), fontsize=6.2)
    synthetic_tag(f, "SYNTHETIC · illustrative latencies, not measured")
    return f


@figure(CH, "tool-call")
def tool_call():
    f, ax = draw.canvas("text", 1.7)
    draw.box(ax, 0.0, 0.9, 1.45, 0.62, "model output", kind="llm", size=6.8, weight="bold", sub='{"tool": "threat_intel",\n "args": {"ip": "185.4.2.9"}}', subsize=5.4)
    draw.arrow(ax, (1.45, 1.21), (1.75, 1.21))
    draw.box(ax, 1.75, 0.9, 1.2, 0.62, "your code runs\nthe tool", kind="neutral", size=6.6)
    draw.arrow(ax, (2.95, 1.21), (3.25, 1.21))
    draw.box(ax, 3.25, 0.9, 1.45, 0.62, "result goes back\ninto the context", kind="data", size=6.6)
    draw.text(ax, 0.0, 0.5, "A “tool call” is two decisions in a trench coat: which tool (a choice among a", size=6.4)
    draw.text(ax, 0.0, 0.3, "fixed list) and what arguments (an extraction). Your code, not the model, does the acting.", size=6.4)
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="An agent is a loop", kind="neutral", h=1.45,
             body="Observe the situation, decide what to do, act, observe what happened. Repeat until done."),
        dict(num=2, title="Tools are how it acts", kind="neutral", h=1.45,
             body="The model names a tool and its arguments; your code runs it and hands back the result. Keep the acting in your code."),
        dict(num=3, title="Most steps are decisions", kind="jev", h=1.45,
             body=(f"In our SOC agent: {rr['per_alert']['decide']:.1f} small typed decisions per alert, "
                   f"{rr['per_alert']['generate']:.1f} pieces of free text.")),
        dict(num=4, title="Errors compound", kind="fail", h=1.45,
             body=f"95% right per step sounds good. Ten steps in a row: {rr['compound_95_10']:.0%}. Twenty: {rr['compound_95_20']:.0%}."),
        dict(num=5, title="Latency compounds too", kind="llm", h=1.45,
             body=(f"With illustrative numbers, making each decision with an LLM call took {rr['t_llm']:.1f} s per alert; "
                   f"with a fast decision model {rr['t_fast']:.1f} s.")),
        dict(num=6, title="Stopping is a decision", kind="jev", h=1.45,
             body="“Do we have enough evidence?” decides cost, speed and quality. Give it a budget and a threshold."),
    ]
    return summary_page(CH, "Agents: Observe, Decide, Act", panels, footer="Next: an honest look at where agents break.")
