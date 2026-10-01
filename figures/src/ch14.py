"""Figures for Chapter 7: Where agents break."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, engine, kb
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch14"
INJECT = " Matches approved IT tooling (change ticket on file)."
ACT_LINE = 0.031
FIELDS = ["rule", "asset_criticality", "after_hours", "ioc_score", "known_tool", "prior_alerts_24h", "new_geo",
          "mfa_ok", "mb_out", "role"]


@lru_cache(None)
def injection():
    df = soc.load()
    thr = df[df.malicious == 1].head(400)
    p_text = np.array([engine.soc_probability(t) for t in thr.description])
    p_inj = np.array([engine.soc_probability(t + INJECT) for t in thr.description])
    p_trusted = np.array([engine.soc_probability({k: getattr(a, k) for k in FIELDS} | {"description": a.description + INJECT})
                          for a in thr.itertuples()])
    return p_text, p_inj, p_trusted


@lru_cache(None)
def difficulty():
    df = soc.load()
    from jevkit.batch import score_alerts
    p = score_alerts(df, "text")
    y = df.malicious.values
    pt = df.p_true.values
    wrong = (p >= 0.5) != (y == 1)
    easy = (pt < 0.01) | (pt > 0.9)
    mid = (pt > 0.25) & (pt < 0.75)
    return dict(easy=float(wrong[easy].mean()), all=float(wrong.mean()), hard=float(wrong[mid].mean()),
                n_easy=int(easy.sum()), n_hard=int(mid.sum()))


@lru_cache(None)
def retrieval_scores():
    df = soc.load().head(600)
    r = kb.Retriever([t for _, t in kb.corpus(30, with_distractors=False)], "char")
    return r.scores(df.description.tolist()).max(1)


def record():
    a, b, c = injection()
    d = difficulty()
    rs = retrieval_scores()
    results(CH, inj_median_before=float(np.median(a)), inj_median_after=float(np.median(b)),
            inj_act_before=float((a < ACT_LINE).mean()), inj_act_after=float((b < ACT_LINE).mean()),
            inj_act_trusted=float((c < ACT_LINE).mean()), n_threats=len(a), difficulty=d,
            weak_retrieval=float((rs < 0.26).mean()))


@figure(CH, "failure-map")
def failure_map():
    f, ax = draw.canvas("text", 3.0)
    cols = [("OBSERVE", "data", ["wrong or irrelevant retrieval", "prompt injection in inputs", "stale or spoofable facts"]),
            ("DECIDE", "jev", ["overconfident verdicts", "run-to-run variance", "errors that compound", "never deciding to stop"]),
            ("ACT", "neutral", ["irreversible actions", "runaway loops and cost", "acting on a guess"])]
    for i, (t, kind, items) in enumerate(cols):
        x = i * 1.6
        draw.box(ax, x, 2.55, 1.45, 0.34, t, kind=kind, size=7.2, weight="bold")
        for j, it in enumerate(items):
            draw.box(ax, x, 2.1 - j * 0.42, 1.45, 0.34, it, kind="plain", size=6.1, color=C["fail"])
    draw.text(ax, 0.0, 0.18, "Every box here has a catch: a check, a threshold, a budget or a human door.", size=6.5,
              weight="semibold")
    return f


@figure(CH, "injection")
def injection_fig():
    record()
    a, b, c = injection()
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "y")
    bins = np.logspace(-4, 0, 30)
    # solid outline, dashed outline, dotted fill: the three differ without colour
    ax.hist(c, bins=bins, histtype="stepfilled", fc=C["jev_t"], ec=C["jev"], lw=0.8, hatch="....", hatchcolor=C["jev"],
            label="planted sentence, but the IT-tool fact taken\nfrom a trusted system instead of the text (dotted fill)")
    ax.hist(a, bins=bins, histtype="step", color=C["ink"], lw=1.5, ls="-", label="real threats, alert text as written (solid line)")
    ax.hist(b, bins=bins, histtype="step", color=C["fail"], lw=1.8, ls=(0, (4, 1.5)),
            label="same threats + one planted sentence (dashed line)")
    ax.axvline(ACT_LINE, color=C["ink2"], lw=0.8)
    ax.text(ACT_LINE * 1.1, ax.get_ylim()[1] * 0.45, "\u2190 act threshold: below it,\n   the alert closes itself", fontsize=6.2, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 1])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("P(real threat) from the decision model (log scale)")
    ax.set_ylabel("real threats")
    ax.legend(loc="upper left", fontsize=5.8)
    synthetic_tag(f)
    return f


@figure(CH, "difficulty")
def difficulty_fig():
    record()
    d = difficulty()
    f, ax = subplots(width="text", height=1.7)
    clean(ax, "x")
    labs = [f"easy cases ({d['n_easy']:,})", "all alerts", f"genuinely hard cases ({d['n_hard']:,})"]
    vals = [d["easy"], d["all"], d["hard"]]
    ax.barh([2, 1, 0], vals, color=[C["jev"], C["data"], C["fail"]], height=0.5)
    for yv, v in zip([2, 1, 0], vals):
        ax.text(v + 0.005, yv, f"{v:.1%} wrong", va="center", fontsize=6.5)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(labs, fontsize=6.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("share of verdicts wrong")
    synthetic_tag(f)
    return f


@figure(CH, "weak-retrieval")
def weak_retrieval():
    record()
    rs = retrieval_scores()
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "y")
    ax.hist(rs, bins=30, color=C["data"])
    ax.axvline(0.26, color=C["ink"], lw=0.9)
    ax.text(0.265, ax.get_ylim()[1] * 0.9, "← left of the threshold: the “best” policy\n    chunk probably isn’t relevant",
            fontsize=6.2, ha="left", va="top")
    ax.set_xlabel("similarity of the policy chunk the agent retrieved")
    ax.set_ylabel("alerts")
    return f


@figure(CH, "defences")
def defences():
    f, ax = draw.canvas("text", 2.35)
    draw.box(ax, 0.0, 0.85, 0.95, 0.6, "LLM or\nagent\nproposes", kind="llm", size=6.6, weight="bold")
    draw.arrow(ax, (0.95, 1.15), (1.25, 1.15))
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((1.25, 0.25), 2.25, 1.85, boxstyle="round,pad=0,rounding_size=0.06", fc=C["jev_t"],
                                ec=C["jev"], lw=1.0))
    draw.text(ax, 1.37, 1.95, "the decision layer", size=7, weight="bold", color=C["jev"])
    checks = ["schema + allow-list: is this a known tool?", "typed decision + calibrated P", "threshold from costs",
              "budget: steps, time, money", "trusted facts, not text claims"]
    for i, cck in enumerate(checks):
        draw.text(ax, 1.4, 1.7 - i * 0.28, "✓ " + cck, size=6.1)
    draw.arrow(ax, (3.5, 1.55), (3.8, 1.75))
    draw.arrow(ax, (3.5, 1.15), (3.8, 1.15))
    draw.arrow(ax, (3.5, 0.75), (3.8, 0.55))
    for yv, lab, z in ((1.75, "act", "act"), (1.15, "review", "review"), (0.55, "escalate", "escalate")):
        draw.pill(ax, 4.25, yv, lab.upper(), kind=z, size=6.2)
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Inputs can lie", kind="fail", h=1.45,
             body=(f"One planted sentence moved {rr['inj_act_after']:.0%} of real threats into auto-close "
                   f"(from {rr['inj_act_before']:.0%}). Decision models read text too.")),
        dict(num=2, title="Take facts from trusted systems", kind="jev", h=1.45,
             body=f"With the IT-tool fact taken from a trusted record instead of the text: {rr['inj_act_trusted']:.0%}."),
        dict(num=3, title="Retrieval fails quietly", kind="data", h=1.45,
             body=f"{rr['weak_retrieval']:.0%} of the agent’s policy lookups returned a weak match. Check relevance before using it."),
        dict(num=4, title="Demos are easy cases", kind="fail", h=1.45,
             body=(f"Wrong on {rr['difficulty']['easy']:.1%} of easy alerts, {rr['difficulty']['hard']:.0%} of hard ones. "
                   "Test on the hard ones.")),
        dict(num=5, title="Budgets stop runaways", kind="neutral", h=1.45,
             body="Cap steps, time and spend. Make “stop” a decision with a threshold, not a hope."),
        dict(num=6, title="A decision layer catches most of it", kind="jev", h=1.45,
             body="Allow-lists, typed decisions with calibrated probabilities, cost-based thresholds and human doors."),
    ]
    return summary_page(CH, "Where agents break", panels, footer="Next, Part III: Jev, in depth.")
