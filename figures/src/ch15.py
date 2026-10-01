"""Figures for Chapter 8: System 1 and System 2."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, calibration as cal
from jevkit.batch import score_alerts
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch15"
CFN, CFP = 10_000, 400
S1_SECONDS, S2_SECONDS = 0.15, 25.0          # illustrative: a fast decision vs a long reasoning pass


@lru_cache(None)
def routing():
    df = soc.load()
    p = score_alerts(df, "text")
    hist, live = soc.history_and_live(df.assign(p=p))
    q = cal.Platt().fit(hist.p, hist.malicious)(live.p.values)
    y, pt = live.malicious.values, live.p_true.values
    t = CFP / (CFP + CFN)
    L = lambda x: np.log(x / (1 - x))
    order = np.argsort(np.abs(L(np.clip(q, 1e-6, 1 - 1e-6)) - L(t)))
    rnd = np.random.default_rng(0).permutation(len(q))

    def cost(dec):
        return float(((dec == 1) & (y == 0)).sum() * CFP + ((dec == 0) & (y == 1)).sum() * CFN) / 7

    fs = np.linspace(0, 1, 41)
    smart, rand = [], []
    for f in fs:
        k = int(round(f * len(q)))
        for idxs, out in ((order[:k], smart), (rnd[:k], rand)):
            dec = (q >= t).astype(int)
            dec[idxs] = (pt[idxs] >= t).astype(int)
            out.append(cost(dec))
    return fs, np.array(smart), np.array(rand)


def record():
    fs, smart, rand = routing()
    i20 = int(np.argmin(np.abs(fs - 0.2)))
    gain = (smart[0] - smart[i20]) / (smart[0] - smart[-1])
    gain_r = (rand[0] - rand[i20]) / (rand[0] - rand[-1])
    results(CH, cost_s1=smart[0], cost_s2=smart[-1], cost_20=smart[i20], cost_20_random=rand[i20],
            share_gain_20=gain, share_gain_20_random=gain_r,
            time_all_s1=S1_SECONDS, time_all_s2=S2_SECONDS, time_20=S1_SECONDS + 0.2 * S2_SECONDS)


@figure(CH, "two-systems")
def two_systems():
    f, ax = draw.canvas("text", 2.3)
    rows = [("", "System 1", "System 2"), ("speed", "fast, automatic", "slow, effortful"),
            ("feels like", "just knowing", "working it out"), ("good at", "familiar patterns,\nat volume", "new problems,\nlong chains of logic"),
            ("fails by", "confident snap\njudgements", "tiring, and slow\nwhen time matters"),
            ("in software", "classifiers, rules,\nSystem One models", "reasoning LLMs,\nanalysts")]
    for i, (a, b, c) in enumerate(rows):
        y = 2.1 - i * 0.37
        if i == 0:
            draw.box(ax, 1.25, y - 0.15, 1.6, 0.3, b, kind="jev", size=7.2, weight="bold")
            draw.box(ax, 3.05, y - 0.15, 1.6, 0.3, c, kind="llm", size=7.2, weight="bold")
            continue
        draw.text(ax, 0.0, y, a, size=6.5, weight="semibold", color=C["ink2"])
        draw.text(ax, 2.05, y, b, size=6.3, ha="center")
        draw.text(ax, 3.85, y, c, size=6.3, ha="center")
        ax.plot([0, 4.7], [y - 0.19, y - 0.19], color=C["grid"], lw=0.6)
    return f


@figure(CH, "task-map")
def task_map():
    f, ax = subplots(width="text", height=2.8)
    clean(ax, "both")
    tasks = [("is this alert real?", 700, 3.3), ("which queue?", 900, 0.5), ("tool call: which tool next?", 3700, 1.2),
             ("is this retrieved chunk relevant?", 2500, 4.0), ("dedupe: same incident?", 1500, 1.9),
             ("severity 0–3", 600, 2.6), ("write the incident report", 3, 6.5),
             ("reconstruct an attack timeline", 2, 8.5), ("tune a detection rule", 0.5, 7.5),
             ("explain an alert to the CISO", 1, 5.5)]
    for name, vol, depth in tasks:
        col = C["jev"] if depth < 5 else C["llm"]
        ax.scatter([vol], [depth], s=26, color=col, edgecolor="white", lw=0.8, zorder=3)
        right = vol > 100
        ax.text(vol / 1.2 if right else vol * 1.2, depth, name, fontsize=6.1, va="center", ha="right" if right else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 60000)
    ax.set_ylim(0, 10)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlabel("times a day at Kestrel (log scale, illustrative)")
    ax.set_ylabel("how much thinking it needs")
    ax.set_yticks([])
    ax.text(0.4, 9.3, "System 2 territory", fontsize=6.6, color=C["llm"], fontweight="bold")
    ax.text(3000, 5.0, "System 1 territory", fontsize=6.6, color=C["jev"], fontweight="bold")
    return f


@figure(CH, "router")
def router():
    f, ax = draw.canvas("text", 1.85)
    draw.box(ax, 0.0, 0.7, 0.8, 0.46, "every\nalert", kind="data", size=6.8, weight="bold")
    draw.arrow(ax, (0.8, 0.93), (1.1, 0.93))
    draw.box(ax, 1.1, 0.62, 1.1, 0.62, "System 1\nfast decision\n+ probability", kind="jev", size=6.6, weight="bold")
    draw.arrow(ax, (2.2, 1.1), (2.75, 1.45), label="confident", labelsize=6, labeloffset=(-0.1, 0.03))
    draw.arrow(ax, (2.2, 0.76), (2.75, 0.42), label="unsure", labelsize=6, labeloffset=(-0.1, -0.2))
    draw.box(ax, 2.75, 1.25, 1.95, 0.42, "decide now (most alerts)", kind="jev", size=6.6)
    draw.box(ax, 2.75, 0.18, 1.95, 0.5, "System 2: reason slowly,\ngather evidence, or ask a person", kind="llm", size=6.4)
    return f


@figure(CH, "routing-curve")
def routing_curve():
    record()
    fs, smart, rand = routing()
    f, ax = subplots(width="text", height=2.35)
    clean(ax, "y")
    ax.plot(fs, smart / 1000, color=C["jev"], lw=2, ls="-", label="send the cases System 1 is least sure about (solid)")
    ax.plot(fs, rand / 1000, color=C["muted"], lw=1.5, ls=(0, (5, 2)), label="send a random share (dashed)")
    ax.axvline(0.2, color=C["ink2"], lw=0.6)
    ax.set_xlabel("share of alerts sent to System 2")
    ax.set_ylabel("expected cost per day ($k)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", fontsize=6.3)
    synthetic_tag(f)
    return f


@figure(CH, "conditions")
def conditions():
    f, ax = draw.canvas("text", 1.75)
    draw.text(ax, 0.0, 1.62, "When can fast judgement be trusted? (after Kahneman and Klein, 2009)", size=6.8, weight="bold")
    items = [("A regular world", "the same cues keep meaning\nthe same things", "jev"),
             ("Lots of practice", "many examples, not a\nhandful of anecdotes", "jev"),
             ("Quick, clear feedback", "you find out whether\nyou were right", "jev"),
             ("Honest doubt", "knowing when the case\nis unfamiliar", "fail")]
    for i, (t, s, kind) in enumerate(items):
        x = i * 1.2
        draw.box(ax, x, 0.72, 1.1, 0.5, t, kind=kind, size=6.3, weight="bold")
        draw.text(ax, x + 0.55, 0.6, s, size=5.8, ha="center", va="top", color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Two ways of thinking", kind="neutral", h=1.45,
             body="System 1 is fast, automatic, intuitive. System 2 is slow, effortful, deliberate. Both are useful; both fail."),
        dict(num=2, title="Most work is System 1", kind="jev", h=1.45,
             body="An analyst triages most alerts in seconds and investigates a few. Agents look the same: many quick calls, few deep ones."),
        dict(num=3, title="Software has mostly had System 2", kind="llm", h=1.45,
             body="General LLMs reason in text: powerful, but slow and costly to use for every small judgement."),
        dict(num=4, title="Route by doubt", kind="jev", h=1.45,
             body=(f"Sending the least-sure 20% to System 2 captured {rr['share_gain_20']:.0%} of the possible gain; a random "
                   f"20% captured {rr['share_gain_20_random']:.0%}.")),
        dict(num=5, title="Doubt must be honest", kind="fail", h=1.45,
             body="Routing only works if System 1 knows when it’s unsure. That means calibrated probabilities."),
        dict(num=6, title="When intuition can be trusted", kind="neutral", h=1.45,
             body="A regular world, lots of practice, quick feedback. SOC triage fits. Novel investigations don’t."),
    ]
    return summary_page(CH, "System 1 and System 2", panels, footer="Next: what’s actually inside Jev, and what isn’t known.")
