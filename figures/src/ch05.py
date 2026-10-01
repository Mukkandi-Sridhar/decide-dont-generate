"""Figures for Chapter 4: From probabilities to actions."""

from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression

from jevkit import soc, policy as pol
from jevkit.figs import figure, draw, C, ZONE, subplots, clean, results, summary_page, synthetic_tag

CH = "ch05"
CFP, CFN = 200, 4000        # wrongly blocking a harmless alert vs letting a real threat through
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
MONEY = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


@lru_cache(None)
def data():
    df = soc.load()
    tr, ca, te = soc.split(df)
    F = lambda d: soc.feature_matrix(d).values
    m = LogisticRegression(C=1e4, max_iter=5000).fit(F(tr), tr.malicious)
    p = m.predict_proba(F(te))[:, 1]
    pos = tr[tr.malicious == 1]
    neg = tr[tr.malicious == 0].sample(len(pos), random_state=0)
    bal = pd.concat([pos, neg])
    mb = LogisticRegression(C=1e4, max_iter=5000).fit(F(bal), bal.malicious)
    pb = mb.predict_proba(F(te))[:, 1]
    return dict(p=p, pb=pb, y=te.malicious.values, te=te)


def cost(pp, y, t):
    f = pp >= t
    return float((f & (y == 0)).sum() * CFP + ((~f) & (y == 1)).sum() * CFN)


def record():
    d = data()
    p, pb, y = d["p"], d["pb"], d["y"]
    t = CFP / (CFP + CFN)
    ts = np.linspace(0.005, 0.9, 300)
    c1 = np.array([cost(p, y, x) for x in ts])
    c2 = np.array([cost(pb, y, x) for x in ts])
    f = p >= t
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(float)
    cov, risk, _ = pol.coverage_risk(conf, correct)
    r = {c: float(risk[int(c * len(cov)) - 1]) for c in (0.5, 0.8, 0.9, 1.0)}
    results(CH, t_formula=t, t_best=float(ts[c1.argmin()]), cost_best=float(c1.min()), cost_formula=cost(p, y, t),
            cost_half=cost(p, y, 0.5), cost_none=cost(p, y, 2.0), cost_all=cost(p, y, -1.0),
            cost_bal_formula=cost(pb, y, t), t_bal_best=float(ts[c2.argmin()]),
            bal_penalty=cost(pb, y, t) / cost(p, y, t) - 1, half_penalty=cost(p, y, 0.5) / cost(p, y, t),
            blocked_frac=float(f.mean()), caught=float((f & (y == 1)).sum() / y.sum()),
            precision=float((f & (y == 1)).sum() / f.sum()), n_test=len(y),
            risk_50=r[0.5], risk_80=r[0.8], risk_90=r[0.9], risk_100=r[1.0],
            umbrella_t=1 / 5, soc_review_t=pol.cost_optimal_thresholds()[0])
    return d, ts, c1, c2


@figure(CH, "umbrella")
def umbrella():
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    p = np.linspace(0, 1, 100)
    carry = np.full_like(p, 1.0)
    leave = p * 4.0
    ax.plot(p, carry, color=C["data"], lw=1.8)
    ax.plot(p, leave, color=C["fail"], lw=1.8)
    ax.fill_between(p, np.minimum(carry, leave), 0, color=C["neutral_t"], zorder=0)
    ax.axvline(0.2, color=C["ink"], lw=0.8)
    ax.text(0.21, 3.6, "carry it whenever the\nchance of rain is above 20%", fontsize=6.5, color=C["ink"])
    ax.text(0.55, 1.12, "carry: always costs 1 (annoying)", fontsize=6.5, color=C["data"], fontweight="semibold")
    ax.text(0.47, 2.35, "leave it: costs 4 (soaked)\n× the chance of rain", fontsize=6.5, color=C["fail"],
            fontweight="semibold", rotation=0)
    ax.set_xlabel("Chance of rain")
    ax.set_ylabel("Expected cost (units of annoyance)")
    ax.xaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 4.2)
    ax.set_xlim(0, 1)
    return f


@figure(CH, "threshold-ratio")
def threshold_ratio():
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    ratio = np.logspace(-1, 3.2, 200)       # cost of a miss / cost of a false alarm
    t = 1 / (1 + ratio)
    ax.plot(ratio, t, color=C["ink"], lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    pts = [(1, "equal costs:\nthe famous 0.5"), (4, "umbrella"), (20, "block a\nsuspicious login"),
           (633, "Kestrel: auto-close\nvs analyst review")]
    for r, lab in pts:
        ax.scatter([r], [1 / (1 + r)], color=C["jev"], s=24, zorder=4, edgecolor="white", lw=0.8)
        ax.text(r * 1.25, 1 / (1 + r) * 1.18, lab, fontsize=6.3, color=C["ink2"])
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
    ax.set_yticks([0.001, 0.01, 0.1, 0.5, 1])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "0.5", "1"])
    ax.set_xlabel("How much worse a miss is than a false alarm (log)")
    ax.set_ylabel("Act when P is above… (log)")
    return f


@figure(CH, "cost-curve")
def cost_curve():
    d, ts, c1, c2 = record()
    t = CFP / (CFP + CFN)
    f, ax = subplots(width="text", height=2.45)
    clean(ax, "y")
    ax.plot(ts, c1, color=C["jev"], lw=1.8, ls="-")              # calibrated: solid
    ax.plot(ts, c2, color=C["fail"], lw=1.6, ls=(0, (5, 2)))     # rebalanced: dashed
    ax.axvline(t, color=C["ink"], lw=0.8)
    ax.text(t + 0.01, 1.2e6, f"formula: {t:.3f}", fontsize=6.4, color=C["ink"], fontweight="bold")
    ax.scatter([t], [cost(d["p"], d["y"], t)], color=C["jev"], s=26, zorder=5, edgecolor="white", lw=0.8)
    ax.scatter([t], [cost(d["pb"], d["y"], t)], color=C["fail"], s=26, zorder=5, edgecolor="white", lw=0.8)
    ax.axvline(0.5, color=C["muted"], lw=0.6)
    ax.text(0.51, 1.2e6, "0.5", fontsize=6.4, color=C["ink2"])
    ax.text(0.62, c1[-60] + 60000, "calibrated model", fontsize=6.5, color=C["jev"], fontweight="semibold")
    ax.text(0.45, 0.2e6, "rebalanced (miscalibrated) model", fontsize=6.5, color=C["fail"], fontweight="semibold")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1.3e6)
    ax.yaxis.set_major_formatter(MONEY)
    ax.set_xlabel("Block the alert when P is at least …")
    ax.set_ylabel(f"Total cost, {len(d['y']):,} test alerts")
    synthetic_tag(f, "SYNTHETIC DATA · illustrative costs")
    return f


@figure(CH, "pr-threshold")
def pr_threshold():
    d = data()
    p, y = d["p"], d["y"]
    ts = np.linspace(0.005, 0.95, 200)
    prec = [((p >= t) & (y == 1)).sum() / max(1, (p >= t).sum()) for t in ts]
    rec = [((p >= t) & (y == 1)).sum() / y.sum() for t in ts]
    vol = [(p >= t).mean() for t in ts]
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "y")
    ax.plot(ts, rec, color=C["jev"], lw=1.8, ls="-", label="recall: share of real threats blocked (solid)")
    ax.plot(ts, prec, color=C["data"], lw=1.8, ls=(0, (5, 2)), label="precision: share of blocks that were real (dashed)")
    ax.plot(ts, vol, color=C["muted"], lw=1.2, ls=":", label="share of all alerts blocked (dotted)")
    ax.legend(loc="center right", fontsize=6.3)
    ax.axvline(CFP / (CFP + CFN), color=C["ink"], lw=0.8)
    ax.set_xlabel("Threshold")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.02)
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "coverage-risk")
def coverage_risk():
    d = data()
    p, y = d["p"], d["y"]
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    for pp, col, lab, ls in ((p, C["jev"], "logistic regression (solid)", "-"),
                             (np.clip(p + np.random.default_rng(0).normal(0, 0.12, len(p)), 0, 1), C["slate"],
                              "a noisier model (dashed)", (0, (5, 2)))):
        conf = np.maximum(pp, 1 - pp)
        correct = ((pp >= 0.5) == (y == 1)).astype(float)
        cov, risk, _ = pol.coverage_risk(conf, correct)
        ax.plot(cov[200:], risk[200:], color=col, lw=1.8, ls=ls, label=lab)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Coverage: share of alerts the machine decides alone (most confident first)")
    ax.set_ylabel("Error rate among them")
    ax.legend(loc="upper left", fontsize=6.4)
    ax.set_ylim(0, 0.12)
    ax.text(0.52, 0.09, "decide everything: take\nwhatever error rate you get", fontsize=6.2, color=C["ink2"])
    ax.annotate("", xy=(0.99, 0.065), xytext=(0.85, 0.085), arrowprops=dict(arrowstyle="-|>", color=C["ink2"], lw=0.6))
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "three-doors")
def three_doors():
    f, ax = draw.canvas("text", 1.55)
    draw.zone_bar(ax, 0.0, 0.62, 4.7, 0.5, 0.2, 0.65, sublabels=("machine handles it", "a person checks",
                                                                  "a person, right now"), ticks=False, size=7.2)
    draw.text(ax, 0.0, 1.38, "P(something needs a human) →", size=6.8, color=C["ink2"])
    for x, lab in ((0.47, "auto-close\nauto-reply\nauto-approve"), (2.0, "analyst queue\nsupport agent\nmoderator"),
                   (3.9, "page on-call\nsupervisor\nlegal")):
        draw.text(ax, x, 0.38, lab, size=6.0, color=C["ink2"], va="top", ha="center")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))

    def mini_formula(ax, x, y, w, h):
        draw.text(ax, x, y + h * 0.6, "act when  P > C_fa / (C_fa + C_miss)", size=7.2, family="JetBrains Mono",
                  weight="bold")
        draw.text(ax, x, y + h * 0.1, "umbrella 0.2  ·  blocking 0.048  ·  auto-close 0.0016", size=6.2,
                  color=C["ink2"])

    def mini_curve(ax, x, y, w, h):
        tt = np.linspace(0, 1, 50)
        ax.plot(x + tt * w * 0.8, y + (0.2 + (tt - 0.08) ** 2 * 1.4) * h * 0.6, color=C["jev"], lw=1.4)
        draw.dot(ax, x + 0.08 * w * 0.8, y + 0.2 * h * 0.6, r=0.035, color=C["ink"])

    def mini_zone(ax, x, y, w, h):
        draw.zone_bar(ax, x, y + 0.05, w, h * 0.6, 0.25, 0.7, ticks=False, size=6.2)

    panels = [
        dict(num=1, title="Two costs make one line", kind="neutral", h=1.5, draw=mini_formula, draw_h=0.45,
             body="If a false alarm costs C_fa and a miss costs C_miss, act whenever P is above their ratio."),
        dict(num=2, title="0.5 is a special case", kind="fail", h=1.5,
             body=(f"It only makes sense when both mistakes cost the same. Here 0.5 cost {rr['half_penalty']:.1f}× more "
                   "than the formula’s line.")),
        dict(num=3, title="The formula needs honest P", kind="jev", h=1.5, draw=mini_curve, draw_h=0.45,
             body=(f"On the calibrated model, theory and practice agree. On the rebalanced one, the same line cost "
                   f"{rr['bal_penalty']:.0%} more.")),
        dict(num=4, title="Precision and recall trade", kind="data", h=1.5,
             body="Lower the line: catch more, bother more. The costs tell you where to stop."),
        dict(num=5, title="“I don’t know” is an action", kind="review", h=1.5,
             body=(f"Decide only the confident cases and the error rate drops: {rr['risk_100']:.1%} deciding all, "
                   f"{rr['risk_80']:.1%} deciding the surest 80%.")),
        dict(num=6, title="Three doors", kind="review", h=1.5, draw=mini_zone, draw_h=0.4,
             body="Act, review, escalate: machines at the easy ends, people in the middle. Chapter 14 runs it for real."),
    ]
    return summary_page(CH, "From probabilities to actions", panels,
                        footer="Next, Part II: how do models learn from raw pixels and words?")
