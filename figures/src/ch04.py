"""Figures for Chapter 3: Calibration: when 0.8 really means 80%."""

from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression

from jevkit import soc, calibration as cal
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch04"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


@lru_cache(None)
def data():
    df = soc.load()
    tr, ca, te = soc.split(df)
    F = lambda d: soc.feature_matrix(d).values
    m = LogisticRegression(C=1e4, max_iter=5000).fit(F(tr), tr.malicious)
    p = m.predict_proba(F(te))[:, 1]
    sq = cal.sigmoid(cal.logit(p) * 0.45 + 1.0)
    pos = tr[tr.malicious == 1]
    neg = tr[tr.malicious == 0].sample(len(pos), random_state=0)
    bal = pd.concat([pos, neg])
    mb = LogisticRegression(C=1e4, max_iter=5000).fit(F(bal), bal.malicious)
    pb, pbc = mb.predict_proba(F(te))[:, 1], mb.predict_proba(F(ca))[:, 1]
    fixes = {n: f.fit(pbc, ca.malicious.values) for n, f in
             (("Platt", cal.Platt()), ("Temperature", cal.Temperature()), ("Isotonic", cal.Isotonic()))}
    cols = [c for c in soc.feature_matrix(tr).columns if not c.startswith("rule_")]
    mg = LogisticRegression(C=1e4, max_iter=5000).fit(soc.feature_matrix(tr)[cols], tr.malicious)
    pg = mg.predict_proba(soc.feature_matrix(te)[cols])[:, 1]
    return dict(tr=tr, ca=ca, te=te, y=te.malicious.values, p=p, sq=sq, pb=pb, pbc=pbc, fixes=fixes, pg=pg)


def record():
    d = data()
    y = d["y"]
    s = {k: cal.summary(d[k], y) for k in ("p", "sq", "pb", "pg")}
    fx = {n: cal.summary(f(d["pb"]), y) for n, f in d["fixes"].items()}
    dec = cal.brier_decomposition(d["p"], y)
    src = {}
    for sname in sorted(d["te"].source.unique()):
        m = (d["te"].source == sname).values
        src[sname] = dict(n=int(m.sum()), pred=float(d["pg"][m].mean()), actual=float(y[m].mean()))
    b = cal.reliability(d["p"], y)
    hi = b.count[-3:].sum()
    results(CH, lr=s["p"], squashed=s["sq"], balanced=s["pb"], generic=s["pg"], fixes=fx, decomposition=dec,
            by_source=src, bal_mean_pred=float(d["pb"].mean()), base=float(y.mean()), n_test=len(y),
            n_high=int(hi), temp_T=float(d["fixes"]["Temperature"].T))
    return d


def reliability_ax(ax, p, y, color, label=None, bins=10, strategy="quantile", ms=4, ls="-", marker="o"):
    b = cal.reliability(p, y, n_bins=bins, strategy=strategy)
    m = b.count > 0
    ax.plot(b.mean_pred[m], b.frac_pos[m], color=color, lw=1.5, ls=ls, marker=marker, ms=ms, mec="white", mew=0.7,
            label=label)
    return b.mean_pred[m], b.frac_pos[m]


def diag(ax, lim=1):
    ax.plot([0, lim], [0, lim], color=C["muted"], lw=0.8, ls=(0, (3, 2)), zorder=1)


@figure(CH, "forecasters")
def forecasters():
    f, ax = subplots(width="text", height=2.4)
    clean(ax, "both")
    x = np.linspace(0.05, 0.95, 10)
    diag(ax)
    ax.plot(x, x + np.array([0.01, -0.02, 0.02, 0.0, -0.01, 0.02, -0.01, 0.01, -0.02, 0.0]), color=C["data"], lw=1.6,
            marker="o", ms=4, mec="white", mew=0.7)
    ax.plot(x, 0.5 + (x - 0.5) * 0.45, color=C["fail"], lw=1.6, marker="o", ms=4, mec="white", mew=0.7)
    ax.text(0.05, 0.84, "a careful forecaster:\non the diagonal", fontsize=6.6, color=C["data"], fontweight="semibold")
    ax.text(0.58, 0.12, "an overconfident one:\n“90%” comes true about\n70% of the time", fontsize=6.6,
            color=C["fail"], fontweight="semibold")
    ax.set_xlabel("What they said")
    ax.set_ylabel("How often it happened")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.99, 0.02, "illustrative", fontsize=5.6, color=C["muted"], ha="right")
    return f


@figure(CH, "same-auc")
def same_auc():
    d = record()
    y = d["y"]
    f, axes = subplots(1, 2, width="text", height=2.3, sharey=True, gridspec_kw=dict(wspace=0.18))
    for ax, key, title, col in ((axes[0], "p", "Model A", C["jev"]), (axes[1], "sq", "Model B", C["fail"])):
        clean(ax, "both")
        diag(ax)
        reliability_ax(ax, d[key], y, col)
        s = cal.summary(d[key], y)
        ax.set_title(title, fontsize=7.6)
        ax.text(0.04, 0.9, f"AUC {s['auc']:.3f}\nECE {s['ece']:.3f}", fontsize=6.6, color=C["ink"], va="top",
                family="JetBrains Mono")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Predicted P(attack)")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])      # the panels meet at 100% | 0%: label only one of them
    axes[0].set_ylabel("Share that were attacks")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "anatomy")
def anatomy():
    d = data()
    y, p = d["y"], d["p"]
    f, (a1, a2) = subplots(2, 1, width="text", height=3.1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    clean(a1, "both")
    diag(a1)
    b = cal.reliability(p, y, n_bins=10, strategy="uniform")
    m = b.count > 0
    a1.plot(b.mean_pred[m], b.frac_pos[m], color=C["jev"], lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8)
    a1.annotate("each dot: all alerts the model\nput in one 10%-wide bin", (b.mean_pred[m][4], b.frac_pos[m][4]),
                xytext=(0.52, 0.18), fontsize=6.3, color=C["ink2"], arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    a1.text(0.06, 0.88, "above the line: the model\nwas too cautious", fontsize=6.3, color=C["ink2"])
    a1.text(0.74, 0.46, "below the line: the\nmodel was too sure", fontsize=6.3, color=C["ink2"])
    a1.set_ylim(0, 1)
    a1.set_ylabel("Share that were attacks")
    a1.yaxis.set_major_formatter(PCT)
    a2.grid(False)
    a2.hist(p, bins=np.linspace(0, 1, 41), color=C["data"], edgecolor="white", lw=0.5)
    a2.set_yscale("log")
    a2.set_yticks([10, 100, 1000])
    a2.set_yticklabels(["10", "100", "1,000"], fontsize=6)
    a2.set_ylabel("alerts", fontsize=6.6)
    a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(PCT)
    a2.set_xlabel("Predicted P(attack), logistic regression from Chapter 2, test alerts")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "rebalanced")
def rebalanced():
    d = data()
    y = d["y"]
    f, ax = subplots(width="text", height=2.45)
    clean(ax, "both")
    diag(ax)
    # solid line with squares, dashed line with circles, and a label on each line (no colour legend)
    x1, y1 = reliability_ax(ax, d["pb"], y, C["fail"], ls="-", marker="s")
    x2, y2 = reliability_ax(ax, d["fixes"]["Platt"](d["pb"]), y, C["jev"], ls=(0, (5, 2)), marker="o")
    k2 = int(np.argmin(abs(x2 - 0.3)))
    ax.annotate("trained on 50/50 rebalanced data\n(solid line, squares)", (x1[-1], y1[-1]), xytext=(0.6, 0.48),
                fontsize=6.4, color=C["ink"], va="bottom", arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.annotate("same model, after Platt scaling\n(dashed line, circles)", (x2[k2], y2[k2]), xytext=(0.05, 0.62),
                fontsize=6.4, color=C["ink"], va="bottom", arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Predicted P(attack)")
    ax.set_ylabel("Share that were attacks")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "fixes")
def fixes():
    d = data()
    f, axes = subplots(1, 3, width="text", height=1.85, sharey=True, gridspec_kw=dict(wspace=0.1))
    x = np.linspace(0.001, 0.999, 300)
    cols = {"Platt": C["jev"], "Temperature": C["slate"], "Isotonic": C["data"]}
    notes = {"Platt": "an S-curve:\nshift + stretch", "Temperature": "one number:\nstretch only",
             "Isotonic": "a staircase:\nany rising shape"}
    for ax, (n, fx) in zip(axes, d["fixes"].items()):
        clean(ax, "both")
        diag(ax)
        ax.plot(x, fx(x), color=cols[n], lw=1.8)
        ax.set_title(n, fontsize=7.4)
        ax.text(0.04, 0.92, notes[n], fontsize=6.0, color=C["ink2"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("raw P", fontsize=6.6)
    axes[0].set_ylabel("corrected P")
    axes[0].set_yticks([0, 0.5, 1])
    return f


@figure(CH, "by-source")
def by_source():
    d = data()
    te, y, pg = d["te"], d["y"], d["pg"]
    srcs = ["Email", "Identity", "Cloud", "DLP", "Network", "EDR"]
    f, axes = subplots(1, 6, width="wide", height=1.35, sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, s in zip(axes, srcs):
        clean(ax, "both")
        m = (te.source == s).values
        diag(ax, 0.3)
        reliability_ax(ax, pg[m], y[m], C["jev"], bins=5, ms=3)
        ax.set_title(s, fontsize=7)
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.3)
        ax.set_xticks([0, 0.3])
        ax.set_xticklabels(["0", "30%"], fontsize=5.6)
        ax.get_xticklabels()[0].set_ha("left")          # keep neighbouring panels' labels apart
        ax.get_xticklabels()[-1].set_ha("right")
        mp, ma = pg[m].mean(), y[m].mean()
        ax.text(0.015, 0.285, f"says {mp:.1%}\nreally {ma:.1%}", fontsize=5.6, va="top",
                color=C["fail"] if abs(mp - ma) > 0.02 else C["ink2"])
    axes[0].set_yticks([0, 0.15, 0.3])
    axes[0].set_yticklabels(["0", "15%", "30%"], fontsize=5.6)
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))

    def mini_rel(ax, x, y, w, h, bent=0.0, col=C["jev"]):
        t = np.linspace(0, 1, 20)
        ax.plot([x, x + h], [y, y + h], color=C["muted"], lw=0.7, ls=(0, (2, 2)))
        ax.plot(x + t * h, y + (t + bent * np.sin(np.pi * t)) * h, color=col, lw=1.4)

    def two(ax, x, y, w, h):
        mini_rel(ax, x, y, w, h)
        mini_rel(ax, x + h + 0.3, y, w, h, bent=-0.25, col=C["fail"])
        draw.text(ax, x + 2 * h + 0.45, y + h * 0.5, "same ranking,\ndifferent honesty", size=5.8, color=C["ink2"])

    def fixes_mini(ax, x, y, w, h):
        for i, (lab, kind) in enumerate((("Platt", "jev"), ("Temperature", "neutral"), ("Isotonic", "data"))):
            draw.pill(ax, x + 0.4 + i * 0.85, y + h * 0.5, lab, kind=kind, size=6.0)

    panels = [
        dict(num=1, title="Calibration = honesty", kind="jev", h=1.5,
             body="A calibrated model’s 80% happens about 80% of the time. You check it by grouping and counting."),
        dict(num=2, title="Ranking ≠ honesty", kind="fail", h=1.5, draw=two, draw_h=0.5,
             body=(f"Two models with the same AUC ({rr['lr']['auc']:.2f}) can have ECE {rr['lr']['ece']:.3f} "
                   f"and {rr['squashed']['ece']:.2f}.")),
        dict(num=3, title="Read the reliability diagram", kind="neutral", h=1.5, draw=lambda ax, x, y, w, h: mini_rel(ax, x, y, w, h, bent=0.08), draw_h=0.5,
             body="Said vs happened, bin by bin. On the diagonal is honest; below it is overconfident. Check the counts."),
        dict(num=4, title="Rebalancing breaks it", kind="fail", h=1.5,
             body=(f"Train on 50/50 data and the model thinks attacks are common: its average P is "
                   f"{rr['bal_mean_pred']:.0%} when the truth is {rr['base']:.0%}.")),
        dict(num=5, title="Fix it on held-out data", kind="data", h=1.5, draw=fixes_mini, draw_h=0.35,
             body="Platt shifts and stretches; temperature only stretches; isotonic bends. Fit on data the model never trained on."),
        dict(num=6, title="Check every slice", kind="neutral", h=1.5,
             body="Honest on average can still be dishonest for Email, or for EDR. Check the groups you’ll act on."),
    ]
    return summary_page(CH, "Calibration: when 0.8 really means 80%", panels,
                        footer="Next: honest probabilities in hand, how do we turn them into actions?")
