"""Figures for Chapter 2: Probability is the language of decisions."""

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from jevkit import soc
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch02"


def clues():
    df = soc.load()
    y = df.malicious.values
    items = [("threat-intel score ≥ 0.7", df.ioc_score.values >= 0.7),
             ("outside business hours", df.after_hours.values),
             ("sign-in from a new country", df.new_geo.values)]
    out = []
    for name, m in items:
        out.append((name, float(m[y == 1].mean() / m[y == 0].mean()), m))
    return df, y, out


def record():
    df, y, cl = clues()
    base = y.mean()
    odds = base / (1 - base)
    o1 = odds * cl[0][1]
    o2 = o1 * cl[1][1]
    o3 = o2 * cl[2][1]
    m12 = cl[0][2] & cl[1][2]
    m123 = m12 & cl[2][2]
    t = df.ioc_score.values >= 0.5
    tp = int((t & (y == 1)).sum()); fp = int((t & (y == 0)).sum())
    fn = int((~t & (y == 1)).sum()); tn = int((~t & (y == 0)).sum())
    results(CH, base=float(base), base_odds=float(odds), lr_ioc=cl[0][1], lr_after=cl[1][1], lr_geo=cl[2][1],
            p1=o1 / (1 + o1), p2=o2 / (1 + o2), p3=o3 / (1 + o3), real12=float(y[m12].mean()), n12=int(m12.sum()),
            real123=float(y[m123].mean()), n123=int(m123.sum()),
            tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp), recall=tp / (tp + fn), accuracy=(tp + tn) / len(y),
            always_benign_acc=float(1 - base), ioc_attack_share=float(cl[0][2][y == 1].mean()),
            ioc_benign_share=float(cl[0][2][y == 0].mean()))
    return df, y, cl


@figure(CH, "forecast")
def forecast():
    f, ax = draw.canvas("text", 1.75)
    rng = np.random.default_rng(5)
    rained = np.zeros(100, bool)
    rained[rng.choice(100, 70, replace=False)] = True
    s = 0.125
    for i in range(100):
        x = (i % 20) * (s + 0.03)
        yv = 1.35 - (i // 20) * (s + 0.05)
        col = C["data"] if rained[i] else "white"
        ax.add_patch(Rectangle((x, yv), s, s, fc=col, ec=C["data"], lw=0.7))
    draw.text(ax, 3.2, 1.36, "100 days when the\nforecast said 70%", size=7.2, weight="bold", va="top")
    draw.text(ax, 3.2, 0.95, "■ it rained (70 days)", size=6.8, color=C["data"])
    draw.text(ax, 3.2, 0.75, "□ it stayed dry (30 days)", size=6.8, color=C["ink2"])
    draw.text(ax, 3.2, 0.42, "A good forecaster’s 70% comes\ntrue about 70 times in 100.", size=6.6, color=C["ink2"],
              style="italic")
    return f


@figure(CH, "base-rate-tree")
def base_rate_tree():
    f, ax = draw.canvas("text", 2.6)
    draw.box(ax, 1.75, 2.1, 1.2, 0.42, "100,000 events", kind="neutral", size=7.2, weight="bold")
    draw.box(ax, 0.35, 1.22, 1.3, 0.46, "100 attacks", kind="fail", size=7, weight="bold", sub="1 in 1,000", subsize=5.8)
    draw.box(ax, 3.05, 1.22, 1.3, 0.46, "99,900 normal", kind="neutral", size=7, weight="bold", sub="everything else",
             subsize=5.8)
    draw.arrow(ax, (2.1, 2.1), (1.1, 1.68))
    draw.arrow(ax, (2.6, 2.1), (3.6, 1.68))
    draw.box(ax, 0.0, 0.3, 0.95, 0.46, "99 alert", kind="fail", size=6.8, weight="bold", sub="caught (99%)", subsize=5.6)
    draw.box(ax, 1.05, 0.3, 0.8, 0.46, "1 silent", kind="plain", size=6.8, sub="missed", subsize=5.6)
    draw.box(ax, 2.65, 0.3, 1.05, 0.46, "4,995 alert", kind="data", size=6.8, weight="bold", sub="false alarms (5%)",
             subsize=5.6)
    draw.box(ax, 3.8, 0.3, 0.9, 0.46, "94,905 silent", kind="plain", size=6.4, sub="correctly quiet", subsize=5.6)
    for a, b in (((0.8, 1.22), (0.48, 0.76)), ((1.2, 1.22), (1.45, 0.76)), ((3.5, 1.22), (3.18, 0.76)),
                 ((3.9, 1.22), (4.25, 0.76))):
        draw.arrow(ax, a, b)
    draw.bracket(ax, 0.0, 3.7, 0.2, "", up=False)
    draw.text(ax, 1.85, 0.03, "5,094 alerts in total, and only 99 are real: about 1 in 51", size=6.8, ha="center",
              weight="semibold", va="center")
    return f


@figure(CH, "icon-array")
def icon_array():
    f, ax = draw.canvas("text", 1.9)
    n_cols, n_rows = 51, 10
    s = 0.075
    k = 0
    real = set([7, 58, 121, 190, 233, 301, 344, 412, 455, 499])
    for r in range(n_rows):
        for c in range(n_cols):
            if k >= 510:
                break
            x = c * (s + 0.017)
            yv = 1.6 - r * (s + 0.03)
            col = C["fail"] if k in real else C["data_t"]
            ec = C["fail"] if k in real else C["data"]
            ax.add_patch(Rectangle((x, yv), s, s, fc=col, ec=ec, lw=0.35))
            k += 1
    draw.text(ax, 0, 0.42, "Every square is 10 alerts from that detector. Red squares are the real attacks.", size=6.8)
    draw.text(ax, 0, 0.2, "If your alarm just went off, you are almost certainly looking at a light blue square.",
              size=6.8, weight="semibold")
    return f


@figure(CH, "odds-update")
def odds_update():
    df, y, cl = record()
    base = y.mean()
    odds = [base / (1 - base)]
    for _, lr, _ in cl:
        odds.append(odds[-1] * lr)
    probs = [o / (1 + o) for o in odds]
    labels = ["Any alert", "…with a bad\nthreat-intel score", "…and outside\nbusiness hours",
              "…and from a\nnew country"]
    mults = [None] + [f"×{lr:.1f}" for _, lr, _ in cl]
    f, ax = subplots(width="text", height=2.35)
    clean(ax, "y")
    xs = np.arange(4)
    ax.bar(xs, probs, width=0.5, color=[C["data"], C["jev"], C["jev"], C["jev"]], edgecolor="white")
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.0%}", ha="center", fontsize=7, fontweight="bold", color=C["ink"])
        if mults[i]:
            ax.annotate("", xy=(i - 0.3, probs[i] * 0.55 + 0.05), xytext=(i - 0.7, probs[i - 1] * 0.55 + 0.05),
                        arrowprops=dict(arrowstyle="-|>", color=C["ink2"], lw=0.8))
            ax.text(i - 0.5, max(probs[i - 1], probs[i]) * 0.55 + 0.1, f"odds {mults[i]}", ha="center", fontsize=6.2,
                    color=C["ink2"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Chance it’s a real attack")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "confusion")
def confusion():
    df, y, cl = record()
    t = df.ioc_score.values >= 0.5
    tp = int((t & (y == 1)).sum()); fp = int((t & (y == 0)).sum())
    fn = int((~t & (y == 1)).sum()); tn = int((~t & (y == 0)).sum())
    f, ax = draw.canvas("text", 2.45)
    x0, y0, w, h = 1.25, 0.25, 1.45, 0.78
    draw.text(ax, x0 + w, 2.35, "What really happened", size=7, weight="bold", ha="center")
    draw.text(ax, x0 + w / 2, 2.12, "real attack", size=6.6, ha="center", color=C["fail"])
    draw.text(ax, x0 + w * 1.5, 2.12, "harmless", size=6.6, ha="center", color=C["ink2"])
    draw.text(ax, 0.05, y0 + h * 1.5, "Rule\nsays\nalert", size=6.6, color=C["ink2"])
    draw.text(ax, 0.05, y0 + h * 0.5, "Rule\nsays\nquiet", size=6.6, color=C["ink2"])
    cells = [(0, 1, tp, "caught", "jev"), (1, 1, fp, "false alarm", "data"), (0, 0, fn, "missed", "fail"),
             (1, 0, tn, "correctly quiet", "neutral")]
    for cx, cy, n, lab, kind in cells:
        draw.box(ax, x0 + cx * (w + 0.05), y0 + cy * (h + 0.05), w, h, f"{n:,}", kind=kind, size=12, weight="bold",
                 sub=lab, subsize=6.4)
        # (sub text drawn by box)
    xr = x0 + 2 * w + 0.25
    draw.text(ax, xr, 1.75, "Precision", size=7, weight="bold")
    draw.text(ax, xr, 1.62, f"of alerts, how many real?\n{tp:,} / {tp + fp:,} = {tp / (tp + fp):.0%}", size=6.4, va="top",
              color=C["ink2"])
    draw.text(ax, xr, 1.05, "Recall", size=7, weight="bold")
    draw.text(ax, xr, 0.92, f"of attacks, how many caught?\n{tp:,} / {tp + fn:,} = {tp / (tp + fn):.0%}", size=6.4,
              va="top", color=C["ink2"])
    return f


@figure(CH, "sample-size")
def sample_size():
    rng = np.random.default_rng(1)
    true = 0.076
    ns = np.unique(np.round(np.logspace(1, 4, 40)).astype(int))
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "y")
    lo, hi = [], []
    for n in ns:
        z = 1.96
        ph = true
        den = 1 + z * z / n
        centre = (ph + z * z / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        lo.append(centre - half)
        hi.append(centre + half)
    ax.fill_between(ns, lo, hi, color=C["data"], alpha=0.12, lw=0)
    ax.plot(ns, lo, color=C["data"], lw=1)
    ax.plot(ns, hi, color=C["data"], lw=1)
    ax.axhline(true, color=C["ink"], lw=1.2)
    sims = [(n, rng.binomial(n, true) / n) for n in (20, 50, 100, 300, 1000, 3000)]
    ax.scatter([s[0] for s in sims], [s[1] for s in sims], color=C["ink"], s=18, zorder=4, edgecolor="white", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Alerts you checked (log scale)")
    ax.set_ylabel("Share that were attacks")
    ax.text(300, 0.25, "95% of honest samples land\ninside the shaded band", fontsize=6.4, color=C["data"])
    ax.text(4000, true + 0.012, "true rate 7.6%", fontsize=6.4, color=C["ink"], ha="center")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))

    def mini_forecast(ax, x, y, w, h):
        for i in range(10):
            ax.add_patch(Rectangle((x + i * 0.2, y + h * 0.3), 0.16, 0.16, fc=C["data"] if i < 7 else "white",
                                   ec=C["data"], lw=0.7))
        draw.text(ax, x, y + 0.05, "“70%” → 7 days in 10", size=6.2, color=C["ink2"])

    def mini_tree(ax, x, y, w, h):
        draw.pill(ax, x + 0.5, y + h * 0.7, "99 real", kind="fail", size=6.2)
        draw.pill(ax, x + 1.4, y + h * 0.7, "4,995 false", kind="data", size=6.2)
        draw.text(ax, x, y + 0.05, "P(attack | alert) ≈ 2%", size=6.6, weight="bold", family="JetBrains Mono")

    def mini_odds(ax, x, y, w, h):
        vals = [rr["base"], rr["p1"], rr["p2"], rr["p3"]]
        for i, v in enumerate(vals):
            ax.add_patch(Rectangle((x + i * 0.45, y), 0.3, v * h * 1.2, fc=C["data"] if i == 0 else C["jev"], ec="none"))
            draw.text(ax, x + i * 0.45 + 0.15, y + v * h * 1.2 + 0.07, f"{v:.0%}", size=5.8, ha="center")

    def mini_conf(ax, x, y, w, h):
        for (cx, cy, kind, lab) in ((0, 1, "jev", "caught"), (1, 1, "data", "false alarm"), (0, 0, "fail", "missed"),
                                    (1, 0, "neutral", "quiet")):
            draw.box(ax, x + cx * 0.85, y + cy * 0.3, 0.8, 0.26, lab, kind=kind, size=5.8)

    panels = [
        dict(num=1, title="A probability is a track record", kind="data", h=1.55, draw=mini_forecast, draw_h=0.4,
             body="“70% chance of rain” means: of days like this, about 7 in 10 get rain. You can check it."),
        dict(num=2, title="Always ask: among what?", kind="data", h=1.55,
             body="P(attack) and P(attack | this score) are different numbers. The bar means “given”: among cases like this one."),
        dict(num=3, title="Base rates rule", kind="fail", h=1.55, draw=mini_tree, draw_h=0.45,
             body="A 99%-accurate detector for a 1-in-1,000 event still raises mostly false alarms."),
        dict(num=4, title="Evidence multiplies the odds", kind="jev", h=1.55, draw=mini_odds, draw_h=0.4,
             body="Each clue multiplies the odds by its likelihood ratio. That’s Bayes’ rule, in one line."),
        dict(num=5, title="Precision and recall", kind="neutral", h=1.55, draw=mini_conf, draw_h=0.6,
             body=f"Accuracy hides the rare class. “Never alert” scores {rr['always_benign_acc']:.0%} here."),
        dict(num=6, title="Small samples wobble", kind="neutral", h=1.55,
             body="3 attacks in 20 alerts could be a 4% rate or a 35% one. Report the range, not just the number."),
    ]
    return summary_page(CH, "Probability is the language of decisions", panels,
                        footer="Next: how does a machine actually learn a probability from examples?")
