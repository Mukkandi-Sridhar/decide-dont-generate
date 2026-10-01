"""Figures for Chapter 1: What "learning" means."""

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, engine
from jevkit.figs import figure, draw, C, ZONE, subplots, clean, results, summary_page, synthetic_tag
from jevkit.figs.style import KIND

CH = "ch01"


def threshold_search():
    df = soc.load()
    y, x = df.malicious.values, df.ioc_score.values
    ts = np.round(np.arange(0, 1.0001, 0.01), 2)
    fn = np.array([((x < t) & (y == 1)).sum() for t in ts])
    fp = np.array([((x >= t) & (y == 0)).sum() for t in ts])
    return df, ts, fn, fp


def record_results():
    df, ts, fn, fp = threshold_search()
    tot = fn + fp
    i = int(tot.argmin())
    y, x = df.malicious.values, df.ioc_score.values
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(x, edges) - 1, 0, 9)
    frac = [float(y[idx == b].mean()) for b in range(10)]
    results(CH, n_alerts=len(df), per_day=len(df) / 28, attacks=int(y.sum()), base_rate=float(y.mean()),
            benign=int(len(df) - y.sum()), best_t=float(ts[i]), best_mistakes=int(tot[i]), best_missed=int(fn[i]),
            best_false=int(fp[i]), never_flag_mistakes=int(y.sum()), never_flag_accuracy=float(1 - y.mean()),
            t05_missed=int(fn[50]), t05_false=int(fp[50]), t05_mistakes=int(tot[50]),
            top_bin_rate=frac[9], bottom_bin_rate=frac[0], bottom_bin_n=int((idx == 0).sum()),
            best_caught=int(y.sum() - fn[i]), mid_bin_rate=frac[5], mock_p_99=round(engine.soc_probability(df.description[99]), 2),
            alert99_malicious=int(df.malicious[99]))
    return frac


@figure(CH, "rules-vs-examples")
def rules_vs_examples():
    f, ax = draw.canvas("text", 2.75)
    W = 4.7
    # ---- left: rules
    draw.text(ax, 0.0, 2.62, "Writing rules", size=8.4, weight="bold")
    draw.text(ax, 0.0, 2.44, "Every new trick needs a new exception", size=6.6, color=C["ink2"])
    rules = [
        ("IF the email says \u201cpassword\u201d", "flag it"),
        ("  \u2026unless it\u2019s IT\u2019s reset tool", "don\u2019t"),
        ("  \u2026unless it says \u201cpassw0rd\u201d", "flag it"),
        ("  \u2026unless it\u2019s in an image", "?"),
        ("  \u2026unless it\u2019s only a link", "?"),
    ]
    y = 2.12
    for i, (r, a) in enumerate(rules):
        fade = 1 - i * 0.13
        kind = "neutral"
        draw.box(ax, 0.0, y - 0.13, 2.05, 0.26, "", kind=kind, radius=0.04, lw=0.6,
                 fill="#FFFFFF" if i else C["neutral_t"])
        draw.text(ax, 0.08, y, r, size=6.5, family="JetBrains Mono", color=C["ink"] if i < 3 else C["ink2"])
        col = C["fail"] if a == "?" else C["ink2"]
        draw.text(ax, 1.97, y, a, size=6.3, ha="right", color=col, weight="bold" if a == "?" else "normal")
        y -= 0.33
    draw.text(ax, 0.02, y + 0.02, "+ hundreds more, and still behind", size=6.6, color=C["fail"], weight="semibold")
    # ---- divider
    ax.plot([2.3, 2.3], [0.1, 2.62], color=C["grid"], lw=0.8)
    # ---- right: examples
    x0 = 2.5
    draw.text(ax, x0, 2.62, "Learning from examples", size=8.4, weight="bold")
    draw.text(ax, x0, 2.44, "Show it cases; let it find the pattern", size=6.6, color=C["ink2"])
    rng = np.random.default_rng(3)
    labels = rng.random(12) < 0.33
    for k in range(12):
        cx = x0 + (k % 4) * 0.34
        cy = 2.0 - (k // 4) * 0.42
        draw.doc_icon(ax, cx, cy - 0.13, w=0.24, h=0.3, kind="data", lines=3)
        draw.dot(ax, cx + 0.2, cy - 0.12, r=0.045, color=C["fail"] if labels[k] else C["muted"], ec="white", lw=0.8)
    draw.text(ax, x0, 0.92, "\u25cf attack", size=6.2, color=C["fail"])
    draw.text(ax, x0 + 0.55, 0.92, "\u25cf fine", size=6.2, color=C["muted"])
    draw.arrow(ax, (x0 + 1.4, 1.62), (x0 + 1.72, 1.62), color=C["ink2"])
    draw.box(ax, x0 + 1.72, 1.3, 0.48, 0.64, "model", kind="neutral", size=7.2, weight="bold")
    draw.arrow(ax, (x0 + 1.96, 1.3), (x0 + 1.96, 0.92), color=C["ink2"])
    draw.box(ax, x0 + 1.38, 0.38, 0.82, 0.52, "new email", kind="data", size=6.6, sub="0.83 chance\nit\u2019s phishing", subsize=6.0)
    return f


@figure(CH, "train-then-use")
def train_then_use():
    f, ax = draw.canvas("text", 2.35)
    # lanes
    for yy, title, sub in ((2.2, "TRAINING", "once, slowly, with answers"), (1.0, "USING", "every alert, fast, no answers")):
        draw.text(ax, 0.0, yy, title, size=6.6, weight="bold", color=C["ink2"])
        draw.text(ax, 0.62 if title == "USING" else 0.86, yy, sub, size=6.4, color=C["muted"])
    # training lane
    for k in range(4):
        draw.box(ax, 0.02 + k * 0.05, 1.3 + k * 0.05, 1.0, 0.56, "", kind="data", radius=0.04, lw=0.7)
    draw.box(ax, 0.17, 1.45, 1.0, 0.56, "alert + answer", kind="data", size=6.8, sub="20,000 of them", subsize=6.0)
    draw.arrow(ax, (1.27, 1.73), (1.78, 1.73))
    draw.box(ax, 1.8, 1.43, 1.05, 0.6, "learning", kind="neutral", size=7.2, weight="bold", sub="adjust the knobs", subsize=6.0)
    draw.arrow(ax, (2.85, 1.73), (3.45, 1.73))
    draw.box(ax, 3.47, 1.43, 1.2, 0.6, "a model", kind="plain", size=7.4, weight="bold", sub="knobs, now set", subsize=6.0, lw=1.2, color=C["ink"])
    # link down
    draw.arrow(ax, (4.07, 1.43), (4.07, 0.95), dashed=True, color=C["muted"], label="same model", labelpos=0.45,
               labeloffset=(0.33, -0.03))
    # using lane
    draw.box(ax, 0.17, 0.08, 1.0, 0.56, "new alert", kind="data", size=6.8, sub="answer unknown", subsize=6.0)
    draw.arrow(ax, (1.17, 0.36), (3.45, 0.36), label="goes in", labelsize=6.0)
    draw.box(ax, 3.47, 0.08, 1.2, 0.56, "", kind="plain", lw=1.2, color=C["ink"])
    draw.text(ax, 3.57, 0.46, "P(attack)", size=6.4, color=C["ink2"])
    draw.gauge(ax, 3.57, 0.2, 0.72, 0.12, kind="data", size=6.2)
    return f


@figure(CH, "threshold-search")
def threshold_search_fig():
    df, ts, fn, fp = threshold_search()
    tot = fn + fp
    i = int(tot.argmin())
    f, ax = subplots(width="text", height=2.55)
    clean(ax, "y")
    # line styles are fixed, not left to colour: false alarms dashed, missed attacks dash-dot, total solid
    ax.plot(ts, fp, color=C["data"], lw=1.5, ls=(0, (5, 2)))
    ax.plot(ts, fn, color=C["fail"], lw=1.5, ls=(0, (5, 1.5, 1, 1.5)))
    ax.plot(ts, tot, color=C["ink"], lw=2.0, ls="-")
    ax.axhline(fn[-1], color=C["muted"], lw=0.7, ls=(0, (1, 1.5)))
    ax.text(0.02, fn[-1] + 150, f"flag nothing at all: {fn[-1]:,} mistakes", fontsize=6.4, color=C["ink2"])
    ax.scatter([ts[i]], [tot[i]], s=28, color=C["ink"], zorder=5, edgecolor="white", linewidth=1.2)
    ax.annotate(f"fewest mistakes: {tot[i]:,}\nat a threshold of {ts[i]:.2f}", (ts[i], tot[i]), xytext=(0.55, 5200),
                fontsize=6.6, color=C["ink"], arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.text(0.2, fp[20] + 900, "false alarms", fontsize=6.6, color=C["data"], fontweight="semibold")
    ax.text(0.33, fn[33] - 380, "missed attacks", fontsize=6.6, color=C["fail"], fontweight="semibold")
    ax.text(0.27, tot[27] + 420, "total mistakes", fontsize=6.6, color=C["ink"], fontweight="semibold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9000)
    ax.set_xlabel("Flag the alert if its threat-intel score is at least \u2026")
    ax.set_ylabel("Mistakes, out of 20,000 alerts")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    synthetic_tag(f, "SYNTHETIC DATA \u00b7 Kestrel Logistics")
    return f


@figure(CH, "not-yes-no")
def not_yes_no():
    frac = record_results()
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "y")
    xs = np.arange(10)
    ax.bar(xs, frac, width=0.56, color=C["data"], edgecolor="white", linewidth=0)
    for k in (0, 5, 9):
        ax.text(k, frac[k] + 0.02, f"{frac[k]:.0%}", ha="center", fontsize=6.6, color=C["ink"], fontweight="semibold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a / 10:.1f}\u2013{(a + 1) / 10:.1f}" for a in range(10)], fontsize=6.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Threat-intel score of the alert")
    ax.set_ylabel("Share that were real attacks")
    synthetic_tag(f, "SYNTHETIC DATA \u00b7 Kestrel Logistics")
    return f


def ladder(ax, x0, y0, w, h, compact=False):
    steps = [
        ("Rules", "if temp > 100\u00b0F:\n    alert()", "data", "too many\nexceptions"),
        ("Machine\nlearning", "learn the pattern\nfrom examples", "data", "features too\nhard to write"),
        ("Deep\nlearning", "learn the features\ntoo", "data", "language is\nthe hard part"),
        ("LLMs", "read and write\nlanguage", "llm", "they don\u2019t know\nyour data"),
        ("RAG", "read your\ndocuments first", "llm", "reading isn\u2019t\ndoing"),
        ("Agents", "observe, decide,\nact", "neutral", None),
    ]
    n = len(steps)
    gap = 0.1
    bw = (w - gap * (n - 1)) / n
    for i, (t, sub, kind, why) in enumerate(steps):
        x = x0 + i * (bw + gap)
        yy = y0 + h * 0.36 + i * (h * 0.09 if not compact else h * 0.07)
        bh = h * 0.34
        draw.box(ax, x, yy, bw, bh, t, kind=kind, size=6.9 if not compact else 6.2, weight="bold")
        if not compact:
            draw.text(ax, x + bw / 2, yy - 0.12, sub, size=5.7, ha="center", va="top", color=C["ink2"],
                      family="JetBrains Mono" if i == 0 else None)
        if why and not compact:
            draw.arrow(ax, (x + bw, yy + bh / 2), (x + bw + gap, yy + bh / 2 + h * 0.09), color=C["muted"], lw=0.8, head=3)
    return bw, gap


STEPS = [
    ("Rules", "You write every rule by hand.", "data", "\u2026but there are too many exceptions"),
    ("Machine learning", "Learn the pattern from labelled examples.", "data", "\u2026but someone still has to pick the features"),
    ("Deep learning", "Learn the features too, in many layers.", "data", "\u2026and language turns out to be learnable at scale"),
    ("LLMs", "Read and write language.", "llm", "\u2026but they don\u2019t know your company\u2019s data"),
    ("RAG", "Look things up in your documents first.", "llm", "\u2026but reading isn\u2019t doing"),
    ("Agents", "Observe, decide, act, in a loop.", "neutral", None),
]


@figure(CH, "ai-story-map")
def ai_story_map():
    H = 3.25
    f, ax = draw.canvas("text", H)
    bw, bh, pitch = 1.18, 0.34, 0.545
    top = H - 0.08
    ys = []
    for i, (t, what, kind, why) in enumerate(STEPS):
        y = top - bh - i * pitch
        ys.append(y)
        draw.box(ax, 0.0, y, bw, bh, t, kind=kind, size=7.2, weight="bold")
        draw.text(ax, bw + 0.14, y + bh / 2, what, size=7.0, color=C["ink"])
        if why:
            draw.arrow(ax, (0.3, y), (0.3, y - (pitch - bh)), color=C["muted"], lw=0.8, head=3, shrink=0)
            draw.text(ax, 0.4, y - (pitch - bh) / 2, why, size=6.2, color=C["ink2"], style="italic")
    # Jev branch
    jy = (ys[2] + ys[5]) / 2 - 0.05
    draw.box(ax, 3.35, jy - 0.05, 1.33, 0.62, "System One\nmodels (Jev)", kind="jev", size=7.0, weight="bold")
    draw.text(ax, 4.015, jy - 0.16, "decide, don\u2019t generate", size=6.2, ha="center", color=C["jev"], weight="semibold")
    draw.arrow(ax, (3.2, ys[2] + bh / 2), (3.9, jy + 0.57), color=C["jev"], lw=1.0, rad=-0.25)
    draw.arrow(ax, (3.9, jy - 0.22), (3.2, ys[5] + bh / 2), color=C["jev"], lw=1.0, rad=-0.25)
    draw.text(ax, 4.1, ys[5] + bh / 2 - 0.02, "the \u201cdecide\u201d step", size=6.2, color=C["jev"])
    return f


@figure(CH, "generate-vs-decide")
def generate_vs_decide():
    f, ax = draw.canvas("text", 1.95)
    draw.text(ax, 0.0, 1.82, "An LLM writes its answer, one piece at a time", size=7.2, weight="bold")
    toks = ["The", "ticket", "is", "about", "billing", "."]
    x = 0.0
    for i, t in enumerate(toks):
        w = 0.16 + 0.075 * len(t)
        draw.box(ax, x, 1.3, w, 0.3, t, kind="llm", size=6.8, radius=0.04, family="JetBrains Mono")
        draw.text(ax, x + w / 2, 1.2, f"step {i + 1}", size=5.4, ha="center", color=C["muted"])
        x += w + 0.08
    draw.text(ax, x + 0.05, 1.45, "\u2026then your code has to\nread the sentence", size=6.2, color=C["ink2"])
    ax.plot([0, 4.7], [0.98, 0.98], color=C["grid"], lw=0.8)
    draw.text(ax, 0.0, 0.84, "Jev returns a typed answer with probabilities, in one pass", size=7.2, weight="bold")
    draw.box(ax, 0.0, 0.08, 1.35, 0.56, "ticket text +\nthe question", kind="data", size=6.4)
    draw.arrow(ax, (1.35, 0.36), (1.75, 0.36))
    draw.box(ax, 1.77, 0.08, 0.52, 0.56, "Jev", kind="jev", size=7.2, weight="bold")
    draw.arrow(ax, (2.29, 0.36), (2.62, 0.36))
    for i, (lab, p) in enumerate((("billing", 0.97), ("technical", 0.02), ("other", 0.01))):
        yy = 0.5 - i * 0.18
        draw.text(ax, 2.68, yy + 0.045, lab, size=6.2, family="JetBrains Mono")
        draw.gauge(ax, 3.48, yy, 0.8, p, kind="jev", h=0.085, size=6.0)
    draw.text(ax, 4.7, 0.0, "illustrative numbers", size=5.4, ha="right", va="bottom", color=C["muted"])
    return f


@figure(CH, "summary")
def summary():
    df, ts, fn, fp = threshold_search()
    tot = fn + fp
    i = int(tot.argmin())

    def mini_rules(ax, x, y, w, h):
        draw.box(ax, x, y + h * 0.3, w * 0.44, h * 0.5, "if \u2026 then \u2026", kind="neutral", size=6.3, family="JetBrains Mono")
        draw.text(ax, x + w * 0.22, y + h * 0.1, "you write the rules", size=5.8, ha="center", color=C["ink2"])
        draw.box(ax, x + w * 0.56, y + h * 0.3, w * 0.44, h * 0.5, "examples", kind="data", size=6.3)
        draw.text(ax, x + w * 0.78, y + h * 0.1, "it finds the rules", size=5.8, ha="center", color=C["ink2"])

    def mini_model(ax, x, y, w, h):
        draw.box(ax, x, y + h * 0.25, w * 0.28, h * 0.6, "input", kind="data", size=6.2)
        draw.arrow(ax, (x + w * 0.28, y + h * 0.55), (x + w * 0.36, y + h * 0.55))
        draw.box(ax, x + w * 0.36, y + h * 0.25, w * 0.28, h * 0.6, "model", kind="plain", size=6.2, weight="bold", color=C["ink"])
        draw.arrow(ax, (x + w * 0.64, y + h * 0.55), (x + w * 0.72, y + h * 0.55))
        draw.box(ax, x + w * 0.72, y + h * 0.25, w * 0.28, h * 0.6, "answer", kind="neutral", size=6.2)

    def mini_curve(ax, x, y, w, h):
        xx = x + ts * w
        yy = y + (tot / tot.max()) * h * 0.9
        ax.plot(xx, yy, color=C["ink"], lw=1.2)
        draw.dot(ax, x + ts[i] * w, y + tot[i] / tot.max() * h * 0.9, r=0.04, color=C["ink"])
        draw.text(ax, x + w * 0.5, y + h * 0.95, "mistakes vs threshold", size=5.6, ha="center", color=C["muted"])

    def mini_prob(ax, x, y, w, h):
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(df.ioc_score.values, edges) - 1, 0, 9)
        fr = [df.malicious.values[idx == b].mean() for b in range(10)]
        bw = w / 10
        for b in range(10):
            ax.add_patch(__import__("matplotlib").patches.Rectangle((x + b * bw + bw * 0.2, y), bw * 0.6, fr[b] * h * 1.4,
                                                                   fc=C["data"], ec="none"))
        draw.text(ax, x + w * 0.5, y - 0.08, "share of attacks rises smoothly", size=5.6, ha="center", color=C["muted"])

    def mini_stats(ax, x, y, w, h):
        for k, (val, lab, col) in enumerate(((tot[i], "mistakes: best rule", C["fail"]), (fn[-1], "mistakes: flag nothing", C["ink2"]))):
            xx = x + k * w / 2
            draw.text(ax, xx, y + h * 0.62, f"{val:,}", size=15, weight="bold", color=C["ink"])
            draw.text(ax, xx, y + h * 0.12, lab, size=6.0, color=col)

    def mini_gen(ax, x, y, w, h):
        toks = ["The", "ticket", "is", "about", "billing"]
        xx = x
        for t in toks:
            ww = 0.1 + 0.055 * len(t)
            draw.box(ax, xx, y + h * 0.55, ww, h * 0.36, t, kind="llm", size=5.6, radius=0.03)
            xx += ww + 0.04
        draw.box(ax, x, y, 0.55, h * 0.36, "Jev", kind="jev", size=6.0, weight="bold")
        draw.text(ax, x + 0.62, y + h * 0.18, "billing  0.97", size=6.2, family="JetBrains Mono", color=C["ink"])

    def mini_ladder(ax, x, y, w, h):
        names = [t for t, *_ in STEPS]
        kinds = [k for _, _, k, _ in STEPS]
        bw = (w - 0.16 * 5) / 6
        for i, (nm, k) in enumerate(zip(names, kinds)):
            xx = x + i * (bw + 0.16)
            draw.box(ax, xx, y, bw, h * 0.6, nm.replace("Machine learning", "ML").replace("Deep learning", "Deep L."),
                     kind=k, size=6.2, weight="bold")
            if i < 5:
                draw.arrow(ax, (xx + bw, y + h * 0.3), (xx + bw + 0.16, y + h * 0.3), head=3)

    panels = [
        dict(num=1, title="Rules vs learning", kind="data", h=1.62, draw=mini_rules, draw_h=0.62,
             body="A rule is you telling the computer exactly what to do. Rules break when the world has too many exceptions."),
        dict(num=2, title="A model is a learned function", kind="data", h=1.62, draw=mini_model, draw_h=0.55,
             body="Trained once on examples with answers, then used on new cases where the answer is unknown."),
        dict(num=3, title="Learning = fewer mistakes", kind="neutral", h=1.62, draw=mini_curve, draw_h=0.6,
             body="Try settings, count mistakes, keep the setting that makes the fewest. That is learning, in miniature."),
        dict(num=4, title="But \u201cfewest\u201d can be useless", kind="fail", h=1.62, draw=mini_stats, draw_h=0.5,
             body=(f"With few attacks, the best-scoring rule flags almost nothing: {tot[i]:,} mistakes, against {fn[-1]:,} "
                   "for flagging nothing at all. We asked the wrong question.")),
        dict(num=5, title="Answers should be chances", kind="jev", h=1.62, draw=mini_prob, draw_h=0.5,
             body="The same score can be an attack or not. A good model says how likely."),
        dict(num=6, title="Generate or decide", kind="llm", h=1.62, draw=mini_gen, draw_h=0.5,
             body="LLMs write answers word by word. System One models like Jev return typed answers with probabilities, in one pass."),
        dict(title="The map: a story, not a list", kind="neutral", span=2, h=1.18, draw=mini_ladder, draw_h=0.5,
             body="Rules weren\u2019t enough, so machines learned from data. Harder problems needed bigger models. Now we build systems that act."),
    ]
    return summary_page(CH, "What \u201clearning\u201d means", panels,
                        footer="Next: that \u201chow likely\u201d idea is the whole book. Chapter 2 makes it precise.")
