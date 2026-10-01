"""Figures for Chapter 13: The bake-off: six ways to make a decision."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import bakeoff as B, calibration as cal, llm
from jevkit.econ import JEV_PRICE, JEV_LATENCY
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch20"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
COL = dict(rules=C["slate"], logistic=C["data"], text_clf=C["gold"], llm_json=C["llm"], llm_judge=C["llm"], jev=C["jev"])
HOLLOW = {"llm_judge"}
SHORT = {"rules": "Rules", "logistic": "Logistic regression", "text_clf": "Text classifier",
         "llm_json": "LLM, JSON", "llm_judge": "LLM-as-judge", "jev": "Jev (mock)"}

# Time and price per decision. Classic models: compute only (an assumption). LLM: this book's illustrative
# figures. Jev: vendor-reported.
JSON_OUT, JUDGE_OUT, TOKENS_IN = 60, 8, 500
SPEED = {
    "rules": (1e-6, 0.3e-6),
    "logistic": (1e-5, 0.5e-6),
    "text_clf": (1e-4, 1e-6),
    "llm_json": (llm.MockLLM.simulated_latency_s(JSON_OUT), llm.MockLLM.simulated_cost_usd(TOKENS_IN, JSON_OUT)),
    "llm_judge": (llm.MockLLM.simulated_latency_s(JUDGE_OUT), llm.MockLLM.simulated_cost_usd(TOKENS_IN + 20, JUDGE_OUT)),
    "jev": ((JEV_LATENCY[0] * JEV_LATENCY[1]) ** 0.5, JEV_PRICE),
}


@lru_cache(None)
def data():
    r = B.run()
    sc = {m: B.scores(r["probs"][m], r["y"], verdict=r["rules_live"] if m == "rules" else None) for m in B.METHODS}
    curve = B.labels_curve()
    flips = B.flip_rate(r["live"].description.tolist())
    return r, sc, curve, flips


def record():
    r, sc, curve, flips = data()
    jt = B.scores(r["jev_text"], r["y"])
    # labels the logistic model needs to match Jev's ranking
    match = next((n for n, a in zip(curve["sizes"], curve["logistic"]) if a >= sc["jev"]["auc"]), None)
    results(CH, **{f"{m}_{k}": v for m in B.METHODS for k, v in sc[m].items()},
            jev_text_auc=jt["auc"], jev_text_ece=jt["ece"], jev_text_caught=jt["caught"],
            parse_fail=float(1 - r["parse_ok"].mean()), flip_rate=flips,
            n_live=len(r["y"]), n_hist=len(r["hist"]), live_threats=int(r["y"].sum()),
            labels_to_match=match, labels_curve=curve,
            llm_json_cost_m=SPEED["llm_json"][1] * 1e6, llm_judge_cost_m=SPEED["llm_judge"][1] * 1e6,
            jev_cost_m=JEV_PRICE * 1e6, llm_json_latency=SPEED["llm_json"][0], llm_judge_latency=SPEED["llm_judge"][0],
            # distinct *stated* confidences: undo the verdict flip (P = c or 1 - c) on the answers that parsed
            conf_levels=int(len(np.unique(np.round(np.maximum(r["probs"]["llm_json"], 1 - r["probs"]["llm_json"])[r["parse_ok"]], 3)))))


@figure(CH, "contenders")
def contenders():
    f, ax = draw.canvas("text", 3.0)
    cards = [("rules", "Hand-written rules", "fields", "yes or no", "an expert’s afternoon"),
             ("logistic", "Logistic regression", "fields", "a probability", "thousands of labels"),
             ("text_clf", "Trained text classifier", "raw text", "a probability", "thousands of labels"),
             ("llm_json", "LLM, structured output", "raw text", "JSON: verdict +\nstated confidence", "a prompt"),
             ("llm_judge", "LLM-as-judge", "raw text", "a 1–10 rating", "a prompt and a rubric"),
             ("jev", "Jev (mock)", "fields as JSON", "a probability", "typed questions")]
    w, h = 1.48, 1.28
    for i, (m, name, inp, out, needs) in enumerate(cards):
        x, y = (i % 3) * (w + 0.13), 1.55 - (i // 3) * (h + 0.2)
        kind = {"llm_json": "llm", "llm_judge": "llm", "jev": "jev"}.get(m, "data" if m == "logistic" else "neutral")
        draw.box(ax, x, y, w, h, "", kind=kind)
        draw.number_badge(ax, x + 0.15, y + h - 0.16, i + 1, color=draw.KIND[kind][0])
        draw.text(ax, x + 0.3, y + h - 0.16, name, size=6.8, weight="bold")
        for j, (lab, val) in enumerate((("reads", inp), ("returns", out), ("needs", needs))):
            yy = y + h - 0.45 - j * 0.3
            draw.text(ax, x + 0.1, yy, lab, size=5.8, color=C["ink2"], weight="semibold")
            draw.text(ax, x + 0.55, yy, val, size=6.0, va="center")
    return f


@figure(CH, "scores")
def score_panels():
    r, sc, curve, flips = data()
    f, axes = subplots(1, 3, width="text", height=2.25, sharey=True)
    order = sorted(B.METHODS, key=lambda m: sc[m]["auc"])
    panels = [("auc", "Ranking (AUC)\nhigher is better", (0.5, 1.0), "{:.2f}"),
              ("caught", "Threats caught when\nreviewing 240 a day", (0, 1), "{:.0%}"),
              ("ece", "Calibration error (ECE)\nlower is better", (0, 0.12), "{:.3f}")]
    for ax, (key, title, lim, fmt) in zip(axes, panels):
        clean(ax, "x")
        for i, m in enumerate(order):
            v = sc[m][key]
            ax.plot([lim[0], v], [i, i], color=C["grid"], lw=1, zorder=1)
            ax.scatter([v], [i], s=34, zorder=3, color="white" if m in HOLLOW else COL[m], edgecolor=COL[m], lw=1.4)
            ax.text(v + (lim[1] - lim[0]) * 0.05, i, fmt.format(v), va="center", fontsize=5.9)
        ax.set_xlim(lim[0], lim[1] * 1.12 if key != "auc" else 1.02)
        ax.set_title(title, fontsize=6.6, loc="left")
        if key == "caught":
            ax.xaxis.set_major_formatter(PCT)
            ax.set_xticks([0, 0.5, 1])
        if key == "ece":
            ax.set_xticks([0, 0.05, 0.1])
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([SHORT[m] for m in order], fontsize=6.4)
    f.subplots_adjust(wspace=0.18, top=0.8)
    synthetic_tag(f)
    return f


@figure(CH, "reliability")
def reliability():
    r, sc, curve, flips = data()
    f, axes = subplots(2, 3, width="text", height=3.3, sharex=True, sharey=True)
    for ax, m in zip(axes.flat, B.METHODS):
        clean(ax, "both")
        p = r["probs"][m]
        ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
        b = cal.reliability(p, r["y"], n_bins=20)
        ok = b.count >= 20
        ax.plot(b.mean_pred[ok], b.frac_pos[ok], color=COL[m], lw=1.6, marker="o", ms=3.2,
                mfc="white" if m in HOLLOW else COL[m])
        ax.set_xlim(0, 0.62)
        ax.set_ylim(0, 0.62)
        ax.set_title(f"{SHORT[m]}\nECE {sc[m]['ece']:.3f}", fontsize=6.2, loc="left")
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_yticks([0, 0.25, 0.5])
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
    for ax in axes[1]:
        ax.set_xlabel("what it said", fontsize=6.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("what happened", fontsize=6.4)
    f.subplots_adjust(hspace=0.55, wspace=0.12)
    synthetic_tag(f)
    return f


@figure(CH, "labels")
def labels():
    r, sc, curve, flips = data()
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    n = curve["sizes"]
    ax.plot(n, curve["logistic"], color=COL["logistic"], lw=2, marker="o", ms=3.5)
    ax.plot(n, curve["text_clf"], color=COL["text_clf"], lw=2, marker="o", ms=3.5)
    ax.text(n[-1] * 1.12, curve["logistic"][-1] + 0.022, "logistic regression\n(fields)", fontsize=6.0, va="center",
            color=C["ink"])
    ax.text(n[-1] * 1.12, curve["text_clf"][-1], "text classifier", fontsize=6.0, va="center", color=C["ink"])
    for m, y, lab in (("jev", sc["jev"]["auc"], "Jev (mock), no labels"),
                      ("llm_judge", sc["llm_judge"]["auc"], "LLM-as-judge, no labels"),
                      ("llm_json", sc["llm_json"]["auc"], "LLM JSON, no labels")):
        ax.axhline(y, color=COL[m], lw=1.1, ls=(0, (4, 2)) if m != "jev" else "-")
        lx, ly, va = {"jev": (80, y + 0.006, "bottom"), "llm_judge": (1500, y + 0.006, "bottom"),
                      "llm_json": (3000, y - 0.006, "top")}[m]
        ax.text(lx, ly, lab, fontsize=5.9, color=COL[m], va=va, fontweight="semibold")
    ax.set_xscale("log")
    ax.set_xlim(70, 6e4)
    ax.set_ylim(0.55, 0.93)
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100", "1,000", "10,000"])
    ax.set_xlabel("labelled alerts used for training (log scale)")
    ax.set_ylabel("ranking on the live week (AUC)")
    f.subplots_adjust(right=0.8)
    synthetic_tag(f)
    return f


@figure(CH, "cost-latency")
def cost_latency():
    f, ax = subplots(width="text", height=2.5)
    clean(ax, "both")
    for m, (t, c) in SPEED.items():
        cm = c * 1e6
        if m == "jev":
            ax.plot(JEV_LATENCY, [cm, cm], color=COL[m], lw=5, solid_capstyle="round", alpha=0.5)
        ax.scatter([t], [cm], s=40, zorder=3, color="white" if m in HOLLOW else COL[m], edgecolor=COL[m], lw=1.4)
        lab = SHORT[m]
        dy = {"rules": 1.5, "logistic": 1.5, "text_clf": 1.5, "llm_judge": 0.62}.get(m, 1.6)
        ax.text(t, cm * dy, lab, fontsize=6.0, ha="center", va="bottom" if dy > 1 else "top")
    ax.axhspan(0.1, 1.2, color=C["grid"], alpha=0.5, zorder=0, lw=0)
    ax.text(3e-6, 0.25, "compute only: under $1 per million (assumption)", fontsize=5.8, color=C["ink2"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e-7, 20)
    ax.set_ylim(0.1, 5000)
    ax.set_xticks([1e-6, 1e-3, 1])
    ax.set_xticklabels(["1 µs", "1 ms", "1 s"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:g}"))
    ax.set_xlabel("time per decision (log scale)")
    ax.set_ylabel("cost per million decisions")
    synthetic_tag(f, "LLM: illustrative · Jev: vendor-reported range · classic: assumption")
    return f


@figure(CH, "variance")
def variance():
    r, sc, curve, flips = data()
    f, (a1, a2) = subplots(1, 2, width="text", height=1.9)
    rows = [("LLM, temperature 0.7", flips, C["llm"]), ("LLM, temperature 0", 0.0, C["llm"]),
            ("Jev (mock)", 0.0, C["jev"]), ("logistic regression", 0.0, C["data"])]
    clean(a1, "x")
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        a1.barh(y, v, color=col, height=0.5)
        a1.text(v + 0.003, y, f"{v:.1%}", va="center", fontsize=6.0)
    a1.set_yticks(range(len(rows))[::-1])
    a1.set_yticklabels([r_[0] for r_ in rows], fontsize=6.2)
    a1.set_xlim(0, 0.1)
    a1.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    a1.xaxis.set_major_formatter(PCT)
    a1.set_title("verdict changes across 5 identical calls", fontsize=6.6, loc="left")
    clean(a2, "x")
    fail = 1 - r["parse_ok"].mean()
    rows2 = [("LLM, free text JSON", fail, C["llm"]), ("Jev (typed answers)", 0.0, C["jev"])]
    for i, (lab, v, col) in enumerate(rows2):
        y = len(rows2) - 1 - i
        a2.barh(y, v, color=col, height=0.45)
        a2.text(v + 0.002, y, f"{v:.1%}", va="center", fontsize=6.0)
    a2.set_yticks(range(len(rows2))[::-1])
    a2.set_yticklabels([r_[0] for r_ in rows2], fontsize=6.2)
    a2.set_ylim(-0.8, 1.8)
    a2.set_xlim(0, 0.06)
    a2.xaxis.set_major_formatter(PCT)
    a2.set_title("answers that couldn’t be parsed", fontsize=6.6, loc="left")
    f.subplots_adjust(wspace=0.9)
    synthetic_tag(f)
    return f


@figure(CH, "scorecard")
def scorecard():
    r, sc, curve, flips = data()
    cols = ["ranking\n(AUC)", "caught at\n240/day", "calibration\n(ECE)", "cost per\nmillion", "time per\ndecision",
            "labels\nneeded", "same answer\ntwice?", "explains\nitself?"]
    lat = lambda t: f"{t * 1e6:.0f} µs" if t < 1e-3 else (f"{t * 1000:.0f} ms" if t < 1 else f"{t:.1f} s")
    cost = lambda c: "< $1" if c * 1e6 < 1 else f"${c * 1e6:,.0f}"
    rows = {
        "rules": [None, None, None, cost(SPEED["rules"][1]), lat(SPEED["rules"][0]), "none", "yes", "fully"],
        "logistic": [None, None, None, cost(SPEED["logistic"][1]), lat(SPEED["logistic"][0]), "~3,000", "yes", "weights"],
        "text_clf": [None, None, None, cost(SPEED["text_clf"][1]), lat(SPEED["text_clf"][0]), "15,000+", "yes", "word weights"],
        "llm_json": [None, None, None, cost(SPEED["llm_json"][1]), lat(SPEED["llm_json"][0]), "none", "not always", "in words*"],
        "llm_judge": [None, None, None, cost(SPEED["llm_judge"][1]), lat(SPEED["llm_judge"][0]), "none", "not always", "in words*"],
        "jev": [None, None, None, f"${JEV_PRICE * 1e6:.0f}†", "70–500 ms†", "none‡", "yes (mock)", "no"],
    }
    for m in B.METHODS:
        rows[m][0] = f"{sc[m]['auc']:.2f}"
        rows[m][1] = f"{sc[m]['caught']:.0%}"
        rows[m][2] = f"{sc[m]['ece']:.3f}"
    best = {0: max(B.METHODS, key=lambda m: sc[m]["auc"]), 1: max(B.METHODS, key=lambda m: sc[m]["caught"]),
            2: min(B.METHODS, key=lambda m: sc[m]["ece"])}
    f, ax = draw.canvas("wide", 3.35)
    x0, cw, rh = 1.25, 0.585, 0.42
    for j, c in enumerate(cols):
        draw.text(ax, x0 + j * cw + cw / 2, 3.1, c, size=5.6, ha="center", weight="semibold", color=C["ink2"])
    for i, m in enumerate(B.METHODS):
        y = 2.62 - i * rh
        if i % 2 == 0:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (0, y - rh / 2), x0 + len(cols) * cw, rh, fc=C["neutral_t"], ec="none", zorder=0))
        draw.text(ax, 0.05, y, SHORT[m], size=6.4, weight="bold", color=C["ink"])
        for j, v in enumerate(rows[m]):
            bold = best.get(j) == m
            draw.text(ax, x0 + j * cw + cw / 2, y, v, size=5.7 if "\n" in v or len(v) > 9 else 6.2, ha="center",
                      weight="bold" if bold else "normal", color=C["jev"] if bold else C["ink"])
    draw.text(ax, 0.05, 0.16, "Bold: best in column.  † vendor-reported.  ‡ none to start; about 300 to check and calibrate.  "
                              "Classic costs: compute only (assumption).", size=5.4, color=C["ink2"])
    draw.text(ax, 0.05, 0.02, "* reasons written after the fact, which may not be how the verdict was reached (Chapter 6).  "
                              "LLM figures: illustrative.", size=5.4, color=C["ink2"])
    synthetic_tag(f)
    return f


@figure(CH, "summary")
def summary():
    record()
    r, sc, curve, flips = data()

    def dots(keys, fmt, lim):
        def paint(ax, x, y, w, h):
            order = keys
            for k, m in enumerate(order):
                yy = y + h - (k + 0.5) * h / len(order)
                v = sc[m]["auc"]
                xx = x + 1.05 + (w - 1.5) * (v - lim[0]) / (lim[1] - lim[0])
                ax.plot([x + 1.05, x + w - 0.45], [yy, yy], color=C["grid"], lw=0.8)
                ax.scatter([xx], [yy], s=22, color="white" if m in HOLLOW else COL[m], edgecolor=COL[m], lw=1.2, zorder=3)
                ax.text(x + 1.0, yy, SHORT[m], ha="right", va="center", fontsize=5.9, color=C["ink2"])
                ax.text(xx + 0.08, yy, fmt.format(v), va="center", fontsize=5.8)
        return paint

    panels = [
        dict(num=1, title="Six contestants, one fair test", kind="neutral", h=2.55, span=1,
             body=(f"Each was trained or tuned on three weeks of Kestrel alerts ({r['hist'].shape[0]:,}) and scored on the "
                   f"fourth ({len(r['y']):,}), on ranking, calibration, cost, speed, stability and labels needed."),
             draw=dots(sorted(B.METHODS, key=lambda m: -sc[m]["auc"]), "{:.2f}", (0.6, 0.95)), draw_h=1.05),
        dict(num=2, title="Where Jev loses", kind="data", h=2.55,
             body=(f"With enough labels and clean fields, logistic regression matches or beats it: "
                   f"AUC {sc['logistic']['auc']:.3f} against {sc['jev']['auc']:.3f}, at almost no cost per decision. "
                   f"Rules win on explainability. Nothing is cheaper than a rule.")),
        dict(num=3, title="Where it wins", kind="jev", h=1.75,
             body=("No labels to start, typed answers that always parse, and probabilities that rank and are close to "
                   "calibrated. It reads raw text better than a bag-of-words classifier.")),
        dict(num=4, title="The LLM’s real problem", kind="llm", h=1.75,
             body=(f"Stated confidences snap to a few values, so they can’t rank. {1 - r['parse_ok'].mean():.1%} of JSON "
                   f"answers broke. At temperature 0.7, {flips:.1%} of verdicts changed on a second ask.")),
        dict(num=5, title="Read the design, not the winner", kind="fail", h=1.75,
             body=("The mock LLM was built as a noisier reader than mock Jev. The accuracy gap between them is a design "
                   "choice, not a finding. Run this contest on your own data.")),
        dict(num=6, title="The scorecard has eight columns", kind="neutral", h=1.75,
             body="Pick the method that wins on the columns your decision cares about. Often that’s two methods together."),
    ]
    return summary_page(CH, "The bake-off: six ways to make a decision", panels,
                        footer="Next: turning probabilities into act, review and escalate.")
