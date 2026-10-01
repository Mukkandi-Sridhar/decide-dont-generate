"""Figures for Chapter 5: Training at scale, and why big models are overconfident."""

import warnings
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.neural_network import MLPClassifier

from jevkit import soc, text, calibration as cal
from jevkit.learn import Standardizer
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch09"
EPOCHS = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100, 130, 160, 200)


@lru_cache(None)
def overtrain():
    df = soc.load()
    tr, ca, te = soc.split(df)
    F = lambda d: soc.feature_matrix(d).values
    st = Standardizer().fit(F(tr))
    Xtr, Xca, Xte = st(F(tr)), st(F(ca)), st(F(te))
    ytr, yca, yte = tr.malicious.values, ca.malicious.values, te.malicious.values
    m = MLPClassifier(hidden_layer_sizes=(128, 128), learning_rate_init=0.002, random_state=0, batch_size=256)
    rows, snaps = [], {}
    for ep in range(1, max(EPOCHS) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.partial_fit(Xtr, ytr, classes=[0, 1])
        if ep in EPOCHS:
            p = m.predict_proba(Xte)[:, 1]
            s = cal.summary(p, yte)
            rows.append(dict(epoch=ep, acc=float(((p > 0.5) == yte).mean()), log_loss=s["log_loss"], ece=s["ece"],
                             conf=float(np.maximum(p, 1 - p).mean()),
                             train_loss=cal.log_loss(m.predict_proba(Xtr)[:, 1], ytr)))
            if ep in (5, 200):
                snaps[ep] = (p.copy(), m.predict_proba(Xca)[:, 1].copy())
    return rows, snaps, yte, yca


def ngram_curve():
    notes = text.notes_corpus(40000, seed=0)
    test = notes[30000:]
    sizes = [30, 100, 300, 1000, 3000, 10000, 30000]
    vocab = sorted({w for n in notes for w in n} | {"<s>", "</s>"})
    V = len(vocab)
    out = []
    for n in sizes:
        big = defaultdict(Counter)
        for s in notes[:n]:
            toks = ["<s>", "<s>"] + s + ["</s>"]
            for a, b, c in zip(toks, toks[1:], toks[2:]):
                big[(a, b)][c] += 1
        tot, cnt = 0.0, 0
        for s in test[:3000]:
            toks = ["<s>", "<s>"] + s + ["</s>"]
            for a, b, c in zip(toks, toks[1:], toks[2:]):
                ctx = big.get((a, b), Counter())
                pr = (ctx[c] + 0.05) / (sum(ctx.values()) + 0.05 * V)
                tot += -np.log(pr)
                cnt += 1
        out.append(tot / cnt)
    return sizes, out


def record():
    rows, snaps, yte, yca = overtrain()
    p200, c200 = snaps[200]
    T = cal.Temperature().fit(c200, yca)
    sizes, losses = ngram_curve()
    pl = cal.Platt().fit(c200, yca)
    results(CH, ece_after_platt=cal.ece(pl(p200), yte), ll_after_platt=cal.log_loss(pl(p200), yte),
            best_ll=min(r["log_loss"] for r in rows), best_ece=[r["ece"] for r in rows if r["epoch"] == 3][0])
    results(CH, first=rows[3], last=rows[-1], best_epoch=min(rows, key=lambda r: r["log_loss"])["epoch"],
            temp_T=T.T, ece_after_T=cal.ece(T(p200), yte), ll_after_T=cal.log_loss(T(p200), yte),
            ngram_sizes=sizes, ngram_losses=[round(float(x), 3) for x in losses])


@figure(CH, "pretrain-task")
def pretrain_task():
    f, ax = draw.canvas("text", 1.75)
    words = ["the", "phishing", "email", "asked", "for", "their"]
    x = 0.0
    for w in words:
        wd = 0.13 + 0.066 * len(w)
        draw.box(ax, x, 1.2, wd, 0.3, w, kind="data", size=6.6, radius=0.04, family="JetBrains Mono")
        x += wd + 0.05
    draw.box(ax, x, 1.2, 0.55, 0.3, "?", kind="llm", size=8, weight="bold")
    guesses = [("password", 0.58), ("credentials", 0.2), ("token", 0.12), ("lunch", 0.01)]
    for i, (w, p) in enumerate(guesses):
        yy = 0.85 - i * 0.18
        draw.text(ax, x - 0.9, yy + 0.04, w, size=6.2, family="JetBrains Mono", ha="left")
        draw.gauge(ax, x + 0.05, yy, 0.8, p, kind="llm", h=0.08, size=6.0)
    draw.text(ax, 0.0, 0.66, "Hide the next word. Guess it.", size=6.8, weight="bold")
    draw.text(ax, 0.0, 0.46, "Check against the real text.", size=6.8, weight="bold")
    draw.text(ax, 0.0, 0.26, "Repeat trillions of times.", size=6.8, weight="bold")
    draw.text(ax, 0.0, 0.03, "The answer is always in the text itself, so nobody has to label anything.", size=6.2,
              color=C["ink2"])
    draw.text(ax, 4.7, 0.03, "illustrative numbers", size=5.4, ha="right", color=C["muted"])
    return f


@figure(CH, "mini-scaling")
def mini_scaling():
    record()
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    sizes, losses = rr["ngram_sizes"], rr["ngram_losses"]
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "both")
    ax.plot(sizes, losses, color=C["llm"], lw=1.8, marker="o", ms=4, mec="white", mew=0.7)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=6.2)
    ax.set_xlabel("Analyst notes seen in training (log scale)")
    ax.set_ylabel("Next-word log loss on new notes")
    ax.text(sizes[1], losses[1] + 0.2, "more text, better guesses,\nwith diminishing returns", fontsize=6.4, color=C["ink2"])
    return f


@figure(CH, "stages")
def stages():
    f, ax = draw.canvas("text", 2.1)
    stg = [("Pretraining", "predict the next word\non a huge pile of text", "knows a lot;\njust continues text", "llm"),
           ("Instruction tuning", "learn from examples of\ngood answers", "follows\nrequests", "llm"),
           ("Preference tuning", "learn from people’s ratings\n(RLHF and relatives)", "helpful, polite;\noften less calibrated", "fail")]
    for i, (t, how, res, kind) in enumerate(stg):
        x = i * 1.6
        draw.box(ax, x, 1.2, 1.4, 0.44, t, kind=kind if kind != "fail" else "llm", size=7, weight="bold")
        draw.text(ax, x + 0.7, 1.08, how, size=6.0, ha="center", va="top", color=C["ink2"])
        draw.text(ax, x + 0.7, 0.38, res, size=6.2, ha="center", va="top",
                  color=C["fail"] if kind == "fail" else C["ink"], weight="semibold")
        if i < 2:
            draw.arrow(ax, (x + 1.4, 1.42), (x + 1.6, 1.42))
    draw.text(ax, 0.0, 1.88, "How a raw text predictor becomes an assistant", size=7, weight="bold")
    return f


@figure(CH, "overtraining")
def overtraining():
    rows, snaps, yte, yca = overtrain()
    ep = [r["epoch"] for r in rows]
    f, axes = subplots(1, 3, width="text", height=1.75, gridspec_kw=dict(wspace=0.45))
    specs = [("acc", "accuracy", C["data"], lambda v, _: f"{v:.0%}"), ("log_loss", "log loss", C["fail"], None),
             ("ece", "calibration error (ECE)", C["fail"], None)]
    for ax, (k, title, col, fmt) in zip(axes, specs):
        clean(ax, "y")
        ax.plot(ep, [r[k] for r in rows], color=col, lw=1.6, marker="o", ms=2.5)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("training epochs", fontsize=6.2)
        if fmt:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt))
            ax.set_ylim(0.85, 1.0)
        else:
            ax.set_ylim(0, None)
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "confidence")
def confidence():
    rows, snaps, yte, yca = overtrain()
    f, ax = subplots(width="text", height=2.0)
    clean(ax, "y")
    bins = np.linspace(0, 1, 41)
    for ep, col, lab in ((5, C["data"], "after 5 epochs"), (200, C["fail"], "after 200 epochs")):
        p = snaps[ep][0]
        ax.hist(p, bins=bins, histtype="step", color=col, lw=1.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted P(attack) on test alerts")
    ax.set_ylabel("alerts (log)")
    ax.legend(loc="upper center", fontsize=6.4)
    ax.text(0.5, 200, "overtrained: piles up at 0 and 1", fontsize=6.3, color=C["fail"], ha="center")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "temperature-fix")
def temperature_fix():
    record()
    rows, snaps, yte, yca = overtrain()
    p200, c200 = snaps[200]
    T = cal.Temperature().fit(c200, yca)
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    for p, col, lab in ((p200, C["fail"], "after 200 epochs"), (T(p200), C["jev"], f"same model, temperature T = {T.T:.1f}")):
        b = cal.reliability(p, yte, n_bins=10, strategy="quantile")
        ax.plot(b.mean_pred, b.frac_pos, color=col, lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6,
                label=f"{lab}  (ECE {cal.ece(p, yte):.3f})")
    ax.set_xlabel("Predicted P(attack)")
    ax.set_ylabel("Share that were attacks")
    ax.legend(loc="upper left", fontsize=6.3)
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Pretraining: guess the next word", kind="llm", h=1.45,
             body="The labels come free from the text itself, so models can learn from trillions of words."),
        dict(num=2, title="More data, better guesses", kind="llm", h=1.45,
             body=("Loss falls smoothly as data, model size and compute grow, with diminishing returns. That "
                   "predictability is what made giant models a safe bet.")),
        dict(num=3, title="Tuning makes an assistant", kind="llm", h=1.45,
             body="Instruction tuning teaches it to follow requests; preference tuning teaches it to please raters. The second can cost calibration."),
        dict(num=4, title="Accuracy hides overconfidence", kind="fail", h=1.45,
             body=(f"Train too long: accuracy barely moves, log loss goes from {rr['first']['log_loss']:.2f} to "
                   f"{rr['last']['log_loss']:.2f}, ECE from {rr['first']['ece']:.3f} to {rr['last']['ece']:.3f}.")),
        dict(num=5, title="Why: squeezing the last bit of loss", kind="fail", h=1.45,
             body="On training data, pushing P towards 0 and 1 always lowers the loss, so the model learns to be sure."),
        dict(num=6, title="Stop early, then calibrate", kind="jev", h=1.45,
             body=(f"Temperature (T = {rr['temp_T']:.1f}) helped; Platt fixed the calibration (ECE {rr['ece_after_platt']:.3f}). "
                   "Stopping early beat both on log loss.")),
    ]
    return summary_page(CH, "Training at scale, and why big models are overconfident", panels,
                        footer="Next, Part II: what an LLM actually does when it writes.")
