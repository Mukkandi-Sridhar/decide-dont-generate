"""Figures for Chapter 5: Neurons to networks."""

from functools import lru_cache

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from jevkit import soc, calibration as cal
from jevkit.learn import TinyNet, fit_logistic, sigmoid, log_loss, Standardizer
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch06"
CMAP = LinearSegmentedColormap.from_list("risk", ["#FFFFFF", "#F4D6D8", "#E59AA0"])


@lru_cache(None)
def toy():
    rng = np.random.default_rng(3)
    n = 700
    X = rng.uniform(-1, 1, (n, 2))
    r = np.sqrt(((X - [0.1, 0.05]) ** 2).sum(1))
    y = (rng.random(n) < sigmoid(12 * (r - 0.92))).astype(float)
    w, b, _ = fit_logistic(X, y, lr=1, steps=3000)
    nets = {h: TinyNet(2, h, seed=1).fit(X, y, lr=0.3, steps=3000) for h in (3, 12)}
    return X, y, (w, b), nets


@lru_cache(None)
def kestrel():
    df = soc.load()
    tr, ca, te = soc.split(df)
    F = lambda d: soc.feature_matrix(d).values
    st = Standardizer().fit(F(tr))
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(st(F(tr)), tr.malicious)
    small = MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, random_state=0).fit(st(F(tr)), tr.malicious)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        big = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=400, random_state=0).fit(st(F(tr)), tr.malicious)
    y = te.malicious.values
    Xt = st(F(te))
    return y, {"logistic regression": lr.predict_proba(Xt)[:, 1], "small network (16 units)": small.predict_proba(Xt)[:, 1],
               "big network (64+64), trained too long": big.predict_proba(Xt)[:, 1]}


def record():
    X, y, (w, b), nets = toy()
    yk, preds = kestrel()
    s = {k: cal.summary(v, yk) for k, v in preds.items()}
    results(CH, toy_lr=log_loss(sigmoid(X @ w + b), y), toy_net3=log_loss(nets[3].predict(X), y),
            toy_net12=log_loss(nets[12].predict(X), y), toy_rate=float(y.mean()),
            k_lr=s["logistic regression"], k_small=s["small network (16 units)"],
            k_big=s["big network (64+64), trained too long"])


@figure(CH, "neuron")
def neuron():
    f, ax = draw.canvas("text", 1.95)
    ins = [("threat-intel score", 0.62), ("after hours", 1.05), ("new country", 1.48)]
    for lab, yy in ins:
        draw.box(ax, 0.0, yy - 0.16, 1.25, 0.32, lab, kind="data", size=6.6)
        draw.arrow(ax, (1.25, yy), (2.2, 1.05), color=C["ink2"], lw=0.8)
    for (lab, yy), wt in zip(ins, ("× w₁", "× w₂", "× w₃")):
        draw.text(ax, 1.55, yy + (1.05 - yy) * 0.3 + 0.07, wt, size=6.4, color=C["ink2"], ha="center")
    from matplotlib.patches import Circle
    ax.add_patch(Circle((2.45, 1.05), 0.27, fc=C["neutral_t"], ec=C["ink"], lw=1.1))
    draw.text(ax, 2.45, 1.05, "Σ + b", size=7.8, weight="bold", ha="center")
    draw.text(ax, 2.45, 0.62, "add up\nthe evidence", size=5.8, ha="center", va="top", color=C["ink2"])
    draw.arrow(ax, (2.72, 1.05), (3.15, 1.05))
    draw.box(ax, 3.15, 0.8, 0.62, 0.5, "", kind="plain", color=C["ink"])
    z = np.linspace(-4, 4, 40)
    ax.plot(3.2 + (z + 4) / 8 * 0.52, 0.86 + sigmoid(z) * 0.38, color=C["jev"], lw=1.3)
    draw.text(ax, 3.46, 0.62, "squash\nto 0–1", size=5.8, ha="center", va="top", color=C["ink2"])
    draw.arrow(ax, (3.77, 1.05), (4.15, 1.05))
    draw.text(ax, 4.2, 1.05, "P(attack)", size=7, weight="bold")
    return f


def regions(ax, predict, X, y, title):
    g = np.linspace(-1, 1, 160)
    GX, GY = np.meshgrid(g, g)
    P = predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    ax.contourf(GX, GY, P, levels=np.linspace(0, 1, 11), cmap=CMAP)
    ax.contour(GX, GY, P, levels=[0.5], colors=C["ink"], linewidths=1.0)
    # the two classes differ by shape as well as colour: dots and crosses
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=4, color=C["data"], lw=0, alpha=0.8)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=7, color=C["fail"], marker="x", lw=0.7)
    ax.set_title(title, fontsize=7.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(C["grid"])


@figure(CH, "line-limit")
def line_limit():
    record()
    X, y, (w, b), nets = toy()
    f, axes = subplots(1, 3, width="text", height=1.75, gridspec_kw=dict(wspace=0.06))
    regions(axes[0], lambda Z: sigmoid(Z @ w + b), X, y, "one neuron")
    regions(axes[1], nets[3].predict, X, y, "3 hidden neurons")
    regions(axes[2], nets[12].predict, X, y, "12 hidden neurons")
    for ax in axes:
        ax.set_aspect("equal")
    axes[0].set_xlabel("clue A", fontsize=6.4)
    axes[0].set_ylabel("clue B", fontsize=6.4)
    return f


@figure(CH, "network")
def network():
    f, ax = draw.canvas("text", 2.15)
    layers = [3, 5, 5, 1]
    xs = [0.45, 1.75, 3.05, 4.2]
    names = ["inputs\n(clues)", "hidden layer 1", "hidden layer 2", "output"]
    pos = []
    for n, x in zip(layers, xs):
        ys = np.linspace(1.9 - (5 - n) * 0.16, 0.45 + (5 - n) * 0.16, n) if n > 1 else [1.18]
        pos.append([(x, yy) for yy in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=C["rule"], lw=0.45, zorder=1)
    kinds = ["data", "neutral", "neutral", "jev"]
    from jevkit.figs.style import KIND
    for layer, k in zip(pos, kinds):
        for (x, yy) in layer:
            draw.dot(ax, x, yy, r=0.1, color=KIND[k][1], ec=KIND[k][0], lw=1.0, zorder=3)
    for x, n in zip(xs, names):
        draw.text(ax, x, 0.12, n, size=6.3, ha="center", color=C["ink2"])
    draw.text(ax, 4.36, 1.18, "P", size=7.5, weight="bold")
    return f


@figure(CH, "hinges")
def hinges():
    f, axes = subplots(1, 3, width="text", height=1.6, gridspec_kw=dict(wspace=0.25))
    x = np.linspace(-2, 2, 300)
    relu = lambda v: np.maximum(0, v)
    axes[0].plot(x, relu(x), color=C["ink"], lw=1.8)
    axes[0].set_title("one hinge (ReLU)", fontsize=7)
    parts = [0.9 * relu(x + 1.2), -1.6 * relu(x - 0.1), 1.4 * relu(x - 1.0)]
    for pp, col in zip(parts, (C["data"], C["llm"], C["jev"])):
        axes[1].plot(x, pp, color=col, lw=1.4)
    axes[1].set_title("three hinges", fontsize=7)
    target = np.sin(x * 1.4) * 0.8
    knots = [-1.5, -0.8, -0.1, 0.6, 1.3]
    A = np.c_[np.ones_like(x), x, *[relu(x - k) for k in knots]]
    fit = A @ np.linalg.lstsq(A, target, rcond=None)[0]
    axes[2].plot(x, target, color=C["muted"], lw=3, alpha=0.5)
    axes[2].plot(x, fit, color=C["ink"], lw=1.4)
    axes[2].set_title("five hinges, added: any bend", fontsize=7)
    for ax in axes:
        clean(ax, "none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    return f


@figure(CH, "backprop")
def backprop():
    f, ax = draw.canvas("text", 1.75)
    xs = [0.0, 1.2, 2.4, 3.6]
    labs = ["input", "layer 1", "layer 2", "P = 0.9"]
    kinds = ["data", "neutral", "neutral", "jev"]
    for x, lab, k in zip(xs, labs, kinds):
        draw.box(ax, x, 0.72, 0.95, 0.42, lab, kind=k, size=6.8, weight="bold")
    for a, b in zip(xs[:-1], xs[1:]):
        draw.arrow(ax, (a + 0.95, 1.03), (b, 1.03), color=C["jev"], lw=1.1)
        draw.arrow(ax, (b, 0.83), (a + 0.95, 0.83), color=C["fail"], lw=1.1)
    draw.text(ax, 0.0, 1.5, "forward: compute a prediction", size=6.8, color=C["jev"], weight="semibold")
    draw.text(ax, 0.0, 0.4, "backward: the error (it was harmless: 0.9 was too high) is shared out as blame,", size=6.6,
              color=C["fail"], weight="semibold")
    draw.text(ax, 0.0, 0.2, "layer by layer, and every weight moves a little to reduce it.", size=6.6, color=C["fail"])
    draw.text(ax, 4.62, 1.5, "truth: harmless", size=6.4, ha="right", color=C["ink2"])
    return f


@figure(CH, "kestrel-nets")
def kestrel_nets():
    record()
    y, preds = kestrel()
    f, ax = subplots(width="text", height=2.4)
    clean(ax, "both")
    ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    for (name, p), col in zip(preds.items(), (C["data"], C["jev"], C["fail"])):
        b = cal.reliability(p, y, n_bins=8, strategy="quantile")
        s = cal.summary(p, y)
        ax.plot(b.mean_pred, b.frac_pos, color=col, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.6,
                label=f"{name}: log loss {s['log_loss']:.3f}")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Predicted P(attack)")
    ax.set_ylabel("Share that were attacks")
    ax.legend(loc="upper left", fontsize=6.2)
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    X, y, (w, b), nets = toy()

    def mini_regions(ax, x0, y0, w_, h_):
        for i, pred in enumerate((lambda Z: sigmoid(Z @ w + b), nets[12].predict)):
            g = np.linspace(-1, 1, 60)
            GX, GY = np.meshgrid(g, g)
            P = pred(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
            ext = [x0 + i * (h_ + 0.25), x0 + i * (h_ + 0.25) + h_, y0, y0 + h_]
            ax.contour(np.linspace(ext[0], ext[1], 60), np.linspace(ext[2], ext[3], 60), P, levels=[0.5],
                       colors=C["ink"], linewidths=1)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((ext[0], ext[2]), h_, h_, fc="none", ec=C["grid"]))

    panels = [
        dict(num=1, title="A neuron is logistic regression", kind="data", h=1.45,
             body="Weigh the inputs, add them up, squash the total. Part I already built one."),
        dict(num=2, title="One neuron draws a straight line", kind="fail", h=1.45, draw=mini_regions, draw_h=0.55,
             body="It can’t separate “unusual in any direction”. A few hidden neurons can bend the line."),
        dict(num=3, title="Hinges make any shape", kind="neutral", h=1.45,
             body="ReLU is a hinge. Add enough hinges, and a network can draw almost any curve. Without them, layers collapse into one line."),
        dict(num=4, title="Backpropagation shares the blame", kind="fail", h=1.45,
             body="Same gradient descent as Chapter 2. The error flows backwards, layer by layer, telling each weight which way to move."),
        dict(num=5, title="Deep = many layers", kind="neutral", h=1.45,
             body="Each layer builds features from the one before. On raw pixels and words, that’s what makes deep learning work."),
        dict(num=6, title="Bigger isn’t automatically better", kind="jev", h=1.45,
             body=(f"On Kestrel’s neat features, logistic regression scored {rr['k_lr']['log_loss']:.3f}; an oversized "
                   f"network {rr['k_big']['log_loss']:.3f}, and it was overconfident.")),
    ]
    return summary_page(CH, "Neurons to networks", panels,
                        footer="Next: networks need numbers. How does a word become a number that means something?")
