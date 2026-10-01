"""Figures for Chapter 2: Data, loss and gradient descent."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, calibration as cal
from jevkit.learn import sigmoid, log_loss, fit_logistic, Standardizer
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch03"


@lru_cache(None)
def data():
    df = soc.load()
    tr, ca, te = soc.split(df)
    X, Xt = soc.feature_matrix(tr).values, soc.feature_matrix(te).values
    st = Standardizer().fit(X)
    w, b, hist = fit_logistic(st(X), tr.malicious.values, lr=0.5, steps=400, record=True)
    pt = sigmoid(st(Xt) @ w + b)
    return dict(df=df, tr=tr, te=te, X=X, Xt=Xt, st=st, w=w, b=b, hist=hist, pt=pt)


def one_feature_loss(bs, ws, x, y):
    L = np.empty((len(bs), len(ws)))
    for i, b in enumerate(bs):
        for j, w in enumerate(ws):
            L[i, j] = log_loss(sigmoid(b + w * x), y)
    return L


def gd_path(x, y, b0=-0.5, w0=-2.0, lr=4.0, steps=60):
    b, w = b0, w0
    path = [(b, w)]
    for _ in range(steps):
        p = sigmoid(b + w * x)
        gb, gw = np.mean(p - y), np.mean((p - y) * x)
        b, w = b - lr * gb, w - lr * gw
        path.append((b, w))
    return np.array(path)


def record():
    d = data()
    te = d["te"]
    y = te.malicious.values
    s_model = cal.summary(d["pt"], y)
    s_truth = cal.summary(te.p_true.values, y)
    from sklearn.linear_model import LogisticRegression
    m2 = LogisticRegression(C=1e4, max_iter=5000).fit(d["X"], d["tr"].malicious.values)
    cols = list(soc.feature_matrix(d["tr"]).columns)
    coef = dict(zip(cols, m2.coef_[0]))
    results(CH, n_train=len(d["tr"]), n_test=len(te), loss_start=d["hist"][0], loss_end=d["hist"][-1],
            test_loss=s_model["log_loss"], truth_loss=s_truth["log_loss"], test_auc=s_model["auc"],
            test_ece=s_model["ece"], gap_pct=(s_model["log_loss"] / s_truth["log_loss"] - 1),
            mult_ioc_tenth=float(np.exp(coef["ioc_score"] * 0.1)), mult_after=float(np.exp(coef["after_hours"])),
            mult_known=float(np.exp(coef["known_tool"])), mult_geo=float(np.exp(coef["new_geo"])),
            mult_mfa=float(np.exp(coef["mfa_ok"])), loss_if_certain_wrong=float(-np.log(0.01)),
            loss_if_right_09=float(-np.log(0.9)))
    return coef


@figure(CH, "sigmoid")
def sigmoid_fig():
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "both")
    z = np.linspace(-6, 6, 300)
    ax.plot(z, sigmoid(z), color=C["jev"], lw=2)
    for zz, lab, tx, ty in ((-3, "evidence says no", -5.8, 0.2), (0, "evenly balanced", 0.45, 0.33),
                            (3, "evidence says yes", 3.4, 0.72)):
        ax.scatter([zz], [sigmoid(zz)], color=C["ink"], s=16, zorder=4, edgecolor="white", lw=0.8)
        ax.text(tx, ty, f"{lab}\nP = {sigmoid(zz):.2f}", fontsize=6.3, color=C["ink2"])
    ax.set_xlabel("Total evidence, in log-odds  (add up each clue’s weight)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.05)
    return f


@figure(CH, "logloss")
def logloss_fig():
    rec = record()
    f, ax = subplots(width="text", height=2.25)
    clean(ax, "y")
    p = np.linspace(0.005, 0.995, 300)
    ax.plot(p, -np.log(p), color=C["fail"], lw=1.8, ls="-")              # it was an attack: solid
    ax.plot(p, -np.log(1 - p), color=C["data"], lw=1.8, ls=(0, (5, 2)))  # it was harmless: dashed
    ax.text(0.08, 3.6, "it WAS an attack:\npenalty = −log(P)", fontsize=6.5, color=C["fail"], fontweight="semibold")
    ax.text(0.62, 3.6, "it was harmless:\npenalty = −log(1−P)", fontsize=6.5, color=C["data"], fontweight="semibold")
    ax.scatter([0.01], [-np.log(0.01)], color=C["fail"], s=18, zorder=4)
    ax.annotate("said 1%, and it happened:\nhuge penalty (4.6)", (0.01, -np.log(0.01)), xytext=(0.12, 4.9), fontsize=6.3,
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.scatter([0.9], [-np.log(0.9)], color=C["fail"], s=18, zorder=4)
    ax.annotate("said 90%, and it happened:\ntiny penalty (0.1)", (0.9, -np.log(0.9)), xytext=(0.56, 1.2), fontsize=6.3,
                arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.set_ylim(0, 5.4)
    ax.set_xlabel("The probability the model gave to “attack”")
    ax.set_ylabel("Log loss (surprise)")
    return f


@figure(CH, "descent-2d")
def descent_2d():
    d = data()
    tr = d["tr"]
    x = tr.ioc_score.values
    y = tr.malicious.values
    bs = np.linspace(-5.2, 0.5, 70)
    ws = np.linspace(-3, 8, 70)
    L = one_feature_loss(bs, ws, x, y)
    path = gd_path(x, y, b0=-0.5, w0=-2.5, lr=6.0, steps=400)
    f, ax = subplots(width="text", height=2.7)
    ax.grid(False)
    levels = np.quantile(L, [0.005, 0.02, 0.05, 0.1, 0.18, 0.28, 0.4, 0.55, 0.7, 0.85])
    cs = ax.contour(ws, bs, L, levels=np.unique(levels), colors=C["rule"], linewidths=0.6)
    ax.contourf(ws, bs, L, levels=np.unique(np.concatenate([[L.min()], levels, [L.max()]])),
                cmap=__import__("matplotlib").colors.LinearSegmentedColormap.from_list("b", ["#DDEBF7", "#FFFFFF"]))
    ax.plot(path[:, 1], path[:, 0], color=C["ink"], lw=1.2)
    ax.scatter(path[::25, 1], path[::25, 0], color=C["ink"], s=10, zorder=4, edgecolor="white", lw=0.6)
    ax.scatter([path[0, 1]], [path[0, 0]], color=C["fail"], s=26, zorder=5, edgecolor="white", lw=0.8)
    ax.scatter([path[-1, 1]], [path[-1, 0]], color=C["jev"], s=34, zorder=5, edgecolor="white", lw=0.8, marker="o")
    ax.text(path[0, 1] + 0.2, path[0, 0] + 0.15, "start: a bad guess", fontsize=6.4, color=C["fail"])
    ax.text(path[-1, 1] + 0.25, path[-1, 0] - 0.35, "bottom of the valley", fontsize=6.4, color=C["jev"], fontweight="semibold")
    ax.set_xlabel("weight on the threat-intel score  (w)")
    ax.set_ylabel("starting point  (b)")
    ax.spines["left"].set_visible(True)
    return f


@figure(CH, "descent-1d")
def descent_1d():
    d = data()
    tr = d["tr"]
    x, y = tr.ioc_score.values, tr.malicious.values
    b = -3.9
    ws = np.linspace(-2, 9, 200)
    L = [log_loss(sigmoid(b + w * x), y) for w in ws]
    f, axes = subplots(1, 3, width="text", height=1.75, sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, lr, title in zip(axes, (2.5, 14, 160), ("steps too small", "about right", "steps too big")):
        clean(ax, "y")
        ax.plot(ws, L, color=C["rule"], lw=1.4)
        w = -1.5
        pts = [w]
        for _ in range(8):
            p = sigmoid(b + w * x)
            w = w - lr * np.mean((p - y) * x)
            pts.append(w)
        pts = np.clip(pts, -2, 9)
        ls = [log_loss(sigmoid(b + q * x), y) for q in pts]
        ax.plot(pts, ls, color=C["ink"], lw=0.8, marker="o", ms=3.5, mfc=C["ink"], mec="white", mew=0.6)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel("weight w", fontsize=6.6)
    axes[0].set_ylabel("loss")
    axes[0].set_yticks([])
    return f


@figure(CH, "training-curve")
def training_curve():
    d = data()
    rec = record()
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    h = d["hist"]
    ax.plot(np.arange(len(h)), h, color=C["jev"], lw=1.8)
    ax.axhline(rr["truth_loss"], color=C["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(len(h) - 5, rr["truth_loss"] + 0.012, "the best any model could do (the generator’s own truth)", fontsize=6.3,
            ha="right", color=C["ink2"])
    ax.text(25, 0.5, "all weights at zero:\nevery alert gets P = 0.5", fontsize=6.3, color=C["ink2"])
    ax.set_ylim(0.1, 0.72)
    ax.set_xlabel("Gradient-descent step")
    ax.set_ylabel("Log loss on training alerts")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "weights")
def weights():
    coef = record()
    items = [("threat-intel score +0.1", coef["ioc_score"] * 0.1), ("outside business hours", coef["after_hours"]),
             ("sign-in from a new country", coef["new_geo"]), ("each related alert (24h)", coef["prior_alerts_24h"]),
             ("admin account", coef["role_admin"]), ("MFA completed", coef["mfa_ok"]),
             ("matches approved IT tooling", coef["known_tool"])]
    items = sorted(items, key=lambda t: t[1])
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "x")
    ys = np.arange(len(items))
    vals = np.exp([v for _, v in items])
    cols = [C["jev"] if v > 1 else C["data"] for v in vals]
    ax.barh(ys, np.log(vals), color=cols, height=0.5)
    ax.axvline(0, color=C["ink"], lw=0.8)
    for yy, v in zip(ys, vals):
        ax.text(np.log(v) + (0.08 if v > 1 else -0.08), yy, f"×{v:.2f}", va="center", ha="left" if v > 1 else "right",
                fontsize=6.4, color=C["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in items], fontsize=6.6)
    ax.set_xticks(np.log([0.1, 0.25, 0.5, 1, 2, 4]))
    ax.set_xticklabels(["×0.1", "×0.25", "×0.5", "×1", "×2", "×4"])
    ax.set_xlim(np.log(0.07), np.log(9))
    ax.set_xlabel("What each clue does to the odds of an attack (learned)")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "overfit")
def overfit():
    from sklearn.tree import DecisionTreeClassifier
    d = data()
    X, Xt, ytr, yte = d["X"], d["Xt"], d["tr"].malicious.values, d["te"].malicious.values
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20]
    trl, tel = [], []
    for dep in depths:
        t = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(X, ytr)
        trl.append(log_loss(np.clip(t.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3), ytr))
        tel.append(log_loss(np.clip(t.predict_proba(Xt)[:, 1], 1e-3, 1 - 1e-3), yte))
    best = depths[int(np.argmin(tel))]
    results(CH, tree_best_depth=best, tree_best_test=float(min(tel)), tree_deep_train=trl[-1], tree_deep_test=tel[-1])
    f, ax = subplots(width="text", height=2.25)
    clean(ax, "y")
    ax.plot(depths, trl, color=C["data"], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6)
    ax.plot(depths, tel, color=C["fail"], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6)
    ax.text(15, trl[-3] + 0.04, "alerts it studied", fontsize=6.5, color=C["data"], fontweight="semibold")
    ax.text(12, tel[-4] + 0.03, "new alerts", fontsize=6.5, color=C["fail"], fontweight="semibold")
    ax.axvline(best, color=C["muted"], lw=0.7)
    ax.text(best + 0.3, 0.55, "learning stops,\nmemorising starts", fontsize=6.3, color=C["ink2"])
    ax.set_xlabel("How many questions the model may ask (decision-tree depth)")
    ax.set_ylabel("Log loss")
    synthetic_tag(f, "SYNTHETIC DATA · Kestrel Logistics")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))

    def mini_sig(ax, x, y, w, h):
        z = np.linspace(-5, 5, 60)
        ax.plot(x + (z + 5) / 10 * w * 0.6, y + sigmoid(z) * h, color=C["jev"], lw=1.4)
        draw.text(ax, x + w * 0.66, y + h * 0.5, "evidence adds up\nin log-odds", size=6.0, color=C["ink2"])

    def mini_loss(ax, x, y, w, h):
        p = np.linspace(0.02, 0.98, 60)
        ax.plot(x + p * w * 0.6, y + (-np.log(p)) / 4 * h, color=C["fail"], lw=1.4)
        draw.text(ax, x + w * 0.66, y + h * 0.5, "confident and\nwrong = ouch", size=6.0, color=C["ink2"])

    def mini_curve(ax, x, y, w, h):
        hh = np.array(data()["hist"])
        ax.plot(x + np.arange(len(hh)) / len(hh) * w, y + (hh - hh.min()) / (hh.max() - hh.min()) * h, color=C["jev"], lw=1.4)

    def mini_over(ax, x, y, w, h):
        t = np.linspace(0, 1, 30)
        ax.plot(x + t * w, y + (1 - t) * h * 0.8, color=C["data"], lw=1.3)
        ax.plot(x + t * w, y + (0.55 - 0.4 * t + 1.2 * np.maximum(0, t - 0.35) ** 1.5) * h, color=C["fail"], lw=1.3)

    panels = [
        dict(num=1, title="A model adds up evidence", kind="jev", h=1.5, draw=mini_sig, draw_h=0.45,
             body="Each clue gets a weight. Add the weights, squash the total into 0–1 with the S-curve. That’s logistic regression."),
        dict(num=2, title="Score probabilities with log loss", kind="fail", h=1.5, draw=mini_loss, draw_h=0.45,
             body="Penalty = surprise. It rewards honest confidence and punishes confident mistakes very hard."),
        dict(num=3, title="Roll downhill", kind="neutral", h=1.5,
             body="Gradient descent: feel the slope of the loss, take a small step down, repeat. Step size matters."),
        dict(num=4, title="It works", kind="jev", h=1.5, draw=mini_curve, draw_h=0.45,
             body=f"Twenty lines of NumPy get within {rr['gap_pct']:.0%} of the best possible log loss on new alerts."),
        dict(num=5, title="Weights are readable", kind="data", h=1.5,
             body=f"Learned odds multipliers: +0.1 threat-intel score ×{rr['mult_ioc_tenth']:.2f}; approved IT tooling ×{rr['mult_known']:.2f}."),
        dict(num=6, title="Test on data it never saw", kind="fail", h=1.5, draw=mini_over, draw_h=0.45,
             body="Flexible models can memorise. Only held-out data tells learning from memorising."),
    ]
    return summary_page(CH, "Data, loss and gradient descent", panels,
                        footer="Next: the model says 0.8. Does 0.8 really mean 80%?")
