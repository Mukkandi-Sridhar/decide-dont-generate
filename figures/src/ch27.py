"""Figures for Chapter 20: Build your own System One model."""

import json
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, tinyjev as tj, calibration as cal, bakeoff
from jevkit.batch import score_alerts
from jevkit.figs import figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch27"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


@lru_cache(None)
def splits():
    A = soc.load()
    H, L = soc.history_and_live(A)
    ts = H.timestamp.astype("datetime64[ns]")
    cut = np.datetime64("2026-09-15")
    Tr, Va = H[ts < cut].reset_index(drop=True), H[ts >= cut].reset_index(drop=True)
    return H, L, Tr, Va, tj.features(Tr) + tj.targets(Tr), tj.features(Va) + tj.targets(Va)


@lru_cache(None)
def good():
    H, L, Tr, Va, ftr, fva = splits()
    p, hist = tj.train(*ftr, steps=1500, val=fva, every=50)
    T = tj.fit_temperatures(p, *fva)
    return p, hist, T


@lru_cache(None)
def over():
    H, L, Tr, Va, ftr, fva = splits()
    p, hist = tj.train(*ftr, steps=6000, l2=0.0, val=fva, d=64, every=100)
    T = tj.fit_temperatures(p, *fva)
    return p, hist, T


def scores():
    H, L, Tr, Va, ftr, fva = splits()
    y = L.malicious.to_numpy()
    p, _, T = good()
    po, _, To = over()
    raw = tj.predict(p, L)[0]
    tem = tj.predict(p, L, T)[0]
    ovr = tj.predict(po, L)[0]
    jev = score_alerts(L)
    lr = bakeoff.fit_logistic(H, H.malicious).predict_proba(soc.feature_matrix(L))[:, 1]
    return y, dict(raw=raw, temp=tem, over=ovr, jev=jev, logistic=lr)


@lru_cache(None)
def plugin_answer():
    """The chapter's "Plug it in" listing, run the same way, so the figure shows what the listing prints."""
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    H, L, Tr, Va, ftr, fva = splits()
    p, _, T = good()
    client = TypeSafeClient(api_key="mock", transport=tj.TinyJevTransport(p, T))
    a = L.iloc[118]
    state = {"alert": a.title, **{k: a[k].item() if hasattr(a[k], "item") else a[k] for k in soc.FEATURE_FIELDS}}
    r = client.system_one(state=state, questions={
        "attack": Noul(instructions="Is this alert a real attack?"),
        "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
        "severity": Score(criteria=soc.SEVERITY_LEVELS)})
    return round(r.nouls["attack"].noul, 4), r.choices["kind"].choice


def record():
    H, L, Tr, Va, ftr, fva = splits()
    y, s = scores()
    p, hist, T = good()
    po, histo, To = over()
    yl, cat, sev = tj.targets(L)
    pn, pc, ps = tj.predict(p, L, T)
    best = min(histo, key=lambda r: r["val"]["noul"])
    results(CH, **{f"{k}_auc": cal.summary(v, y)["auc"] for k, v in s.items()},
            **{f"{k}_ece": cal.ece(v, y) for k, v in s.items()},
            T_noul=T[0], T_choice=T[1], T_score=T[2], T_over=To[0],
            cat_acc=float((pc.argmax(1) == cat).mean()), sev_acc=float((ps.argmax(1) == sev).mean()),
            sev_within1=float((np.abs(ps.argmax(1) - sev) <= 1).mean()),
            train_noul=hist[-1]["train"]["noul"], val_noul=hist[-1]["val"]["noul"],
            over_train_noul=histo[-1]["train"]["noul"], over_val_noul=histo[-1]["val"]["noul"],
            over_best_step=best["step"], over_best_val=best["val"]["noul"],
            n_train=len(Tr), n_val=len(Va), n_params=int(sum(np.size(v) for v in p.values())),
            over_ece_temp=cal.ece(tj.predict(po, L, To)[0], y), plugin_noul=plugin_answer()[0])


@figure(CH, "architecture")
def architecture():
    f, ax = draw.canvas("text", 2.6)
    draw.box(ax, 0.0, 1.55, 1.0, 0.55, "10 numeric\nfields", kind="data", size=6.1)
    draw.box(ax, 0.0, 0.7, 1.0, 0.55, "rule name →\nembedding (8)", kind="data", size=6.1)
    draw.arrow(ax, (1.0, 1.82), (1.3, 1.45))
    draw.arrow(ax, (1.0, 0.97), (1.3, 1.25))
    draw.box(ax, 1.3, 0.95, 1.25, 0.8, "shared encoder\n2 layers × 32\n(ReLU)", kind="neutral", size=6.2,
             weight="semibold")
    heads = [("noul head", "sigmoid → P(attack)", 1.9), ("choice head", "softmax → P(category)", 1.15),
             ("score head", "ordinal → P(severity level)", 0.4)]
    for t, s, y in heads:
        draw.arrow(ax, (2.55, 1.35), (2.95, y + 0.2), head=3, color=C["muted"])
        draw.box(ax, 2.95, y, 1.75, 0.4, "", kind="jev")
        draw.text(ax, 3.05, y + 0.26, t, size=6.2, weight="bold")
        draw.text(ax, 3.05, y + 0.1, s, size=5.6, color=C["ink2"])
    draw.text(ax, 0.0, 0.25, "one forward pass · every head trained with log loss", size=6.0, color=C["ink2"],
              style="italic")
    return f


@figure(CH, "heads")
def heads_fig():
    f, (a1, a2, a3) = subplots(1, 3, width="text", height=1.8)
    z = np.linspace(-6, 6, 200)
    clean(a1, "both")
    a1.plot(z, 1 / (1 + np.exp(-z)), color=C["jev"], lw=2)
    a1.set_title("noul → P(yes)", fontsize=6.4, loc="left")
    a1.set_xlabel("score z", fontsize=6)
    a1.yaxis.set_major_formatter(PCT)
    clean(a2, "y")
    logits = np.array([2.1, 0.4, 0.1, -0.5, -1.2])
    pr = np.exp(logits) / np.exp(logits).sum()
    a2.bar(range(5), pr, color=C["jev"], width=0.6)
    a2.set_xticks(range(5))
    a2.set_xticklabels(["benign", "phish", "malw", "cred", "other"], fontsize=5.2, rotation=30)
    a2.set_title("choice → softmax", fontsize=6.4, loc="left")
    a2.yaxis.set_major_formatter(PCT)
    clean(a3, "both")
    th = np.array([-1.0, 0.8, 2.2])
    above = 1 / (1 + np.exp(-(z[:, None] - th[None, :])))
    cum = np.concatenate([np.ones((len(z), 1)), above, np.zeros((len(z), 1))], 1)
    probs = cum[:, :-1] - cum[:, 1:]
    cols = ["#B7DCC7", "#7FC19E", C["jev"], "#1F6B47"]
    a3.stackplot(z, probs.T, colors=cols)
    for k, t in enumerate(th):
        a3.axvline(t, color="white", lw=0.8)
    a3.set_xlim(-4, 5)
    a3.set_ylim(0, 1)
    a3.set_title("score → ordered cut points", fontsize=6.4, loc="left")
    a3.set_xlabel("score z", fontsize=6)
    a3.yaxis.set_major_formatter(PCT)
    for x, lab in ((-2.8, "info"), (-0.1, "low"), (1.5, "med"), (3.6, "high")):
        a3.text(x, 0.5, lab, fontsize=5.6, ha="center", color="white" if lab in ("med", "high") else C["ink"])
    f.subplots_adjust(wspace=0.45)
    return f


@figure(CH, "curves")
def curves():
    _, hist, _ = good()
    _, histo, _ = over()
    f, (a1, a2) = subplots(1, 2, width="text", height=2.0, sharey=True)
    for ax, hh, title in ((a1, hist, "TinyJev: 1,500 steps, small weight penalty"),
                          (a2, histo, "a bigger net, 6,000 steps, no penalty")):
        clean(ax, "both")
        st = [r["step"] for r in hh]
        ax.plot(st, [r["train"]["noul"] for r in hh], color=C["data"], lw=1.6, label="training weeks")
        ax.plot(st, [r["val"]["noul"] for r in hh], color=C["jev"] if ax is a1 else C["fail"], lw=1.6,
                label="held-out week")
        ax.set_title(title, fontsize=6.4, loc="left")
        ax.set_xlabel("training step", fontsize=6)
    best = min(histo, key=lambda r: r["val"]["noul"])
    a2.axvline(best["step"], color=C["ink"], lw=0.8, ls=(0, (3, 2)))
    a2.text(best["step"] + 150, 0.45, "should have\nstopped here", fontsize=5.6)
    a1.set_ylabel("log loss, noul head", fontsize=6)
    a1.legend(fontsize=5.8, frameon=False)
    a1.set_ylim(0, 0.56)
    f.subplots_adjust(wspace=0.12)
    synthetic_tag(f)
    return f


@figure(CH, "calibration")
def calibration():
    y, s = scores()
    f, axes = subplots(1, 3, width="text", height=1.9, sharey=True)
    for ax, key, title, col in zip(axes, ("raw", "temp", "over"),
                                   ("TinyJev as trained", "after temperature", "overtrained, no temperature"),
                                   (C["data"], C["jev"], C["fail"])):
        clean(ax, "both")
        b = cal.reliability(s[key], y, n_bins=15)
        ok = b.count >= 20
        ax.plot([0, 0.7], [0, 0.7], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
        ax.plot(b.mean_pred[ok], b.frac_pos[ok], color=col, lw=1.6, marker="o", ms=3)
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 0.7)
        ax.set_title(f"{title}\nECE {cal.ece(s[key], y):.3f}", fontsize=6.3, loc="left")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("what it said", fontsize=6)
    axes[0].set_ylabel("what happened", fontsize=6)
    f.subplots_adjust(wspace=0.15, top=0.78)
    synthetic_tag(f)
    return f


@figure(CH, "compare")
def compare():
    y, s = scores()
    f, (a1, a2) = subplots(1, 2, width="text", height=1.9)
    rows = [("logistic regression\n(3 weeks of labels)", "logistic", C["data"]), ("mock Jev\n(no labels)", "jev", C["jev"]),
            ("TinyJev\n(2 weeks + 1 to calibrate)", "temp", C["gold"]), ("the overtrained net\n(no temperature)", "over", C["fail"])]
    for ax, fn, title, lim in ((a1, lambda v: cal.summary(v, y)["auc"], "ranking (AUC)", (0.7, 0.92)),
                               (a2, lambda v: cal.ece(v, y), "calibration error (ECE)", (0, 0.08))):
        clean(ax, "x")
        for i, (lab, k, col) in enumerate(rows):
            v = fn(s[k])
            yy = len(rows) - 1 - i
            ax.plot([lim[0], v], [yy, yy], color=C["grid"], lw=1)
            ax.scatter([v], [yy], color=col, s=34, zorder=3)
            ax.text(v + (lim[1] - lim[0]) * 0.03, yy, f"{v:.3f}", fontsize=5.8, va="center")
        ax.set_xlim(lim[0], lim[1] * 1.08)
        ax.set_yticks(range(len(rows))[::-1])
        ax.set_yticklabels([r[0] for r in rows] if ax is a1 else [""] * 4, fontsize=5.8)
        ax.set_title(title, fontsize=6.6, loc="left")
    f.subplots_adjust(wspace=0.12, left=0.28)
    synthetic_tag(f)
    return f


@figure(CH, "plugin")
def plugin():
    f, ax = draw.canvas("text", 1.9)
    draw.box(ax, 0.0, 1.05, 1.5, 0.55, "client.system_one(…)\nthe same book code", kind="neutral", size=5.9,
             family="JetBrains Mono")
    draw.box(ax, 1.8, 1.05, 1.25, 0.55, "official SDK", kind="jev", size=6.2)
    draw.box(ax, 3.35, 1.05, 1.35, 0.55, "TinyJevTransport\nyour weights", kind="data", size=6.0, family="JetBrains Mono")
    draw.arrow(ax, (1.5, 1.32), (1.8, 1.32))
    draw.arrow(ax, (3.05, 1.32), (3.35, 1.32))
    noul, kind = plugin_answer()
    lines = ['"model": "tinyjev-your-own"', f'"attack": {{"noul": {noul}}}', f'"kind": {{"choice": "{kind}", …}}',
             '"severity": {"probabilities": {…}}']
    draw.box(ax, 1.8, 0.05, 2.9, 0.8, "", kind="plain")
    for i, l in enumerate(lines):
        draw.text(ax, 1.9, 0.7 - i * 0.17, l, size=5.4, family="JetBrains Mono")
    draw.arrow(ax, (4.0, 1.05), (3.6, 0.85), head=3, color=C["muted"])
    draw.text(ax, 0.0, 0.45, "Every chapter’s code\nruns against your model\nunchanged.", size=6.3, weight="semibold")
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Three heads, one pass", kind="jev", h=1.5,
             body="A shared encoder reads the fields once; a sigmoid, a softmax and an ordinal head answer the three types."),
        dict(num=2, title="Log loss everywhere", kind="neutral", h=1.5,
             body="Each head is trained with a proper scoring rule, so honest probabilities are what training rewards."),
        dict(num=3, title="And still overconfident", kind="fail", h=1.5,
             body=(f"On unseen weeks the network was a little too sure: temperatures of about {rr['T_noul']:.1f} "
                   "for each head fixed it.")),
        dict(num=4, title="Stop before it memorises", kind="fail", h=1.5,
             body="Train too long without a penalty and held-out loss climbs. A temperature can’t fully repair that."),
        dict(num=5, title="Respectable, not magic", kind="data", h=1.5,
             body=(f"TinyJev: AUC {rr['temp_auc']:.3f}, ECE {rr['temp_ece']:.3f}. Logistic regression and mock Jev "
                   "still rank a little better here.")),
        dict(num=6, title="Plug it in", kind="jev", h=1.5,
             body="A transport answers the SDK’s requests with your model. Every lab in the book runs against it."),
    ]
    return summary_page(CH, "Build your own System One model", panels,
                        footer="Next: wrap a decision model in a service you could run in production.")
