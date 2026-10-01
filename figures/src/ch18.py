"""Figures for Chapter 11: Testing Jev's calibration yourself."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import Choice

from jevkit import soc, calibration as cal, client as make_client
from jevkit.batch import score_alerts
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch18"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


@lru_cache(None)
def companies():
    K = soc.load()
    pk = score_alerts(K)
    H = soc.to_frame(soc.generate(n=8000, seed=21, logit_shift={r: -1.3 for r in soc.RULES}))
    ph = score_alerts(H)
    return K, pk, H, ph


@lru_cache(None)
def fixes():
    K, pk, H, ph = companies()
    y = H.malicious.values
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 300, replace=False)
    rest = np.setdiff1d(np.arange(len(y)), idx)
    est = float(y[idx].mean())
    adj = cal.prior_shift(ph, K.malicious.mean(), est)
    platt = cal.Platt().fit(ph[idx], y[idx])
    return idx, rest, est, adj, platt(ph)


@lru_cache(None)
def ece_by_n():
    K, pk, H, ph = companies()
    y = K.malicious.values
    rng = np.random.default_rng(1)
    ns = [100, 200, 400, 800, 1600, 3200, 6400]
    out = []
    for n in ns:
        vals = []
        for _ in range(200):
            i = rng.choice(len(pk), n, replace=False)
            vals.append(cal.ece(pk[i], y[i]))
        out.append(np.percentile(vals, [5, 50, 95]))
    return ns, np.array(out), cal.ece(pk, y)


@lru_cache(None)
def top_label():
    c = make_client()
    df = soc.load().sample(2500, random_state=4)
    conf, ok = [], []
    for a in df.itertuples():
        state = {k: getattr(a, k) for k in soc.FEATURE_FIELDS}
        state = {k: (v.item() if hasattr(v, "item") else v) for k, v in state.items()}
        r = c.system_one(state=state, questions={"k": Choice(criteria={x: None for x in soc.CATEGORIES})})
        conf.append(r.choices["k"].confidence)
        ok.append(r.choices["k"].choice == a.category)
    return np.array(conf), np.array(ok, float)


def record():
    K, pk, H, ph = companies()
    idx, rest, est, adj, plat = fixes()
    yH = H.malicious.values
    ns, band, full = ece_by_n()
    conf, ok = top_label()
    results(CH, kestrel_base=float(K.malicious.mean()), harbor_base=float(yH.mean()),
            kestrel_mean_p=float(pk.mean()), harbor_mean_p=float(ph.mean()),
            ece_kestrel=cal.ece(pk, K.malicious.values), ece_harbor=cal.ece(ph, yH),
            ece_prior_known=cal.ece(cal.prior_shift(ph, K.malicious.mean(), yH.mean()), yH),
            ece_prior_est=cal.ece(adj[rest], yH[rest]), ece_platt300=cal.ece(plat[rest], yH[rest]),
            est_base=est, ece_n100=float(band[0][1]), ece_n100_hi=float(band[0][2]), ece_n3200=float(band[5][1]),
            ece_full=float(full), top_conf_mean=float(conf.mean()), top_acc=float(ok.mean()),
            top_ece=cal.ece(conf, ok))


@figure(CH, "two-claims")
def two_claims():
    f, ax = draw.canvas("text", 1.6)
    draw.box(ax, 0.0, 0.75, 1.9, 0.6, "“Jev returns calibrated\nprobabilities”", kind="llm", size=6.8,
             sub="the vendor", subsize=5.8)
    draw.box(ax, 2.8, 0.75, 1.9, 0.6, "“Jev can’t be calibrated\nfor your data”", kind="fail", size=6.8,
             sub="a critic", subsize=5.8)
    draw.box(ax, 1.4, 0.05, 1.9, 0.45, "test it on your labels", kind="jev", size=7, weight="bold")
    draw.arrow(ax, (0.95, 0.75), (1.9, 0.5))
    draw.arrow(ax, (3.75, 0.75), (2.8, 0.5))
    return f


@figure(CH, "two-companies")
def two_companies():
    record()
    K, pk, H, ph = companies()
    f, axes = subplots(1, 2, width="text", height=2.3, sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, p, y, name in ((axes[0], pk, K.malicious.values, "Kestrel Logistics"),
                           (axes[1], ph, H.malicious.values, "Harbor Pharma")):
        clean(ax, "both")
        ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
        b = cal.reliability(p, y, n_bins=10, strategy="quantile")
        ax.plot(b.mean_pred, b.frac_pos, color=C["jev"], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6)
        ax.set_title(f"{name}: attacks {y.mean():.1%}, ECE {cal.ece(p, y):.3f}", fontsize=6.6)
        ax.set_xlabel("P(attack) from the same model")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share that were attacks")
    synthetic_tag(f)
    return f


@figure(CH, "harbor-fixes")
def harbor_fixes():
    K, pk, H, ph = companies()
    idx, rest, est, adj, plat = fixes()
    y = H.malicious.values
    f, ax = subplots(width="text", height=2.35)
    clean(ax, "both")
    ax.plot([0, 0.5], [0, 0.5], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    # each line has its own dash pattern and marker, named in the legend, so none depends on colour
    for p, col, lab, ls, mk in ((ph[rest], C["fail"], "as returned (solid, squares)", "-", "s"),
                                (adj[rest], C["data"], "odds adjusted for Harbor’s base rate (dashed, circles)", (0, (5, 2)), "o"),
                                (plat[rest], C["jev"], "Platt scaling on 300 labels (dotted, triangles)", ":", "^")):
        b = cal.reliability(p, y[rest], n_bins=8, strategy="quantile")
        ax.plot(b.mean_pred, b.frac_pos, color=col, lw=1.6, ls=ls, marker=mk, ms=3.8, mec="white", mew=0.5,
                label=f"{lab}: ECE {cal.ece(p, y[rest]):.3f}")
    ax.legend(loc="upper left", fontsize=6.0, handlelength=2.6)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("P(attack)")
    ax.set_ylabel("share that were attacks")
    synthetic_tag(f)
    return f


@figure(CH, "ece-noise")
def ece_noise():
    ns, band, full = ece_by_n()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    # the band keeps visible dotted edges, so it shows even where a photocopy loses the light fill
    ax.fill_between(ns, band[:, 0], band[:, 2], color=C["data"], alpha=0.15, lw=0)
    for k in (0, 2):
        ax.plot(ns, band[:, k], color=C["data"], lw=0.8, ls=":")
    ax.plot(ns, band[:, 1], color=C["data"], lw=1.6, marker="o", ms=3.5)
    ax.axhline(full, color=C["ink"], lw=0.9)
    ax.text(ns[-1], full - 0.004, f"ECE on all 20,000 alerts: {full:.3f}", ha="right", va="top", fontsize=6.3)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=6.2)
    ax.set_xlabel("labelled alerts you checked (log scale)")
    ax.set_ylabel("ECE you would measure")
    ax.text(ns[0] * 1.1, band[0, 2], "90% of random samples\nland in the band", fontsize=6.2, color=C["data"], va="top")
    synthetic_tag(f)
    return f


@figure(CH, "top-label")
def top_label_fig():
    conf, ok = top_label()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "both")
    ax.plot([0.3, 1], [0.3, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    b = cal.reliability(conf, ok, n_bins=8, strategy="quantile")
    ax.plot(b.mean_pred, b.frac_pos, color=C["jev"], lw=1.6, marker="o", ms=4, mec="white", mew=0.6)
    ax.set_xlabel("confidence of the top category")
    ax.set_ylabel("share where the top category\nwas right", fontsize=6.8)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    synthetic_tag(f)
    return f


@figure(CH, "protocol")
def protocol():
    f, ax = draw.canvas("text", 2.4)
    steps = [("Collect labels", "a few hundred to a few thousand\nrecent cases with known outcomes"),
             ("Plot, don’t just score", "reliability diagram + ECE with a range;\nlook near your thresholds"),
             ("Slice", "every group you act on\ndifferently: source, customer, language"),
             ("Fix", "base-rate adjustment if you know it;\nPlatt or temperature on labels"),
             ("Re-test", "on a schedule, and after any model\nversion or data change")]
    for i, (t, s) in enumerate(steps):
        y = 2.15 - i * 0.44
        draw.number_badge(ax, 0.12, y, i + 1, color=C["jev"])
        draw.text(ax, 0.32, y + 0.02, t, size=7, weight="bold")
        draw.text(ax, 1.95, y + 0.02, s, size=6.1, color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Both claims can be true", kind="neutral", h=1.45,
             body="A model can be calibrated where it was checked, and miscalibrated on data that differs. Calibration belongs to model and data together."),
        dict(num=2, title="Same model, two companies", kind="fail", h=1.45,
             body=(f"ECE {rr['ece_kestrel']:.3f} at Kestrel ({rr['kestrel_base']:.1%} attacks), {rr['ece_harbor']:.3f} at Harbor "
                   f"({rr['harbor_base']:.1%}). Harbor’s risks come out about twice too high.")),
        dict(num=3, title="Fix the base rate first", kind="data", h=1.45,
             body=(f"Adjusting the odds for Harbor’s base rate cut ECE to {rr['ece_prior_known']:.3f}; Platt scaling on 300 "
                   f"labels to {rr['ece_platt300']:.3f}.")),
        dict(num=4, title="Small samples lie", kind="fail", h=1.45,
             body=(f"With 100 labels, a model whose true ECE is {rr['ece_full']:.3f} typically measures about {rr['ece_n100']:.3f}. "
                   "Report a range, and get more labels.")),
        dict(num=5, title="Check choices too", kind="jev", h=1.45,
             body="For a choice: does the top label’s confidence match how often it’s right? Then check each label."),
        dict(num=6, title="Scores until proven otherwise", kind="jev", h=1.45,
             body="Treat new model outputs as scores. Once tested and fixed on your data, treat them as probabilities."),
    ]
    return summary_page(CH, "Testing Jev’s calibration yourself", panels,
                        footer="Next: what happens to an industry when a decision becomes almost free.")
