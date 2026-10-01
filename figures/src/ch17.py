"""Figures for Chapter 10: The type system: choice, score, noul."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import Choice, Noul, Score

from jevkit import soc, client as make_client
from jevkit.figs import hatch_kw, figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch17"
THREATS = [k for k in soc.CATEGORIES if k not in ("benign", "policy_violation")]
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


@lru_cache(None)
def forced():
    c = make_client()
    df = soc.load()
    harm = df[df.malicious == 0].head(400)
    full, forced_ = [], []
    for t in harm.description:
        r1 = c.system_one(state=t, questions={"k": Choice(criteria={k: None for k in soc.CATEGORIES})})
        r2 = c.system_one(state=t, questions={"k": Choice(criteria={k: None for k in THREATS})})
        full.append((r1.choices["k"].choice, r1.choices["k"].confidence))
        forced_.append((r2.choices["k"].choice, r2.choices["k"].confidence))
    return full, forced_


@lru_cache(None)
def severity(ids=(118, 5, 318)):
    c = make_client()
    df = soc.load()
    out = []
    for i in ids:
        r = c.system_one(state=df.description[i], questions={
            "s": Score(instructions="How severe is this?", criteria=soc.SEVERITY_LEVELS),
            "a": Noul(instructions="Is this a real attack?")})
        pr = r.scores["s"].probabilities
        out.append((i, r.nouls["a"].noul, r.scores["s"].score, [pr[k] for k in range(4)]))
    return out


@lru_cache(None)
def kinds(ids=(3, 99, 270, 57)):
    c = make_client()
    df = soc.load()
    out = []
    for i in ids:
        r = c.system_one(state=df.description[i], questions={"k": Choice(criteria={k: None for k in soc.CATEGORIES})})
        out.append((i, df.rule[i], r.choices["k"].probabilities))
    return out


def record():
    full, fo = forced()
    sev = severity()
    results(CH, full_benign_share=float(np.mean([ch == "benign" for ch, _ in full])),
            forced_threat_share=1.0, forced_conf_median=float(np.median([cf for _, cf in fo])),
            forced_conf_mean=float(np.mean([cf for _, cf in fo])), n_harmless=len(full),
            sev_example={"id": sev[0][0], "p_attack": sev[0][1], "expected": sev[0][2], "probs": sev[0][3],
                         "p_high": sev[0][3][2] + sev[0][3][3]})


@figure(CH, "three-types")
def three_types():
    f, ax = draw.canvas("text", 2.45)
    cards = [("noul", "a yes/no question", 'Noul(instructions=\n  "Is this a real attack?")', "noul = 0.68", "one probability: of “yes”"),
             ("choice", "pick one label", 'Choice(criteria={\n  "phishing": …,\n  "malware": …,\n  "benign": …})',
              "choice = \"benign\"\nprobabilities = {…}", "a probability per label;\nthey add up to 1"),
             ("score", "a level on a scale", 'Score(criteria=[\n  "Informational",\n  "Low", "Medium",\n  "High"])',
              "score = 1.05\nprobabilities = {0: …}", "a probability per level,\nplus the average level")]
    for i, (name, what, q, a, note) in enumerate(cards):
        x = i * 1.6
        draw.box(ax, x, 2.02, 1.45, 0.32, name, kind="jev", size=7.4, weight="bold", family="JetBrains Mono")
        draw.text(ax, x + 0.72, 1.93, what, size=6.0, ha="center", va="top", color=C["ink2"], style="italic")
        draw.box(ax, x, 0.98, 1.45, 0.72, "", kind="data", radius=0.04)
        draw.text(ax, x + 0.06, 1.62, q, size=5.3, family="JetBrains Mono", va="top")
        draw.box(ax, x, 0.35, 1.45, 0.52, "", kind="jev", radius=0.04)
        draw.text(ax, x + 0.06, 0.8, a, size=5.4, family="JetBrains Mono", va="top")
        draw.text(ax, x + 0.72, 0.27, note, size=5.7, ha="center", va="top", color=C["ink2"])
    return f


@figure(CH, "choice-dists")
def choice_dists():
    ks = kinds()
    f, axes = subplots(1, len(ks), width="text", height=2.0, sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, (i, rule, pr) in zip(axes, ks):
        clean(ax, "x")
        labs = list(pr)
        vals = [pr[l] for l in labs]
        top = int(np.argmax(vals))
        # the top label is solid; the rest are light and striped, so the top one stands out without colour
        ys = list(range(len(labs))[::-1])
        for j in range(len(labs)):
            if j == top:
                ax.barh(ys[j], vals[j], color=C["jev"], height=0.6)
            else:
                ax.barh(ys[j], vals[j], height=0.6, **hatch_kw(1, C["neutral_t"]))
        ax.set_yticks(range(len(labs))[::-1])
        ax.set_yticklabels([l.replace("_", " ") for l in labs], fontsize=6.0)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"], fontsize=5.8)
        short = {"unsigned_temp_binary": "temp binary", "mfa_fatigue": "MFA fatigue", "lsass_access": "LSASS read",
                 "public_bucket": "public bucket"}.get(rule, rule.replace("_", " "))
        ax.set_title(short, fontsize=6.3)
    synthetic_tag(f)
    return f


@figure(CH, "forced")
def forced_fig():
    record()
    full, fo = forced()
    f, ax = subplots(width="text", height=2.0)
    clean(ax, "y")
    bins = np.linspace(0.2, 1, 17)
    ax.hist([cf for _, cf in fo], bins=bins, color=C["fail"], alpha=0.85,
            label="options: threat categories only, no “benign” (solid)")
    ax.hist([cf for ch, cf in full if ch != "benign"], bins=bins, histtype="stepfilled", fc=C["jev_t"], ec=C["jev"],
            lw=1.0, hatch="////", hatchcolor=C["jev"],
            label="options include “benign”, only the few labelled as threats shown (striped)")
    ax.set_xlabel("confidence of the top label, harmless alerts only")
    ax.set_ylabel("alerts")
    ax.legend(loc="upper left", fontsize=6.0)
    synthetic_tag(f)
    return f


@figure(CH, "severity")
def severity_fig():
    sev = severity()
    f, axes = subplots(1, len(sev), width="text", height=1.9, sharey=True, gridspec_kw=dict(wspace=0.12))
    labs = ["info", "low", "med", "high"]
    for ax, (i, pa, sc, pr) in zip(axes, sev):
        clean(ax, "y")
        ax.bar(range(4), pr, color=[C["data"], C["data"], C["fail"], C["fail"]], width=0.6)
        ax.axvline(sc, color=C["ink"], lw=1)
        ax.text(sc + 0.08, 0.72, f"average\n{sc:.2f}", fontsize=5.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labs, fontsize=6)
        ax.set_title(f"P(attack) = {pa:.2f}", fontsize=6.6)
        ax.set_ylim(0, 0.85)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("probability")
    synthetic_tag(f)
    return f


@figure(CH, "expected-cost")
def expected_cost():
    f, ax = draw.canvas("text", 2.15)
    labels = ["phishing", "malware", "credential abuse", "benign"]
    probs = [0.15, 0.1, 0.35, 0.4]
    actions = ["email team", "endpoint team", "identity team", "close"]
    cost = np.array([[0, 300, 300, 50], [300, 0, 300, 50], [300, 300, 0, 50], [2000, 3000, 5000, 0]])
    draw.text(ax, 0.0, 2.02, "Probabilities from a choice question", size=6.6, weight="bold")
    for i, (l, p) in enumerate(zip(labels, probs)):
        y = 1.72 - i * 0.2
        draw.text(ax, 0.0, y, l, size=6.2)
        draw.gauge(ax, 0.95, y - 0.04, 0.6, p, kind="jev", h=0.08, size=6)
    draw.text(ax, 2.2, 2.02, "Expected cost of each action", size=6.6, weight="bold")
    exp = cost @ np.array(probs)
    best = int(np.argmin(exp))
    for i, (a, e) in enumerate(zip(actions, exp)):
        y = 1.72 - i * 0.2
        draw.text(ax, 2.2, y, a, size=6.2, weight="bold" if i == best else "normal")
        draw.text(ax, 3.6, y, f"${e:,.0f}", size=6.2, family="JetBrains Mono", weight="bold" if i == best else "normal",
                  color=C["jev"] if i == best else C["ink"])
    draw.text(ax, 4.7, 0.2, "illustrative numbers", size=5.4, ha="right", color=C["muted"])
    return f


@figure(CH, "maths")
def maths():
    f, ax = draw.canvas("text", 1.45)
    items = [("noul", "the S-curve of Chapter 2", "one number → P(yes)"),
             ("choice", "the softmax of Chapter 5", "one number per label → P per label"),
             ("score", "an ordered choice", "P per level → average and tails")]
    for i, (t, m, s) in enumerate(items):
        x = i * 1.6
        draw.box(ax, x, 0.72, 1.45, 0.42, t, kind="jev", size=7, weight="bold", family="JetBrains Mono")
        draw.text(ax, x + 0.72, 0.6, m, size=6.1, ha="center", va="top")
        draw.text(ax, x + 0.72, 0.38, s, size=5.8, ha="center", va="top", color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    ex = rr["sev_example"]
    panels = [
        dict(num=1, title="Three types cover a lot", kind="jev", h=1.45,
             body="Yes/no (noul), pick one (choice), a level on a scale (score). Most small decisions are one of these, or a few together."),
        dict(num=2, title="Say what yes means", kind="neutral", h=1.45,
             body="One fact per noul. Describe the yes and no cases. Avoid double negatives and bundled questions."),
        dict(num=3, title="Give every case a label", kind="fail", h=1.45,
             body=(f"Remove “benign” and harmless alerts still get a threat label, with median confidence "
                   f"{rr['forced_conf_median']:.2f}. Probabilities sum to 1 over the options you gave.")),
        dict(num=4, title="Use the whole distribution", kind="jev", h=1.45,
             body="Pick the action with the lowest expected cost, not the label with the highest probability."),
        dict(num=5, title="Averages hide tails", kind="fail", h=1.45,
             body=(f"One alert’s average severity was {ex['expected']:.2f} (“low”), with a "
                   f"{ex['p_high']:.0%} chance of medium or high. Act on P(level ≥ 2), not the mean.")),
        dict(num=6, title="Ask several at once", kind="neutral", h=1.45,
             body="One call, one state, many typed questions, and a typed response model in your code."),
    ]
    return summary_page(CH, "The type system: choice, score, noul", panels,
                        footer="Next: testing whether Jev’s probabilities deserve to be trusted.")
