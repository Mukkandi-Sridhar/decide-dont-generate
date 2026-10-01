"""Figures for Chapter 6: Structured outputs and JSON mode."""

import json
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import soc, llm, calibration as cal
from jevkit.batch import score_alerts
from jevkit.figs import hatch_kw, figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch11"


@lru_cache(None)
def data():
    df = soc.load()
    df["p_jev_text"] = score_alerts(df, "text")
    tr, ca, te = soc.split(df)
    outcomes = {}
    for mode in (False, True):
        m = llm.MockLLM(json_mode=mode)
        c = {"valid, right shape": 0, "wrapped in chat text": 0, "broken JSON": 0, "valid, extra field": 0}
        for t in te.description:
            raw = m.structured(t)
            try:
                o = json.loads(raw)
                c["valid, extra field" if set(o) - {"verdict", "confidence", "rule", "threat_intel_score",
                                                    "after_hours", "related_alerts"} else "valid, right shape"] += 1
            except json.JSONDecodeError:
                c["wrapped in chat text" if raw.startswith("Sure") else "broken JSON"] += 1
        outcomes["JSON mode" if mode else "prompt only"] = c
    m = llm.MockLLM()
    verb, tok = [], []
    for t in te.description:
        r = m.classify(t)
        verb.append(r["confidence"] if r["label"] == "malicious" else 1 - r["confidence"])
        tok.append(m.label_token_prob(t))
    return dict(te=te, y=te.malicious.values, outcomes=outcomes, verb=np.array(verb), tok=np.array(tok),
                jev=te.p_jev_text.values, stated=[m.classify(t)["confidence"] for t in te.description])


def record():
    d = data()
    y = d["y"]
    s = {k: cal.summary(d[k], y) for k in ("verb", "tok", "jev")}
    st = np.array(d["stated"])
    results(CH, outcomes=d["outcomes"], n=len(y), verbal=s["verb"], token=s["tok"], jev_text=s["jev"],
            share_stated_ge_09=float((st >= 0.9).mean()),
            prompt_fail_share=sum(v for k, v in d["outcomes"]["prompt only"].items() if k != "valid, right shape") / len(y),
            prompt_fail_per_day=700 * sum(v for k, v in d["outcomes"]["prompt only"].items() if k != "valid, right shape") / len(y), distinct_stated=sorted(set(float(x) for x in st)), n_distinct_stated=len(set(float(x) for x in st)))


@figure(CH, "json-failures")
def json_failures():
    record()
    d = data()
    cats = ["wrapped in chat text", "broken JSON", "valid, extra field"]
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "y")
    x = np.arange(len(cats))
    for k, (mode, col) in enumerate((("prompt only", C["llm"]), ("JSON mode", C["slate"]))):
        vals = [d["outcomes"][mode][c] for c in cats]
        ax.bar(x + (k - 0.5) * 0.3, vals, width=0.28, label=mode + (" (striped)" if k else " (solid)"), **hatch_kw(k, col))
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * 0.3, v + 1.5, f"{v}", ha="center", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.8)
    n = sum(d["outcomes"]["prompt only"].values())
    ax.set_ylabel(f"responses (of {n:,})")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper center", fontsize=6.4, ncol=2)
    synthetic_tag(f, "SYNTHETIC \u00b7 llm-mock-synthetic, not a real model")
    return f


@figure(CH, "constrained")
def constrained():
    f, ax = draw.canvas("text", 1.9)
    so_far = '{"verdict": "'
    draw.text(ax, 0.0, 1.78, "Output so far:", size=6.6, weight="bold")
    draw.box(ax, 0.95, 1.63, 1.35, 0.28, so_far, kind="data", size=6.6, family="JetBrains Mono", align="left")
    draw.text(ax, 0.0, 1.35, "The model’s next-token scores, and what the schema allows:", size=6.6, weight="bold")
    toks = [("mal", 0.46, True), ("ben", 0.31, True), ("Sure", 0.09, False), ("I", 0.07, False), ("unk", 0.04, False),
            ("```", 0.03, False)]
    for i, (t, p, ok) in enumerate(toks):
        y = 1.1 - i * 0.17
        draw.text(ax, 0.1, y, t, size=6.4, family="JetBrains Mono", color=C["ink"] if ok else C["muted"])
        ax.add_patch(__import__("matplotlib").patches.Rectangle((0.7, y - 0.04), 1.6 * p, 0.08,
                                                               fc=C["llm"] if ok else C["neutral_t"], ec="none"))
        draw.text(ax, 0.75 + 1.6 * p, y, "allowed" if ok else "masked out", size=5.8,
                  color=C["jev"] if ok else C["fail"])
    draw.text(ax, 2.7, 1.05, "Schema: verdict must be\n\"malicious\" or \"benign\"", size=6.4, family="JetBrains Mono",
              color=C["ink2"], va="top")
    draw.text(ax, 2.7, 0.55, "Disallowed tokens get zero chance.\nThe output can’t break the format.\nIt can still be wrong.",
              size=6.4, color=C["ink"], va="top")
    return f


@figure(CH, "three-confidences")
def three_confidences():
    d = data()
    y = d["y"]
    f, ax = subplots(width="text", height=2.55)
    clean(ax, "both")
    ax.plot([0, 1], [0, 1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    # each line has its own dash pattern and marker, named in the legend, so none depends on colour
    specs = [("verb", C["gold"], "confidence the LLM wrote in its JSON", ":", "^", "dotted, triangles"),
             ("tok", C["llm"], "LLM token probability of “malicious”", (0, (5, 2)), "s", "dashed, squares"),
             ("jev", C["jev"], "jev-mock-synthetic noul, same raw text", "-", "o", "solid, circles")]
    for k, col, lab, ls, mk, how in specs:
        b = cal.reliability(d[k], y, n_bins=10, strategy="quantile")
        s = cal.summary(d[k], y)
        ax.plot(b.mean_pred, b.frac_pos, color=col, lw=1.6, ls=ls, marker=mk, ms=3.8, mec="white", mew=0.5,
                label=f"{lab} ({how})  AUC {s['auc']:.2f}, ECE {s['ece']:.3f}")
    ax.set_xlabel("Probability of “malicious”")
    ax.set_ylabel("Share that were attacks")
    ax.legend(loc="upper left", fontsize=5.9)
    synthetic_tag(f)
    return f


@figure(CH, "stated")
def stated():
    d = data()
    st = np.array(d["stated"])
    vals, counts = np.unique(st, return_counts=True)
    f, ax = subplots(width="text", height=1.7)
    clean(ax, "y")
    ax.bar([f"{v:.2f}" for v in vals], counts, color=C["gold"], width=0.55)
    for i, c in enumerate(counts):
        ax.text(i, c + 30, f"{c:,}", ha="center", fontsize=6.2)
    ax.set_xlabel("“confidence” value written in the JSON")
    ax.set_ylabel("alerts")
    ax.set_ylim(0, counts.max() * 1.18)
    synthetic_tag(f, "SYNTHETIC · llm-mock-synthetic, not a real model")
    return f


@figure(CH, "surface-forms")
def surface_forms():
    f, ax = draw.canvas("text", 1.75)
    draw.text(ax, 0.0, 1.62, "First-token probabilities for one alert (illustrative)", size=6.8, weight="bold")
    forms = [("malicious", 0.41, "fail"), ("Malicious", 0.12, "fail"), (" malicious", 0.07, "fail"),
             ("The", 0.14, "neutral"), ("This", 0.09, "neutral"), ("benign", 0.12, "data"), ("Benign", 0.05, "data")]
    for i, (t, p, kind) in enumerate(forms):
        y = 1.38 - i * 0.18
        draw.text(ax, 0.0, y, repr(t), size=6.2, family="JetBrains Mono")
        from jevkit.figs.style import KIND
        ax.add_patch(__import__("matplotlib").patches.Rectangle((1.05, y - 0.045), 2.2 * p, 0.09, fc=KIND[kind][0],
                                                               ec="none", alpha=0.85))
        draw.text(ax, 1.1 + 2.2 * p, y, f"{p:.2f}", size=5.8, color=C["ink2"])
    draw.text(ax, 2.55, 1.2, "“malicious” as a first token: 0.41", size=6.3, family="JetBrains Mono")
    draw.text(ax, 2.55, 1.0, "every way of saying malicious: 0.60", size=6.3, family="JetBrains Mono")
    draw.text(ax, 2.55, 0.8, "answer not started yet (“The…”): 0.23", size=6.3, family="JetBrains Mono")
    draw.text(ax, 2.55, 0.5, "The probability of a token isn’t the\nprobability of an answer.", size=6.4,
              weight="semibold", va="top")
    return f


@figure(CH, "typed-vs-text")
def typed_vs_text():
    f, ax = draw.canvas("text", 2.05)
    draw.text(ax, 0.0, 1.95, "Ask an LLM for JSON", size=6.8, weight="bold", color=C["llm"])
    draw.box(ax, 0.0, 0.25, 2.25, 1.55, "", kind="llm", radius=0.05)
    lines = ['{', '  "verdict": "malicious",', '  "confidence": 0.95,', '  "rule": "mfa_fatigue"', '}']
    for i, l in enumerate(lines):
        draw.text(ax, 0.1, 1.62 - i * 0.2, l, size=6.2, family="JetBrains Mono")
    draw.text(ax, 0.1, 0.4, "text you parse, a number it wrote", size=5.8, color=C["ink2"])
    draw.text(ax, 2.45, 1.95, "Ask a typed decision question", size=6.8, weight="bold", color=C["jev"])
    draw.box(ax, 2.45, 0.25, 2.25, 1.55, "", kind="jev", radius=0.05)
    lines = ['answers["attack"]:', '  type="noul"', '  noul=0.68', 'answers["kind"]:', '  choice="credential_abuse"',
             '  probabilities={...}']
    for i, l in enumerate(lines):
        draw.text(ax, 2.55, 1.62 - i * 0.2, l, size=6.2, family="JetBrains Mono")
    draw.text(ax, 2.55, 0.4, "typed data, a probability per option", size=5.8, color=C["ink2"])
    draw.text(ax, 4.7, 0.05, "illustrative values", size=5.4, ha="right", color=C["muted"])
    return f


@figure(CH, "summary")
def summary():
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    po = rr["outcomes"]["prompt only"]
    n = sum(po.values())
    panels = [
        dict(num=1, title="Asking for JSON mostly works", kind="llm", h=1.45,
             body=(f"Prompt only: {po['valid, right shape'] / n:.1%} came back valid and the right shape. The rest were "
                   "wrapped in chat, broken, or had extra fields.")),
        dict(num=2, title="Constrained decoding fixes the format", kind="jev", h=1.45,
             body="Tokens that would break the schema get zero chance. The format becomes guaranteed. The answer doesn’t."),
        dict(num=3, title="Validate anyway", kind="neutral", h=1.45,
             body="Parse into a strict type, reject unknown fields, retry or fail safe. Treat model output as untrusted input."),
        dict(num=4, title="A stated confidence is text too", kind="fail", h=1.45,
             body=(f"The mock wrote only {len(rr['distinct_stated'])} different values, {rr['share_stated_ge_09']:.0%} of them 0.9 "
                   f"or more. Ranking power: AUC {rr['verbal']['auc']:.2f}.")),
        dict(num=5, title="A token probability isn’t an answer probability", kind="fail", h=1.45,
             body=(f"It’s split across spellings and starts. In the mock it ranked better (AUC {rr['token']['auc']:.2f}) "
                   f"but was overconfident (ECE {rr['token']['ece']:.3f}).")),
        dict(num=6, title="Measure, whatever you use", kind="jev", h=1.45,
             body=(f"A model built to return probabilities scored ECE {rr['jev_text']['ece']:.3f} on the same text. Research "
                   "on real LLMs is mixed. Check yours.")),
    ]
    return summary_page(CH, "Structured outputs and JSON mode", panels,
                        footer="Next: giving an LLM your own documents to read, and remembering what it learned.")
