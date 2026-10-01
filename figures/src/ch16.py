"""Figures for Chapter 9: Inside Jev: what we know and what we don't."""

import numpy as np

from jevkit import llm
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page

CH = "ch16"
JEV_PRICE_IN = 0.042            # $ per million input tokens, vendor-reported (OpenRouter listing)
DOOM_DOLLARS_PER_HOUR = 7.0     # vendor-reported demo figure
DOOM_DECISIONS_PER_S = 10


def record():
    tokens_in = 500
    jev_cost = tokens_in * JEV_PRICE_IN / 1e6
    llm_cost = llm.MockLLM.simulated_cost_usd(tokens_in, 40)
    decisions_per_hour = DOOM_DECISIONS_PER_S * 3600
    per_decision = DOOM_DOLLARS_PER_HOUR / decisions_per_hour
    implied_tokens = per_decision / (JEV_PRICE_IN / 1e6)
    results(CH, tokens_in=tokens_in, jev_cost_per_decision=jev_cost, jev_cost_per_million=jev_cost * 1e6,
            llm_cost_per_decision=llm_cost, llm_cost_per_million=llm_cost * 1e6, ratio=llm_cost / jev_cost,
            doom_decisions_per_hour=decisions_per_hour, doom_cost_per_decision=per_decision,
            doom_implied_tokens=implied_tokens, llm_price_in=llm.LLM_PRICE_IN_PER_M, llm_price_out=llm.LLM_PRICE_OUT_PER_M)


@figure(CH, "ledger")
def ledger():
    f, ax = draw.canvas("text", 3.45)
    cols = [("VERIFIED", "read in the public SDK", "jev", [
                "POST /v1/systemone", "state + model + questions", "three types: choice,\nscore, noul",
                "a probability for every\noption; default model\n“jev-latest”", "output tokens currently\nfree (SDK docs)"]),
            ("VENDOR-REPORTED", "claimed; not independently\nchecked here", "llm", [
                "“parallel sampler”: all\noutputs in one pass", "trained with RLCD:\nreinforcement learning\nfor calibrated decisions",
                "70–500 ms latency", "$0.042 per million\ninput tokens", "two orders of magnitude\nfaster than LLMs"]),
            ("UNKNOWN", "not published", "fail", [
                "model size and layers", "training data", "how RLCD’s reward\nis defined",
                "calibration on\nyour data", "failure modes on\nadversarial input"])]
    for i, (t, sub, kind, items) in enumerate(cols):
        x = i * 1.6
        draw.box(ax, x, 3.0, 1.45, 0.36, t, kind=kind, size=7, weight="bold")
        draw.text(ax, x + 0.72, 2.91, sub, size=5.6, ha="center", va="top", color=C["ink2"], style="italic")
        for j, it in enumerate(items):
            draw.text(ax, x + 0.05, 2.5 - j * 0.5, "• " + it, size=6.1, va="top")
    return f


@figure(CH, "anatomy")
def anatomy():
    f, ax = draw.canvas("text", 2.9)
    req = ['POST /v1/systemone', '{', '  "state": "I was charged twice…",', '  "model": "jev-latest",', '  "questions": {',
           '    "topic":  {"type": "choice", "criteria": {…}},', '    "urgent": {"type": "noul", "instructions": …},',
           '    "tone":   {"type": "score", "criteria": […]}', '  }', '}']
    res = ['{', '  "model": "jev-mock-synthetic",', '  "answers": {', '    "topic":  {"choice": "billing",',
           '               "probabilities": {…}},', '    "urgent": {"noul": 0.61},', '    "tone":   {"score": 0.22,',
           '               "probabilities": {…}}', '  },', '  "usage": {"input_tokens": 89, …}', '}']
    draw.box(ax, 0.0, 0.05, 2.3, 2.7, "", kind="data", radius=0.05)
    draw.box(ax, 2.4, 0.05, 2.3, 2.7, "", kind="jev", radius=0.05)
    draw.text(ax, 0.08, 2.62, "request (what the SDK sends)", size=6.4, weight="bold", color=C["data"])
    draw.text(ax, 2.48, 2.62, "response", size=6.4, weight="bold", color=C["jev"])
    for i, l in enumerate(req):
        draw.text(ax, 0.08, 2.38 - i * 0.22, l, size=5.5, family="JetBrains Mono")
    for i, l in enumerate(res):
        draw.text(ax, 2.48, 2.38 - i * 0.2, l, size=5.5, family="JetBrains Mono")
    return f


@figure(CH, "latency")
def latency():
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "x")
    rows = [("Jev, vendor-reported range", 0.07, 0.5, C["jev"]),
            ("LLM, small JSON (illustrative)", llm.MockLLM.simulated_latency_s(40), None, C["llm"]),
            ("LLM, reasoning (illustrative)", llm.MockLLM.simulated_latency_s(1500), None, C["llm"])]
    for i, (lab, a, b, col) in enumerate(rows):
        y = len(rows) - 1 - i
        if b:
            ax.plot([a, b], [y, y], color=col, lw=6, solid_capstyle="butt")
            ax.text(b * 1.3, y, f"{a * 1000:.0f}–{b * 1000:.0f} ms", va="center", fontsize=6.4)
        else:
            ax.scatter([a], [y], color=col, s=40, zorder=3)
            ax.text(a * 1.3, y, f"{a:.1f} s", va="center", fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1 s", "1 s", "10 s", "100 s"])
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("time per decision (log scale)")
    return f


@figure(CH, "cost")
def cost():
    record()
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    f, ax = subplots(width="text", height=1.6)
    clean(ax, "x")
    vals = [("Jev, vendor-reported price", rr["jev_cost_per_million"], C["jev"]),
            ("LLM, illustrative price", rr["llm_cost_per_million"], C["llm"])]
    for i, (lab, v, col) in enumerate(vals):
        ax.scatter([v], [1 - i], color=col, s=60, zorder=3)
        ax.text(v * 1.25, 1 - i, f"${v:,.0f}", va="center", fontsize=6.6)
    ax.set_xscale("log")
    ax.set_xlim(5, 5000)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([v[0] for v in vals], fontsize=6.6)
    ax.set_xlabel(f"cost per million decisions ({rr['tokens_in']} input tokens each; log scale)")
    return f


@figure(CH, "doom-check")
def doom_check():
    record()
    import json
    from jevkit.figs import ROOT
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    f, ax = draw.canvas("text", 1.5)
    steps = [("10 decisions a second", "vendor demo"), (f"= {rr['doom_decisions_per_hour']:,} an hour", "arithmetic"),
             ("$7 an hour", "vendor demo"), (f"= ${rr['doom_cost_per_decision']:.5f} each", "arithmetic"),
             (f"≈ {rr['doom_implied_tokens']:,.0f} input tokens", "at $0.042 per million")]
    for i, (a, b) in enumerate(steps):
        x = i * 0.95
        draw.box(ax, x, 0.6, 0.85, 0.5, a, kind="llm" if b.startswith("vendor") else "neutral", size=6.1, wrapw=16)
        draw.text(ax, x + 0.42, 0.48, b, size=5.6, ha="center", va="top", color=C["ink2"])
        if i < 4:
            draw.arrow(ax, (x + 0.85, 0.85), (x + 0.95, 0.85), head=3)
    return f


@figure(CH, "guess")
def guess():
    f, ax = draw.canvas("text", 2.2)
    draw.text(ax, 0.0, 2.08, "ONE POSSIBLE SHAPE · the author’s guess, not TypeSafe’s description", size=6.4,
              weight="bold", color=C["fail"])
    draw.box(ax, 0.0, 1.15, 1.05, 0.55, "state\n(text or JSON)", kind="data", size=6.4)
    draw.box(ax, 0.0, 0.45, 1.05, 0.55, "typed questions\n+ options", kind="data", size=6.4)
    draw.arrow(ax, (1.05, 1.42), (1.4, 1.1))
    draw.arrow(ax, (1.05, 0.72), (1.4, 1.0))
    draw.box(ax, 1.4, 0.7, 1.35, 0.72, "a reader:\nsees everything\nat once", kind="neutral", size=6.4, weight="bold")
    heads = [("choice head", "softmax over options"), ("score head", "a probability per level"), ("noul head", "one probability")]
    for i, (h, s) in enumerate(heads):
        y = 1.45 - i * 0.5
        draw.arrow(ax, (2.75, 1.06), (3.05, y + 0.18), head=3, color=C["muted"])
        draw.box(ax, 3.05, y, 1.6, 0.36, h, kind="jev", size=6.3, weight="bold")
        draw.text(ax, 3.85, y - 0.04, s, size=5.6, ha="center", va="top", color=C["ink2"])
    draw.text(ax, 0.0, 0.12, "One forward pass, no token-by-token writing. Chapter 20 builds a small model of exactly this shape.",
              size=6.2, color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Sort every claim", kind="neutral", h=1.45,
             body="Verified, vendor-reported, unknown. Most of what’s written about new models mixes the three."),
        dict(num=2, title="The interface is public", kind="jev", h=1.45,
             body="State plus typed questions in; a probability for every option out. You can read it in the SDK."),
        dict(num=3, title="The speed claim has a mechanism", kind="llm", h=1.45,
             body="All answers in one pass instead of token by token. Vendor-reported: 70–500 ms per request."),
        dict(num=4, title="The price is the story", kind="llm", h=1.45,
             body=(f"At the listed price, a million 500-token decisions cost about ${rr['jev_cost_per_million']:.0f} "
                   f"(vendor-reported), against ${rr['llm_cost_per_million']:,.0f} for our illustrative LLM.")),
        dict(num=5, title="Check claims against each other", kind="neutral", h=1.45,
             body=f"The Doom demo’s numbers imply about {rr['doom_implied_tokens']:,.0f} tokens per decision. Plausible, and checkable."),
        dict(num=6, title="Calibration can’t be taken on trust", kind="fail", h=1.45,
             body="“Calibrated” is a claim about data. Yours isn’t theirs. Chapter 11 tests it."),
    ]
    return summary_page(CH, "Inside Jev: what we know and what we don’t", panels,
                        footer="Next: the three question types, one at a time.")
