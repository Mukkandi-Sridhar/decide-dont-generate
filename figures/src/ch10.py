"""Figures for Chapter 6: How an LLM writes, one token at a time."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import llm, text, soc
from jevkit.figs import hatch_kw, figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch10"


@lru_cache(None)
def bpe():
    notes = [" ".join(n) for n in text.notes_corpus(5000)]
    return llm.BPE().fit(notes, merges=150)


@lru_cache(None)
def lm():
    return llm.TrigramLM().fit(text.notes_corpus(30000))


@lru_cache(None)
def variance():
    df = soc.load()
    m0 = llm.MockLLM()
    q = np.array([m0.label_token_prob(t) for t in df.description.iloc[:3000]])
    cand = np.where((q > 0.2) & (q < 0.8))[0]
    rng = np.random.default_rng(3)
    idx = rng.choice(cand, 40, replace=False)
    rows = []
    for i in idx:
        t = df.description.iloc[i]
        m = llm.MockLLM(temperature=0.7, seed=int(i))
        answers = [m.classify(t)["label"] for _ in range(20)]
        rows.append((sum(a == "malicious" for a in answers) / 20, int(df.malicious.iloc[i])))
    return rows


def record():
    b = bpe()
    rows = variance()
    split = sum(1 for s, _ in rows if 0 < s < 1)
    d = lm().next_distribution("the", "phishing")
    top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
    results(CH, tokens_ransomware=b.tokenize("ransomware"), tokens_date=b.tokenize("2026-09-14"),
            n_split=split, n_alerts=len(rows), top_next=[(w, round(p, 3)) for w, p in top],
            ttft=llm.LLM_TIME_TO_FIRST_TOKEN_S, tps=llm.LLM_TOKENS_PER_SECOND,
            lat_1=llm.MockLLM.simulated_latency_s(1), lat_40=llm.MockLLM.simulated_latency_s(40),
            lat_300=llm.MockLLM.simulated_latency_s(300), lat_1500=llm.MockLLM.simulated_latency_s(1500))


@figure(CH, "tokens")
def tokens():
    b = bpe()
    f, ax = draw.canvas("text", 1.95)
    items = ["the ransomware was quarantined", "passw0rd reset", "2026-09-14"]
    y = 1.65
    for s in items:
        toks = b.tokenize(s)
        draw.text(ax, 0.0, y, s, size=6.6, family="JetBrains Mono", color=C["ink2"])
        x = 1.95
        for i, t in enumerate(toks):
            w = 0.09 + 0.062 * len(t)
            if x + w > 4.7:
                break
            draw.box(ax, x, y - 0.11, w, 0.22, t, kind="llm" if i % 2 == 0 else "data", size=6.0, radius=0.03,
                     family="JetBrains Mono")
            x += w + 0.03
        draw.text(ax, 4.7, y - 0.22, f"{len(toks)} tokens", size=5.6, ha="right", color=C["muted"])
        y -= 0.55
    draw.text(ax, 0.0, 0.12, "A tokenizer learned from our synthetic notes. Common chunks become one token; unfamiliar words, typos",
              size=6.0, color=C["ink2"])
    draw.text(ax, 0.0, -0.04, "and numbers get chopped into many small ones.", size=6.0, color=C["ink2"])
    ax.set_ylim(-0.1, 1.95)
    return f


@figure(CH, "loop")
def loop():
    f, ax = draw.canvas("text", 2.25)
    ctx = ["the", "phishing"]
    steps = [("email", [("email", 0.44), ("attempt", 0.42), ("was", 0.02)]),
             ("asked", [("asked", 0.9), ("was", 0.03), ("the", 0.02)]),
             ("the", [("the", 0.95), ("for", 0.02), ("a", 0.01)])]
    y0 = 1.85
    for k, (pick, dist) in enumerate(steps):
        y = y0 - k * 0.62
        x = 0.0
        for w in ctx:
            wd = 0.1 + 0.06 * len(w)
            draw.box(ax, x, y - 0.12, wd, 0.24, w, kind="data", size=5.8, radius=0.03, family="JetBrains Mono")
            x += wd + 0.03
        draw.arrow(ax, (x + 0.02, y), (x + 0.3, y), head=3)
        draw.box(ax, x + 0.3, y - 0.14, 0.5, 0.28, "model", kind="llm", size=6.2, weight="bold")
        xd = x + 0.95
        for i, (w, p) in enumerate(dist):
            draw.text(ax, xd, y + 0.14 - i * 0.12, w, size=5.4, family="JetBrains Mono", va="center")
            ax.add_patch(__import__("matplotlib").patches.Rectangle((xd + 0.45, y + 0.11 - i * 0.12), 0.5 * p, 0.07,
                                                                   fc=C["llm"] if w == pick else C["llm_t"], ec="none"))
        draw.text(ax, xd + 1.05, y, f"pick “{pick}”, append, repeat", size=6.0, color=C["ink2"])
        ctx = ctx + [pick]
    return f


@figure(CH, "temperature")
def temperature():
    d = lm().next_distribution("the", "phishing")
    top = sorted(d.items(), key=lambda kv: -kv[1])[:8]
    words = [w for w, _ in top]
    logp = np.log([p for _, p in top])
    f, axes = subplots(1, 3, width="text", height=1.9, sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, T in zip(axes, (0.3, 1.0, 2.0)):
        clean(ax, "y")
        z = logp / T
        pr = np.exp(z - z.max())
        pr /= pr.sum()
        ax.bar(range(len(words)), pr, color=C["llm"], width=0.6)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=60, ha="right", fontsize=5.6, family="JetBrains Mono")
        ax.set_title(f"temperature {T:g}", fontsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("chance of being picked")
    axes[0].text(0.0, -0.62, "next word after “the phishing” (top 8), from the trigram model",
                 transform=axes[0].transAxes, fontsize=5.8, color=C["muted"])
    return f


@figure(CH, "latency")
def latency():
    record()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "x")
    cases = [("one-word label", 1), ("small JSON object", 40), ("label + explanation", 300),
             ("step-by-step reasoning", 1500)]
    ys = np.arange(len(cases))[::-1]
    for y, (lab, n) in zip(ys, cases):
        t = llm.MockLLM.simulated_latency_s(n)
        ax.barh(y, llm.LLM_TIME_TO_FIRST_TOKEN_S, color=C["llm_t"], height=0.5, hatch="////", hatchcolor=C["llm"],
                edgecolor=C["llm"], lw=0.6)
        ax.barh(y, t - llm.LLM_TIME_TO_FIRST_TOKEN_S, left=llm.LLM_TIME_TO_FIRST_TOKEN_S, color=C["llm"], height=0.5)
        ax.text(t + 0.3, y, f"{t:.1f} s  ({n:,} token{'s' if n != 1 else ''})", va="center", fontsize=6.4, color=C["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in cases], fontsize=6.6)
    ax.set_xlabel("seconds per answer")
    ax.set_xlim(0, 34)
    ax.text(0.99, -0.34, f"illustrative: {llm.LLM_TIME_TO_FIRST_TOKEN_S:.2f} s before the first token (light, striped), then "
            f"{llm.LLM_TOKENS_PER_SECOND:.0f} tokens per second (dark, solid)", transform=ax.transAxes, fontsize=5.6,
            color=C["muted"], ha="right")
    return f


@figure(CH, "variance")
def variance_fig():
    record()
    rows = sorted(variance(), key=lambda r: r[0])
    f, ax = subplots(width="text", height=2.0)
    clean(ax, "y")
    x = np.arange(len(rows))
    share = [r[0] for r in rows]
    # real threats solid, harmless alerts striped: the two kinds differ without colour
    from matplotlib.patches import Patch
    thr = np.array([bool(r[1]) for r in rows])
    ax.bar(x[thr], np.array(share)[thr], color=C["fail"], width=0.7)
    ax.bar(x[~thr], np.array(share)[~thr], width=0.7, **hatch_kw(1, C["neutral_t"]))
    ax.bar(x[~thr], np.array(share)[~thr], width=0.7, fill=False, edgecolor=C["slate"], lw=0.6)
    ax.set_xticks([])
    ax.set_xlabel("40 borderline alerts, each asked 20 times at temperature 0.7 (sorted)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of answers:\n“malicious”", fontsize=6.6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(handles=[Patch(color=C["fail"], label="really a threat (solid)"),
                       Patch(fc=C["neutral_t"], ec=C["slate"], hatch="////", lw=0.6, label="really harmless (striped)")],
              loc="upper left", fontsize=6.2, handlelength=1.4)
    synthetic_tag(f, "SYNTHETIC · llm-mock-synthetic, not a real model")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Text becomes tokens", kind="llm", h=1.45,
             body=f"Chunks, not words. “ransomware” became {len(rr['tokens_ransomware'])} tokens in our toy tokenizer. You pay per token."),
        dict(num=2, title="One token at a time", kind="llm", h=1.45,
             body="Read everything so far, get a probability for every possible next token, pick one, append it, repeat."),
        dict(num=3, title="Temperature sets the randomness", kind="llm", h=1.45,
             body="Divide the scores by T before the softmax. Low T: predictable. High T: creative, then nonsense."),
        dict(num=4, title="Length is time", kind="neutral", h=1.45,
             body=f"Each token waits for the last. With our illustrative numbers: 1 token {rr['lat_1']:.1f} s, 1,500 tokens {rr['lat_1500']:.0f} s."),
        dict(num=5, title="Same question, different answers", kind="fail", h=1.45,
             body=f"Sampling means repeat calls can disagree: {rr['n_split']} of {rr['n_alerts']} borderline alerts got both answers across 20 asks."),
        dict(num=6, title="Fluent isn’t true", kind="fail", h=1.45,
             body="The loop optimises for a plausible next token, not a correct answer. Grounding and checking come from outside."),
    ]
    return summary_page(CH, "How an LLM writes, one token at a time", panels,
                        footer="Next: what happens when you ask an LLM for a decision, not an essay.")
