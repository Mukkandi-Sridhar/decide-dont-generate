"""Figures for Chapter 7: RAG and memory."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import kb
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page

CH = "ch12"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def recall_at(mode, size, ks=(1, 2, 3, 4, 5)):
    ch = kb.corpus(size)
    texts = [t for _, t in ch]
    r = kb.Retriever(texts, mode)
    S = r.scores([q for _, q in kb.QUESTIONS])
    out = []
    for k in ks:
        hits = 0
        for i, (d, q) in enumerate(kb.QUESTIONS):
            top = np.argsort(-S[i])[:k]
            hits += any(kb.ANSWER_KEY[q] in texts[j] for j in top)
        out.append(hits / len(kb.QUESTIONS))
    return out


@lru_cache(None)
def top_scores():
    ch = kb.corpus(30)
    r = kb.Retriever([t for _, t in ch], "char")
    a = r.scores([q for _, q in kb.QUESTIONS]).max(1)
    u = r.scores(kb.UNANSWERABLE).max(1)
    return a, u


def record():
    rec = {m: recall_at(m, 30) for m in ("word", "char", "hybrid")}
    sizes = [0, 15, 30, 60, 120]
    by_size = [recall_at("char", s, ks=(3,))[0] for s in sizes]
    a, u = top_scores()
    t = float((np.sort(u)[-2] + np.sort(u)[-1]) / 2)
    results(CH, recall=rec, sizes=sizes, recall3_by_size=by_size, n_questions=len(kb.QUESTIONS),
            n_unanswerable=len(kb.UNANSWERABLE), n_chunks=len(kb.corpus(30)), n_docs=len(kb.DOCS),
            abstain_t=t, answered_share=float((a >= t).mean()), unans_answered=float((u >= t).mean()))


@figure(CH, "pipeline")
def pipeline():
    f, ax = draw.canvas("text", 2.2)
    draw.box(ax, 0.0, 1.55, 1.1, 0.46, "question", kind="data", size=7, weight="bold", sub="“is svc-backup allowed…?”", subsize=5.4)
    draw.arrow(ax, (1.1, 1.78), (1.4, 1.78))
    draw.box(ax, 1.4, 1.55, 1.05, 0.46, "retrieve", kind="neutral", size=7, weight="bold", sub="top 3 chunks", subsize=5.6)
    from jevkit.figs.draw import cylinder
    cylinder(ax, 1.55, 0.35, 0.75, 0.75, kind="data", label="policies,\nrunbooks", size=6.2)
    draw.arrow(ax, (1.92, 1.1), (1.92, 1.55), color=C["muted"])
    draw.arrow(ax, (2.45, 1.78), (2.75, 1.78))
    draw.box(ax, 2.75, 1.35, 1.1, 0.86, "prompt =\nquestion +\nretrieved text", kind="plain", size=6.4, color=C["ink2"])
    draw.arrow(ax, (3.85, 1.78), (4.1, 1.78))
    draw.box(ax, 4.1, 1.55, 0.6, 0.46, "LLM", kind="llm", size=7, weight="bold")
    draw.arrow(ax, (4.4, 1.55), (4.4, 1.15))
    draw.box(ax, 3.5, 0.5, 1.2, 0.62, "answer, citing\nthe chunk it\nused", kind="llm", size=6.2)
    draw.text(ax, 0.0, 0.62, "Retrieve first,\nthen generate:\nthe model reads\nyour documents\nbefore answering.",
              size=6.3, color=C["ink2"], va="center")
    return f


@figure(CH, "retrieval")
def retrieval():
    record()
    ks = [1, 2, 3, 4, 5]
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    for m, col, lab in (("word", C["data"], "whole words"), ("char", C["slate"], "character pieces"),
                        ("hybrid", C["jev"], "both combined")):
        ax.plot(ks, recall_at(m, 30), color=col, lw=1.7, marker="o", ms=4, mec="white", mew=0.7, label=lab)
    ax.set_xticks(ks)
    ax.set_xlabel("chunks retrieved (k)")
    ax.set_ylabel("questions whose answer\nwas in the top k", fontsize=6.8)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=6.4)
    return f


@figure(CH, "chunking")
def chunking():
    sizes = [0, 15, 30, 60, 120]
    vals = [recall_at("char", s, ks=(3,))[0] for s in sizes]
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "y")
    labs = ["one\nsentence", "15\nwords", "30\nwords", "60\nwords", "120 words\n(whole doc)"]
    ax.plot(range(len(vals)), vals, color=C["data"], lw=1.6, marker="o", ms=5, mec="white", mew=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.0%}", ha="center", fontsize=6.4)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("answer in top 3")
    ax.set_xlabel("chunk size")
    return f


@figure(CH, "abstain")
def abstain():
    record()
    a, u = top_scores()
    import json
    from jevkit.figs import ROOT
    t = json.load(open(ROOT / "results" / f"{CH}.json"))["abstain_t"]
    f, ax = subplots(width="text", height=2.1)
    clean(ax, "y")
    bins = np.linspace(0, 0.8, 25)
    # solid bars against a striped outline: the two groups differ without colour
    ax.hist(a, bins=bins, color=C["jev"], alpha=0.85, label="the knowledge base has the answer (solid)")
    ax.hist(u, bins=bins, histtype="stepfilled", fc="none", ec=C["fail"], lw=1.3, hatch="////", hatchcolor=C["fail"],
            label="it doesn’t (striped)")
    ax.axvline(t, color=C["ink"], lw=0.9)
    ax.annotate("below this threshold:\nsay “I don’t know”", xy=(t, ax.get_ylim()[1] * 0.7), xytext=(0.6, ax.get_ylim()[1] * 0.55), fontsize=6.3, arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.set_xlabel("similarity of the best-matching chunk")
    ax.set_ylabel("questions")
    ax.legend(loc="upper right", fontsize=6.3)
    return f


@figure(CH, "context-budget")
def context_budget():
    f, ax = subplots(width="text", height=1.55)
    ax.grid(False)
    parts = [("instructions", 600, C["neutral_t"]), ("retrieved chunks", 1800, C["data_t"]),
             ("case memory", 900, C["jev_t"]), ("conversation so far", 2400, C["llm_t"]),
             ("the alert", 300, C["fail_t"])]
    left = 0
    for name, n, col in parts:
        ax.barh(0, n, left=left, color=col, edgecolor="white", height=0.55)
        up = parts.index((name, n, col)) % 2 == 0
        ax.text(left + n / 2, 0.42 if up else -0.42, f"{name} ({n:,})", ha="center", va="bottom" if up else "top",
                fontsize=5.9)
        left += n
    ax.axvline(8000, color=C["fail"], lw=1)
    ax.text(8080, 0, "the window’s\nlimit", fontsize=6.2, color=C["fail"], va="center")
    ax.set_xlim(0, 9200)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("tokens (illustrative)", labelpad=2)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    for s in ("left",):
        ax.spines[s].set_visible(False)
    return f


@figure(CH, "memory")
def memory():
    f, ax = draw.canvas("text", 2.0)
    items = [("Working memory", "the context window: what the model\ncan see right now", "llm"),
             ("Reference memory", "documents fetched on demand:\npolicies, runbooks, tickets (RAG)", "data"),
             ("Case memory", "what happened last time: past alerts,\nverdicts and analyst notes", "jev")]
    for i, (t, sub, kind) in enumerate(items):
        x = i * 1.6
        draw.box(ax, x, 1.05, 1.45, 0.6, t, kind=kind, size=7, weight="bold")
        draw.text(ax, x + 0.72, 0.92, sub, size=5.9, ha="center", va="top", color=C["ink2"])
    draw.text(ax, 0.0, 1.88, "Three kinds of memory an agent can use", size=7, weight="bold")
    draw.text(ax, 0.0, 0.1, "Every write and every read is a small decision: keep this? fetch that? is it relevant enough to trust?",
              size=6.2, color=C["ink"], weight="semibold")
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Retrieve, then generate", kind="data", h=1.45,
             body="Find the passages that match the question and put them in the prompt. The model answers from your documents."),
        dict(num=2, title="Retrieval is the weak link", kind="fail", h=1.45,
             body=(f"If the right chunk isn’t retrieved, nothing downstream can fix it. Our simple retriever put the answer "
                   f"first {rr['recall']['char'][0]:.0%} of the time, in the top 3 {rr['recall']['char'][2]:.0%}.")),
        dict(num=3, title="Chunk size matters", kind="neutral", h=1.45,
             body="Too small loses context; too big dilutes the match. Test sizes on your own questions."),
        dict(num=4, title="Know when you don’t know", kind="jev", h=1.45,
             body=(f"A similarity threshold let the system abstain on {1 - rr['unans_answered']:.0%} of questions it "
                   f"couldn’t answer while answering {rr['answered_share']:.0%} of those it could.")),
        dict(num=5, title="Context is a budget", kind="llm", h=1.45,
             body="Instructions, documents, memory and history all compete for the same window. Every token costs time and money."),
        dict(num=6, title="Memory is full of decisions", kind="jev", h=1.45,
             body="What to store, what to fetch, whether it’s relevant, whether to trust it: small typed decisions, at high volume."),
    ]
    return summary_page(CH, "RAG and memory", panels, footer="Next: letting the model act, in a loop.")
