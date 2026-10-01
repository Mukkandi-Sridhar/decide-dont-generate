"""Figures for Chapter 5: Attention and transformers, visually."""

from functools import lru_cache

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression

from jevkit import attention as at
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page

CH = "ch08"
GREENS = LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#CFE9DA", C["jev"]])


@lru_cache(None)
def model():
    X, y, _ = at.make_data(4000, 0)
    Xt, yt, _ = at.make_data(1000, 1)
    hist = []
    p = at.train(X, y, steps=4000, lr=0.01, seed=1, history=hist)

    def bow(M):
        B = np.zeros((len(M), len(at.VOCAB)))
        for i, r in enumerate(M):
            for t in r:
                B[i, t] += 1
        return B[:, 1:]
    lr = LogisticRegression(max_iter=2000).fit(bow(X), y)
    acc_bow = lr.score(bow(Xt), yt)
    acc_att = float(((at.forward(p, Xt) > 0.5) == yt).mean())
    return p, hist, acc_bow, acc_att


def record():
    p, hist, acc_bow, acc_att = model()
    ex = {}
    for w in (["invoice", "not", "phishing"], ["phishing", "not", "invoice"], ["lunch", "and", "malware"]):
        q, A = at.forward(p, at.encode(w), return_attn=True)
        ex[" ".join(w)] = float(q[0])
    q, A = at.forward(p, at.encode(["invoice", "not", "phishing"]), return_attn=True)
    results(CH, acc_bow=acc_bow, acc_att=acc_att, examples=ex, att_phishing_to_not=float(A[0, 2, 1]),
            final_loss=hist[-1][1])


@figure(CH, "context-problem")
def context_problem():
    f, ax = draw.canvas("text", 1.9)
    draw.text(ax, 0.0, 1.8, "Same word, different meaning", size=7.2, weight="bold")
    for i, (sent, hi) in enumerate(((["she", "sat", "by", "the", "river", "bank"], 5),
                                    (["the", "bank", "froze", "my", "card"], 1))):
        x = 0.0
        for j, w in enumerate(sent):
            wd = 0.13 + 0.065 * len(w)
            draw.box(ax, x, 1.38 - i * 0.38, wd, 0.26, w, kind="fail" if j == hi else "plain", size=6.4, radius=0.04,
                     family="JetBrains Mono")
            x += wd + 0.05
    draw.text(ax, 2.55, 1.8, "Same words, opposite meaning", size=7.2, weight="bold")
    for i, (sent, lab, col) in enumerate(((["invoice", "not", "phishing"], "harmless", C["jev"]),
                                          (["phishing", "not", "invoice"], "a threat", C["fail"]))):
        x = 2.55
        for w in sent:
            wd = 0.13 + 0.065 * len(w)
            draw.box(ax, x, 1.38 - i * 0.38, wd, 0.26, w, kind="plain", size=6.4, radius=0.04, family="JetBrains Mono")
            x += wd + 0.05
        draw.text(ax, x + 0.03, 1.51 - i * 0.38, lab, size=6.4, color=col, weight="semibold")
    draw.text(ax, 0.0, 0.3, "One fixed vector per word can’t tell these apart. Neither can a bag of words,", size=6.6,
              color=C["ink2"])
    draw.text(ax, 0.0, 0.1, "which only knows which words appear, not where or next to what.", size=6.6, color=C["ink2"])
    return f


@figure(CH, "qkv")
def qkv():
    f, ax = draw.canvas("text", 2.35)
    words = ["invoice", "not", "phishing"]
    xs = [0.2, 1.55, 2.9]
    for x, w in zip(xs, words):
        draw.box(ax, x, 1.85, 1.1, 0.3, w, kind="data", size=6.8, family="JetBrains Mono")
    # the word doing the looking: phishing
    draw.box(ax, 3.0, 1.2, 0.9, 0.3, "query", kind="jev", size=6.4, weight="bold")
    draw.text(ax, 3.45, 1.08, "“who changes my meaning?”", size=5.8, ha="center", va="top", color=C["ink2"])
    for x, w in zip(xs, words):
        draw.box(ax, x + 0.05, 0.62, 0.48, 0.26, "key", kind="neutral", size=6.0)
        draw.box(ax, x + 0.57, 0.62, 0.48, 0.26, "value", kind="neutral", size=6.0)
    weights = [0.01, 0.98, 0.01]
    for x, wt in zip(xs, weights):
        draw.arrow(ax, (3.2, 1.2), (x + 0.29, 0.88), color=C["jev"], lw=0.4 + 2.5 * wt, head=3)
        draw.text(ax, x + 0.29, 0.5, f"{wt:.2f}", size=6.4, ha="center", va="top", weight="bold",
                  color=C["ink"] if wt > 0.5 else C["muted"])
    ax.set_ylim(-0.12, 2.35)
    return f


@figure(CH, "heatmaps")
def heatmaps():
    record()
    p, *_ = model()
    sents = [["invoice", "not", "phishing"], ["phishing", "not", "invoice"], ["lunch", "and", "malware"]]
    f, axes = subplots(1, 3, width="text", height=1.7, gridspec_kw=dict(wspace=0.55))
    for ax, s in zip(axes, sents):
        q, A = at.forward(p, at.encode(s), return_attn=True)
        n = len(s)
        M = A[0, :n, :n]
        ax.imshow(M, cmap=GREENS, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if M[i, j] > 0.05:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                            color="white" if M[i, j] > 0.6 else C["ink"])
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(s, rotation=45, ha="right", fontsize=5.8)
        ax.set_yticklabels(s, fontsize=5.8)
        ax.grid(False)
        ax.set_title(f"P(threat) = {float(q[0]):.2f}", fontsize=6.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("word looking →", fontsize=6)
    return f


@figure(CH, "bow-vs-attention")
def bow_vs_attention():
    p, hist, acc_bow, acc_att = model()
    f, (a1, a2) = subplots(1, 2, width="text", height=1.9, gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.4]))
    clean(a1, "x")
    a1.barh([1, 0], [acc_bow, acc_att], color=[C["data"], C["jev"]], height=0.5)
    a1.set_yticks([1, 0])
    a1.set_yticklabels(["bag of words", "one attention layer"], fontsize=6.6)
    for yy, v in ((1, acc_bow), (0, acc_att)):
        a1.text(v - 0.01, yy, f"{v:.0%}", ha="right", va="center", color="white", fontsize=6.6, fontweight="bold")
    a1.set_xlim(0.5, 1)
    a1.set_xlabel("accuracy on new notes", fontsize=6.6)
    clean(a2, "y")
    t, l = zip(*hist)
    a2.plot(t, l, color=C["jev"], lw=1.6)
    a2.set_yscale("log")
    a2.set_xlabel("training step", fontsize=6.6)
    a2.set_ylabel("log loss (log scale)", fontsize=6.6)
    return f


@figure(CH, "block")
def block():
    f, ax = draw.canvas("text", 2.6)
    toks = ["the", "email", "was", "not", "phishing"]
    for i, t in enumerate(toks):
        draw.box(ax, 0.1 + i * 0.62, 0.1, 0.56, 0.28, t, kind="data", size=6.2, family="JetBrains Mono")
        draw.arrow(ax, (0.38 + i * 0.62, 0.38), (0.38 + i * 0.62, 0.62), head=3, color=C["muted"])
    draw.text(ax, 3.3, 0.24, "word vectors + position", size=6.2, color=C["ink2"])
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.05, 0.62), 3.1, 1.52, boxstyle="round,pad=0,rounding_size=0.08", fc="white",
                                ec=C["ink"], lw=1.0, ls=(0, (3, 2))))
    draw.box(ax, 0.2, 0.78, 2.8, 0.42, "attention: every word looks at every word", kind="jev", size=6.6, weight="bold")
    draw.box(ax, 0.2, 1.5, 2.8, 0.42, "a small network, one word at a time", kind="neutral", size=6.6, weight="bold")
    draw.arrow(ax, (1.6, 1.2), (1.6, 1.5), head=3)
    draw.text(ax, 3.3, 1.38, "one transformer block", size=6.8, weight="bold")
    draw.text(ax, 3.3, 1.18, "(plus shortcuts that add each\nstep’s input back to its output,\nand normalisation)", size=5.9,
              color=C["ink2"], va="top")
    draw.text(ax, 1.6, 2.3, "× stack 12, 24 or 100+ blocks", size=6.8, ha="center", weight="bold", color=C["ink"])
    draw.arrow(ax, (1.6, 2.14), (1.6, 2.22), head=3)
    return f


@figure(CH, "encoder-decoder")
def encoder_decoder():
    f, axes = subplots(1, 2, width="text", height=1.9, gridspec_kw=dict(wspace=0.5))
    n = 5
    toks = ["the", "email", "was", "not", "phishing"]
    full = np.ones((n, n))
    causal = np.tril(np.ones((n, n)))
    for ax, M, title in ((axes[0], full, "Reader (encoder):\nevery word sees every word"),
                         (axes[1], causal, "Writer (decoder):\neach word sees only the past")):
        ax.imshow(M, cmap=LinearSegmentedColormap.from_list("m", ["#FFFFFF", "#CFE9DA"]), vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=5.8)
        ax.set_yticklabels(toks, fontsize=5.8)
        ax.set_title(title, fontsize=6.8)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                ax.add_patch(__import__("matplotlib").patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none",
                                                                        ec="white", lw=1.5))
    return f


@figure(CH, "summary")
def summary():
    import json
    from jevkit.figs import ROOT
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Meaning depends on context", kind="fail", h=1.45,
             body="“bank”, “not phishing” vs “phishing, not invoice”. One vector per word can’t see it."),
        dict(num=2, title="Attention = a soft lookup", kind="jev", h=1.45,
             body="Each word sends a query, matches every key, and takes a weighted mix of the values. Weights sum to 1 (softmax)."),
        dict(num=3, title="It learns what to look at", kind="jev", h=1.45,
             body=(f"Trained on toy notes, “phishing” learned to look at the word before it: "
                   f"{rr['att_phishing_to_not']:.0%} of its attention on “not”.")),
        dict(num=4, title="Order matters now", kind="data", h=1.45,
             body=f"Bag of words: {rr['acc_bow']:.0%} on new notes. One attention layer with positions: {rr['acc_att']:.0%}."),
        dict(num=5, title="A transformer stacks blocks", kind="neutral", h=1.45,
             body="Attention, then a small network per word, with shortcuts. Repeat dozens of times. It runs in parallel, so it scales."),
        dict(num=6, title="Readers and writers", kind="llm", h=1.45,
             body="Readers see the whole input at once: good for deciding. Writers see only the past: good for generating, one word at a time."),
    ]
    return summary_page(CH, "Attention and transformers, visually", panels,
                        footer="Next: what happens when you train this at enormous scale, and why it gets overconfident.")
