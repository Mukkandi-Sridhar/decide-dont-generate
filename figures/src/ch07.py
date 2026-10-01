"""Figures for Chapter 5: Embeddings: meaning as geometry."""

from functools import lru_cache

import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from jevkit import soc, text, calibration as cal
from jevkit.figs import figure, draw, C, subplots, clean, results, summary_page, synthetic_tag

CH = "ch07"
GROUP = {"lure": "threat", "malware": "threat", "exfil": "threat", "secret": "threat",
         "money": "business", "office": "business", "action": "other", "role": "other", "device": "other"}
GCOL = {"threat": C["fail"], "business": C["data"], "other": C["slate"]}
GSTYLE = {"threat": (1.8, "-"), "business": (0.7, "-"), "other": (0.9, (0, (3, 2)))}


@lru_cache(None)
def alerts_knn():
    df = soc.load()
    tr, ca, te = soc.split(df)
    emb = text.alert_embedder(tr.description.tolist())
    Ztr, Zte = emb(tr.description.tolist()), emb(te.description.tolist())
    nn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(Ztr)
    d, ix = nn.kneighbors(Zte)
    pk = np.clip(tr.malicious.values[ix].mean(1), 0.005, 0.995)
    F = lambda x: soc.feature_matrix(x).values
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(F(tr), tr.malicious)
    pl = lr.predict_proba(F(te))[:, 1]
    return dict(tr=tr, te=te, emb=emb, Ztr=Ztr, Zte=Zte, pk=pk, pl=pl, ix=ix, d=d)


def record():
    k = alerts_knn()
    y = k["te"].malicious.values
    results(CH, knn=cal.summary(k["pk"], y), lr=cal.summary(k["pl"], y),
            near_phishing=text.note_nearest("phishing", 4), near_upload=text.note_nearest("upload", 4),
            near_download=text.note_nearest("download", 4), n_notes=30000,
            query_desc=k["te"].description.iloc[5], query_mal=int(y[5]))


@figure(CH, "representations")
def representations():
    f, ax = draw.canvas("text", 2.0)
    words = ["invoice", "bill", "lunch", "trojan"]
    cols = [0.0, 1.55, 3.1]
    titles = ["An ID number", "One-hot: a slot per word", "An embedding: a few meaningful numbers"]
    for x, t in zip(cols, titles):
        draw.text(ax, x, 1.88, t, size=6.8, weight="bold")
    emb = {"invoice": [0.8, 0.1, -0.6], "bill": [0.75, 0.15, -0.55], "lunch": [0.1, 0.9, 0.2], "trojan": [-0.7, -0.2, 0.8]}
    for i, w in enumerate(words):
        yy = 1.5 - i * 0.36
        draw.text(ax, cols[0], yy, f"{w:<8}", size=6.6, family="JetBrains Mono")
        draw.text(ax, cols[0] + 0.72, yy, f"{i + 101}", size=6.6, family="JetBrains Mono", color=C["ink2"])
        for j in range(8):
            on = j == i * 2
            ax.add_patch(Rectangle((cols[1] + j * 0.16, yy - 0.07), 0.13, 0.14, fc=C["ink"] if on else "white",
                                   ec=C["rule"], lw=0.5))
        for j, v in enumerate(emb[w]):
            col = C["jev"] if v > 0 else C["llm"]
            ax.add_patch(Rectangle((cols[2] + j * 0.42, yy - 0.07), 0.36 * abs(v), 0.14, fc=col, ec="none"))
            draw.text(ax, cols[2] + j * 0.42 + 0.02, yy + 0.13, f"{v:+.1f}", size=5.2, color=C["muted"])
    draw.text(ax, cols[0], 0.08, "numbers, but 102 isn’t\n“near” 101 in meaning", size=5.9, color=C["ink2"])
    draw.text(ax, cols[1], 0.08, "every pair of words is\nequally different", size=5.9, color=C["ink2"])
    draw.text(ax, cols[2], 0.08, "invoice and bill get similar\nnumbers; lunch and trojan don’t", size=5.9, color=C["ink2"])
    return f


@figure(CH, "company")
def company():
    f, ax = draw.canvas("text", 1.5)
    rows = [["the", "phishing", "email", "asked", "for", "their", "password"],
            ["the", "spoofed", "email", "asked", "for", "their", "token"]]
    for r, sent in enumerate(rows):
        x = 0.0
        yy = 1.2 - r * 0.5
        for i, w in enumerate(sent):
            wd = 0.14 + 0.068 * len(w)
            kind = "fail" if i == 1 else ("data" if 0 < abs(i - 1) <= 2 else "plain")
            draw.box(ax, x, yy - 0.14, wd, 0.28, w, kind=kind, size=6.6, radius=0.04, family="JetBrains Mono")
            x += wd + 0.06
    draw.text(ax, 0.0, 0.18, "Different words, same neighbours (blue). Count enough sentences and their context", size=6.4,
              color=C["ink2"])
    draw.text(ax, 0.0, 0.02, "profiles line up, so their vectors end up close together.", size=6.4, color=C["ink2"])
    return f


@figure(CH, "word-map")
def word_map():
    from sklearn.manifold import MDS
    vocab, vec, _ = text.note_vectors()
    classes = list(text.WORD_CLASSES)
    cent = np.array([vec[[vocab.index(w) for w in text.WORD_CLASSES[c]]].mean(0) for c in classes])
    D = 1 - (cent / np.linalg.norm(cent, axis=1, keepdims=True)) @ (cent / np.linalg.norm(cent, axis=1, keepdims=True)).T
    P = MDS(n_components=2, dissimilarity="precomputed", random_state=2, normalized_stress="auto").fit_transform(D)
    P = (P - P.min(0)) / (P.max(0) - P.min(0))
    f, ax = subplots(width="text", height=2.9)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib.patches import FancyBboxPatch
    for (x, y), c in zip(P, classes):
        col = GCOL[GROUP[c]]
        lw, ls = GSTYLE[GROUP[c]]
        ws = text.WORD_CLASSES[c]
        ax.add_patch(FancyBboxPatch((x - 0.085, y - 0.105), 0.17, 0.21, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="white", ec=col, lw=lw, ls=ls))
        for k, w in enumerate(ws):
            ax.text(x, y + 0.075 - k * 0.037, w, fontsize=5.6, color=C["ink"], ha="center", va="center")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.14, 1.14)
    # the groups differ by border as well as colour: thick, thin, dashed
    from matplotlib.patches import Patch
    keys = [("threat", "threat words (thick border)"), ("business", "everyday business (thin border)"),
            ("other", "actions, roles, devices (dashed border)")]
    ax.legend(handles=[Patch(fc="white", ec=GCOL[g], lw=GSTYLE[g][0], ls=GSTYLE[g][1], label=s) for g, s in keys],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.0, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.2)
    return f


@figure(CH, "cosine")
def cosine():
    f, ax = draw.canvas("text", 1.7)
    o = (0.6, 0.2)
    vecs = [("invoice", (1.9, 1.0), C["data"]), ("bill", (1.75, 1.25), C["data"]), ("trojan", (-0.3, 1.35), C["fail"])]
    for name, (dx, dy), col in vecs:
        ax.add_patch(FancyArrowPatch(o, (o[0] + dx * 0.55, o[1] + dy * 0.55), arrowstyle="-|>,head_length=4,head_width=2.5",
                                     color=col, lw=1.3))
        draw.text(ax, o[0] + dx * 0.58, o[1] + dy * 0.58 + 0.04, name, size=6.6, color=col, weight="semibold")
    draw.text(ax, 2.2, 1.35, "Similar meaning = small angle between arrows.", size=6.6)
    draw.text(ax, 2.2, 1.12, "cosine similarity = cos(angle)", size=6.4, family="JetBrains Mono", color=C["ink2"])
    draw.text(ax, 2.2, 0.85, "invoice vs bill:   close to 1", size=6.4, color=C["ink2"])
    draw.text(ax, 2.2, 0.65, "invoice vs trojan: near 0 or below", size=6.4, color=C["ink2"])
    draw.text(ax, 2.2, 0.38, "The arrows’ lengths don’t matter,\nonly the directions.", size=6.2, color=C["muted"])
    return f


@figure(CH, "alert-map")
def alert_map():
    from sklearn.manifold import TSNE
    k = alerts_knn()
    tr = k["tr"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(k["Ztr"]), 2000, replace=False)
    P = TSNE(n_components=2, random_state=0, perplexity=40, init="pca").fit_transform(k["Ztr"][sel])
    rules = tr.rule.values[sel]
    mal = tr.malicious.values[sel] == 1
    f, ax = subplots(width="text", height=3.2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(P[~mal, 0], P[~mal, 1], s=3, color=C["muted"], alpha=0.35, lw=0)
    ax.scatter(P[mal, 0], P[mal, 1], s=6, color=C["fail"], lw=0)
    for r in np.unique(rules):
        m = rules == r
        if m.sum() < 110:
            continue
        cx, cy = np.median(P[m], 0)
        ax.text(cx, cy, soc.RULES[r]["title"], fontsize=5.2, color=C["ink"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(0.0, 1.0, "\u25cf real threat", transform=ax.transAxes, fontsize=6.3, color=C["fail"])
    ax.text(0.17, 1.0, "\u25cf harmless", transform=ax.transAxes, fontsize=6.3, color=C["muted"])
    synthetic_tag(f, "SYNTHETIC DATA \u00b7 Kestrel Logistics")
    return f


@figure(CH, "knn-vs-fields")
def knn_vs_fields():
    record()
    k = alerts_knn()
    y = k["te"].malicious.values
    f, ax = subplots(width="text", height=2.4)
    clean(ax, "both")
    ax.plot([0, 0.6], [0, 0.6], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    for p, col, lab in ((k["pk"], C["llm"], "50 nearest alerts by text meaning"),
                        (k["pl"], C["data"], "logistic regression on structured fields")):
        b = cal.reliability(p, y, n_bins=8, strategy="quantile")
        s = cal.summary(p, y)
        ax.plot(b.mean_pred, b.frac_pos, color=col, lw=1.6, marker="o", ms=3.5, mec="white", mew=0.6,
                label=f"{lab}  (AUC {s['auc']:.2f})")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
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

    def mini_arrows(ax, x, y, w, h):
        o = (x + 0.3, y + 0.05)
        for (dx, dy), col in (((0.9, 0.35), C["data"]), ((0.85, 0.45), C["data"]), ((-0.1, 0.5), C["fail"])):
            ax.add_patch(FancyArrowPatch(o, (o[0] + dx, o[1] + dy), arrowstyle="-|>,head_length=3,head_width=2", color=col, lw=1.1))

    panels = [
        dict(num=1, title="Words need numbers", kind="data", h=1.45,
             body="IDs and one-hot slots are numbers, but they carry no meaning. “invoice” is as far from “bill” as from “trojan”."),
        dict(num=2, title="Know a word by its company", kind="data", h=1.45,
             body="Words that appear in the same contexts get similar vectors. Nobody labels them: counting does it."),
        dict(num=3, title="Meaning becomes geometry", kind="neutral", h=1.45, draw=mini_arrows, draw_h=0.5,
             body="Similar meaning = small angle. Cosine similarity measures it, from about −1 to 1."),
        dict(num=4, title="Alerts can be embedded too", kind="neutral", h=1.45,
             body="Whole alerts become vectors. Similar alerts land together, which powers search, de-duplication and RAG."),
        dict(num=5, title="Neighbours give probabilities", kind="jev", h=1.45,
             body=(f"“Among the 50 most similar past alerts, how many were attacks?” is honest (ECE "
                   f"{rr['knn']['ece']:.3f}), but blunt (AUC {rr['knn']['auc']:.2f}).")),
        dict(num=6, title="Meaning isn’t the numbers", kind="fail", h=1.45,
             body=(f"Text similarity knows what an alert is about, not its threat score. Structured fields scored "
                   f"AUC {rr['lr']['auc']:.2f}. Keep the numbers as numbers.")),
    ]
    return summary_page(CH, "Embeddings: meaning as geometry", panels,
                        footer="Next: “bank” means different things in different sentences. Attention fixes that.")
