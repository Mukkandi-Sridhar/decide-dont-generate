"""Figures for Chapter 19: An applications gallery."""

import json
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import Choice, Noul

from jevkit import gallery, calibration as cal, client as make_client
from jevkit.figs import ink_on, figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch26"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
ILLUS = "ILLUSTRATIVE · volumes and costs are assumptions"
TOPIC = Choice(instructions="Which team should handle this ticket?",
               criteria={"billing": "Payments, refunds, invoices, charges",
                         "technical": "Bugs, errors, crashes, the app not working",
                         "account": "Login, password, access, profile", "other": None})
URGENT = Noul(instructions="Does this need a reply today?")


@lru_cache(None)
def ticket_run():
    c = make_client()
    T = gallery.tickets()
    top, conf, pu = [], [], []
    for t in T:
        r = c.system_one(state=t["text"], questions={"topic": TOPIC, "urgent": URGENT})
        top.append(r.choices["topic"].choice)
        conf.append(r.choices["topic"].confidence)
        pu.append(r.nouls["urgent"].noul)
    truth = np.array([t["topic"] for t in T])
    return T, np.array(top), np.array(conf), np.array(pu), truth, np.array([t["urgent"] for t in T])


def record():
    T, top, conf, pu, truth, urg = ticket_run()
    ok = top == truth
    per = {k: float(ok[truth == k].mean()) for k in gallery.TOPICS}
    m6 = conf >= 0.6
    confusion = {k: {j: int(((truth == k) & (top == j)).sum()) for j in gallery.TOPICS} for k in gallery.TOPICS}
    results(CH, n_tickets=len(T), topic_acc=float(ok.mean()), per_topic=per, cov60=float(m6.mean()),
            acc60=float(ok[m6].mean()), urgent_auc=cal.summary(pu, urg)["auc"], urgent_ece=cal.ece(pu, urg),
            confusion=confusion, account_as_technical=confusion["account"]["technical"],
            lines={d.name: gallery.line(d) for d in gallery.DOMAINS})


@figure(CH, "domains")
def domain_map():
    f, ax = subplots(width="text", height=2.6)
    clean(ax, "both")
    for d in gallery.DOMAINS:
        # shape as well as colour: squares where a person decides, circles where the model decides
        col = C["fail"] if d.human_final else C["jev"]
        ax.scatter([d.per_day], [d.cost_miss], s=46, color=col, zorder=3, edgecolor="white", lw=0.8,
                   marker="s" if d.human_final else "o")
        dx, dy, ha, va = {"Support tickets": (1, 0.62, "center", "top"), "Content moderation": (1, 0.62, "center", "top"),
                          "Payment fraud": (1, 1.6, "center", "bottom")}.get(d.name, (1.25, 1, "left", "center"))
        ax.text(d.per_day * dx, d.cost_miss * dy, d.name, fontsize=6.1, va=va, ha=ha)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, 2e7)
    ax.set_ylim(10, 3e5)
    ax.set_xticks([100, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["100", "1k", "10k", "100k", "1M", "10M"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("decisions a day (log scale)")
    ax.set_ylabel("cost of one missed problem (log scale)")
    ax.text(3e5, 1.1e4, "circles: decision model decides;\npeople check samples", fontsize=5.8, color=C["jev"], ha="center")
    ax.text(250, 1.2e5, "squares: decision model orders and flags;\na person makes the call", fontsize=5.8, color=C["fail"])
    synthetic_tag(f, ILLUS)
    return f


@figure(CH, "cards")
def cards():
    f, ax = draw.canvas("wide", 4.4)
    w, h = 1.9, 2.05
    for i, d in enumerate(gallery.DOMAINS):
        x, y = (i % 3) * (w + 0.12), 2.25 - (i // 3) * (h + 0.14)
        kind = "fail" if d.human_final else "jev"
        draw.box(ax, x, y, w, h, "", kind="plain")
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (x, y + h - 0.05), w, 0.05, fc=draw.KIND[kind][0], ec="none"))
        draw.text(ax, x + 0.1, y + h - 0.2, d.name, size=7.0, weight="bold")
        draw.text(ax, x + 0.1, y + h - 0.4, "reads: " + d.state, size=5.5, color=C["ink2"], wrapw=44, va="top")
        yy = y + h - 0.78
        for typ, q in d.questions:
            draw.pill(ax, x + 0.1, yy, typ, kind="jev", size=5.2, ha="left")
            draw.text(ax, x + 0.52, yy, q, size=5.5, wrapw=36, va="center")
            yy -= 0.33
        draw.text(ax, x + 0.1, y + 0.3, f"must include: “{d.none_option}”", size=5.4, color=C["ink2"], style="italic")
        who = "a person decides; the model orders the queue" if d.human_final else "the model decides; people audit"
        draw.text(ax, x + 0.1, y + 0.12, who, size=5.4, color=draw.KIND[kind][0], weight="semibold")
    return f


@figure(CH, "lines")
def lines_fig():
    f, ax = subplots(width="text", height=2.1)
    clean(ax, "x")
    ds = sorted(gallery.DOMAINS, key=gallery.line)
    for i, d in enumerate(ds):
        v = gallery.line(d)
        ax.barh(i, v, color=C["fail"] if d.human_final else C["jev"], height=0.55)
        ax.text(v * 1.03 + 0.0005, i, f"{v:.1%}   (miss \\${d.cost_miss:,.0f} · false alarm \\${d.cost_false:,.0f})",
                va="center", fontsize=5.8)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([d.name for d in ds], fontsize=6.3)
    ax.set_xlim(0, 0.09)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("flag a case once P(problem) passes this threshold (Chapter 4)")
    synthetic_tag(f, ILLUS)
    return f


@figure(CH, "tickets")
def tickets_fig():
    T, top, conf, pu, truth, urg = ticket_run()
    f, (a1, a2) = subplots(1, 2, width="text", height=2.1, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ks = list(gallery.TOPICS)
    M = np.array([[((truth == k) & (top == j)).sum() for j in ks] for k in ks], float)
    M = M / M.sum(1, keepdims=True)
    a1.imshow(M, cmap="Greens", vmin=0, vmax=1)
    a1.grid(False)
    for i in range(4):
        for j in range(4):
            a1.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=5.8,
                    color=ink_on(plt.get_cmap("Greens")(M[i, j])), fontweight="semibold")
    a1.set_xticks(range(4))
    a1.set_xticklabels(ks, fontsize=5.6, rotation=30, ha="right")
    a1.set_yticks(range(4))
    a1.set_yticklabels(ks, fontsize=5.8)
    a1.set_xlabel("model’s top label", fontsize=6)
    a1.set_ylabel("true team", fontsize=6)
    a1.set_title("where tickets went", fontsize=6.6, loc="left")
    clean(a2, "both")
    ths = np.linspace(0.3, 0.95, 14)
    cov = [(conf >= t).mean() for t in ths]
    acc = [(top == truth)[conf >= t].mean() for t in ths]
    a2.plot(cov, acc, color=C["jev"], lw=2, marker="o", ms=3)
    a2.set_xlim(1.02, 0.3)
    a2.set_ylim(0.6, 1)
    a2.xaxis.set_major_formatter(PCT)
    a2.yaxis.set_major_formatter(PCT)
    a2.set_xlabel("share routed automatically", fontsize=6)
    a2.set_ylabel("routed correctly", fontsize=6)
    a2.set_title("route only when sure", fontsize=6.6, loc="left")
    f.subplots_adjust(wspace=0.55)
    synthetic_tag(f)
    return f


@figure(CH, "human-final")
def human_final():
    f, ax = draw.canvas("text", 2.0)
    draw.box(ax, 0.0, 0.8, 0.95, 0.55, "case\narrives", kind="data", size=6.2)
    draw.box(ax, 1.25, 0.8, 1.05, 0.55, "decision model:\nP(urgent), P(risk)", kind="jev", size=6.0)
    draw.box(ax, 2.6, 0.8, 1.0, 0.55, "queue ordered\nby probability", kind="review", size=6.0, textcolor="white",
             fill=draw.KIND["review"][0])
    draw.box(ax, 3.9, 0.8, 0.8, 0.55, "a person\ndecides", kind="neutral", size=6.2, weight="semibold")
    for a, b in ((0.95, 1.25), (2.3, 2.6), (3.6, 3.9)):
        draw.arrow(ax, (a, 1.07), (b, 1.07))
    draw.box(ax, 1.25, 0.05, 2.35, 0.45, "log: what the model said, what the person decided, and why", kind="plain",
             size=5.8)
    draw.arrow(ax, (4.3, 0.8), (3.6, 0.3), color=C["muted"], head=3)
    return f


@figure(CH, "fit")
def fit():
    f, ax = draw.canvas("text", 2.3)
    good = ["many similar cases a day", "a fixed set of answers", "outcomes you can learn from",
            "a cost you can put on a mistake", "time or money pressure per case"]
    bad = ["one-off judgements with no precedent", "answers that must be written", "rare cases with no labels",
           "decisions nobody may take without a named person", "problems that need a plan, not a verdict"]
    draw.box(ax, 0.0, 0.05, 2.28, 2.0, "", kind="jev")
    draw.box(ax, 2.42, 0.05, 2.28, 2.0, "", kind="fail")
    draw.text(ax, 0.1, 1.88, "A GOOD FIT", size=6.6, weight="bold", color=C["jev"])
    draw.text(ax, 2.52, 1.88, "A POOR FIT, OR PEOPLE DECIDE", size=6.6, weight="bold", color=C["fail"])
    for i, (g, b) in enumerate(zip(good, bad)):
        draw.text(ax, 0.1, 1.55 - i * 0.3, "• " + g, size=6.1)
        draw.text(ax, 2.52, 1.55 - i * 0.3, "• " + b, size=6.1)
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="Same arithmetic, different lines", kind="neutral", h=1.5,
             body="The cost of a miss against a false alarm sets the line: about 5% for support tickets, 0.3% for clinical intake."),
        dict(num=2, title="Volume decides who decides", kind="jev", h=1.5,
             body="Millions of cheap decisions a day: the model decides and people audit. A few costly ones: people decide."),
        dict(num=3, title="Every choice needs a way out", kind="neutral", h=1.5,
             body="Other, none, not enough information: each domain needs the option that catches what fits nowhere."),
        dict(num=4, title="Measure before you trust", kind="fail", h=1.5,
             body=(f"On synthetic support tickets the mock routed {rr['topic_acc']:.0%} correctly and barely ranked urgency. "
                   "Without labels, you’d never have known.")),
        dict(num=5, title="Route only when sure", kind="jev", h=1.5,
             body=(f"Routing only confident tickets: {rr['cov60']:.0%} of them, {rr['acc60']:.0%} right. "
                   "The rest go to a person.")),
        dict(num=6, title="High stakes: order, don’t decide", kind="fail", h=1.5,
             body="Where a person must make the call, the model’s job is the queue: who is seen first, and what to look at."),
    ]
    return summary_page(CH, "An applications gallery", panels, footer="Next: build a small System One model of your own.")
