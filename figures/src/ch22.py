"""Figures for Chapter 15: A catalogue of decision patterns."""

import json
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import Choice

from jevkit import soc, llm, calibration as cal, client as make_client
from jevkit.batch import score_alerts
from jevkit.figs import figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch22"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def extracted(live):
    """Replace each live alert's fields with what the mock LLM extracted from its text."""
    mock = llm.MockLLM()
    e = live.copy()
    for i, d in enumerate(live.description):
        for k, v in mock.extract(d).items():
            e.at[i, k] = v
    return e


@lru_cache(None)
def extract_scores():
    alerts = soc.load()
    _, live = soc.history_and_live(alerts)
    y = live.malicious.to_numpy()
    e = extracted(live)
    # the LLM deciding on its own: the bake-off's JSON answers (Chapter 13), so both chapters quote one number
    from jevkit import bakeoff
    rows = {"llm_decides": bakeoff.run()["probs"]["llm_json"], "jev_text": score_alerts(live, "text"),
            "extract_then_jev": score_alerts(e, "structured"), "jev_fields": score_alerts(live, "structured")}
    misread = float((e.ioc_score != live.ioc_score).mean())
    return {k: cal.summary(v, y) for k, v in rows.items()}, misread


@lru_cache(None)
def router():
    alerts = soc.load()
    _, live = soc.history_and_live(alerts)
    sub = live.sample(2000, random_state=22).reset_index(drop=True)
    c = make_client()
    q = Choice(instructions="Which kind of threat or activity is this alert about?",
               criteria={k: None for k in soc.CATEGORIES})
    top, conf = [], []
    for d in sub.description:
        a = c.system_one(state=d, questions={"kind": q}).choices["kind"]
        top.append(a.choice)
        conf.append(max(a.probabilities.values()))
    ok = np.array(top) == sub.category.to_numpy()
    conf = np.array(conf)
    lines = np.linspace(0, 0.95, 20)
    cov = np.array([(conf >= t).mean() for t in lines])
    acc = np.array([ok[conf >= t].mean() for t in lines])
    return lines, cov, acc, float(ok.mean())


def record():
    sc, misread = extract_scores()
    lines, cov, acc, overall = router()
    i8 = int(np.argmin(np.abs(lines - 0.8)))
    ch13 = json.load(open(ROOT / "results" / "ch13.json"))
    ch14 = json.load(open(ROOT / "results" / "ch14.json"))
    results(CH, **{f"{k}_auc": v["auc"] for k, v in sc.items()}, **{f"{k}_ece": v["ece"] for k, v in sc.items()},
            misread=misread, route_all_acc=overall, route_line=float(lines[i8]), route_cov=float(cov[i8]),
            route_acc=float(acc[i8]), decide_per_alert=ch13["per_alert"]["decide"],
            generate_per_alert=ch13["per_alert"]["generate"], inj_before=ch14["inj_act_before"],
            inj_after=ch14["inj_act_after"], inj_trusted=ch14["inj_act_trusted"])


# ---------------------------------------------------------------- glyphs shared by the catalogue and the chapter
def glyph(ax, kind, x, y, w, h):
    """A tiny schematic of each pattern, drawn inside the box (x, y, w, h)."""
    s = 5.4
    bw, bh = w * 0.27, min(h * 0.72, 0.36)
    mid = y + h / 2 - bh / 2
    if kind == "extract":
        items = [("text", "data"), ("LLM\nextracts", "llm"), ("Jev\ndecides", "jev")]
    elif kind == "oda":
        items = [("observe", "data"), ("decide", "jev"), ("act, then\nobserve", "neutral")]
    elif kind == "gate":
        items = [("proposed\naction", "llm"), ("Jev\ngate", "jev"), ("allow /\nblock", "neutral")]
    elif kind == "judge":
        items = [("LLM\nwrites", "llm"), ("Jev\nchecks", "jev"), ("ship /\nredo", "neutral")]
    elif kind == "router":
        items = [("case", "data"), ("Jev\nchoice", "jev"), ("queue\nA · B · C", "neutral")]
    else:
        items = [("new\nfact", "data"), ("Jev:\nkeep?", "jev"), ("memory", "neutral")]
    gap = (w - 3 * bw) / 2
    for i, (t, k) in enumerate(items):
        xx = x + i * (bw + gap)
        draw.box(ax, xx, mid, bw, bh, t, kind=k, size=s, radius=0.04)
        if i < 2:
            draw.arrow(ax, (xx + bw, mid + bh / 2), (xx + bw + gap, mid + bh / 2), head=2.5)


@figure(CH, "catalog")
def catalog():
    f, ax = draw.canvas("text", 3.9)
    cards = [("extract", "1 · Extract, then decide", "The LLM turns messy input into fields; Jev decides."),
             ("oda", "2 · The decide step", "Every small choice inside an agent loop goes to Jev."),
             ("gate", "3 · Guardrail gate", "Before an action runs, a typed check on trusted facts."),
             ("judge", "4 · Check the writer", "The LLM writes; Jev checks it; low scores escalate."),
             ("router", "5 · Router", "A choice sends each case to the right queue, or to a person."),
             ("memory", "6 · Memory controller", "Decide what to store, fetch and forget.")]
    w, h = 2.28, 1.15
    for i, (k, t, sub) in enumerate(cards):
        x, y = (i % 2) * (w + 0.14), 2.7 - (i // 2) * (h + 0.14)
        draw.box(ax, x, y, w, h, "", kind="plain")
        draw.text(ax, x + 0.08, y + h - 0.13, t, size=6.8, weight="bold")
        draw.text(ax, x + 0.08, y + h - 0.3, sub, size=5.7, color=C["ink2"])
        glyph(ax, k, x + 0.12, y + 0.08, w - 0.24, h - 0.45)
    return f


@figure(CH, "extract")
def extract_diagram():
    f, ax = draw.canvas("text", 1.9)
    draw.doc_icon(ax, 0.05, 0.75, w=0.34, h=0.44, kind="data")
    draw.text(ax, 0.22, 0.62, "raw alert\ntext", size=5.8, ha="center", va="top")
    draw.arrow(ax, (0.45, 0.97), (0.7, 0.97))
    draw.box(ax, 0.7, 0.72, 0.9, 0.5, "LLM extracts\nfields", kind="llm", size=6.2, weight="semibold")
    draw.arrow(ax, (1.6, 0.97), (1.85, 0.97))
    fields = ['rule: "unsigned_temp_binary"', "ioc_score: 0.38", "after_hours: false", "prior_alerts_24h: 1"]
    draw.box(ax, 1.85, 0.42, 1.38, 1.1, "", kind="data")
    for i, t in enumerate(fields):
        draw.text(ax, 1.9, 1.4 - i * 0.2, t, size=4.9, family="JetBrains Mono")
    draw.arrow(ax, (3.23, 0.97), (3.4, 0.97))
    draw.box(ax, 3.4, 0.72, 0.58, 0.5, "Jev\nnoul", kind="jev", size=6.2, weight="semibold")
    draw.arrow(ax, (3.98, 0.97), (4.12, 0.97))
    draw.box(ax, 4.12, 0.72, 0.58, 0.5, "policy\nthresholds", kind="review", size=6.0)
    draw.text(ax, 1.15, 0.55, "writes, never decides", size=5.6, ha="center", color=C["llm"], style="italic")
    draw.text(ax, 3.69, 0.55, "decides, never writes", size=5.6, ha="center", color=C["jev"], style="italic")
    return f


@figure(CH, "extract-results")
def extract_results():
    sc, misread = extract_scores()
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "x")
    rows = [("LLM decides on its own (stated confidence)", "llm_decides", C["llm"]),
            ("Jev reads the raw text", "jev_text", C["jev"]),
            ("LLM extracts fields, Jev decides", "extract_then_jev", C["jev"]),
            ("Jev reads the SIEM’s own fields", "jev_fields", C["data"])]
    for i, (lab, k, col) in enumerate(rows):
        y = len(rows) - 1 - i
        v = sc[k]["auc"]
        ax.plot([0.5, v], [y, y], color=C["grid"], lw=1)
        ax.scatter([v], [y], color=col, s=40, zorder=3)
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=6.2)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("ranking on Kestrel’s live week (AUC)")
    synthetic_tag(f)
    return f


@figure(CH, "oda")
def oda():
    ch13 = json.load(open(ROOT / "results" / "ch13.json"))["per_alert"]
    f, ax = draw.canvas("text", 2.3)
    draw.box(ax, 0.1, 0.95, 1.1, 0.6, "OBSERVE\nread a tool", kind="data", size=6.4, weight="semibold")
    draw.box(ax, 1.8, 0.95, 1.1, 0.6, "DECIDE\na typed question", kind="jev", size=6.4, weight="semibold")
    draw.box(ax, 3.5, 0.95, 1.1, 0.6, "ACT\ncall a tool", kind="neutral", size=6.4, weight="semibold")
    draw.arrow(ax, (1.2, 1.25), (1.8, 1.25))
    draw.arrow(ax, (2.9, 1.25), (3.5, 1.25))
    draw.arrow(ax, (4.05, 0.95), (0.65, 0.95), rad=-0.28, color=C["muted"], label="until done", labelpos=0.5,
               labeloffset=(0, -0.42))
    draw.box(ax, 1.8, 1.8, 1.1, 0.36, "LLM: write a note", kind="llm", size=5.9, dashed=True)
    draw.arrow(ax, (2.35, 1.55), (2.35, 1.8), head=2.5, color=C["llm"], dashed=True)
    draw.text(ax, 2.35, 0.82, f"≈ {ch13['decide']:.1f} decisions per alert", size=5.9, ha="center", va="top",
              color=C["jev"])
    draw.text(ax, 3.0, 2.0, f"only ≈ {ch13['generate']:.2f} per alert\n(Chapter 7)", size=5.6, color=C["llm"],
              va="center")
    return f


@figure(CH, "gate")
def gate():
    f, ax = draw.canvas("text", 2.4)
    draw.box(ax, 0.0, 1.45, 1.2, 0.5, "agent proposes:\nclose the alert", kind="llm", size=6.0)
    draw.box(ax, 0.0, 0.55, 1.2, 0.62, "trusted facts only:\nSIEM fields, threat\nintel, asset list", kind="data", size=5.8)
    draw.text(ax, 0.6, 0.38, "not the alert’s free text", size=5.4, ha="center", va="top", color=C["fail"], style="italic")
    draw.arrow(ax, (1.2, 1.7), (1.6, 1.3))
    draw.arrow(ax, (1.2, 0.86), (1.6, 1.1))
    draw.box(ax, 1.6, 0.9, 1.0, 0.6, "Jev noul:\nP(safe to\nclose)", kind="jev", size=6.0, weight="semibold")
    draw.arrow(ax, (2.6, 1.2), (2.9, 1.2))
    draw.zone_bar(ax, 2.9, 1.0, 1.75, 0.4, 0.3, 0.75, names=("block", "ask", "allow"), size=6.0, tick_size=5.4)
    draw.text(ax, 3.77, 1.75, "gate on P(safe): high allows, low blocks, middle asks a person", size=5.5,
              ha="center", color=C["ink2"])
    return f


@figure(CH, "judge")
def judge():
    f, ax = draw.canvas("text", 2.0)
    draw.box(ax, 0.0, 0.8, 0.95, 0.55, "LLM drafts the\nincident note", kind="llm", size=6.0)
    draw.arrow(ax, (0.95, 1.07), (1.25, 1.07))
    draw.box(ax, 1.25, 0.8, 1.15, 0.55, "Jev: is every claim\nsupported by the\nevidence?", kind="jev", size=5.8)
    draw.arrow(ax, (2.4, 1.07), (2.7, 1.07))
    outs = [("ship it", "act", 1.55), ("redraft with\nthe gaps named", "review", 1.0), ("a person\nwrites it", "escalate", 0.45)]
    for t, k, y in outs:
        draw.box(ax, 3.05, y - 0.2, 1.3, 0.4, t, kind=k, size=5.8, textcolor="white" if k != "act" else None,
                 fill=draw.KIND[k][0] if k != "act" else None)
        draw.arrow(ax, (2.7, 1.07), (3.05, y), head=2.5, color=C["muted"])
    draw.arrow(ax, (3.05, 0.9), (0.48, 0.8), rad=-0.3, dashed=True, color=C["muted"], head=2.5)
    draw.text(ax, 1.7, 0.12, "redraft at most once, then a person", size=5.5, ha="center", color=C["ink2"], style="italic")
    return f


@figure(CH, "router")
def router_fig():
    lines, cov, acc, overall = router()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "both")
    ax.plot(cov, acc, color=C["jev"], lw=2, marker="o", ms=3)
    for t in (0.0, 0.6, 0.8, 0.9):
        i = int(np.argmin(np.abs(lines - t)))
        ax.annotate(f"route if top ≥ {lines[i]:.1f}" if t else "route everything", (cov[i], acc[i]),
                    xytext=(4, -12), textcoords="offset points", ha="left", fontsize=5.9)
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(0.85, 1.0)
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.invert_xaxis()
    ax.set_xlabel("share of alerts routed automatically (the rest go to a general queue)")
    ax.set_ylabel("routed to the right team")
    synthetic_tag(f)
    return f


@figure(CH, "memory")
def memory():
    f, ax = draw.canvas("text", 2.2)
    draw.box(ax, 0.0, 1.35, 1.0, 0.5, "new fact from\nthis alert", kind="data", size=6.0)
    draw.box(ax, 0.0, 0.35, 1.0, 0.5, "new question\nfrom the agent", kind="data", size=6.0)
    draw.box(ax, 1.35, 1.35, 1.25, 0.5, "Jev: worth keeping?\n(noul)", kind="jev", size=5.9)
    draw.box(ax, 1.35, 0.35, 1.25, 0.5, "Jev: which store?\n(choice)", kind="jev", size=5.9)
    draw.arrow(ax, (1.0, 1.6), (1.35, 1.6))
    draw.arrow(ax, (1.0, 0.6), (1.35, 0.6))
    for i, t in enumerate(("case notes", "host history", "policies", "none")):
        y = 1.55 - i * 0.37
        draw.cylinder(ax, 3.1, y - 0.12, 0.62, 0.3, kind="neutral" if t != "none" else "plain", label=t, size=5.4)
    draw.arrow(ax, (2.6, 1.6), (3.1, 1.55), head=2.5)
    draw.arrow(ax, (2.6, 0.6), (3.1, 1.17), head=2.5, color=C["muted"])
    draw.arrow(ax, (3.72, 1.0), (4.05, 1.0), head=2.5)
    draw.box(ax, 4.05, 0.75, 0.65, 0.5, "LLM\nreasons", kind="llm", size=5.9)
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))

    def g(kind):
        return lambda ax, x, y, w, h: glyph(ax, kind, x, y, w, h)

    panels = [
        dict(num=1, title="Extract, then decide", kind="jev", h=1.95, draw=g("extract"), draw_h=0.6,
             body=(f"LLM fields into Jev: AUC {rr['extract_then_jev_auc']:.3f}, against {rr['llm_decides_auc']:.3f} "
                   f"when the LLM decides alone. Real fields still win: {rr['jev_fields_auc']:.3f}.")),
        dict(num=2, title="The decide step", kind="jev", h=1.95, draw=g("oda"), draw_h=0.6,
             body=f"About {rr['decide_per_alert']:.1f} small decisions per alert. Send them to a decision model; keep the LLM for writing."),
        dict(num=3, title="Guardrail gate", kind="jev", h=1.95, draw=g("gate"), draw_h=0.6,
             body=(f"Check actions on trusted facts only. In Chapter 7, that kept auto-close at "
                   f"{rr['inj_trusted']:.0%} under injection, not {rr['inj_after']:.0%}.")),
        dict(num=4, title="Check the writer", kind="jev", h=1.95, draw=g("judge"), draw_h=0.6,
             body="Ask typed questions about the draft. Redraft once; then a person."),
        dict(num=5, title="Router", kind="jev", h=1.95, draw=g("router"), draw_h=0.6,
             body=(f"Route when the top label is at least {rr['route_line']:.1f}: {rr['route_cov']:.0%} routed, "
                   f"{rr['route_acc']:.0%} to the right team.")),
        dict(num=6, title="Memory controller", kind="jev", h=1.95, draw=g("memory"), draw_h=0.6,
             body="Store, fetch and forget are decisions too, and there are lots of them."),
    ]
    return summary_page(CH, "A catalogue of decision patterns", panels,
                        footer="Next: Part V. Your first real calls, and the mock that makes them free.")
