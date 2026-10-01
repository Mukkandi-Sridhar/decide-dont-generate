"""Figures for Chapter 22: What changes now."""

import json

import numpy as np

from jevkit import econ, llm
from jevkit.econ import TASK, task_cost
from jevkit.figs import figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag
from jevkit.figs.bookmap import PARTS

CH = "ch29"

def record():
    ch24 = json.load(open(ROOT / "results" / "ch24.json"))
    tl, cl = task_cost("llm")
    tj, cj = task_cost("jev")
    results(CH, task_decisions=sum(n for _, k, n in TASK if k == "decide"), task_steps=sum(n for _, _, n in TASK),
            task_llm_s=tl, task_jev_s=tj, task_llm_cost=cl, task_jev_cost=cj,
            ch24_speedup=ch24["speedup"])


@figure(CH, "stack")
def stack():
    f, ax = draw.canvas("text", 2.7)
    draw.text(ax, 0.0, 2.55, "ONE MODEL DOES EVERYTHING", size=6.4, weight="bold", color=C["ink2"])
    draw.text(ax, 2.5, 2.55, "EACH LAYER DOES ONE JOB", size=6.4, weight="bold", color=C["ink2"])
    draw.box(ax, 0.0, 1.55, 2.1, 0.8, "an LLM reads, decides,\nwrites and acts,\nall as text", kind="llm", size=6.4)
    draw.box(ax, 0.0, 0.95, 2.1, 0.45, "tools", kind="data", size=6.2)
    layers = [("LLM: reads messy input, plans, writes for people", "llm", 1.95),
              ("decision layer: typed questions, calibrated probabilities", "jev", 1.45),
              ("policy: thresholds, rules, capacity, fail-safes", "review", 0.95),
              ("tools, logs, monitors, people", "data", 0.45)]
    for t, k, y in layers:
        draw.box(ax, 2.5, y, 2.2, 0.4, t, kind=k, size=5.8, textcolor="white" if k == "review" else None,
                 fill=draw.KIND["review"][0] if k == "review" else None)
    draw.arrow(ax, (2.15, 1.7), (2.45, 1.7), head=4)
    return f


@figure(CH, "task")
def task():
    f, (a1, a2) = subplots(1, 2, width="text", height=1.7)
    tl, cl = task_cost("llm")
    tj, cj = task_cost("jev")
    for ax, vals, title, fmt in ((a1, (tl, tj), "seconds per task", "{:.1f} s"),
                                 (a2, (cl * 1000, cj * 1000), "cost per 1,000 tasks", "${:.2f}")):
        clean(ax, "x")
        ax.barh([1, 0], vals, color=[C["llm"], C["jev"]], height=0.5)
        for y, v in zip((1, 0), vals):
            ax.text(v * 1.02, y, fmt.format(v), va="center", fontsize=6.2)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["LLM decides", "decision model\ndecides"] if ax is a1 else ["", ""], fontsize=6.2)
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_title(title, fontsize=6.6, loc="left")
    f.subplots_adjust(wspace=0.15, left=0.2)
    synthetic_tag(f, "ILLUSTRATIVE · LLM figures assumed · Jev price and latency vendor-reported")
    return f


@figure(CH, "skills")
def skills():
    rows = [("Thinking in probabilities", "2, 3", "jev"), ("Putting a price on mistakes", "4, 14", "jev"),
            ("Testing calibration on your own data", "3, 11", "jev"), ("Designing typed questions", "10, 15", "jev"),
            ("Sizing queues and people", "14, 18", "review"), ("Monitoring a decision system", "17, 21", "review"),
            ("Writing prompts for everything", "6, 7", "llm"), ("Picking the biggest model", "5, 13", "llm")]
    f, ax = draw.canvas("text", 2.9)
    draw.text(ax, 0.0, 2.78, "MATTERS MORE", size=6.4, weight="bold", color=C["jev"])
    draw.text(ax, 0.0, 0.7, "MATTERS LESS ON ITS OWN", size=6.4, weight="bold", color=C["llm"])
    y = 2.5
    for i, (t, chs, k) in enumerate(rows):
        if i == 6:
            y = 0.42
        draw.box(ax, 0.0, y - 0.14, 0.12, 0.28, "", kind=k)
        draw.text(ax, 0.22, y, t, size=6.4)
        draw.text(ax, 4.7, y, f"Chapters {chs}", size=5.8, ha="right", color=C["ink2"])
        y -= 0.3
    return f


@figure(CH, "unknowns")
def unknowns():
    f, ax = draw.canvas("text", 2.4)
    items = [("Does Jev’s calibration hold on data like yours?", "Chapter 11 tells you how to find out."),
             ("How fast do System One models improve, and do they drift between versions?", "Pin versions; re-test."),
             ("Will prices stay this low?", "Early-access prices change. Chapter 12’s model shows what depends on it."),
             ("Who else builds decision models?", "The interface pattern outlives any one vendor."),
             ("How will rules on automated decisions evolve?", "Keep people deciding where stakes are high.")]
    for i, (q, a) in enumerate(items):
        y = 2.2 - i * 0.44
        draw.text(ax, 0.0, y, "?", size=11, weight="bold", color=C["fail"])
        draw.text(ax, 0.25, y + 0.06, q, size=6.4, weight="semibold")
        draw.text(ax, 0.25, y - 0.13, a, size=5.9, color=C["ink2"])
    return f


@figure(CH, "monday")
def monday():
    f, ax = draw.canvas("text", 2.3)
    phases = [("WEEK 1", "Audit", "List every decision your system makes. Count them. Mark which are small, typed and frequent."),
              ("WEEKS 2–4", "Measure", "Pick one. Get a few hundred labels. Test ranking and calibration: Chapters 3, 11 and 13."),
              ("MONTH 2", "Shadow", "Put costs on mistakes, set thresholds, run in shadow: Chapters 4, 14 and 21."),
              ("MONTH 3", "Switch, and watch", "Switch on with a fail-safe, an audit and a daily monitor. Then the next decision.")]
    w = 1.12
    for i, (when, what, detail) in enumerate(phases):
        x = i * (w + 0.07)
        draw.box(ax, x, 0.1, w, 1.85, "", kind="jev" if i == 3 else "neutral")
        draw.text(ax, x + 0.08, 1.78, when, size=5.8, weight="bold", color=C["ink2"])
        draw.text(ax, x + 0.08, 1.55, what, size=7.2, weight="bold")
        draw.text(ax, x + 0.08, 1.35, detail, size=5.7, wrapw=24, va="top", color=C["ink"])
    return f


@figure(CH, "journey")
def journey():
    f, ax = draw.canvas("text", 2.9)
    story = [("I", "We learned to speak in probabilities, make them calibrated and turn them into actions."),
             ("II", "Networks learned patterns, LLMs learned to write, and agents to act."),
             ("III", "A new kind of model decided in one pass, with probabilities we could test."),
             ("IV", "We compared methods, set thresholds from costs and named the patterns."),
             ("V", "We built it: calls, an agent, a case study, our own model, a service."),
             ("VI", "Now decisions are cheap. The question is which ones deserve them.")]
    for i, (r, t) in enumerate(story):
        y = 2.7 - i * 0.38
        draw.number_badge(ax, 0.15, y, r, color=C["jev"] if i >= 2 else C["data"], r=0.13, size=5.4)
        draw.text(ax, 0.4, y, t, size=6.6)
        if i < len(story) - 1:
            ax.plot([0.15, 0.15], [y - 0.13, y - 0.25], color=C["rule"], lw=1)
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="A new layer in the stack", kind="jev", h=1.5,
             body="Writers write, deciders decide, policies act. Each layer is smaller, testable and replaceable."),
        dict(num=2, title="Decisions get cheap", kind="jev", h=1.5,
             body=(f"One illustrative agent task: {rr['task_llm_s']:.0f} s with an LLM deciding, "
                   f"{rr['task_jev_s']:.0f} s with a decision model.")),
        dict(num=3, title="Judgement gets valuable", kind="neutral", h=1.5,
             body="Costs, lines, calibration checks and queues: the skills that decide whether cheap decisions are good ones."),
        dict(num=4, title="Much is still unknown", kind="fail", h=1.5,
             body="Calibration on your data, drift between versions, prices, regulation. Each is testable or plannable."),
        dict(num=5, title="Start on Monday", kind="jev", h=1.5,
             body="Audit your decisions. Measure one. Shadow it. Switch it on with a fail-safe and a monitor."),
        dict(num=6, title="The story", kind="neutral", h=1.5,
             body="Rules weren’t enough, so we learned from data. Now we can decide cheaply, so we must decide well."),
    ]
    return summary_page(CH, "What changes now", panels, footer="The end of the book. The start of your first audit.")
