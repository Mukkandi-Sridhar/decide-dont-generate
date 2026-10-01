"""Figures for Chapter 18: Case study: SOC alert triage."""

import json

import numpy as np
from matplotlib.ticker import FuncFormatter

from jevkit import ops
from jevkit.figs import figure, draw, C, ZONE, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch25"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def record():
    r = ops.before_after()
    so, sn = r["s_old"], r["s_new"]
    rows = ops.campaign()
    base = [x for x in rows if not x["respond"]]
    resp = [x for x in rows if x["respond"]]
    y = r["y"]
    ro, rn = r["old"][0], r["new"][0]
    results(CH, old=so, new=sn, low=r["policy"].low, high=r["policy"].high,
            old_flag_per_day=float((ro == "queue").sum() / 7), new_act_per_day=float((rn == "act").sum() / 7),
            new_review_per_day=float((rn == "review").sum() / 7), new_esc_per_day=float((rn == "escalate").sum() / 7),
            old_threats_closed_per_day=float(((ro == "closed") & (y == 1)).sum() / 7),
            new_threats_closed_per_day=float(((rn == "act") & (y == 1)).sum() / 7),
            camp_missed_base=float(sum(x["missed"] for x in base)), camp_missed_resp=float(sum(x["missed"] for x in resp)),
            camp_peak_reviews=max(x["reviews"] for x in base), camp_days=7, capacity=ops.CAPACITY,
            page_response_min=ops.PAGE_RESPONSE_MIN, **queue_order(), **forecast())


def forecast():
    """What the history weeks said to expect before the live week: threats auto-closed per day."""
    hist, live, platt, policy = ops.setup()
    z = policy.decide_many(hist.pc.to_numpy())
    act = z == "act"
    return dict(forecast_closed_hist=float((act & (hist.malicious.to_numpy() == 1)).sum() / 21),
                forecast_closed_expected=float(hist.pc.to_numpy()[act].sum() / 21))


def queue_order():
    r = ops.before_after()
    y = r["y"].astype(bool)
    rev = r["new"][0] == "review"
    t = ops.minutes(r["live"].timestamp)[rev]
    p = r["live"].pc.to_numpy()[rev]
    yt = y[rev]
    fifo, prio = ops.serve(t), ops.serve(t, priority=p)
    return dict(fifo_wait=float(np.median(fifo[yt & np.isfinite(fifo)])),
                prio_wait=float(np.median(prio[yt & np.isfinite(prio)])))


@figure(CH, "system")
def system():
    f, ax = draw.canvas("wide", 3.1)
    boxes = [
        (0.0, 2.3, "SIEM alert", "fields + text", "data", "Ch 1"),
        (1.25, 2.3, "Jev: P(attack),\nkind, severity", "typed questions", "jev", "Ch 10"),
        (2.5, 2.3, "calibrate", "Platt on history", "neutral", "Ch 3, 11"),
        (3.75, 2.3, "policy", "act · review · escalate", "review", "Ch 14"),
        (3.75, 1.2, "queue by P\nor page on-call", "capacity 240 / 40", "neutral", "Ch 14"),
        (2.5, 1.2, "LLM writes the\ncase note", "only for people", "llm", "Ch 15, 17"),
        (1.25, 1.2, "decision log", "state, answers,\npolicy version", "neutral", "Ch 14"),
        (0.0, 1.2, "monitor", "zone rates, queue,\ncalibration, audit", "fail", "Ch 11, 17"),
    ]
    w, h = 1.05, 0.62
    for x, y, t, sub, k, ref in boxes:
        draw.box(ax, x, y, w, h, t, kind=k, size=6.1, weight="semibold")
        draw.text(ax, x + w / 2, y - 0.06, sub, size=5.3, ha="center", va="top", color=C["ink2"])
        draw.pill(ax, x + w - 0.12, y + h, ref, kind="plain", size=4.9)
    for a, b in (((1.05, 2.61), (1.25, 2.61)), ((2.3, 2.61), (2.5, 2.61)), ((3.55, 2.61), (3.75, 2.61)),
                 ((4.27, 2.3), (4.27, 1.82)), ((3.75, 1.51), (3.55, 1.51)), ((2.5, 1.51), (2.3, 1.51)),
                 ((1.25, 1.51), (1.05, 1.51))):
        draw.arrow(ax, a, b, head=3)
    draw.text(ax, 1.9, 0.62, "labels from reviews and a random audit feed back into calibration", size=5.6,
              ha="center", color=C["ink2"], style="italic")
    return f


@figure(CH, "routes")
def routes():
    r = ops.before_after()
    y = r["y"].astype(bool)
    f, (a1, a2) = subplots(1, 2, width="text", height=2.0)
    ro, rn = r["old"][0], r["new"][0]
    # every part has its own pattern as well as its colour, named in the legend
    specs = [("old", [("closed unseen (back-striped)", ro == "closed", C["fail"], "\\\\\\\\"),
                      ("queue, first come, first served (dotted)", ro == "queue", C["slate"], "....")]),
             ("new", [("auto-closed (plain)", rn == "act", ZONE["act"], ""), ("review queue (striped)", rn == "review", ZONE["review"], "////"),
                      ("page on-call (cross-hatched)", rn == "escalate", ZONE["escalate"], "xxxx")])]
    for ax, mask, title in ((a1, np.ones_like(y), "where each day’s alerts go"), (a2, y, "where the real threats go")):
        clean(ax, "x")
        for row, (name, parts) in enumerate(specs):
            left = 0.0
            tot = mask.sum()
            for lab, m, col, hat in parts:
                v = (m & mask).sum() / tot
                ax.barh(1 - row, v, left=left, color=col, height=0.55, edgecolor="white", lw=1, hatch=hat or None)
                if v > 0.1:
                    ax.text(left + v / 2, 1 - row, f"{v:.0%}", ha="center", va="center", fontsize=6.0,
                            color="white" if col not in (ZONE["act"],) else C["ink"],
                            bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))
                left += v
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["before:\nrules", "after:\nJev + policy"] if ax is a1 else ["", ""], fontsize=6.2)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PCT)
        ax.set_title(title, fontsize=6.8, loc="left")
    handles = [(lab, col, hat) for _, parts in specs for lab, _, col, hat in parts]
    for lab, col, hat in handles:
        a1.barh(0, 0, color=col, ec="white", hatch=hat or None, label=lab)
    a1.legend(loc="upper left", bbox_to_anchor=(0, -0.3), ncol=3, fontsize=5.6, frameon=False)
    f.subplots_adjust(wspace=0.35, bottom=0.32)
    synthetic_tag(f)
    return f


@figure(CH, "time-to-human")
def time_to_human():
    r = ops.before_after()
    y = r["y"].astype(bool)
    f, ax = subplots(width="text", height=2.3)
    clean(ax, "both")
    grid = np.logspace(0, np.log10(7 * 24 * 60), 200)
    for (lab, (_, w), col) in (("before: rules + first come, first served", r["old"], C["slate"]),
                               ("after: pages at once, queue by probability", r["new"], C["jev"])):
        wt = w[y]
        share = [(wt <= g).mean() for g in grid]
        ax.plot(grid, share, color=col, lw=2)
        ax.text(grid[-1] * 1.1, share[-1], f"{lab}\n{share[-1]:.0%} ever seen", fontsize=5.8, va="center")
    ax.set_xscale("log")
    ax.set_xticks([1, 15, 60, 240, 1440, 10080])
    ax.set_xticklabels(["1 min", "15 min", "1 h", "4 h", "1 day", "1 week"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("time from alert to a person looking (log scale)")
    ax.set_ylabel("share of real threats")
    f.subplots_adjust(right=0.68)
    synthetic_tag(f)
    return f


@figure(CH, "shadow")
def shadow():
    r = ops.before_after()
    y = r["y"].astype(bool)
    rn = r["new"][0]
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "none")
    zones = ["act", "review", "escalate"]
    rows = [("harmless", ~y), ("real threat", y)]
    for j, z in enumerate(zones):
        for i, (lab, m) in enumerate(rows):
            n = int(((rn == z) & m).sum())
            v = n / 7
            miss = i == 1 and z == "act"      # real threats the system would close: red, outlined and striped
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (j, 1 - i), 0.96, 0.92, fc=ZONE[z] if not miss else C["fail"], alpha=0.9,
                ec=C["ink"] if miss else "none", lw=2.2 if miss else 0, hatch="////" if miss else None,
                hatchcolor="white"))
            ax.text(j + 0.48, 1 - i + 0.46, f"{v:,.1f}\na day" + ("\nmissed" if miss else ""), ha="center", va="center",
                    fontsize=6.4, color="white" if (z != "act" or i == 1) else C["ink"], fontweight="semibold",
                    bbox=dict(boxstyle="square,pad=0.15", fc=C["fail"], ec="none") if miss else None)
    ax.set_xlim(-0.05, 3)
    ax.set_ylim(-0.05, 2)
    ax.set_xticks([0.48, 1.48, 2.48])
    ax.set_xticklabels(["would auto-close", "would queue", "would page"], fontsize=6.3)
    ax.set_yticks([1.46, 0.46])
    ax.set_yticklabels(["harmless", "real threat"], fontsize=6.3)
    ax.tick_params(length=0)
    synthetic_tag(f)
    return f


@figure(CH, "campaign")
def campaign_fig():
    rows = ops.campaign()
    f, (a1, a2) = subplots(1, 2, width="text", height=2.1)
    days = np.arange(7)
    for respond, col, lab in ((False, C["fail"], "no response"), (True, C["jev"], "respond on day 3")):
        rr = [x for x in rows if x["respond"] == respond]
        a1.plot(days + 1, [x["reviews"] for x in rr], color=col, lw=1.8, marker="o", ms=3, label=lab)
        a2.plot(days + 1, np.cumsum([x["missed"] for x in rr]), color=col, lw=1.8, marker="o", ms=3, label=lab)
    base = [x for x in rows if not x["respond"]]
    a1.plot(days + 1, [ops.CAPACITY] * 7, color=C["fail"], lw=0.8, ls=(0, (3, 2)))
    resp_cap = [x["capacity"] for x in rows if x["respond"]]
    a1.plot(days + 1, resp_cap, color=C["jev"], lw=0.8, ls=(0, (3, 2)))
    a1.text(7, ops.CAPACITY - 12, "normal capacity", fontsize=5.4, va="top", ha="right", color=C["fail"])
    a1.text(7, resp_cap[-1] - 12, "with overtime", fontsize=5.4, va="top", ha="right", color=C["jev"])
    for ax, t in ((a1, "alerts sent to review per day"), (a2, "real threats never seen, cumulative")):
        clean(ax, "y")
        ax.set_title(t, fontsize=6.7, loc="left")
        ax.set_xticks(days + 1)
        ax.set_xlabel("day of the campaign week")
    a2.legend(fontsize=5.8, frameon=False, loc="upper left")
    a1.set_ylim(0, max(x["reviews"] for x in rows) * 1.25)
    f.subplots_adjust(wspace=0.35)
    synthetic_tag(f)
    return f


@figure(CH, "scorecard")
def scorecard():
    r = ops.before_after()
    so, sn = r["s_old"], r["s_new"]
    rows = [("real threats a person sees", f"{so['threats_seen_share']:.0%}", f"{sn['threats_seen_share']:.0%}"),
            ("real threats never seen, per day", f"{so['threats_unseen_per_day']:.1f}", f"{sn['threats_unseen_per_day']:.1f}"),
            ("real threats seen within an hour", f"{so['threats_within_hour_share']:.0%}",
             f"{sn['threats_within_hour_share']:.0%}"),
            ("median wait for a seen threat", fmt_min(so["median_wait_threat_min"]), fmt_min(sn["median_wait_threat_min"])),
            ("90th-percentile wait", fmt_min(so["p90_wait_threat_min"]), fmt_min(sn["p90_wait_threat_min"])),
            ("alerts seen per day (analysts + on-call)", f"{so['human_looks_per_day']:.0f}", f"{sn['human_looks_per_day']:.0f}"),
            ("analysts", str(ops.ANALYSTS), str(ops.ANALYSTS))]
    f, ax = draw.canvas("text", 2.65)
    draw.text(ax, 2.55, 2.2, "before", size=6.6, weight="bold", ha="center", color=C["slate"])
    draw.text(ax, 3.85, 2.2, "after", size=6.6, weight="bold", ha="center", color=C["jev"])
    for i, (lab, a, b) in enumerate(rows):
        y = 1.9 - i * 0.3
        if i % 2 == 0:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (0, y - 0.14), 4.7, 0.28, fc=C["neutral_t"], ec="none", zorder=0))
        draw.text(ax, 0.05, y, lab, size=6.3)
        draw.text(ax, 2.55, y, a, size=6.5, ha="center")
        draw.text(ax, 3.85, y, b, size=6.5, ha="center", weight="semibold")
    synthetic_tag(f)
    return f


def fmt_min(m):
    if not np.isfinite(m):
        return "–"
    if m < 90:
        return f"{m:.0f} min"
    if m < 48 * 60:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} days"


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    o, n = rr["old"], rr["new"]
    panels = [
        dict(num=1, title="The old way hid its misses", kind="fail", h=1.5,
             body=(f"Rules closed most alerts unseen. {o['threats_unseen_per_day']:.1f} real threats a day never reached "
                   "a person, and nobody could see which.")),
        dict(num=2, title="The same team, a different order", kind="jev", h=1.5,
             body=(f"With the same six analysts, the threats never seen fell to {n['threats_unseen_per_day']:.1f} a day, "
                   "and the likeliest threats were looked at first.")),
        dict(num=3, title="Shadow mode first", kind="neutral", h=1.5,
             body="For a week the system decided in silence while analysts worked as before. Compare, then switch on."),
        dict(num=4, title="Pages beat queues", kind="review", h=1.5,
             body=f"Escalated threats reach a person in about {rr['page_response_min']:.0f} minutes; queues are measured in hours."),
        dict(num=5, title="Campaigns break base rates", kind="fail", h=1.5,
             body=(f"A phishing campaign pushed reviews to {rr['camp_peak_reviews']} a day. Re-estimating the base rate "
                   "and adding overtime on day 3 cut the misses.")),
        dict(num=6, title="The log is the product", kind="neutral", h=1.5,
             body="Every decision is recorded with its state, answers and policy version. That is what you show an auditor."),
    ]
    return summary_page(CH, "Case study: SOC alert triage", panels,
                        footer="Next: the same decision layer, far from the SOC.")
