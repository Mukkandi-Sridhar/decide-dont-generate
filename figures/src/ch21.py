"""Figures for Chapter 14: Act, review, or escalate."""

from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.patches import Rectangle

from jevkit import soc, calibration as cal, policy as pol
from jevkit.batch import score_alerts
from jevkit.figs import figure, draw, C, ZONE, ZONE_T, ZONE_HATCH, subplots, clean, results, summary_page, synthetic_tag
from jevkit.figs.style import ZONE_TEXT

CH = "ch21"
DAYS_LIVE = 7
REVIEWS_PER_ANALYST = 40      # 8 hours x 60 min / 12 min per review
PAGES_PER_DAY = 40            # what the on-call responders can absorb
ANALYSTS = 6


@lru_cache(None)
def data():
    df = soc.load()
    df["p"] = score_alerts(df)
    df["p_text"] = score_alerts(df, "text")
    hist, live = soc.history_and_live(df)
    per_day = len(hist) / 21
    platt = cal.Platt().fit(hist.p.values, hist.malicious.values)
    hist = hist.assign(pc=platt(hist.p.values))
    live = live.assign(pc=platt(live.p.values))
    cap = ANALYSTS * REVIEWS_PER_ANALYST / per_day
    esc = PAGES_PER_DAY / per_day
    policy = pol.best_policy_with_capacity(hist.pc.values, hist.malicious.values, cap, max_escalate_rate=esc)
    return dict(df=df, hist=hist, live=live, per_day=per_day, platt=platt, policy=policy, cap=cap, esc=esc)


def per_day(r):
    return {k: r[k] / DAYS_LIVE for k in ("act", "review", "escalate", "missed_by_automation", "false_pages",
                                            "attacks_escalated", "attacks", "cost")}


def record():
    d = data()
    live, hist, P = d["live"], d["hist"], d["policy"]
    y = live.malicious.values
    naive = per_day(pol.evaluate(pol.ThreeZonePolicy(0.5, 0.5), live.p.values, y))
    lo, hi = pol.cost_optimal_thresholds()
    ideal = per_day(pol.evaluate(pol.ThreeZonePolicy(lo, hi), live.pc.values, y))
    chosen = per_day(pol.evaluate(P, live.pc.values, y))
    act_raw = live.p.values < P.low
    act_cal = live.pc.values < P.low
    # misses under each model at 6 analysts
    Pt = pol.best_policy_with_capacity(hist.p_text.values, hist.malicious.values, d["cap"], max_escalate_rate=d["esc"])
    text = per_day(pol.evaluate(Pt, live.p_text.values, y))
    # hiring: 8 analysts
    P8 = pol.best_policy_with_capacity(hist.pc.values, hist.malicious.values, 8 * REVIEWS_PER_ANALYST / d["per_day"],
                                       max_escalate_rate=d["esc"])
    eight = per_day(pol.evaluate(P8, live.pc.values, y))
    # the act-zone blind spot: raw scores believed vs reality
    low_raw = live.p.values < 0.02
    results(CH,
            live_alerts=len(live), live_per_day=len(live) / DAYS_LIVE, live_attacks_per_day=float(y.sum() / DAYS_LIVE),
            naive=naive, ideal=ideal, chosen=chosen, text6=text, eight=eight,
            ideal_low=lo, ideal_high=hi, low=P.low, high=P.high, low8=P8.low,
            capacity=ANALYSTS * REVIEWS_PER_ANALYST, pages_cap=PAGES_PER_DAY, analysts=ANALYSTS,
            ideal_reviews_analysts=ideal["review"] / REVIEWS_PER_ANALYST,
            raw_believed=float(live.p.values[low_raw].mean()), raw_actual=float(y[low_raw].mean()),
            raw_ratio=float(y[low_raw].mean() / live.p.values[low_raw].mean()),
            platt_a=d["platt"].a, platt_b=d["platt"].b,
            act_cal_believed=float(live.pc.values[act_cal].mean()), act_cal_actual=float(y[act_cal].mean()),
            ece_raw=cal.ece(live.p.values, y), ece_cal=cal.ece(live.pc.values, y),
            saved_per_day_by_two=chosen["missed_by_automation"] - eight["missed_by_automation"],
            saved_money_per_day=round((chosen["missed_by_automation"] - eight["missed_by_automation"]) * 10_000, -3),
            )
    return d


def logx(ax, lo=5e-4):
    ax.set_xscale("log")
    ax.set_xlim(lo, 1)
    ticks = [0.001, 0.01, 0.1, 1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))


def zone_bands(ax, lo, hi, x0=5e-4, alpha=1.0, labels=True, ytext=None):
    for a, b, z in ((x0, lo, "act"), (lo, hi, "review"), (hi, 1, "escalate")):
        ax.axvspan(a, b, color=ZONE_T[z], zorder=0, lw=0)
        if labels:
            yy = ytext if ytext is not None else ax.get_ylim()[1]
            ax.text(np.sqrt(a * b), yy, z.upper(), ha="center", va="top", fontsize=6.4, fontweight="bold",
                    color=ZONE["escalate"] if z != "act" else "#6E5BA8")


@figure(CH, "zones-strip")
def zones_strip():
    d = record()
    live, P = d["live"], d["policy"]
    rng = np.random.default_rng(4)
    sample = live.sample(420, random_state=2)
    f, ax = subplots(width="text", height=2.15)
    ax.set_ylim(-1.3, 1.25)
    logx(ax)
    zone_bands(ax, P.low, P.high, ytext=1.18)
    ax.grid(False)
    for spine in ("left",):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    x = np.clip(sample.pc.values, 6e-4, 1)
    jit = rng.uniform(-0.85, 0.85, len(x))
    fine = sample.malicious.values == 0
    # harmless alerts are small open circles, real threats are bold crosses: they differ without colour
    ax.scatter(x[fine], jit[fine], s=9, facecolor="none", edgecolor=C["muted"], alpha=0.8, lw=0.5, zorder=2)
    ax.scatter(x[~fine], jit[~fine], s=16, color=C["fail"], marker="x", lw=1.1, zorder=3)
    for t in (P.low, P.high):
        ax.axvline(t, color=ZONE["escalate"], lw=0.8)
        ax.text(t, -1.12, f"{t:.3g}", ha="center", va="center", fontsize=6.2, color=ZONE["escalate"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.set_xlabel("P(real threat) after calibration  (log scale)")
    ax.scatter([7.2e-4], [-0.9], s=9, facecolor="none", edgecolor=C["muted"], lw=0.5, zorder=4)
    ax.text(8.6e-4, -0.9, "harmless (open circle)", fontsize=6.2, color=C["ink"], va="center")
    ax.scatter([7.2e-4], [-0.6], s=16, color=C["fail"], marker="x", lw=1.1, zorder=4)
    ax.text(8.6e-4, -0.6, "real threat (cross)", fontsize=6.2, color=C["ink"], va="center")
    synthetic_tag(f)
    return f


@figure(CH, "one-vs-three")
def one_vs_three():
    d = data()
    rr = __import__("json").load(open(__import__("jevkit").figs.ROOT / "results" / "ch21.json"))
    rows = [("One threshold at 0.5", rr["naive"]), (f"Three zones, {ANALYSTS} analysts", rr["chosen"])]
    f, axes = subplots(1, 2, width="text", height=1.75, gridspec_kw=dict(wspace=0.1))
    live = d["live"]
    y = live.malicious.values
    for col, (title, who) in enumerate((("Real threats per day", 1), ("Harmless alerts per day", 0))):
        ax = axes[col]
        clean(ax, "x")
        ax.set_title(title, fontsize=7.6)
        for i, (name, _) in enumerate(rows):
            P = pol.ThreeZonePolicy(0.5, 0.5) if i == 0 else d["policy"]
            pp = live.p.values if i == 0 else live.pc.values
            z = P.decide_many(pp)
            m = y == who
            vals = [np.sum((z == k) & m) / DAYS_LIVE for k in ("act", "review", "escalate")]
            left = 0
            for k, v in zip(("act", "review", "escalate"), vals):
                ax.barh(i, v, left=left, height=0.5, color=ZONE[k], edgecolor="white", linewidth=1.2,
                        hatch=ZONE_HATCH[k] or None)
                if v > (4 if who == 1 else 60):
                    ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.4, color=ZONE_TEXT[k],
                            fontweight="bold", bbox=dict(boxstyle="square,pad=0.15", fc=ZONE[k], ec="none"))
                left += v
        ax.set_ylim(1.6, -0.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r[0] for r in rows] if col == 0 else [], fontsize=6.6)
        ax.tick_params(axis="y", length=0)
    handles = [Rectangle((0, 0), 1, 1, fc=ZONE[k], ec="white", hatch=ZONE_HATCH[k] or None) for k in ("act", "review", "escalate")]
    f.legend(handles, ["act (auto-close; plain)", "review (analyst; striped)", "escalate (page on-call; cross-hatched)"],
             loc="lower center",
             ncol=3, bbox_to_anchor=(0.55, -0.2), fontsize=6.4)
    synthetic_tag(f)
    return f


@figure(CH, "cost-lines")
def cost_lines():
    c = pol.Costs()
    lo, hi = pol.cost_optimal_thresholds(c)
    p = np.logspace(np.log10(5e-4), 0, 400)
    act = p * c.auto_close_miss
    rev = c.review_cost + p * c.review_miss_rate * c.auto_close_miss
    esc = (1 - p) * c.escalate_false_alarm
    f, ax = subplots(width="text", height=2.45)
    clean(ax, "y")
    logx(ax)
    ax.set_yscale("log")
    ax.set_ylim(3, 20000)
    env = np.minimum(np.minimum(act, rev), esc)
    # act solid, review dashed, escalate dotted; the cheapest action is a wide grey band under them
    ax.plot(p, env, color=C["ink"], lw=5.5, ls="-", alpha=0.3, solid_capstyle="round")
    ax.plot(p, act, color="#9D89D6", lw=1.6, ls="-")
    ax.plot(p, rev, color=ZONE["review"], lw=1.6, ls=(0, (5, 2)))
    ax.plot(p, esc, color=ZONE["escalate"], lw=1.6, ls=":")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.text(0.09, 1500, "act (solid)", fontsize=7, color="#6E5BA8", ha="right", fontweight="bold")
    ax.text(0.02, 34, "review (dashed)", fontsize=7, color=ZONE["review"], fontweight="bold")
    ax.text(0.004, 520, "escalate (dotted)", fontsize=7, color=ZONE["escalate"], fontweight="bold")
    for t, lab in ((lo, f"{lo:.4f}"), (hi, f"{hi:.2f}")):
        ax.axvline(t, color=C["ink2"], lw=0.6)
        ax.text(t, 13000, lab, fontsize=6.2, ha="center", color=C["ink"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.set_xlabel("P(real threat)  (log scale)")
    ax.set_ylabel("Expected cost of the action (log)")
    ax.text(0.0035, 5.0, "grey band = the cheapest action at each P", fontsize=6.1, color=C["ink2"])
    return f


@figure(CH, "capacity")
def capacity():
    d = data()
    hist, live = d["hist"], d["live"]
    y = live.malicious.values
    ks = np.arange(3, 15)
    out = {}
    for name, hcol, lcol in (("structured", "pc", "pc"), ("text", "p_text", "p_text")):
        misses = []
        for k in ks:
            P = pol.best_policy_with_capacity(hist[hcol].values, hist.malicious.values,
                                              k * REVIEWS_PER_ANALYST / d["per_day"], max_escalate_rate=d["esc"])
            misses.append(pol.evaluate(P, live[lcol].values, y)["missed_by_automation"] / DAYS_LIVE)
        out[name] = np.array(misses)
    results(CH, capacity_curve={"analysts": ks.tolist(), "structured": out["structured"].round(2).tolist(),
                                "text": out["text"].round(2).tolist()})
    f, ax = subplots(width="text", height=2.35)
    clean(ax, "y")
    ax.plot(ks, out["text"], color=C["slate"], lw=1.5, ls="-", marker="s", ms=3.5, mec="white", mew=0.8)
    ax.plot(ks, out["structured"], color=C["jev"], lw=2, ls=(0, (5, 2)), marker="o", ms=4, mec="white", mew=0.8)
    ax.axvline(ANALYSTS, color=C["muted"], lw=0.7)
    ax.text(ANALYSTS + 0.12, ax.get_ylim()[1] * 0.93 if False else max(out["text"]) * 0.98, "Kestrel today:\n6 analysts",
            fontsize=6.3, color=C["ink2"], va="top")
    ax.text(8.3, out["text"][5] + 1.0, "mock reads the raw alert text (solid, squares)", fontsize=6.4, color=C["slate"], ha="left", va="bottom")
    ax.text(4.3, 5.2, "mock reads structured fields (dashed, circles)", fontsize=6.4, color=C["jev"], ha="left",
            va="top", fontweight="semibold")
    ax.set_ylim(0, max(out["text"]) * 1.08)
    ax.set_xlabel("Analysts on the review queue (40 reviews each per day)")
    ax.set_ylabel("Real threats auto-closed per day")
    ax.set_xticks(ks)
    synthetic_tag(f)
    return f


@figure(CH, "calibration-zones")
def calibration_zones():
    d = data()
    df, P = d["df"], d["policy"]
    y = df.malicious.values
    raw = df.p.values
    # cross-fitted Platt: fit on even days, apply to odd days and vice versa (so no alert calibrates itself)
    day = (df.timestamp.astype("datetime64[ns]") - np.datetime64("2026-09-01")).dt.days.values
    fixed = np.empty_like(raw)
    for part in (0, 1):
        m = day % 2 == part
        fixed[~m] = cal.Platt().fit(raw[m], y[m])(raw[~m])
    f, ax = subplots(width="text", height=2.45)
    clean(ax, "both")
    for pp, col, lab, ls, mk in ((raw, C["fail"], "raw mock scores (solid, squares)", "-", "s"),
                                 (fixed, C["jev"], "after Platt scaling (dashed, circles)", (0, (5, 2)), "o")):
        m = pp < 0.1
        qs = np.quantile(pp[m], np.linspace(0, 1, 9))
        idx = np.clip(np.searchsorted(qs, pp[m], side="right") - 1, 0, 7)
        mp = np.array([pp[m][idx == k].mean() for k in range(8)])
        fr = np.array([y[m][idx == k].mean() for k in range(8)])
        ax.plot(mp, fr, color=col, lw=1.5, ls=ls, marker=mk, ms=4, mec="white", mew=0.8, label=lab)
    ax.plot([0, 0.1], [0, 0.1], color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    ax.text(0.062, 0.058, "perfectly calibrated", fontsize=6.2, color=C["muted"], rotation=30)
    ax.axvspan(0, P.low, color=ZONE_T["act"], zorder=0)
    ax.axvline(P.low, color=ZONE["review"], lw=0.9, zorder=1)          # the zone's edge shows without its tint
    ax.text(P.low / 2, 0.075, "ACT", ha="center", fontsize=6.4, fontweight="bold", color="#6E5BA8")
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 0.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}" if v else "0"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}" if v else "0"))
    ax.set_xlabel("What the model said (average P in each group of alerts)")
    ax.set_ylabel("What actually happened")
    ax.legend(loc="lower right", fontsize=6.4)
    synthetic_tag(f)
    return f


@figure(CH, "production-loop")
def production_loop():
    f, ax = draw.canvas("text", 3.0)
    # main flow
    draw.box(ax, 0.0, 2.2, 0.8, 0.5, "alert", kind="data", size=7, weight="bold", sub="SIEM", subsize=5.8)
    draw.arrow(ax, (0.8, 2.45), (1.05, 2.45))
    draw.box(ax, 1.05, 2.2, 0.8, 0.5, "Jev", kind="jev", size=7, weight="bold", sub="P(threat)", subsize=5.8)
    draw.arrow(ax, (1.85, 2.45), (2.1, 2.45))
    draw.box(ax, 2.1, 2.2, 0.95, 0.5, "calibrate", kind="neutral", size=7, weight="bold", sub="Platt, v7", subsize=5.8)
    draw.arrow(ax, (3.05, 2.45), (3.3, 2.45))
    P = data()["policy"]
    draw.box(ax, 3.3, 2.2, 1.38, 0.5, "policy v3", kind="neutral", size=7, weight="bold", sub=f"{P.low:.3f} / {P.high:.2f}",
             subsize=5.8, color=C["ink"])
    # zones
    for i, (z, lab, sub) in enumerate((("act", "ACT", "auto-close"), ("review", "REVIEW", "analyst queue"),
                                      ("escalate", "ESCALATE", "page on-call"))):
        x = 2.25 + i * 0.83
        draw.box(ax, x, 1.2, 0.76, 0.5, lab, kind=z, size=6.6, weight="bold", fill=ZONE[z], textcolor=ZONE_TEXT[z],
                 sub=None)
        draw.text(ax, x + 0.38, 1.12, sub, size=5.8, ha="center", va="top", color=C["ink2"])
        draw.arrow(ax, (3.99, 2.2), (x + 0.38, 1.7), color=C["muted"], lw=0.7, head=3)
    # decision log
    draw.box(ax, 0.0, 1.2, 1.45, 0.5, "decision log", kind="plain", size=6.8, weight="bold",
             sub="what it saw, said, did", subsize=5.6, color=C["ink2"])
    draw.arrow(ax, (3.3, 2.3), (1.3, 1.7), color=C["muted"], lw=0.7, dashed=True, head=3, rad=0.1)
    # outcomes & feedback
    draw.box(ax, 2.25, 0.2, 2.42, 0.48, "outcomes become labels", kind="neutral", size=6.8, weight="bold",
             sub="reviews + escalations + a random 3% audit of ACT", subsize=5.6)
    for i in range(3):
        draw.arrow(ax, (2.63 + i * 0.83, 0.98), (2.63 + i * 0.83, 0.68), color=C["muted"], lw=0.7, head=3)
    draw.box(ax, 0.0, 0.2, 1.9, 0.48, "weekly check", kind="fail", size=6.8, weight="bold",
             sub="calibration drift? zone rates?", subsize=5.6)
    draw.arrow(ax, (2.25, 0.44), (1.9, 0.44), color=C["ink2"], lw=0.8)
    draw.arrow(ax, (1.75, 0.68), (2.4, 2.2), color=C["fail"], lw=0.9)
    draw.text(ax, 1.62, 0.95, "refit calibration,\nmove the thresholds", size=5.9, color=C["fail"], ha="right")
    return f


@figure(CH, "drift-watch")
def drift_watch():
    d = data()
    df, platt = d["df"], d["platt"]
    cw = soc.campaign_week()
    cw["p"] = score_alerts(cw)
    allw = __import__("pandas").concat([df[["timestamp", "p", "malicious"]], cw[["timestamp", "p", "malicious"]]])
    allw["pc"] = platt(allw.p.values)
    allw["day"] = (allw.timestamp.astype("datetime64[ns]") - np.datetime64("2026-09-01")).dt.days + 1
    g = allw.groupby("day").agg(pred=("pc", "mean"), actual=("malicious", "mean"), n=("pc", "size"))
    P = d["policy"]
    allw["zone"] = P.decide_many(allw.pc.values)
    rev = allw[allw.zone == "review"].groupby("day").size()
    results(CH, campaign_alerts_per_day=len(cw) / 7, campaign_actual=float(cw.malicious.mean()),
            campaign_pred=float(platt(cw.p.values).mean()),
            campaign_reviews_per_day=float((P.decide_many(platt(cw.p.values)) == "review").sum() / 7))
    f, (a1, a2) = subplots(2, 1, width="text", height=3.0, sharex=True, gridspec_kw=dict(hspace=0.3))
    for a in (a1, a2):
        clean(a, "y")
        a.axvspan(28.5, 35.5, color=C["fail_t"], zorder=0, lw=0)
        for xx in (28.5, 35.5):                                       # the campaign's edges show without its tint
            a.axvline(xx, color=C["fail"], lw=0.8, ls=":", zorder=1)
    a1.plot(g.index, g.actual, color=C["ink"], lw=1.5, ls="-")
    a1.plot(g.index, g.pred, color=C["jev"], lw=1.5, ls=(0, (5, 2)))
    a1.text(1, 0.135, "what really happened (solid line)", fontsize=6.3, color=C["ink"])
    a1.text(1, 0.035, "what the calibrated model expected (dashed line)", fontsize=6.3, color=C["jev"], fontweight="semibold")
    a1.text(32, 0.153, "phishing\ncampaign", fontsize=6.3, color=C["fail"], ha="center", va="top", fontweight="semibold")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_ylim(0, 0.16)
    a1.set_ylabel("Share of alerts\nthat were threats", fontsize=6.8)
    a2.bar(rev.index, rev.values, color=ZONE["review"], width=0.62)
    a2.axhline(ANALYSTS * REVIEWS_PER_ANALYST, color=C["ink"], lw=0.8)
    a2.text(1, ANALYSTS * REVIEWS_PER_ANALYST + 12, f"capacity: {ANALYSTS * REVIEWS_PER_ANALYST}/day", fontsize=6.3,
            color=C["ink"])
    a2.set_ylabel("Alerts sent\nto review", fontsize=6.8)
    a2.set_xlabel("Day")
    a2.set_ylim(0, 420)
    synthetic_tag(f)
    return f


@figure(CH, "summary")
def summary():
    import json
    rr = json.load(open(__import__("jevkit").figs.ROOT / "results" / "ch21.json"))
    d = data()
    P = d["policy"]

    def mini_bar(ax, x, y, w, h):
        draw.zone_bar(ax, x, y + h * 0.35, w, h * 0.45, 0.2, 0.62, sublabels=("auto-close", "analyst", "page"),
                      ticks=False, size=6.4)
        draw.text(ax, x, y + 0.02, f"Kestrel: act below {P.low:.3g}, escalate at {P.high:.2g} and above", size=6.1,
                  color=C["ink2"])

    def mini_costs(ax, x, y, w, h):
        for i, (lab, z) in enumerate((("P × miss", "act"), ("review + misses", "review"), ("(1−P) × page", "escalate"))):
            draw.pill(ax, x + 0.45 + i * (w / 3), y + h * 0.5, lab, kind=z, size=6.0)

    def mini_cap(ax, x, y, w, h):
        cc = rr["capacity_curve"]
        ks, ms = np.array(cc["analysts"]), np.array(cc["structured"])
        xx = x + (ks - ks.min()) / (ks.max() - ks.min()) * w
        yy = y + ms / ms.max() * h
        ax.plot(xx, yy, color=C["jev"], lw=1.4)
        draw.text(ax, x, y + h + 0.05, "threats auto-closed / day", size=5.6, color=C["muted"])
        draw.text(ax, x + w, y - 0.08, "more analysts →", size=5.6, color=C["muted"], ha="right")

    def mini_split(ax, x, y, w, h):
        tot = rr["naive"]["attacks"]
        for i, (lab, who) in enumerate((("one line", rr["naive"]), ("three zones", rr["chosen"]))):
            yy = y + h * (0.62 - i * 0.55)
            draw.text(ax, x, yy + 0.07, lab, size=5.8, color=C["ink2"])
            left = x + 0.7
            ww = w - 0.7
            vals = (who["missed_by_automation"], who["attacks"] - who["missed_by_automation"] - who["attacks_escalated"],
                    who["attacks_escalated"])
            for v, z in zip(vals, ("act", "review", "escalate")):
                if v <= 0:
                    continue
                ax.add_patch(Rectangle((left, yy), ww * v / tot, 0.14, fc=ZONE[z], ec="white", lw=1))
                left += ww * v / tot
        draw.text(ax, x + 0.7, y - 0.02, "real threats: auto-closed | reviewed | escalated", size=5.4, color=C["muted"])

    def mini_rel(ax, x, y, w, h):
        ax.plot([x, x + w * 0.5], [y, y + h], color=C["muted"], lw=0.7, ls=(0, (2, 2)))
        xs = np.linspace(0, 1, 8)
        ax.plot(x + xs * w * 0.5, y + np.minimum(1, xs * 1.7) * h, color=C["fail"], lw=1.2)
        ax.plot(x + xs * w * 0.5, y + xs * 1.03 * h, color=C["jev"], lw=1.2)
        draw.text(ax, x + w * 0.56, y + h * 0.75, "raw: says 0.6%,\nreally 1.4%", size=5.8, color=C["fail"])
        draw.text(ax, x + w * 0.56, y + h * 0.2, "recalibrated", size=5.8, color=C["jev"])

    def mini_week(ax, x, y, w, h):
        items = ["zone rates", "queue vs capacity", "calibration (ECE)", "audit 3% of ACT", "fail safe = REVIEW"]
        xx = x
        for it in items:
            t = draw.pill(ax, xx, y + h / 2, it, kind="neutral", size=6.0, ha="left")
            xx += 0.2 + 0.052 * len(it)

    def mini_log(ax, x, y, w, h):
        draw.box(ax, x, y, w, h, "{alert, P, calibrated P,\n policy v3, action, time}", kind="plain", size=6.0,
                 family="JetBrains Mono", color=C["ink2"])

    panels = [
        dict(num=1, title="One line is not enough", kind="fail", h=1.6, draw=mini_split, draw_h=0.55,
             body=(f"A single cut at 0.5 auto-closes {rr['naive']['missed_by_automation']:.0f} real threats a day. "
                   "Three zones let machines handle the easy ends and people handle the middle.")),
        dict(num=2, title="Act, review, escalate", kind="review", h=1.6, draw=mini_bar, draw_h=0.55,
             body="Act: the machine closes it alone. Review: an analyst checks. Escalate: someone is paged now."),
        dict(num=3, title="Costs set the lines", kind="neutral", h=1.6, draw=mini_costs, draw_h=0.4,
             body="Write down what each action costs as a function of P. The cheapest line at each P is your zone."),
        dict(num=4, title="Capacity moves the lines", kind="jev", h=1.6, draw=mini_cap, draw_h=0.42,
             body=(f"Six analysts can review {rr['capacity']} alerts a day, not {rr['ideal']['review']:.0f}. "
                   "Queue size is a business decision.")),
        dict(num=5, title="Only honest P can be thresholded", kind="data", h=1.6, draw=mini_rel, draw_h=0.5,
             body=(f"The raw mock’s “almost surely fine” alerts were {rr['raw_ratio']:.1f}× riskier than it "
                   "claimed. Recalibrate on recent history before you trust any line.")),
        dict(num=6, title="Log every decision", kind="neutral", h=1.6, draw=mini_log, draw_h=0.42,
             body="Jev gives no reasons, so audit the policy and the evidence: what it saw, what it said, what you did."),
        dict(title="Watch it every week", kind="fail", span=2, h=1.3, draw=mini_week, draw_h=0.3,
             body=("Zone rates, queue size and calibration drift. Audit a random slice of ACT, because otherwise you "
                   "never see the mistakes the machine closed on its own. Fail safe to REVIEW, never to ACT.")),
    ]
    return summary_page(CH, "Act, review, or escalate", panels,
                        footer="Next: the patterns that put a decision layer inside real systems.")
