"""Figures for Chapter 21: Capstone: a production decision service."""

import json
from functools import lru_cache

import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jevkit import soc, service as S, MockJevTransport
from jevkit.batch import score_alerts
from jevkit.figs import hatch_kw, figure, draw, C, ZONE, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch28"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
CAP = 240


@lru_cache(None)
def base():
    A = soc.load()
    A["p"] = score_alerts(A)
    H, L = soc.history_and_live(A)
    v1 = S.fit_config(H, version="triage-2026.10.1", account_for_rules=False)
    v2 = S.fit_config(H, version="triage-2026.10.2")
    return H, L, v1, v2


def client(**kw):
    return TypeSafeClient(api_key="mock", transport=MockJevTransport(**kw), retry=RetryPolicy(max_retries=0))


def days(df):
    t = df.timestamp.astype("datetime64[ns]").to_numpy()
    return ((t - t.min()) // np.timedelta64(1, "D")).astype(int)


@lru_cache(None)
def run(version: str, week: str = "live"):
    H, L, v1, v2 = base()
    cfg = v1 if version == "v1" else v2
    df = L if week == "live" else soc.campaign_week()
    svc = S.DecisionService(cfg, client())
    recs = [svc.decide(a) for _, a in df.iterrows()]
    return df.reset_index(drop=True), recs


def per_day_reports(version, week):
    df, recs = run(version, week)
    d = days(df)
    out = []
    y = df.malicious.to_numpy()
    for k in range(7):
        idx = np.where(d == k)[0]
        rr = [recs[i] for i in idx]
        labels = {recs[i].alert_id: int(y[i]) for i in idx if recs[i].zone != "act" or recs[i].audit}
        out.append(S.daily_report(rr, labels))
    return out


@lru_cache(None)
def failsafe():
    H, L, v1, v2 = base()
    sub = L.iloc[:1500]
    y = sub.malicious.to_numpy()
    out = {}
    for name, fb in (("review", "review"), ("act", "act")):
        cfg = S.Config(**{**v2.__dict__, "fallback": fb})
        svc = S.DecisionService(cfg, client(fail_rate=0.1, seed=28))
        recs = [svc.decide(a) for _, a in sub.iterrows()]
        fbm = np.array([r.reason.startswith("fallback") for r in recs])
        out[name] = dict(fallback_rate=float(fbm.mean()), threats_in_fallback=int((fbm & (y == 1)).sum()),
                         threats_closed_by_fallback=int((fbm & (y == 1)).sum()) if fb == "act" else 0)
    return out


def moved_threats():
    df, a = run("v1")
    _, b = run("v2")
    y = df.malicious.to_numpy()
    return int(sum(1 for i, (u, v) in enumerate(zip(a, b)) if u.zone == "review" and v.zone == "act" and y[i]))


def _example(rec) -> dict:
    """The example decision record. Its latency is measured live and varies from run to run, so the printed record
    keeps the value already in results/ch28.json; everything else is recomputed."""
    d = json.loads(rec.model_dump_json())
    path = ROOT / "results" / f"{CH}.json"
    if path.exists():
        d["latency_ms"] = json.loads(path.read_text()).get("example", {}).get("latency_ms", d["latency_ms"])
    return d


def record():
    H, L, v1, v2 = base()
    r1 = per_day_reports("v1", "live")
    r2 = per_day_reports("v2", "live")
    rc = per_day_reports("v2", "campaign")
    _, a = run("v1")
    _, b = run("v2")
    sh = S.shadow_compare(a, b)
    fs = failsafe()
    rec = b[0]
    results(CH, v1_low=v1.low, v2_low=v2.low, high=v2.high,
            v1_reviews=[x["reviews"] for x in r1], v2_reviews=[x["reviews"] for x in r2],
            v1_mean_reviews=float(np.mean([x["reviews"] for x in r1])),
            v2_mean_reviews=float(np.mean([x["reviews"] for x in r2])),
            campaign_flag_days=sum(bool(x["flags"]) for x in rc), live_flag_days_v2=sum(bool(x["flags"]) for x in r2),
            campaign_first_flag=next((i + 1 for i, x in enumerate(rc) if x["flags"]), None),
            campaign_flags_day1=rc[0]["flags"], shadow_agree=sh["agree"], shadow_matrix=sh["matrix"].tolist(),
            shadow_review_to_act_day=float(sh["matrix"][1, 0] / 7), shadow_moved_threats_day=moved_threats() / 7, fallback_rate=fs["review"]["fallback_rate"],
            fallback_threats=fs["review"]["threats_in_fallback"], fallback_act_closed=fs["act"]["threats_closed_by_fallback"],
            example=_example(rec), v2_fingerprint=v2.fingerprint())


@figure(CH, "architecture")
def architecture():
    f, ax = draw.canvas("wide", 2.8)
    steps = [("alert\narrives", "data"), ("build state:\ntrusted fields", "data"), ("Jev: three\ntyped questions", "jev"),
             ("calibrate\n(config v…)", "neutral"), ("policy + rules\n+ capacity", "review"), ("record +\nrespond", "neutral")]
    w, gap = 0.82, 0.2
    for i, (t, k) in enumerate(steps):
        x = i * (w + gap)
        draw.box(ax, x, 1.6, w, 0.6, t, kind=k, size=5.9, textcolor="white" if k == "review" else None,
                 fill=draw.KIND["review"][0] if k == "review" else None)
        if i < len(steps) - 1:
            draw.arrow(ax, (x + w, 1.9), (x + w + gap, 1.9), head=3)
    draw.box(ax, 2.04, 0.2, 1.6, 0.5, "any error or timeout →\nfallback: REVIEW", kind="fail", size=5.8)
    draw.arrow(ax, (2.45, 1.6), (2.6, 0.7), head=3, color=C["fail"], dashed=True)
    draw.cylinder(ax, 5.15, 0.65, 0.8, 0.55, kind="neutral", label="decision\nlog", size=5.6)
    draw.arrow(ax, (5.55, 1.6), (5.55, 1.2), head=3)
    draw.box(ax, 3.9, 0.05, 1.0, 0.45, "daily monitor\n+ flags", kind="neutral", size=5.8)
    draw.box(ax, 5.05, 0.05, 0.9, 0.45, "shadow:\nnext version", kind="jev", size=5.8, dashed=True)
    draw.arrow(ax, (5.2, 0.65), (4.6, 0.5), head=3, color=C["muted"])
    draw.box(ax, 0.0, 0.7, 1.7, 0.5, "config: model, calibration,\nlines, rules — one version", kind="plain", size=5.6)
    draw.arrow(ax, (1.7, 0.95), (3.4, 1.6), head=3, color=C["muted"], dashed=True)
    return f


@figure(CH, "record")
def record_fig():
    _, b = run("v2")
    rec = _example(b[0])
    f, ax = draw.canvas("text", 2.6)
    draw.box(ax, 0.0, 0.0, 2.9, 2.55, "", kind="plain")
    for i, (k, v) in enumerate(rec.items()):
        vv = f"{v:.4f}" if isinstance(v, float) else json.dumps(v)
        draw.text(ax, 0.1, 2.4 - i * 0.165, f'"{k}": {vv[:34]}', size=5.2, family="JetBrains Mono")
    notes = [(2.4 - 2 * 0.165, "which config decided, exactly"), (2.4 - 5 * 0.165, "trace a call with TypeSafe"),
             (2.4 - 7 * 0.165, "what the policy saw"), (2.4 - 10 * 0.165, "why, in words"),
             (2.4 - 11 * 0.165, "picked for the random audit?")]
    for y, t in notes:
        draw.arrow(ax, (3.55, y), (2.95, y), head=2.5, color=C["muted"])
        draw.text(ax, 3.6, y, t, size=5.8, color=C["ink2"])
    return f


@figure(CH, "dry-run")
def dry_run():
    H, L, v1, v2 = base()
    r1 = per_day_reports("v1", "live")
    r2 = per_day_reports("v2", "live")
    f, ax = subplots(width="text", height=2.0)
    clean(ax, "y")
    x = np.arange(1, 8)
    ax.bar(x - 0.2, [r["reviews"] for r in r1], width=0.38, label=f"thresholds fitted without the rule (low {v1.low:.3f}; solid)",
           **hatch_kw(0, C["fail"]))
    ax.bar(x + 0.2, [r["reviews"] for r in r2], width=0.38, label=f"thresholds fitted with the rule (low {v2.low:.3f}; striped)",
           **hatch_kw(1, C["jev"]))
    ax.axhline(CAP, color=C["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(0.45, CAP + 4, "capacity", fontsize=5.6, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xlabel("day of the live week")
    ax.set_ylabel("alerts sent to review")
    ax.set_ylim(0, max(r["reviews"] for r in r1) * 1.35)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=1)
    synthetic_tag(f)
    return f


@figure(CH, "failsafe")
def failsafe_fig():
    fs = failsafe()
    f, ax = subplots(width="text", height=1.6)
    clean(ax, "x")
    rows = [("fallback = REVIEW", fs["review"]["threats_in_fallback"], 0, C["jev"]),
            ("fallback = ACT (auto-close)", 0, fs["act"]["threats_closed_by_fallback"], C["fail"])]
    for i, (lab, seen, closed, col) in enumerate(rows):
        y = 1 - i
        v = seen or closed
        ax.barh(y, v, color=col, height=0.5)
        ax.text(v + 0.3, y, f"{v} real threats " + ("sent to a person" if seen else "closed unseen"), va="center",
                fontsize=6.0)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(0, max(r[1] or r[2] for r in rows) * 2.6)
    ax.set_xlabel(f"1,500 alerts with 10% of calls failing ({fs['review']['fallback_rate']:.0%} fell back)")
    synthetic_tag(f, "SIMULATED FAILURES")
    return f


@figure(CH, "monitor")
def monitor():
    rl = per_day_reports("v2", "live")
    rc = per_day_reports("v2", "campaign")
    reps = rl + rc
    f, (a1, a2) = subplots(2, 1, width="text", height=2.9, sharex=True)
    x = np.arange(1, 15)
    clean(a1, "y")
    clean(a2, "y")
    for xi, r in zip(x, reps):          # flagged days are red and striped, so they show without colour
        a1.bar(xi, r["reviews"], width=0.6, **hatch_kw(1 if r["flags"] else 0, C["fail"] if r["flags"] else C["jev"]))
    a1.axhline(CAP, color=C["ink"], lw=0.8, ls=(0, (3, 2)))
    a1.set_ylabel("reviews", fontsize=6)
    a1.set_title("review queue per day (red, striped: a flag was raised)", fontsize=6.6, loc="left")
    a2.plot(x, [r["predicted"] for r in reps], color=C["jev"], lw=1.6, ls="-", marker="o", ms=3, label="predicted share real (solid, circles)")
    a2.plot(x, [r["observed"] for r in reps], color=C["ink"], lw=1.6, ls=(0, (5, 2)), marker="s", ms=3,
            label="confirmed by people (dashed, squares)")
    a2.set_title("alerts people saw: predicted against confirmed", fontsize=6.6, loc="left")
    a2.yaxis.set_major_formatter(PCT)
    a2.legend(fontsize=5.8, frameon=False, loc="upper left")
    for ax in (a1, a2):
        ax.axvspan(7.5, 14.5, color=C["fail_t"], zorder=0, lw=0)
        ax.axvline(7.5, color=C["fail"], lw=0.8, ls=":", zorder=1)       # the campaign's start shows without its tint
    a1.set_ylim(0, 380)
    a1.text(11, 350, "phishing campaign week", fontsize=5.8, ha="center", color=C["fail"])
    a2.set_xticks(x)
    a2.set_xlabel("day")
    f.subplots_adjust(hspace=0.35)
    synthetic_tag(f)
    return f


@figure(CH, "shadow")
def shadow():
    _, a = run("v1")
    _, b = run("v2")
    m = S.shadow_compare(a, b)["matrix"]
    f, ax = subplots(width="text", height=1.9)
    clean(ax, "none")
    z = ["act", "review", "escalate"]
    for i in range(3):
        for j in range(3):
            v = m[i, j] / 7
            col = ZONE[z[j]] if i == j else (C["gold"] if v > 0 else C["neutral_t"])
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((j, 2 - i), 0.96, 0.92, fc=col))
            ax.text(j + 0.48, 2 - i + 0.46, f"{v:.1f}", ha="center", va="center", fontsize=6.4,
                    color="white" if (i == j and j > 0) else C["ink"])
    ax.set_xlim(-0.05, 3)
    ax.set_ylim(-0.05, 3)
    ax.set_xticks([0.48, 1.48, 2.48])
    ax.set_xticklabels([f"new: {t}" for t in z], fontsize=6.2)
    ax.set_yticks([2.46, 1.46, 0.46])
    ax.set_yticklabels([f"current: {t}" for t in z], fontsize=6.2)
    ax.tick_params(length=0)
    ax.set_title("alerts per day, current version against the candidate in shadow", fontsize=6.6, loc="left")
    synthetic_tag(f)
    return f


@figure(CH, "checklist")
def checklist():
    items = [("Pinned", "a dated model name, not an alias; the SDK version in the lockfile"),
             ("Versioned", "calibration, thresholds and rules in one config, with a fingerprint on every record"),
             ("Fail-safe", "every error has a decided meaning; test it by making calls fail"),
             ("Logged", "state, answers, probabilities, zone and reason for every decision"),
             ("Audited", "a random share of auto-closed cases goes to a person"),
             ("Monitored", "zone rates, queue, fallback rate, predicted against confirmed, every day"),
             ("Shadowed", "each new version runs silently beside the current one first"),
             ("Owned", "a named person reads the flags and can switch the service to review-everything")]
    f, ax = draw.canvas("text", 2.75)
    for i, (h, t) in enumerate(items):
        y = 2.6 - i * 0.33
        draw.box(ax, 0.0, y - 0.13, 0.26, 0.26, "✓", kind="jev", size=7)
        draw.text(ax, 0.38, y, h, size=6.6, weight="bold")
        draw.text(ax, 1.25, y, t, size=6.0, color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="One config, one fingerprint", kind="neutral", h=1.5,
             body="Model, calibration, thresholds and rules live in one versioned config. Every record says which one decided."),
        dict(num=2, title="The dry run found a bug", kind="fail", h=1.5,
             body=(f"A rule added after the lines were fitted pushed reviews to about {rr['v1_mean_reviews']:.0f} a day. "
                   f"Fitting with the rule brought them to {rr['v2_mean_reviews']:.0f}.")),
        dict(num=3, title="Failure has a meaning", kind="jev", h=1.5,
             body=(f"With 10% of calls failing, fallback to REVIEW sent {rr['fallback_threats']} real threats to a person. "
                   f"Fallback to ACT would have closed them unseen.")),
        dict(num=4, title="The record is the explanation", kind="neutral", h=1.5,
             body="State, answers, calibrated probability, zone and reason, for every decision, forever."),
        dict(num=5, title="Monitors that raise flags", kind="fail", h=1.5,
             body="Queue over capacity, zone rates out of band, confirmed threats above prediction: each becomes a flag a person reads."),
        dict(num=6, title="Shadow before switch", kind="jev", h=1.5,
             body=f"The new version agreed with the old on {rr['shadow_agree']:.0%} of alerts. Look at the rest before you switch."),
    ]
    return summary_page(CH, "Capstone: a production decision service", panels,
                        footer="Next: what changes now, for the industry and for you.")
