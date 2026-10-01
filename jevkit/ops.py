"""Chapter 18's case study: Kestrel's SOC before and after the decision layer.

Everything here is a simulation on synthetic alerts. The pieces are the book's own: the old way is the
hand-written rules from Chapter 13 feeding a first-come-first-served queue; the new way is mock Jev,
Platt-calibrated on history, with Chapter 14's capacity-aware three-zone policy.
"""

from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np

from jevkit import soc, calibration as cal, policy as pol
from jevkit.bakeoff import rules
from jevkit.batch import score_alerts

ANALYSTS = 6
REVIEWS_PER_ANALYST = 40            # per day: 8 hours / 12 minutes
CAPACITY = ANALYSTS * REVIEWS_PER_ANALYST
PAGES_PER_DAY = 40
PAGE_RESPONSE_MIN = 15.0            # minutes from a page to a person looking (assumption)


def minutes(ts) -> np.ndarray:
    t = ts.astype("datetime64[ns]").to_numpy()
    return (t - t.min()).astype("timedelta64[s]").astype(float) / 60


def serve(arrivals, priority=None, per_day: int = CAPACITY, max_age_min: float = 24 * 60):
    """A queue worked around the clock at `per_day` items a day. Returns minutes waited (inf if never reached).

    priority=None serves first come, first served; otherwise higher priority first among waiting items (ties by
    arrival). Items still waiting after `max_age_min` age out: they are closed without anyone looking.
    """
    arrivals = np.asarray(arrivals, float)
    order = np.argsort(arrivals, kind="stable")
    service = 24 * 60 / per_day
    wait = np.full(len(arrivals), np.inf)
    heap, i, t = [], 0, 0.0
    # the week ends at the end of the last arrival's day: work doesn't carry on into an eighth day, so no more than
    # `per_day` items a day are ever served, and anything still waiting then counts as never reached
    end = np.ceil((arrivals.max() + 1e-9) / (24 * 60)) * 24 * 60 if len(arrivals) else 0.0
    while (i < len(order) or heap) and t < end:
        while i < len(order) and arrivals[order[i]] <= t:
            k = order[i]
            key = -priority[k] if priority is not None else arrivals[k]
            heapq.heappush(heap, (key, arrivals[k], k))
            i += 1
        if not heap:
            t = arrivals[order[i]]
            continue
        _, a, k = heapq.heappop(heap)
        if t - a > max_age_min:
            continue                      # aged out: nobody looks, and no analyst time is spent
        wait[k] = t - a
        t += service
    return wait


@lru_cache(None)
def setup():
    df = soc.load()
    df["p"] = score_alerts(df)
    hist, live = soc.history_and_live(df)
    platt = cal.Platt().fit(hist.p.to_numpy(), hist.malicious.to_numpy())
    hist = hist.assign(pc=platt(hist.p.to_numpy()))
    live = live.assign(pc=platt(live.p.to_numpy()))
    per_day = len(hist) / 21
    policy = pol.best_policy_with_capacity(hist.pc.to_numpy(), hist.malicious.to_numpy(), CAPACITY / per_day,
                                           max_escalate_rate=PAGES_PER_DAY / per_day)
    return hist, live, platt, policy


def old_way(df):
    """Every alert joins one queue. Alerts the hand-written rules flag go first; otherwise first come, first
    served. Whatever nobody reaches within a day ages out, closed unseen."""
    flag = rules(df)
    t = minutes(df.timestamp)
    wait = serve(t, priority=flag.astype(float))
    route = np.where(np.isfinite(wait), "queue", "closed")
    return route, wait


def new_way(df, pc, policy):
    """Escalations page someone; reviews queue by probability, highest first; the rest are auto-closed."""
    z = policy.decide_many(pc)
    t = minutes(df.timestamp)
    wait = np.full(len(df), np.inf)
    esc = z == "escalate"
    wait[esc] = PAGE_RESPONSE_MIN
    rev = z == "review"
    wait[rev] = serve(t[rev], priority=np.asarray(pc)[rev])
    return z, wait


def summary(route, wait, y, days: float = 7) -> dict:
    y = np.asarray(y).astype(bool)
    seen = np.isfinite(wait)
    return dict(
        threats_per_day=float(y.sum() / days),
        threats_unseen_per_day=float((y & ~seen).sum() / days),
        threats_seen_share=float((y & seen).sum() / y.sum()),
        median_wait_threat_min=float(np.median(wait[y & seen])) if (y & seen).any() else float("nan"),
        p90_wait_threat_min=float(np.percentile(wait[y & seen], 90)) if (y & seen).any() else float("nan"),
        human_looks_per_day=float(seen.sum() / days),
        threats_within_hour_share=float((y & (wait <= 60)).sum() / y.sum()),
    )


@lru_cache(None)
def before_after():
    hist, live, platt, policy = setup()
    y = live.malicious.to_numpy()
    r_old, w_old = old_way(live)
    r_new, w_new = new_way(live, live.pc.to_numpy(), policy)
    return dict(live=live, y=y, old=(r_old, w_old), new=(r_new, w_new), policy=policy,
                s_old=summary(r_old, w_old, y), s_new=summary(r_new, w_new, y))


@lru_cache(None)
def campaign(respond_day: int = 2, extra_analysts: int = 2):
    """The phishing-campaign week, day by day, without and with a response from `respond_day` (0-based)."""
    hist, live, platt, policy = setup()
    cw = soc.campaign_week()
    cw["p"] = score_alerts(cw)
    cw["pc"] = platt(cw.p.to_numpy())
    day = (minutes(cw.timestamp) // (24 * 60)).astype(int)
    y = cw.malicious.to_numpy()
    link = cw.rule.isin(["suspicious_link", "lookalike_domain"]).to_numpy()
    rows = []
    for respond in (False, True):
        for d in range(7):
            m = day == d
            p = cw.pc.to_numpy()[m].copy()
            cap, P = CAPACITY, policy
            if respond and d >= respond_day:
                # re-estimate the link alerts' base rate from what reviewers confirmed so far, and add overtime
                seen = (day < d) & link
                b_hist = float(hist.malicious[hist.rule.isin(["suspicious_link", "lookalike_domain"])].mean())
                b_now = float(y[seen].mean())
                lm = link[m]
                p[lm] = cal.prior_shift(p[lm], b_hist, b_now)
                cap = (ANALYSTS + extra_analysts) * REVIEWS_PER_ANALYST
                P = pol.best_policy_with_capacity(hist.pc.to_numpy(), hist.malicious.to_numpy(),
                                                  cap / (len(hist) / 21), max_escalate_rate=PAGES_PER_DAY / (len(hist) / 21))
            z = P.decide_many(p)
            yy = y[m]
            reviews = int((z == "review").sum())
            reviewed_share = min(1.0, cap / reviews) if reviews else 1.0
            missed = float(((z == "act") & (yy == 1)).sum() + ((z == "review") & (yy == 1)).sum() * (1 - reviewed_share))
            rows.append(dict(respond=respond, day=d, alerts=int(m.sum()), reviews=reviews, capacity=cap,
                             pages=int((z == "escalate").sum()), predicted=float(p.mean()), actual=float(yy.mean()),
                             missed=missed, link_share=float(link[m].mean())))
    return rows
