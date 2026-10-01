"""Turning probabilities into actions: act, review, or escalate.

In the SOC running example:
    act       -> the system closes the alert by itself (no human)
    review    -> an analyst looks at it in the queue
    escalate  -> page the on-call responder now
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ZONES = ("act", "review", "escalate")


def bayes_threshold(cost_fp: float, cost_fn: float) -> float:
    """For a yes/no action, act on 'yes' when p > c_fp / (c_fp + c_fn)."""
    return cost_fp / (cost_fp + cost_fn)


def expected_cost_of_action(p: float, cost_if_wrong_yes: float, cost_if_wrong_no: float) -> dict:
    """Expected cost of saying yes vs no when the chance of 'yes being true' is p."""
    return {"say_yes": (1 - p) * cost_if_wrong_yes, "say_no": p * cost_if_wrong_no}


@dataclass
class Costs:
    """What each outcome costs, in dollars. Defaults are the book's SOC assumptions (illustrative, not industry data)."""
    auto_close_miss: float = 10_000.0     # expected loss when we auto-close a real attack
    auto_close_ok: float = 0.0
    review_minutes: float = 12.0          # analyst time per reviewed alert
    analyst_per_hour: float = 75.0
    review_miss_rate: float = 0.05        # tired analysts miss some attacks too
    escalate_false_alarm: float = 400.0   # paging on-call for nothing
    escalate_hit: float = 0.0

    @property
    def review_cost(self) -> float:
        return self.review_minutes / 60 * self.analyst_per_hour


@dataclass
class ThreeZonePolicy:
    """p < low -> act (auto-close); low <= p < high -> review; p >= high -> escalate."""
    low: float = 0.02
    high: float = 0.6
    version: str = "v1"
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("need 0 <= low <= high <= 1")

    def decide(self, p: float) -> str:
        if p < self.low:
            return "act"
        if p < self.high:
            return "review"
        return "escalate"

    def decide_many(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return np.where(p < self.low, "act", np.where(p < self.high, "review", "escalate"))


def evaluate(policy: ThreeZonePolicy, p, y, costs: Costs | None = None) -> dict:
    """Run a policy over predictions `p` with ground truth `y` and count what happens."""
    costs = costs or Costs()
    p, y = np.asarray(p, float), np.asarray(y, int)
    z = policy.decide_many(p)
    n = len(p)
    act, rev, esc = z == "act", z == "review", z == "escalate"
    missed_auto = int(np.sum(act & (y == 1)))
    missed_review = float(np.sum(rev & (y == 1)) * costs.review_miss_rate)
    false_pages = int(np.sum(esc & (y == 0)))
    cost = (missed_auto * costs.auto_close_miss + rev.sum() * costs.review_cost
            + missed_review * costs.auto_close_miss + false_pages * costs.escalate_false_alarm)
    attacks = max(1, int(y.sum()))
    return dict(
        n=n,
        act=int(act.sum()), review=int(rev.sum()), escalate=int(esc.sum()),
        act_rate=float(act.mean()), review_rate=float(rev.mean()), escalate_rate=float(esc.mean()),
        missed_by_automation=missed_auto,
        attacks=int(y.sum()),
        attacks_escalated=int(np.sum(esc & (y == 1))),
        attack_recall_escalate=float(np.sum(esc & (y == 1)) / attacks),
        false_pages=false_pages,
        page_precision=float(np.sum(esc & (y == 1)) / max(1, esc.sum())),
        cost=float(cost),
        cost_per_alert=float(cost / n),
    )


def cost_surface(p, y, lows, highs, costs: Costs | None = None) -> np.ndarray:
    """Total cost for every (low, high) pair. NaN where low > high."""
    out = np.full((len(lows), len(highs)), np.nan)
    for i, lo in enumerate(lows):
        for j, hi in enumerate(highs):
            if lo <= hi:
                out[i, j] = evaluate(ThreeZonePolicy(lo, hi), p, y, costs)["cost"]
    return out


def best_policy(p, y, costs: Costs | None = None, grid=None) -> ThreeZonePolicy:
    grid = grid if grid is not None else np.round(np.concatenate([np.linspace(0.002, 0.05, 25), np.linspace(0.06, 0.95, 90)]), 4)
    surf = cost_surface(p, y, grid, grid, costs)
    i, j = np.unravel_index(np.nanargmin(surf), surf.shape)
    return ThreeZonePolicy(float(grid[i]), float(grid[j]))


def coverage_risk(confidence, correct):
    """Sort by confidence. For each coverage level (fraction automated), the error rate among automated cases."""
    c = np.asarray(confidence, float)
    ok = np.asarray(correct, float)
    order = np.argsort(-c)
    errs = np.cumsum(1 - ok[order])
    k = np.arange(1, len(c) + 1)
    return k / len(c), errs / k, c[order]


def queue_load(review_count_per_day: float, minutes_per_review: float, analysts: int, hours_per_shift: float = 8.0) -> float:
    """Fraction of analyst capacity the review queue uses. Above ~0.8, queues grow and people burn out."""
    return review_count_per_day * minutes_per_review / (analysts * hours_per_shift * 60)


def best_policy_with_capacity(p, y, max_review_rate: float, costs: Costs | None = None, highs=None,
                              max_escalate_rate: float = 1.0) -> ThreeZonePolicy:
    """Cheapest policy whose review queue holds at most `max_review_rate` of all alerts.

    For each candidate `high`, the best `low` is the smallest one that still fits the queue,
    because every alert pulled out of 'act' and into 'review' lowers the expected cost.
    """
    p = np.asarray(p, float)
    srt = np.sort(p)
    cap = int(np.floor(max_review_rate * len(p)))
    highs = highs if highs is not None else np.round(np.linspace(0.05, 0.95, 91), 3)
    best, best_cost = None, np.inf
    for hi in highs:
        k = int(np.searchsorted(srt, hi, side="left"))      # alerts with p < hi
        if (len(p) - k) > max_escalate_rate * len(p):
            continue                                          # the on-call team can't take that many pages
        lo = 0.0 if k <= cap else float(srt[k - cap])
        lo = min(lo, float(hi))
        pol = ThreeZonePolicy(lo, float(hi))
        c = evaluate(pol, p, y, costs)["cost"]
        if c < best_cost:
            best, best_cost = pol, c
    return best


def cost_optimal_thresholds(costs: Costs | None = None) -> tuple[float, float]:
    """Where the expected-cost lines cross, assuming the probabilities are calibrated.

    act costs      p * miss
    review costs   review + p * miss_rate * miss
    escalate costs (1 - p) * false_page
    """
    c = costs or Costs()
    low = c.review_cost / (c.auto_close_miss * (1 - c.review_miss_rate))
    high = (c.escalate_false_alarm - c.review_cost) / (c.escalate_false_alarm + c.review_miss_rate * c.auto_close_miss)
    return low, high
