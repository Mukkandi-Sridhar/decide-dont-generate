"""The economics of cheap decisions (Chapter 12).

A deliberately simple, *illustrative* model of the decisions Kestrel could make in a day. Every number here
is an assumption chosen to be plausible, stated in the open, and easy to change. None is a measurement.

Each pool is a kind of decision. Every decision in a pool has a value: the expected loss it avoids, in dollars.
A decision is worth making when its value is above its price and it can be answered inside its time budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from jevkit import llm

JEV_PRICE_IN_PER_M = 0.042          # $ per million input tokens, vendor-reported
TOKENS_PER_DECISION = 500
JEV_PRICE = TOKENS_PER_DECISION * JEV_PRICE_IN_PER_M / 1e6            # $ per decision
LLM_PRICE = llm.MockLLM.simulated_cost_usd(TOKENS_PER_DECISION, 40)   # $ per decision (illustrative)
JEV_LATENCY = (0.07, 0.5)            # seconds, vendor-reported range
LLM_LATENCY = llm.MockLLM.simulated_latency_s(40)                      # seconds, illustrative


@dataclass(frozen=True)
class Pool:
    name: str
    per_day: int           # how many candidate decisions of this kind arise each day
    median_value: float    # $ of expected loss avoided by a typical decision
    spread: float          # standard deviation of log10(value)
    budget_s: float        # how long the decision may take before it's useless
    reviewable: bool       # can a person be asked to look at it?


POOLS = [
    Pool("alert triage", 714, 6.0, 0.6, 60.0, True),
    Pool("agent steps", 3_700, 0.5, 0.7, 2.0, False),
    Pool("checking LLM outputs", 20_000, 0.01, 0.7, 3.0, False),
    Pool("inbound emails", 60_000, 0.002, 0.8, 5.0, True),
    Pool("logins", 150_000, 0.0005, 0.8, 0.3, True),
    Pool("raw log events", 2_000_000, 0.00002, 0.9, 10.0, True),
]


@lru_cache(None)
def values(seed: int = 19) -> dict[str, np.ndarray]:
    """One simulated day: the value of every candidate decision, by pool."""
    rng = np.random.default_rng(seed)
    return {p.name: 10 ** rng.normal(np.log10(p.median_value), p.spread, p.per_day) for p in POOLS}


NEW_JOB = Pool("external file shares", 5_000_000, 0.00003, 0.8, 2.0, True)


def _pool_values(pools, seed):
    vals = dict(values(seed))
    for i, p in enumerate(pools):
        if p.name not in vals:
            rng = np.random.default_rng(seed + 101 + i)
            vals[p.name] = 10 ** rng.normal(np.log10(p.median_value), p.spread, p.per_day)
    return vals


def day(price: float, latency: float = 0.0, seed: int = 19, extra: tuple = ()) -> dict:
    """What a day looks like at a given price and latency per decision. `extra` adds pools (new jobs)."""
    pools = list(POOLS) + list(extra)
    vals = _pool_values(pools, seed)
    out = dict(price=price, latency=latency, decisions=0, spend=0.0, value=0.0, by_pool={})
    for p in pools:
        v = vals[p.name]
        made = (v > price) & (latency <= p.budget_s)
        n = int(made.sum())
        out["by_pool"][p.name] = n
        out["decisions"] += n
        out["spend"] += n * price
        out["value"] += float((v[made] - price).sum())
    return out


def flags(price: float, latency: float = 0.0, top_share: float = 0.001, review_cost: float = 15.0,
          seed: int = 19) -> dict:
    """Two ways to decide what goes to a person: a fixed share of everything decided, or a cost line."""
    vals = values(seed)
    decided = [vals[p.name][(vals[p.name] > price) & (latency <= p.budget_s)] for p in POOLS if p.reviewable]
    decided = np.concatenate(decided)
    return dict(decided=len(decided), top_share=int(round(top_share * len(decided))),
                cost_line=int((decided > review_cost).sum()))


def elastic_spend(price_ratio, elasticity: float):
    """Constant-elasticity demand: quantity ~ price^-e, so spend ~ price^(1-e). Returns spend relative to today."""
    return np.asarray(price_ratio, float) ** (1 - elasticity)


# An illustrative "decision audit" of one agent task (Chapter 22), reused by its lab.
TASK = [("read the ticket", "observe", 1), ("which tool next?", "decide", 4), ("is this enough?", "decide", 3),
        ("fetch records", "observe", 3), ("is this safe to do?", "decide", 1), ("do it", "act", 1),
        ("write the reply", "generate", 1)]


def task_cost(decider: str):
    """Seconds and dollars for one task, deciding with an LLM or a decision model (illustrative + vendor-reported)."""
    t = c = 0.0
    for _, kind, n in TASK:
        if kind == "decide":
            if decider == "llm":
                t += n * llm.MockLLM.simulated_latency_s(40)
                c += n * llm.MockLLM.simulated_cost_usd(500, 40)
            else:
                t += n * JEV_LATENCY[1]
                c += n * JEV_PRICE
        elif kind == "generate":
            t += llm.MockLLM.simulated_latency_s(150)
            c += llm.MockLLM.simulated_cost_usd(800, 150)
        else:
            t += n * 0.3
    return t, c
