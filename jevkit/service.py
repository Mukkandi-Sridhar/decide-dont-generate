"""A small, production-shaped decision service (Chapter 21).

    svc = DecisionService(config, client=jevkit.client())
    record = svc.decide(alert_fields)          # a DecisionRecord: what it saw, said and did, and why

Framework-agnostic: wrap `decide` in any web framework. Everything a production service needs is here in
miniature: a pinned, versioned config; typed questions; calibration; a policy with capacity; a fail-safe for every
failure; a decision log; monitors that raise flags; and a shadow mode for trying a new version safely.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from typesafe_sdk import Choice, Noul, Score, TypeSafeError

from jevkit import soc, calibration as cal, policy as pol


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Config:
    """Everything that changes a decision, in one versioned place."""
    version: str = "triage-2026.10.1"
    model: str = "jev-latest"                  # pin a dated model in production
    platt_a: float = 1.0
    platt_b: float = 0.0
    low: float = 0.03
    high: float = 0.34
    critical_asset_review: bool = True         # never auto-close critical assets
    fallback: str = "review"                   # what a failed call means
    audit_rate: float = 0.03                   # share of auto-closed alerts sent to a random audit

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:10]


def fit_config(history, version: str = "triage-2026.10.1", capacity_per_day: int = 240, pages_per_day: int = 40,
               p_col: str = "p", account_for_rules: bool = True) -> Config:
    """Calibrate on history and choose capacity-aware lines, as in Chapter 14.

    With `account_for_rules`, the critical-asset rule (which turns some 'act' into 'review') is included when the
    lines are chosen: the queue budget left for the model shrinks until the whole queue, rule included, fits.
    """
    platt = cal.Platt().fit(history[p_col].to_numpy(), history.malicious.to_numpy())
    pc = platt(history[p_col].to_numpy())
    y = history.malicious.to_numpy()
    crit = history.asset_criticality.to_numpy() >= 3
    per_day = len(history) / 21
    budget = capacity_per_day
    for _ in range(8):
        P = pol.best_policy_with_capacity(pc, y, budget / per_day, max_escalate_rate=pages_per_day / per_day)
        if not account_for_rules:
            break
        z = P.decide_many(pc)
        reviews = ((z == "review") | ((z == "act") & crit)).sum() / 21
        if reviews <= capacity_per_day:
            break
        budget -= reviews - capacity_per_day
    return Config(version=version, platt_a=platt.a, platt_b=platt.b, low=P.low, high=P.high)


# ----------------------------------------------------------------------------- the record
class DecisionRecord(BaseModel):
    decision_id: str
    alert_id: str
    config_version: str
    config_fingerprint: str
    model: str
    request_id: str | None
    p_raw: float | None
    p_calibrated: float | None
    kind: str | None
    severity: float | None
    zone: str
    reason: str
    audit: bool
    latency_ms: float


QUESTIONS = {
    "attack": Noul(instructions="Is this alert a real attack?"),
    "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
    "severity": Score(instructions="How severe is this?", criteria=soc.SEVERITY_LEVELS),
}


def state_of(alert) -> dict:
    """The state sent to the model: trusted fields only, never the free text (Chapter 7)."""
    keys = ["title", *soc.FEATURE_FIELDS]
    s = {k: (alert[k].item() if hasattr(alert[k], "item") else alert[k]) for k in keys}
    s["alert"] = s.pop("title")
    return s


# ----------------------------------------------------------------------------- the service
@dataclass
class DecisionService:
    config: Config
    client: object
    log_path: Path | None = None
    records: list = field(default_factory=list)

    def decide(self, alert) -> DecisionRecord:
        c = self.config
        t0 = time.perf_counter()
        aid = str(alert["alert_id"])
        base = dict(alert_id=aid, config_version=c.version, config_fingerprint=c.fingerprint(),
                    decision_id=hashlib.sha1(f"{aid}|{c.version}".encode()).hexdigest()[:12])
        try:
            r = self.client.system_one(state=state_of(alert), questions=QUESTIONS, model=c.model)
            p = float(r.nouls["attack"].noul)
            z = c.platt_a * np.log(max(p, 1e-6) / max(1 - p, 1e-6)) + c.platt_b
            pc = float(1 / (1 + np.exp(-z)))
            zone = pol.ThreeZonePolicy(c.low, c.high).decide(pc)
            reason = f"P={pc:.3f} against lines {c.low:.3f}/{c.high:.2f}"
            if zone == "act" and c.critical_asset_review and int(alert["asset_criticality"]) >= 3:
                zone, reason = "review", reason + "; critical asset is never auto-closed"
            audit = zone == "act" and _audit_draw(aid, c.audit_rate)
            rec = DecisionRecord(**base, model=getattr(r, "model", c.model), request_id=r.request_id, p_raw=p,
                                 p_calibrated=pc, kind=r.choices["kind"].choice, severity=r.scores["severity"].score,
                                 zone=zone, reason=reason, audit=audit, latency_ms=0.0)
        except TypeSafeError as e:
            rec = DecisionRecord(**base, model=c.model, request_id=None, p_raw=None, p_calibrated=None, kind=None,
                                 severity=None, zone=c.fallback, reason=f"fallback: {type(e).__name__}", audit=False,
                                 latency_ms=0.0)
        rec.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.records.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(rec.model_dump_json() + "\n")
        return rec


def _audit_draw(alert_id: str, rate: float) -> bool:
    """Deterministic 'random' audit: the same alert is always in or out, so audits are reproducible."""
    return int(hashlib.sha256(alert_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


# ----------------------------------------------------------------------------- monitoring
BANDS = dict(act=(0.45, 0.75), review=(0.25, 0.40), escalate=(0.02, 0.09), fallback=(0.0, 0.01))


def daily_report(records, labels: dict | None = None, capacity: int = 240, bands=BANDS, tolerance: float = 0.1) -> dict:
    """One day's health check. `labels` maps alert_id -> 0/1 for alerts a person has seen (reviews, pages, audits)."""
    n = len(records)
    zones = [r.zone for r in records]
    rates = {z: zones.count(z) / n for z in ("act", "review", "escalate")}
    rates["fallback"] = sum(r.reason.startswith("fallback") for r in records) / n
    flags = [f"{k} rate {v:.1%} outside {bands[k][0]:.0%}–{bands[k][1]:.0%}" for k, v in rates.items()
             if not bands[k][0] <= v <= bands[k][1]]
    reviews = zones.count("review")
    if reviews > capacity * (1 + tolerance):          # a little over is noise; a flag nobody trusts gets ignored
        flags.append(f"review queue {reviews} over capacity {capacity}")
    out = dict(n=n, rates=rates, reviews=reviews, flags=flags)
    if labels:
        seen = [r for r in records if r.alert_id in labels and r.p_calibrated is not None]
        if seen:
            p = np.array([r.p_calibrated for r in seen])
            y = np.array([labels[r.alert_id] for r in seen])
            out["labelled"] = len(seen)
            out["predicted"] = float(p.mean())
            out["observed"] = float(y.mean())
            if abs(out["observed"] - out["predicted"]) > max(0.02, 0.25 * out["predicted"]):
                flags.append(f"seen alerts: predicted {out['predicted']:.1%} real, observed {out['observed']:.1%}")
    return out


def shadow_compare(current: list, candidate: list) -> dict:
    """How often a candidate version would have decided differently, and in which direction."""
    order = ["act", "review", "escalate"]
    m = np.zeros((3, 3), int)
    for a, b in zip(current, candidate):
        if a.zone in order and b.zone in order:
            m[order.index(a.zone), order.index(b.zone)] += 1
    return dict(matrix=m, agree=float(np.trace(m) / m.sum()))
