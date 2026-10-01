"""A small SOC triage agent that runs an Observe -> Decide -> Act loop (Chapter 7 onward).

Every step is recorded with its kind:
    observe   - a tool that reads something (the alert, threat intel, host history, a policy)
    decide    - a small typed decision (noul / choice / score) answered by the decision model
    generate  - free text written by the LLM (a case note, a page message)
    act       - a tool that changes something (close, queue, page)

Decisions go through the official TypeSafe SDK (mock transport by default); generation goes
through the synthetic MockLLM. Nothing here is a real product; it's a teaching harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from typesafe_sdk import Choice, Noul, Score

from . import soc, kb, llm, policy as pol, client as make_client

JEV_PRICE_IN_PER_M = 0.042      # $ per million input tokens, vendor-reported

EVIDENCE = {
    "threat_intel": "Look up the reputation of the addresses and domains in the alert",
    "host_history": "Check what this host and user did in the last 24 hours",
    "policy": "Check whether an approved tool or change ticket explains it",
    "none": "No more evidence needed",
}


@dataclass
class Step:
    kind: str
    name: str
    detail: str = ""
    latency_s: float = 0.0


@dataclass
class Trace:
    alert_id: str
    steps: list[Step] = field(default_factory=list)
    action: str = ""
    p: float | None = None
    cost_usd: float = 0.0
    parse_failed: bool = False

    def add(self, kind, name, detail="", latency_s=0.0):
        self.steps.append(Step(kind, name, detail, latency_s))

    def count(self, kind):
        return sum(s.kind == kind for s in self.steps)


class SOCAgent:
    def __init__(self, policy: pol.ThreeZonePolicy | None = None, client=None, max_evidence: int = 3,
                 decide_latency_s: float | None = None, decider: str = "jev", enough_line: float = 0.5,
                 always: tuple = ()):
        """decider="jev": typed decisions go to the decision model (the hybrid agent).
        decider="llm": the same loop, but every decision is charged as an LLM JSON call, and the verdict is the
        LLM's own JSON answer (a broken answer falls back to review). Evidence steps are kept identical so the two
        agents differ only in who decides."""
        self.decider = decider
        self.enough_line = enough_line
        self.always = tuple(always)      # evidence gathered unconditionally, before any decision
        self.policy = policy or pol.ThreeZonePolicy(0.03, 0.34)
        self.client = client or make_client()
        self.llm = llm.MockLLM()
        self.max_evidence = max_evidence
        self.retriever = kb.Retriever([t for _, t in kb.corpus(30, with_distractors=False)], "char")
        self.decide_latency_s = decide_latency_s

    def _decide(self, trace, name, state, questions):
        r = self.client.system_one(state=state, questions=questions)
        if self.decider == "llm":
            tokens_in = r.usage.input_tokens + 150            # plus a prompt asking for JSON
            lat = llm.MockLLM.simulated_latency_s(40 * len(questions))
            trace.cost_usd += llm.MockLLM.simulated_cost_usd(tokens_in, 40 * len(questions))
        else:
            lat = self.decide_latency_s
            if lat is None:
                lat = float(r.raw_http_response.headers.get("x-jevkit-simulated-latency-ms", "150")) / 1000
            trace.cost_usd += r.usage.input_tokens * JEV_PRICE_IN_PER_M / 1e6
        trace.add("decide", name, ", ".join(f"{k}" for k in questions), lat)
        return r

    def _observe(self, t, state, a, pick):
        if pick == "threat_intel":
            state["ioc_score"] = float(a.ioc_score)
            t.add("observe", "threat_intel", f"score {a.ioc_score:.2f}", 0.3)
        elif pick == "host_history":
            state["prior_alerts_24h"] = int(a.prior_alerts_24h)
            state["after_hours"] = bool(a.after_hours)
            t.add("observe", "host_history", f"{a.prior_alerts_24h} related alert{'s' if a.prior_alerts_24h != 1 else ''}", 0.4)
        else:
            hits = self.retriever.search(a.description, k=1)
            state["known_tool"] = bool(a.known_tool)
            state["policy_note"] = hits[0][1][:120]
            t.add("observe", "policy_lookup", hits[0][1][:50], 0.2)

    def run(self, alert_row) -> Trace:
        a = alert_row
        t = Trace(alert_id=a.alert_id)
        state = {"alert": a.title, "rule": a.rule, "description": a.description, "department": a.department,
                 "asset_criticality": int(a.asset_criticality), "role": a.role}
        t.add("observe", "get_alert", a.title, 0.05)
        gathered = set()
        for pick in self.always:
            gathered.add(pick)
            self._observe(t, state, a, pick)
        for _ in range(self.max_evidence):
            options = {k: v for k, v in EVIDENCE.items() if k not in gathered}
            r = self._decide(t, "next_evidence", state, {"next": Choice(instructions="What evidence should we gather next?",
                                                                          criteria=options)})
            pick = r.choices["next"].choice
            t.steps[-1].detail = f"-> {pick}  (p={r.choices['next'].confidence:.2f})"
            if pick == "none" or len(options) == 1:
                break
            gathered.add(pick)
            self._observe(t, state, a, pick)
            r = self._decide(t, "enough_evidence", state, {"enough": Noul(
                instructions="Is there enough evidence to judge this alert?")})
            t.steps[-1].detail = f"P(enough) = {r.nouls['enough'].noul:.2f}"
            if r.nouls["enough"].noul > self.enough_line:
                break
        # the main verdict, with the extra typed answers the policy needs
        for k in soc.FEATURE_FIELDS:
            state.setdefault(k, getattr(a, k) if k in ("rule", "role", "asset_criticality", "mb_out", "new_geo", "mfa_ok")
                             else state.get(k))
        state = {k: (v.item() if hasattr(v, "item") else v) for k, v in state.items() if v is not None}
        r = self._decide(t, "verdict", state, {
            "attack": Noul(instructions="Is this alert a real attack?"),
            "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
            "severity": Score(instructions="How severe is this?", criteria=soc.SEVERITY_LEVELS)})
        p = r.nouls["attack"].noul
        if self.decider == "llm":
            raw = self.llm.structured(a.description)
            try:
                o = json.loads(raw)
                p = float(o["confidence"]) if o["verdict"] == "malicious" else 1 - float(o["confidence"])
            except (ValueError, KeyError, TypeError):
                p, t.parse_failed = None, True
        t.p = p
        if p is None:
            t.steps[-1].detail = "LLM answer could not be parsed: send to review"
            action = "review"
        else:
            t.steps[-1].detail = f"P(attack) = {p:.3f}, {r.choices['kind'].choice}, severity {r.scores['severity'].score:.1f}"
            action = self.policy.decide(p)
        if action == "escalate" or (action == "act" and int(a.asset_criticality) >= 3):
            action = "escalate" if action == "escalate" else "review"
        t.action = action
        if action == "act":
            t.add("act", "close_alert", "auto-closed", 0.05)
        elif action == "review":
            t.add("generate", "case_note", "summary for the analyst", self.llm.simulated_latency_s(120))
            t.cost_usd += llm.MockLLM.simulated_cost_usd(600, 120)
            t.add("act", "queue_for_analyst", "", 0.05)
        else:
            t.add("generate", "page_message", "what, where, why now", self.llm.simulated_latency_s(80))
            t.cost_usd += llm.MockLLM.simulated_cost_usd(600, 80)
            t.add("act", "page_oncall", r.choices["kind"].choice, 0.1)
        return t
