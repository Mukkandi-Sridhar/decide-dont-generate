"""Chapter 19: decision layers outside the SOC.

`DOMAINS` is an *illustrative* table: volumes and costs are round, plausible assumptions chosen to show how the
same arithmetic lands differently in different jobs. None is a measurement of any real organisation.
`tickets()` generates synthetic support tickets with known labels, for the one runnable demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    name: str
    state: str                   # what the model reads
    questions: tuple             # (type, text)
    per_day: int                 # decisions a day at a mid-sized organisation (illustrative)
    cost_miss: float             # $ cost of acting as if harmless when it wasn't (illustrative)
    cost_false: float            # $ cost of treating a harmless case as a problem (illustrative)
    human_final: bool            # should a person make the final call?
    none_option: str             # the option a choice must include so every case has somewhere to go


DOMAINS = [
    Domain("Support tickets", "the ticket text and customer tier",
           (("choice", "Which team? billing / technical / account / other"), ("noul", "Needs a reply today?"),
            ("score", "How upset? calm … angry")), 20_000, 60, 3, False, "other"),
    Domain("Content moderation", "a post and its thread",
           (("choice", "Which policy, if any? spam / harassment / … / none"), ("noul", "Remove before a person checks?")),
           2_000_000, 50, 2, False, "none"),
    Domain("Payment fraud", "the transaction and the account’s history",
           (("noul", "Is this payment fraudulent?"), ("choice", "Pattern? card testing / takeover / … / none")),
           500_000, 120, 4, False, "none"),
    Domain("Insurance claims", "the claim form and policy",
           (("noul", "Straight-through payment?"), ("score", "Fraud risk: low … high")), 3_000, 4_000, 60, True, "needs more info"),
    Domain("Clinical intake", "a triage note written by a nurse",
           (("score", "Urgency: routine … emergency"), ("noul", "Needs a clinician within the hour?")),
           400, 50_000, 150, True, "not enough information"),
    Domain("Code review gate", "a diff and its test results",
           (("noul", "Safe to merge without a second reviewer?"), ("choice", "Risk: none / security / data / performance")),
           1_500, 2_000, 30, False, "none"),
]


def line(d: Domain) -> float:
    """Chapter 4's line: act as if it's a problem when P > cost_false / (cost_false + cost_miss)."""
    return d.cost_false / (d.cost_false + d.cost_miss)


TOPICS = {
    "billing": ["I was charged twice for my {plan} plan", "My invoice shows the wrong amount",
                "Please refund last month, I cancelled", "Why did my card get charged again?",
                "The payment failed but money left my account"],
    "technical": ["The app crashes when I open {page}", "{page} shows an error every time",
                  "Export to PDF stopped working", "The sync is stuck at 90 percent", "Uploads fail with a timeout"],
    "account": ["I can’t log in to my account", "The password reset email never arrives",
                "Please change the email on my profile", "Two-factor codes are rejected", "I was locked out of my account"],
    "other": ["Do you have a dark mode?", "Is there a student discount?", "Where are you based?",
              "Can I suggest a feature?", "Do you sell gift cards?"],
}
URGENT = ["I need this fixed today.", "This is blocking my whole team.", "Client demo in an hour!"]
CALM = ["No rush.", "Whenever you get a chance.", "Thanks!"]


def tickets(n: int = 400, seed: int = 26):
    """Synthetic tickets with a true topic and a true 'needs a reply today' label."""
    rng = np.random.default_rng(seed)
    out = []
    topics = list(TOPICS)
    for _ in range(n):
        t = topics[rng.integers(len(topics))]
        body = TOPICS[t][rng.integers(len(TOPICS[t]))].format(plan=rng.choice(["Pro", "Team", "Basic"]),
                                                              page=rng.choice(["settings", "reports", "the dashboard"]))
        urgent = bool(rng.random() < 0.3)
        tail = URGENT[rng.integers(3)] if urgent else CALM[rng.integers(3)]
        out.append(dict(text=f"{body}. {tail}", topic=t, urgent=urgent))
    return out
