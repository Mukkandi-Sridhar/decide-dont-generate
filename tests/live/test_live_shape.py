"""Runs only with JEVKIT_LIVE=1 and TYPESAFE_API_KEY set. Checks the API shape, never quotes numbers."""
import os

import pytest
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

pytestmark = pytest.mark.skipif(not (os.environ.get("JEVKIT_LIVE") == "1" and os.environ.get("TYPESAFE_API_KEY")),
                                reason="no live API key")


def test_three_types_round_trip():
    with TypeSafeClient() as c:
        r = c.system_one(state="I was charged twice. Please fix this ASAP.", questions={
            "category": Choice(criteria={"billing": None, "technical": None}),
            "urgent": Noul(instructions="Is this urgent?"),
            "urgency": Score(criteria=["Can wait", "This week", "Today"]),
        })
    assert set(r.answers) == {"category", "urgent", "urgency"}
    assert 0 <= r.nouls["urgent"].noul <= 1
    assert abs(sum(r.choices["category"].probabilities.values()) - 1) < 0.02
