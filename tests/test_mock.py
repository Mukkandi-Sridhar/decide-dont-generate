import pytest
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, TypeSafeUnprocessableEntityError, RetryPolicy
from typesafe_sdk import TypeSafeInternalServerError, TypeSafeAPIError

from jevkit import MockJevTransport, MODEL_NAME, client as make_client
from jevkit import soc


def client(**kw):
    return TypeSafeClient(api_key="mock", transport=MockJevTransport(**kw))


def test_official_client_roundtrip():
    c = client()
    r = c.system_one(state="I was charged twice. Please fix this ASAP.",
                     questions={"category": Choice(criteria={"billing": None, "technical": None, "other": None})})
    assert r.model == MODEL_NAME
    a = r.choices["category"]
    assert a.choice == "billing"
    assert abs(sum(a.probabilities.values()) - 1) < 1e-6
    assert a.confidence == max(a.probabilities.values())
    assert r.usage.input_tokens > 0


def test_deterministic():
    q = {"x": Noul(instructions="Is this spam?"), "s": Score(criteria=["low", "mid", "high"])}
    a = client().system_one(state="Click here to claim your FREE prize", questions=q)
    b = client().system_one(state="Click here to claim your FREE prize", questions=q)
    assert a.answers == b.answers


def test_all_types_and_ranges():
    alert = soc.generate(50)[3]
    r = client().system_one(state=alert.state(), questions={
        "malicious": Noul(instructions="Is this a real attack?"),
        "verdict": Choice(criteria={"malicious": None, "benign": None}),
        "category": Choice(criteria={c: None for c in soc.CATEGORIES}),
        "severity": Score(criteria=soc.SEVERITY_LEVELS),
    })
    p = r.nouls["malicious"].noul
    assert 0 <= p <= 1
    assert abs(r.choices["verdict"].probabilities["malicious"] - p) < 0.01
    s = r.scores["severity"]
    assert set(s.probabilities) == {0, 1, 2, 3}
    assert 0 <= s.score <= 3


def test_validation_errors_match_api_shape():
    with pytest.raises(TypeSafeUnprocessableEntityError):
        client().system_one(state="x", questions={"q": {"type": "choice", "criteria": {}}})


def test_unknown_model_is_404():
    with pytest.raises(TypeSafeAPIError):
        client().system_one(state="x", questions={"q": Noul(instructions="?")}, model="gpt-9")


def test_retries_on_simulated_503():
    t = MockJevTransport(fail_every=2)
    c = TypeSafeClient(api_key="mock", transport=t)
    for _ in range(4):
        c.system_one(state="hello", questions={"q": Noul(instructions="Is this a greeting?")})
    assert t.calls > 4  # some calls were retried


def test_no_retries_raises():
    t = MockJevTransport(fail_every=1)
    c = TypeSafeClient(api_key="mock", transport=t, retry=RetryPolicy(max_retries=0))
    with pytest.raises(TypeSafeInternalServerError):
        c.system_one(state="hello", questions={"q": Noul(instructions="?")})


def test_models_list_and_factory():
    c = make_client()
    names = [m.name for m in c.models.list().models]
    assert names == [MODEL_NAME]
