import numpy as np
from jevkit import calibration as cal, policy as pol


def test_perfect_and_bad_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert cal.ece(p, y) < 0.01
    over = np.clip(p ** 0.3, 0, 1)
    assert cal.ece(over, y) > 0.1
    t = cal.Temperature().fit(cal.sigmoid(cal.logit(p) * 3), y)
    assert 2.5 < t.T < 3.5


def test_brier_decomposition_adds_up():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    d = cal.brier_decomposition(p, y, n_bins=20)
    assert abs(d["reliability"] - d["resolution"] + d["uncertainty"] - d["brier"]) < 0.01


def test_policy_zones():
    P = pol.ThreeZonePolicy(0.1, 0.7)
    assert [P.decide(x) for x in (0.05, 0.3, 0.9)] == ["act", "review", "escalate"]
    assert pol.bayes_threshold(1, 9) == 0.1


def test_evaluate_counts():
    p = np.array([0.01, 0.2, 0.9, 0.95])
    y = np.array([0, 1, 1, 0])
    r = pol.evaluate(pol.ThreeZonePolicy(0.05, 0.8), p, y)
    assert (r["act"], r["review"], r["escalate"]) == (1, 1, 2)
    assert r["false_pages"] == 1
