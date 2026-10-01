import numpy as np
from jevkit import soc


def test_deterministic_and_pinned():
    a = soc.generate(500, seed=7)
    b = soc.generate(500, seed=7)
    assert [x.description for x in a] == [x.description for x in b]
    df = soc.load()
    # These numbers are quoted in the book. If they move, the book is wrong.
    assert len(df) == 20000
    assert abs(df.malicious.mean() - 0.0758) < 0.002


def test_truth_is_calibrated():
    df = soc.load()
    from jevkit.calibration import ece
    assert ece(df.p_true, df.malicious) < 0.01


def test_feature_matrix():
    X = soc.feature_matrix(soc.load(1000))
    assert X.shape[0] == 1000 and not np.isnan(X.values).any()
