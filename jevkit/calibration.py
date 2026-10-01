"""Calibration tools: is a 0.8 really an 80%?

Small, readable implementations. Every function takes plain arrays:
`p` are predicted probabilities for the positive class, `y` are 0/1 outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12


def _arr(p, y=None):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if y is None:
        return p
    return p, np.asarray(y, float)


@dataclass
class Bins:
    edges: np.ndarray
    mean_pred: np.ndarray     # average predicted probability in each bin
    frac_pos: np.ndarray      # observed frequency of positives in each bin
    count: np.ndarray


def reliability(p, y, n_bins: int = 10, strategy: str = "uniform") -> Bins:
    """Group predictions into bins and compare 'what we said' with 'what happened'."""
    p, y = _arr(p, y)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    count = np.bincount(idx, minlength=k).astype(float)
    sp = np.bincount(idx, weights=p, minlength=k)
    sy = np.bincount(idx, weights=y, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Bins(edges, sp / count, sy / count, count)


def ece(p, y, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: the average gap between confidence and reality, weighted by bin size."""
    b = reliability(p, y, n_bins, strategy)
    m = b.count > 0
    return float(np.sum(b.count[m] * np.abs(b.mean_pred[m] - b.frac_pos[m])) / b.count.sum())


def mce(p, y, n_bins: int = 10) -> float:
    """Maximum calibration error: the worst bin's gap."""
    b = reliability(p, y, n_bins)
    m = b.count > 0
    return float(np.max(np.abs(b.mean_pred[m] - b.frac_pos[m])))


def brier(p, y) -> float:
    """Mean squared error between probability and outcome. Lower is better. 0.25 is a coin flip on balanced data."""
    p, y = _arr(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    """Average surprise. Punishes confident mistakes very hard."""
    p, y = _arr(p, y)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_decomposition(p, y, n_bins: int = 10) -> dict:
    """Murphy's decomposition: Brier ~= reliability - resolution + uncertainty."""
    p, y = _arr(p, y)
    b = reliability(p, y, n_bins)
    m = b.count > 0
    n = len(p)
    base = y.mean()
    rel = np.sum(b.count[m] * (b.mean_pred[m] - b.frac_pos[m]) ** 2) / n
    res = np.sum(b.count[m] * (b.frac_pos[m] - base) ** 2) / n
    unc = base * (1 - base)
    return dict(reliability=float(rel), resolution=float(res), uncertainty=float(unc), brier=brier(p, y))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


class Platt:
    """Fit p' = sigmoid(a * logit(p) + b) on held-out data."""

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        z = logit(p).reshape(-1, 1)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y).astype(int))
        self.a, self.b = float(m.coef_[0, 0]), float(m.intercept_[0])
        return self

    def __call__(self, p):
        return sigmoid(self.a * logit(p) + self.b)


class Temperature:
    """One number, T, that softens (T>1) or sharpens (T<1) every prediction: sigmoid(logit/T)."""

    def fit(self, p, y):
        z, yy = logit(p), np.asarray(y, float)

        def nll(t):
            q = np.clip(sigmoid(z / t), EPS, 1 - EPS)
            return -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))

        self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self

    def __call__(self, p):
        return sigmoid(logit(p) / self.T)


class Isotonic:
    """A monotone step function learned from data. Flexible, needs more data than Platt."""

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), np.asarray(y, float))
        return self

    def __call__(self, p):
        return self.m.predict(np.asarray(p, float))


def temperature_multiclass(logits, labels) -> float:
    """Fit a temperature for softmax logits (n, k) against integer labels."""
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, int)

    def nll(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(labels)), labels].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def summary(p, y) -> dict:
    """The four numbers we report for every model in the book."""
    from sklearn.metrics import roc_auc_score
    p, y = _arr(p, y)
    return dict(auc=float(roc_auc_score(y, p)), brier=brier(p, y), log_loss=log_loss(p, y), ece=ece(p, y))


def prior_shift(p, base_rate_trained: float, base_rate_yours: float):
    """Adjust probabilities for a different base rate: multiply the odds by the ratio of the base-rate odds.

    This is the classic correction for label (prior) shift. It assumes that, within each class, alerts look
    the same at both places; only how common each class is has changed.
    """
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    o = p / (1 - p) * (base_rate_yours / (1 - base_rate_yours)) / (base_rate_trained / (1 - base_rate_trained))
    return o / (1 + o)


def ece_interval(p, y, n_boot: int = 300, seed: int = 0, q=(5, 95)):
    """Bootstrap range for ECE. With few labels, ECE is noisy and biased upwards: report the range."""
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p, float), np.asarray(y, float)
    vals = [ece(p[i], y[i]) for i in (rng.integers(0, len(p), len(p)) for _ in range(n_boot))]
    return tuple(float(v) for v in np.percentile(vals, q))
