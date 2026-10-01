"""TinyJev: a small System One model you can read in one sitting (Chapter 20).

A shared encoder reads an alert's fields; three typed heads answer three questions in one forward pass:

    noul    P(attack)                       sigmoid of one number
    choice  P(category) for each category   softmax over one number per label
    score   P(severity) for each level      ordinal: P(level > k) = sigmoid(z - t_k), thresholds t_k increasing

Every head is trained with log loss, a proper scoring rule, so honesty is what training rewards. After training,
one temperature per head is fitted on held-out data (Chapter 3), because proper scoring alone doesn't stop a
network from overfitting into overconfidence.

Plain NumPy + autograd. Nothing here is TypeSafe's architecture; it's a design the interface suggests.
"""

from __future__ import annotations

import json

import autograd.numpy as anp
import httpx2
import numpy as np
from autograd import grad
from scipy.optimize import minimize_scalar

from jevkit import soc
from jevkit.mock import MockJevTransport

NUMERIC = ["ioc_score", "after_hours", "asset_criticality", "known_tool", "prior_alerts_24h", "new_geo", "mfa_ok",
           "log_mb_out", "role_admin", "role_service"]
RULES = list(soc.RULES)
CATS = list(soc.CATEGORIES)
LEVELS = list(soc.SEVERITY_LEVELS)


# ----------------------------------------------------------------------------- inputs
def features(df):
    """Numbers for the numeric fields, and an index for the rule (which the model embeds)."""
    X = soc.feature_matrix(df)
    num = X[NUMERIC].to_numpy(float).copy()
    num[:, NUMERIC.index("prior_alerts_24h")] /= 6.0
    num[:, NUMERIC.index("log_mb_out")] /= 8.0
    num[:, NUMERIC.index("asset_criticality")] /= 3.0
    rule = np.array([RULES.index(r) for r in df["rule"]])
    return num, rule


def targets(df):
    y = df["malicious"].to_numpy(float)
    cat = np.array([CATS.index(c) for c in df["category"]])
    sev = df["severity"].to_numpy()
    sev = np.array([LEVELS.index(s) if not isinstance(s, (int, np.integer)) else int(s) for s in sev])
    return y, cat, sev


# ----------------------------------------------------------------------------- model
def init(d: int = 32, e: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_in = len(NUMERIC) + e
    s = lambda a, b: rng.normal(0, np.sqrt(2 / a), (a, b))
    return dict(
        R=rng.normal(0, 0.3, (len(RULES), e)),               # rule embedding (Chapter 5)
        W1=s(n_in, d), b1=np.zeros(d), W2=s(d, d), b2=np.zeros(d),   # shared encoder (Chapter 5)
        wn=rng.normal(0, 0.1, d), bn=0.0,                        # noul head
        Wc=rng.normal(0, 0.1, (d, len(CATS))), bc=np.zeros(len(CATS)),   # choice head
        ws=rng.normal(0, 0.1, d), ts=np.linspace(-1, 1, len(LEVELS) - 1),  # score head: one number + thresholds
    )


def encode(p, num, rule):
    h = anp.concatenate([num, p["R"][rule]], axis=1)
    h = anp.maximum(0, h @ p["W1"] + p["b1"])
    return anp.maximum(0, h @ p["W2"] + p["b2"])


def _sorted_thresholds(ts):
    # keep thresholds in order: first one free, then positive gaps
    return anp.concatenate([ts[:1], ts[:1] + anp.cumsum(anp.log1p(anp.exp(ts[1:])))])


def heads(p, num, rule, T=(1.0, 1.0, 1.0)):
    """Raw scores for all three heads, divided by their temperatures."""
    h = encode(p, num, rule)
    zn = (h @ p["wn"] + p["bn"]) / T[0]
    zc = (h @ p["Wc"] + p["bc"]) / T[1]
    zs = (h @ p["ws"])[:, None] / T[2] - _sorted_thresholds(p["ts"])[None, :] / T[2]
    return zn, zc, zs


def probabilities(zn, zc, zs):
    pn = 1 / (1 + anp.exp(-zn))
    zc = zc - anp.max(zc, axis=1, keepdims=True)
    pc = anp.exp(zc) / anp.sum(anp.exp(zc), axis=1, keepdims=True)
    above = 1 / (1 + anp.exp(-zs))                              # P(level > k), k = 0..K-2
    ones, zeros = anp.ones((zs.shape[0], 1)), anp.zeros((zs.shape[0], 1))
    cum = anp.concatenate([ones, above, zeros], axis=1)          # P(level > -1) = 1, P(level > K-1) = 0
    ps = anp.clip(cum[:, :-1] - cum[:, 1:], 1e-9, 1)
    return pn, pc, ps


def loss(p, num, rule, y, cat, sev, weights=(1.0, 1.0, 1.0), l2=1e-4):
    pn, pc, ps = probabilities(*heads(p, num, rule))
    pn = anp.clip(pn, 1e-7, 1 - 1e-7)
    ln = -anp.mean(y * anp.log(pn) + (1 - y) * anp.log(1 - pn))
    lc = -anp.mean(anp.log(anp.clip(pc[anp.arange(len(cat)), cat], 1e-9, 1)))
    ls = -anp.mean(anp.log(ps[anp.arange(len(sev)), sev]))
    reg = l2 * sum(anp.sum(p[k] ** 2) for k in ("W1", "W2", "Wc"))
    return weights[0] * ln + weights[1] * lc + weights[2] * ls + reg


def head_losses(p, num, rule, y, cat, sev, T=(1.0, 1.0, 1.0)):
    pn, pc, ps = probabilities(*heads(p, num, rule, T))
    pn = np.clip(pn, 1e-7, 1 - 1e-7)
    return dict(noul=float(-np.mean(y * np.log(pn) + (1 - y) * np.log(1 - pn))),
                choice=float(-np.mean(np.log(np.clip(pc[np.arange(len(cat)), cat], 1e-9, 1)))),
                score=float(-np.mean(np.log(ps[np.arange(len(sev)), sev]))))


def train(num, rule, y, cat, sev, steps: int = 1500, lr: float = 0.01, batch: int = 512, seed: int = 0,
          l2: float = 1e-4, val=None, every: int = 50, d: int = 32, weights=(1.0, 1.0, 1.0)):
    """Adam on minibatches. Returns (params, history) where history has per-head losses on train and val."""
    p = init(d=d, seed=seed)
    g = grad(loss)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng(seed)
    hist = []
    for t in range(1, steps + 1):
        idx = rng.choice(len(y), min(batch, len(y)), replace=False)
        gr = g(p, num[idx], rule[idx], y[idx], cat[idx], sev[idx], weights=weights, l2=l2)
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]
            v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
            p[k] = p[k] - lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % every == 0:
            row = dict(step=t, train=head_losses(p, num, rule, y, cat, sev))
            if val is not None:
                row["val"] = head_losses(p, *val)
            hist.append(row)
    return p, hist


def fit_temperatures(p, num, rule, y, cat, sev):
    """One temperature per head, each chosen to minimise that head's log loss on held-out data."""
    out = []
    for i, name in enumerate(("noul", "choice", "score")):
        def f(t):
            T = [1.0, 1.0, 1.0]
            T[i] = t
            return head_losses(p, num, rule, y, cat, sev, T)[name]
        out.append(float(minimize_scalar(f, bounds=(0.2, 5.0), method="bounded").x))
    return tuple(out)


def predict(p, df, T=(1.0, 1.0, 1.0)):
    num, rule = features(df)
    pn, pc, ps = probabilities(*heads(p, num, rule, T))
    return np.asarray(pn), np.asarray(pc), np.asarray(ps)


# ----------------------------------------------------------------------------- plug into the official SDK
class TinyJevTransport(MockJevTransport):
    """The mock transport, but SOC states with fields are answered by *your* model.

    Questions are matched by type: a noul gets P(attack), a choice over the category labels gets the category head,
    and a score over the severity levels gets the severity head. Anything else falls back to the mock engine.
    """

    def __init__(self, params, T=(1.0, 1.0, 1.0), **kw):
        super().__init__(**kw)
        self.params, self.T = params, T

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        resp = super().handle_request(request)
        if resp.status_code != 200 or request.url.path.rstrip("/").split("/")[-1] != "systemone":
            return resp
        body = json.loads(request.content)
        state = body["state"]
        if not (isinstance(state, dict) and state.get("rule") in soc.RULES):
            return resp
        import pandas as pd
        row = {k: state.get(k) for k in soc.FEATURE_FIELDS}
        defaults = dict(asset_criticality=1, after_hours=False, ioc_score=0.2, known_tool=False, prior_alerts_24h=0,
                        new_geo=False, mfa_ok=True, mb_out=0.3, role="employee")
        row = {k: (defaults.get(k) if v is None else v) for k, v in row.items()}
        pn, pc, ps = predict(self.params, pd.DataFrame([row]), self.T)
        payload = json.loads(resp.content)
        for name, q in body["questions"].items():
            a = payload["answers"][name]
            if q["type"] == "noul":
                a["noul"] = float(pn[0])
            elif q["type"] == "choice" and set(q["criteria"]) == set(CATS):
                probs = {c: float(pc[0][CATS.index(c)]) for c in q["criteria"]}
                a["probabilities"] = probs
                a["choice"] = max(probs, key=probs.get)
                a["confidence"] = max(probs.values())
            elif q["type"] == "score" and list(q["criteria"]) == LEVELS:
                probs = {str(i): float(v) for i, v in enumerate(ps[0])}
                a["probabilities"] = probs
                a["score"] = float(np.dot(np.arange(len(LEVELS)), ps[0]) / (len(LEVELS) - 1))
                a["confidence"] = float(ps[0].max())
        payload["model"] = "tinyjev-your-own"
        return httpx2.Response(200, json=payload, request=request, headers=dict(resp.headers))
