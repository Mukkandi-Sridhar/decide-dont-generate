"""A one-layer, one-head self-attention classifier in NumPy + autograd (Chapter 5).

The toy task: short analyst notes such as "invoice not phishing" or "phishing not invoice".
A note is a threat if it contains a threat word that is NOT directly preceded by "not".
Both example notes contain the same words, so a bag-of-words model cannot tell them apart.
Attention with position information can.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from autograd import grad

THREAT = ["phishing", "malware", "ransomware", "spoofed"]
BENIGN = ["invoice", "meeting", "lunch", "printer"]
VOCAB = ["<pad>", "not", "and", "the"] + THREAT + BENIGN
IDX = {w: i for i, w in enumerate(VOCAB)}
LENGTH = 6


def make_data(n: int = 4000, seed: int = 0, noise: float = 0.0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, LENGTH), int)
    y = np.zeros(n)
    sents = []
    for i in range(n):
        words = []
        k = int(rng.integers(2, 4))
        for _ in range(k):
            w = str(rng.choice(THREAT if rng.random() < 0.4 else BENIGN))
            if rng.random() < 0.45:
                words += ["not", w]
            else:
                words += [w]
        if rng.random() < 0.3:
            words.insert(int(rng.integers(0, len(words) + 1)), "and")
        words = words[:LENGTH]
        label = any(w in THREAT and (j == 0 or words[j - 1] != "not") for j, w in enumerate(words))
        if noise and rng.random() < noise:          # real labels are messy: some are simply wrong
            label = not label
        X[i, :len(words)] = [IDX[w] for w in words]
        y[i] = float(label)
        sents.append(words)
    return X, y, sents


def init(d: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = 0.3
    return dict(E=rng.normal(0, s, (len(VOCAB), d)), P=rng.normal(0, s, (LENGTH, d)),
                Wq=rng.normal(0, s, (d, d)), Wk=rng.normal(0, s, (d, d)), Wv=rng.normal(0, s, (d, d)),
                w=rng.normal(0, s, d), b=0.0)


def forward(p, X, return_attn=False):
    mask = (X != 0)
    H = p["E"][X] + p["P"][None, :, :]                          # word meaning + position
    Q, K, V = H @ p["Wq"], H @ p["Wk"], H @ p["Wv"]
    scores = anp.einsum("nid,njd->nij", Q, K) / anp.sqrt(Q.shape[-1])
    scores = scores + anp.where(mask[:, None, :], 0.0, -1e9)    # ignore padding
    scores = scores - anp.max(scores, axis=-1, keepdims=True)
    A = anp.exp(scores)
    A = A / anp.sum(A, axis=-1, keepdims=True)                  # softmax: weights sum to 1
    Hc = anp.einsum("nij,njd->nid", A, V) + H                   # mix in the words it attended to
    token_logit = Hc @ p["w"] + p["b"]
    token_logit = anp.where(mask, token_logit, -1e9)
    # the note is a threat if ANY word says so: a smooth max over positions
    z = anp.log(anp.sum(anp.exp(token_logit - 5.0), axis=1)) + 5.0
    prob = 1 / (1 + anp.exp(-z))
    return (prob, A) if return_attn else prob


def loss(p, X, y):
    q = anp.clip(forward(p, X), 1e-7, 1 - 1e-7)
    return -anp.mean(y * anp.log(q) + (1 - y) * anp.log(1 - q))


def train(X, y, steps: int = 400, lr: float = 0.05, seed: int = 0, batch: int = 256, history=None, callback=None, every: int = 100):
    """Adam on minibatches. Deterministic for a given seed."""
    p = init(seed=seed)
    g = grad(loss)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng(seed)
    for t in range(1, steps + 1):
        idx = rng.choice(len(X), batch, replace=False)
        gr = g(p, X[idx], y[idx])
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]
            v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
            mh, vh = m[k] / (1 - 0.9 ** t), v2[k] / (1 - 0.999 ** t)
            p[k] = p[k] - lr * mh / (np.sqrt(vh) + 1e-8)
        if history is not None and t % 20 == 0:
            history.append((t, float(loss(p, X, y))))
        if callback is not None and t % every == 0:
            callback(t, p)
    return p


def encode(words):
    X = np.zeros((1, LENGTH), int)
    X[0, :len(words)] = [IDX[w] for w in words]
    return X
