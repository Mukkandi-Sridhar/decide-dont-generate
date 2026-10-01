"""Tiny, readable learning code used in Part I (logistic regression from scratch)."""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Standardizer:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_logistic(X, y, lr=0.5, steps=300, record=False):
    """Plain gradient descent on log loss. Returns (w, b, history)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    b = 0.0
    hist = []
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
        if record:
            hist.append(log_loss(p, y))
    return w, b, hist


class TinyNet:
    """A one-hidden-layer network in plain NumPy: ReLU hidden units, a sigmoid output, log loss.

    Trained with full-batch gradient descent plus momentum. Small enough to read in one sitting.
    """

    def __init__(self, n_in, n_hidden=8, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), n_hidden)
        self.b2 = 0.0

    def forward(self, X):
        self.X = X
        self.h_in = X @ self.W1 + self.b1
        self.h = np.maximum(0, self.h_in)             # ReLU: the hinge
        return sigmoid(self.h @ self.W2 + self.b2)

    def fit(self, X, y, lr=0.1, steps=2000, momentum=0.9):
        vel = [0, 0, 0, 0]
        for _ in range(steps):
            p = self.forward(X)
            d_out = (p - y) / len(y)                   # blame at the output
            gW2 = self.h.T @ d_out
            gb2 = d_out.sum()
            d_h = np.outer(d_out, self.W2) * (self.h_in > 0)   # blame flows back through the hinge
            gW1 = X.T @ d_h
            gb1 = d_h.sum(0)
            for i, (name, g) in enumerate((("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2))):
                vel[i] = momentum * vel[i] - lr * g
                setattr(self, name, getattr(self, name) + vel[i])
        return self

    def predict(self, X):
        return self.forward(X)
