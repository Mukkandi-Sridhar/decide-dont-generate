"""Chapter 13's bake-off: six ways to decide whether a Kestrel alert is a real threat.

Every contestant is trained or tuned on the history weeks and scored on the live week. Jev answers come from
`jev-mock-synthetic`, and the LLM is `jevkit.llm.MockLLM`, so every number here is synthetic.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np

from jevkit import calibration as cal, llm, soc
from jevkit.batch import score_alerts

REVIEWS_PER_DAY = 240          # Kestrel's review capacity (Chapter 14)

METHODS = ["rules", "logistic", "text_clf", "llm_json", "llm_judge", "jev"]
NAMES = {"rules": "Hand-written rules", "logistic": "Logistic regression",
         "text_clf": "Trained text classifier", "llm_json": "LLM, structured output",
         "llm_judge": "LLM-as-judge", "jev": "Jev (mock)"}
INPUTS = {"rules": "fields", "logistic": "fields", "text_clf": "text", "llm_json": "text", "llm_judge": "text",
          "jev": "fields as JSON"}


def rules(df) -> np.ndarray:
    """The kind of rule a senior analyst writes on a whiteboard. True means 'treat as a threat'."""
    return ((df.ioc_score >= 0.7)
            | (df.new_geo & ~df.mfa_ok.astype(bool))
            | ((df.role == "admin") & df.after_hours & (df.ioc_score >= 0.4))).to_numpy()


def _norm(t: str) -> str:
    """Turn numbers into coarse tokens a bag-of-words model can use ('0.38' -> 'dec3')."""
    t = re.sub(r"\d{4}-\d\d-\d\d (\d\d):\d\d", lambda m: f" hour{int(m.group(1)) // 3} ", t)
    t = re.sub(r"\b0\.(\d)\d*", lambda m: f" dec{m.group(1)} ", t)
    t = re.sub(r"[\d,]+(\.\d+)?", lambda m: f" mag{len(m.group(0).replace(',', '').split('.')[0])} ", t)
    return t.lower()


class TextClassifier:
    """TF-IDF over words and word pairs, then logistic regression: a classic trained text classifier.

    It stands in for a fine-tuned model. A fine-tuned transformer would read the text better; this is the
    honest thing we can train on a laptop in seconds.
    """

    def fit(self, texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.tf = TfidfVectorizer(preprocessor=_norm, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        self.m = LogisticRegression(C=4, max_iter=3000).fit(self.tf.fit_transform(list(texts)), y)
        return self

    def __call__(self, texts):
        return self.m.predict_proba(self.tf.transform(list(texts)))[:, 1]


def fit_logistic(df, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000).fit(soc.feature_matrix(df), y)


def llm_json_probability(raw: str):
    """Parse the LLM's JSON. Returns (P(threat), ok). A broken answer returns (nan, False)."""
    try:
        o = json.loads(raw)
        c = float(o["confidence"])
        return (c if o["verdict"] == "malicious" else 1 - c), True
    except (ValueError, KeyError, TypeError):
        return float("nan"), False


def jev_state(df):
    return df.assign(jev=score_alerts(df, "structured"))["jev"].to_numpy()


@lru_cache(None)
def run(n_labels: int | None = None) -> dict:
    """Train on history, score the live week. Returns probabilities per method, plus extras."""
    alerts = soc.load()
    alerts["jev"] = score_alerts(alerts, "structured")
    alerts["jev_text"] = score_alerts(alerts, "text")
    hist, live = soc.history_and_live(alerts)
    if n_labels:
        hist = hist.iloc[:n_labels]
    yh, yl = hist.malicious.to_numpy(), live.malicious.to_numpy()

    r_h, r_l = rules(hist), rules(live)
    p_rules = np.where(r_l, yh[r_h].mean(), yh[~r_h].mean())    # each branch's hit rate in history

    p_log = fit_logistic(hist, yh).predict_proba(soc.feature_matrix(live))[:, 1]
    p_txt = TextClassifier().fit(hist.description, yh)(live.description)

    mock = llm.MockLLM()
    parsed = [llm_json_probability(mock.structured(d)) for d in live.description]
    p_json = np.array([p for p, _ in parsed])
    parse_ok = np.array([ok for _, ok in parsed])
    p_json_filled = np.where(parse_ok, p_json, yh.mean())     # a broken answer falls back to the base rate
    p_judge = np.array([(mock.judge(d, "malicious") - 1) / 9 for d in live.description])

    probs = dict(rules=p_rules, logistic=p_log, text_clf=p_txt, llm_json=p_json_filled, llm_judge=p_judge,
                 jev=live.jev.to_numpy())
    return dict(probs=probs, y=yl, live=live, hist=hist, rules_live=r_l, parse_ok=parse_ok,
                jev_text=live.jev_text.to_numpy())


def caught_at_capacity(p, y, days: int = 7, per_day: int = REVIEWS_PER_DAY) -> float:
    """Share of real threats among the alerts a team reviews if it takes the top `per_day` each day."""
    k = min(len(p), days * per_day)
    top = np.argsort(-np.asarray(p))[:k]
    return float(np.asarray(y)[top].sum() / np.asarray(y).sum())


def f1_at(p, y, line: float = 0.5) -> float:
    pred = np.asarray(p) >= line
    y = np.asarray(y).astype(bool)
    tp = (pred & y).sum()
    return float(2 * tp / (pred.sum() + y.sum())) if (pred.sum() + y.sum()) else 0.0


def scores(p, y, verdict=None) -> dict:
    """The numeric part of the scorecard. `verdict` overrides the 0.5 line for methods that give a yes/no."""
    s = cal.summary(p, y)
    if verdict is not None:
        v, yy = np.asarray(verdict, bool), np.asarray(y, bool)
        f1 = float(2 * (v & yy).sum() / (v.sum() + yy.sum()))
    else:
        f1 = f1_at(p, y)
    s.update(caught=caught_at_capacity(p, y), f1=f1)
    return s


def labels_curve(sizes=(100, 300, 1000, 3000, 10000), repeats: int = 5) -> dict:
    """AUC on the live week for the two trained methods, as the number of history labels grows."""
    r = run()
    hist, live, y = r["hist"], r["live"], r["y"]
    out = {"sizes": list(sizes) + [len(hist)], "logistic": [], "text_clf": []}
    rng = np.random.default_rng(20)
    from sklearn.metrics import roc_auc_score
    for n in out["sizes"]:
        a, b = [], []
        for _ in range(repeats if n < len(hist) else 1):
            idx = rng.choice(len(hist), n, replace=False)
            h = hist.iloc[idx]
            if h.malicious.nunique() < 2:
                continue
            a.append(roc_auc_score(y, fit_logistic(h, h.malicious).predict_proba(soc.feature_matrix(live))[:, 1]))
            b.append(roc_auc_score(y, TextClassifier().fit(h.description, h.malicious)(live.description)))
        out["logistic"].append(float(np.mean(a)))
        out["text_clf"].append(float(np.mean(b)))
    return out


def flip_rate(texts, runs: int = 5, temperature: float = 0.7) -> float:
    """Share of alerts whose verdict changes across repeated calls to the same LLM."""
    verdicts = []
    for s in range(runs):
        m = llm.MockLLM(temperature=temperature, seed=s)
        verdicts.append([m.classify(t)["label"] for t in texts])
    v = np.array(verdicts)
    return float((v != v[0]).any(axis=0).mean())
