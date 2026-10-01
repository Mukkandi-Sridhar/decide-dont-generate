"""Score many alerts through the official SDK + mock, with an on-disk cache.

    from jevkit.batch import score_alerts
    p = score_alerts(alerts)            # P(malicious) for every row, from jev-mock-synthetic

The cache key includes the engine settings, so changing the mock invalidates it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import engine

CACHE = Path(__file__).resolve().parents[1] / ".cache"

QUESTION = {"attack": {"type": "noul", "instructions": "Is this alert a real attack?"}}


def _key(n: int, mode: str) -> str:
    cfg = [engine.SOC_SHARPEN, engine.SOC_NOISE, engine.SOC_TEXT_NOISE, engine.SOC_BIAS, n, mode, 3]
    return hashlib.sha1(json.dumps(cfg).encode()).hexdigest()[:12]


def score_alerts(alerts, mode: str = "structured", client=None, use_cache: bool = True) -> np.ndarray:
    """P(attack) for every alert (a DataFrame from `soc.load()`), asked through the SDK."""
    from . import soc
    import pandas as pd
    CACHE.mkdir(exist_ok=True)
    fields = ["alert", "source", "rule", "host", "user", "department"] + [f for f in soc.FEATURE_FIELDS if f != "rule"] + ["description"]
    # The key covers everything the state is built from, so edited fields never hit a stale cache.
    cols = [c for c in ["alert_id", "title"] + fields if c in alerts.columns]
    content = hashlib.sha1(pd.util.hash_pandas_object(alerts[cols].astype(str), index=False).values.tobytes()).hexdigest()[:10]
    path = CACHE / f"scores-{_key(len(alerts), mode)}-{content}.npy"
    if use_cache and path.exists():
        return np.load(path)
    if client is None:
        from . import client as make_client
        client = make_client()
    out = np.empty(len(alerts))
    for i, row in enumerate(alerts.itertuples(index=False)):
        if mode == "text":
            state = row.description
        else:
            d = row._asdict()
            state = {"alert": d["title"], **{k: d[k] for k in fields if k in d}}
            state = {k: (v.item() if hasattr(v, "item") else v) for k, v in state.items()}
        r = client.system_one(state=state, questions=QUESTION)
        out[i] = r.nouls["attack"].noul
    np.save(path, out)
    return out
