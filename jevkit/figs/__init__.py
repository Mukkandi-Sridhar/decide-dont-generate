"""Figure system. Each chapter's figures live in figures/src/chNN.py and register with @figure."""

from __future__ import annotations

import json
from pathlib import Path

from .style import (C, ZONE, ZONE_T, ZONE_HATCH, KIND, SERIES, HATCH, setup, fig, subplots, save, synthetic_tag, clean, pct,
                    luma, ink_on, hatch_kw, ROOT, TEXT_W, WIDE_W)
from . import draw
from .common import you_are_here, summary_page

_REGISTRY: dict[str, list] = {}


def figure(chapter: str, name: str):
    """Register a function that returns a Matplotlib figure."""
    def deco(fn):
        _REGISTRY.setdefault(chapter, []).append((name, fn))
        return fn
    return deco


def registered(chapter: str):
    return _REGISTRY.get(chapter, [])


def results(chapter: str, **values):
    """Merge numbers into results/<chapter>.json so the text can quote them via {{< num >}}."""
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{chapter}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(values)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data
