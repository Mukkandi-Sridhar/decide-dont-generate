"""jevkit: the companion toolkit for *Decide, Don't Generate*.

- `MockJevTransport`: a deterministic, synthetic stand-in for the Jev API that
  plugs into the official `typesafe_sdk.TypeSafeClient` via `transport=`.
- `soc`: the synthetic Kestrel Logistics SOC alert generator.
- `calibration`, `policy`: the probability and decision tools used all book long.

Everything jevkit returns is synthetic. Nothing here was measured on real Jev.
"""

from __future__ import annotations

import os

from .mock import MockJevTransport, RecordingTransport, ReplayTransport
from .engine import MODEL_NAME

__version__ = "1.0.0"
SYNTHETIC_LABEL = "Synthetic — not measured on real Jev"

__all__ = ["MockJevTransport", "RecordingTransport", "ReplayTransport", "MODEL_NAME", "SYNTHETIC_LABEL", "client"]


def client(**kwargs):
    """Return a `TypeSafeClient`.

    Uses the mock unless JEVKIT_LIVE=1 *and* TYPESAFE_API_KEY is set, in which
    case the same code talks to the real API.
    """
    from typesafe_sdk import TypeSafeClient

    if os.environ.get("JEVKIT_LIVE") == "1" and os.environ.get("TYPESAFE_API_KEY"):
        return TypeSafeClient(**kwargs)
    kwargs.setdefault("api_key", "mock")
    kwargs.setdefault("transport", MockJevTransport())
    return TypeSafeClient(**kwargs)
