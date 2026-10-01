"""`MockJevTransport`: plug the synthetic engine into the official SDK.

    from typesafe_sdk import TypeSafeClient
    from jevkit import MockJevTransport

    client = TypeSafeClient(api_key="mock", transport=MockJevTransport())

The official client builds the real HTTP request (POST /v1/systemone with
{state, model, questions}). Instead of sending it over the network, httpx2
hands it to this transport, which answers it locally. Your code cannot tell
the difference, which is the point: swap the transport out and the same code
talks to the real API.
"""

from __future__ import annotations

import hashlib
import json
import random
import math
import threading

import httpx2

from . import engine

SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
ACCEPTED_MODELS = {"jev-latest", "jev", engine.MODEL_NAME}


def simulated_latency_ms(input_tokens: int, n_questions: int) -> float:
    """A made-up latency model for the mock. SYNTHETIC: not measured on real Jev."""
    return round(120.0 + 0.05 * input_tokens + 8.0 * n_questions, 1)


def _err(status: int, detail, request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(status, json={"detail": detail}, request=request)


def _validation(loc, msg, typ="value_error"):
    return [{"loc": ["body", *loc], "msg": msg, "type": typ}]


def validate_body(body) -> list | None:
    """Mirror the real API's request schema closely enough to catch the same mistakes."""
    if not isinstance(body, dict):
        return _validation([], "Input should be a valid dictionary", "dict_type")
    if "state" not in body:
        return _validation(["state"], "Field required", "missing")
    if not isinstance(body["state"], (str, dict, list)):
        return _validation(["state"], "Input should be a string, object or array")
    if not isinstance(body.get("model"), str):
        return _validation(["model"], "Field required", "missing")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        return _validation(["questions"], "Dictionary should have at least 1 item after validation", "too_short")
    for name, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            return _validation(["questions", name], "Input tag must be one of 'noul', 'score', 'choice'", "union_tag_invalid")
        t = q["type"]
        if t == "choice":
            c = q.get("criteria")
            if not isinstance(c, dict) or not c:
                return _validation(["questions", name, "choice", "criteria"], "Field required", "missing")
        if t == "score":
            c = q.get("criteria")
            if not isinstance(c, list) or not c:
                return _validation(["questions", name, "score", "criteria"], "List should have at least 1 item after validation", "too_short")
        if t == "noul" and q.get("criteria") is not None:
            c = q["criteria"]
            if not isinstance(c, dict) or set(c) - {"true", "false"}:
                return _validation(["questions", name, "noul", "criteria"], "Extra inputs are not permitted", "extra_forbidden")
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            return _validation(["questions", name, t, sorted(extra)[0]], "Extra inputs are not permitted", "extra_forbidden")
    return None


class MockJevTransport(httpx2.BaseTransport):
    """An httpx2 transport that answers TypeSafe API calls with the synthetic engine.

    Args:
        fail_every: if set, every Nth request returns HTTP 503 (for retry lessons).
        rate_limit_every: if set, every Nth request returns HTTP 429 with retry-after.
        fail_rate: if set, each request independently returns HTTP 503 with this probability (seeded).
        log: if a list, every (request_body, response_body) pair is appended to it.
    """

    def __init__(self, *, fail_every: int | None = None, rate_limit_every: int | None = None, log: list | None = None,
                 fail_rate: float | None = None, seed: int = 0):
        self.fail_every = fail_every
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)
        self.statuses: list[int] = []
        self.rate_limit_every = rate_limit_every
        self.log = log
        self.calls = 0
        self._lock = threading.Lock()

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        with self._lock:
            self.calls += 1
            n = self.calls
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or len(auth) <= 7:
            return _err(401, "Invalid or missing API key", request)
        if self.rate_limit_every and n % self.rate_limit_every == 0:
            r = _err(429, "Rate limit exceeded (simulated)", request)
            r.headers["retry-after-ms"] = "50"
            return r
        if (self.fail_every and n % self.fail_every == 0) or (self.fail_rate and self._rng.random() < self.fail_rate):
            self.statuses.append(503)
            return _err(503, "Service unavailable (simulated)", request)
        self.statuses.append(200)

        path = request.url.path
        if path == MODELS_PATH and request.method == "GET":
            return httpx2.Response(200, json={"models": [
                {"name": engine.MODEL_NAME, "description": engine.MODEL_DESCRIPTION, "release_date": engine.MODEL_RELEASE}
            ]}, request=request)
        if path != SYSTEM_ONE_PATH or request.method != "POST":
            return _err(404, "Not Found", request)
        try:
            body = json.loads(request.content or b"null")
        except json.JSONDecodeError:
            return _err(422, _validation([], "JSON decode error", "json_invalid"), request)
        problem = validate_body(body)
        if problem:
            return _err(422, problem, request)
        if body["model"] not in ACCEPTED_MODELS:
            return _err(404, f"Model '{body['model']}' not found. Try 'jev-latest'.", request)

        answers = engine.answer(body["state"], body["questions"])
        in_tok = engine.count_tokens({"state": body["state"], "questions": body["questions"]})
        out_tok = sum(2 + len(a.get("probabilities", {})) for a in answers.values())
        payload = {"model": engine.MODEL_NAME, "answers": answers,
                   "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
        rid = "req_mock_" + hashlib.sha256(request.content).hexdigest()[:16]
        resp = httpx2.Response(200, json=payload, request=request, headers={
            "x-typesafe-request-id": rid,
            "x-jevkit-synthetic": "true",
            "x-jevkit-simulated-latency-ms": str(simulated_latency_ms(in_tok, len(answers))),
        })
        if self.log is not None:
            self.log.append((body, payload))
        return resp


class RecordingTransport(httpx2.BaseTransport):
    """Wrap any transport and write each exchange to a JSON-lines file (for fixtures)."""

    def __init__(self, inner: httpx2.BaseTransport, path: str):
        self.inner, self.path = inner, path

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        resp = self.inner.handle_request(request)
        resp.read()
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"path": request.url.path, "request": json.loads(request.content or b"null"),
                                 "status": resp.status_code, "response": json.loads(resp.content or b"null")}) + "\n")
        return resp

    def close(self):
        self.inner.close()


class ReplayTransport(httpx2.BaseTransport):
    """Answer requests from a recorded JSON-lines file. Unknown requests raise."""

    def __init__(self, path: str):
        self.table = {}
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                self.table[json.dumps(rec["request"], sort_keys=True)] = (rec["status"], rec["response"])

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        key = json.dumps(json.loads(request.content or b"null"), sort_keys=True)
        if key not in self.table:
            raise KeyError("No recorded response for this request")
        status, body = self.table[key]
        return httpx2.Response(status, json=body, request=request)
