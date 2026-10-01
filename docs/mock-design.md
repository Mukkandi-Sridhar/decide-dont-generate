# How the book's mock works

Every Jev answer in *Decide, Don't Generate* comes from `jev-mock-synthetic`, a mock that runs on your own
machine. This page explains how it's built, so you can judge what the book's numbers do and don't show.
**Every number the mock produces is synthetic: not measured on real Jev.**

## What is copied from the real thing

The mock copies Jev's *interface*, not its behaviour.

- **The client is real.** You use TypeSafe's official Python library, `typesafe-sdk` 0.7.x from PyPI, exactly as
  you would with a key. `MockJevTransport` (in `jevkit/mock.py`) plugs into the client's `transport=` argument and
  answers the HTTP request locally instead of sending it. Remove the transport and add a key, and the same code
  calls the real API.
- **Requests** are checked against the same schema as the real endpoint (`POST /v1/systemone` with `state`,
  `model` and `questions`), so the mock rejects the same mistakes with the same kind of validation error.
- **Responses** follow the SDK's own data models. A `choice` answer carries `{choice, confidence, probabilities}`.
  A `score` answer carries `{score, confidence, legend, probabilities}`. A `noul` answer carries only the
  probability of "yes"; like the real API, it has no `confidence`.
- **Model name.** Requests may ask for `jev-latest`; the mock answers as `jev-mock-synthetic`, so a response can
  never be mistaken for a real one. Responses also carry an `x-jevkit-synthetic` header.

## Where the mock differs on purpose

**Confidence.** TypeSafe's documentation describes `confidence` as a statistic computed from the *shape* of the
answer's probability distribution: high when the probability is concentrated on one option, low when it's spread
out. Its quick-start example shows a confidence of 0.78 next to a top probability of 0.85, so the real value is
not simply the top probability. The exact formula isn't published. The mock uses the simplest stand-in: the
probability of the top option (for `choice`) or of the most likely level (for `score`). Don't carry that
definition over to the real API; read its `probabilities` and test its `confidence` on your own data.

**SOC alerts.** Kestrel Logistics' alerts come from a generator with a known true probability for every alert
(`jevkit/soc.py`). For those alerts the mock starts from the true log-odds, then makes them imperfect in stated
ways (`jevkit/engine.py`):

| Setting | Value | Effect |
|---|---|---|
| `SOC_BIAS` | +0.10 on the log-odds | slightly too keen on "malicious" |
| `SOC_SHARPEN` | ×1.18 on the log-odds | overconfident at the extremes, like most trained models |
| `SOC_NOISE` | σ 0.35, seeded per alert | the model isn't an oracle |
| `SOC_TEXT_NOISE` | σ 0.55 extra | reading raw text is harder than reading fields |

The result is a model that ranks well and is close to, but not exactly, calibrated. That's a teaching choice,
not a claim about Jev.

**Everything else.** Outside the SOC (support tickets, the gallery, the pattern catalogue) the mock uses a simple
word-matching engine. It is weak on purpose, and Chapter 19 shows how weak. Chapters that rely on it say so.

**Latency and price.** The mock's latency is a stated formula, `120 ms + 0.05 ms per input token + 8 ms per
question` (`simulated_latency_ms`). Prices use TypeSafe's vendor-reported list price. Charts that use either
are labelled synthetic or vendor-reported.

## The mock LLM

Chapters that compare a decision model with an LLM use `MockLLM` (`jevkit/llm.py`), which is also synthetic:

- Its belief is mock Jev's raw-text probability plus extra noise (σ 0.45 on the log-odds), so it is built to be a
  *noisier reader* than mock Jev. Any accuracy gap between the two is a design choice, not evidence.
- Its label probability is sharpened (temperature 0.45), so it is overconfident.
- Its stated confidence snaps to a few round values, mostly between 0.9 and 0.99.
- About 2% of free-text JSON answers are wrapped or cut short, and 1.5% add an invented field.
- Latency is 0.45 s to the first token, then 60 output tokens a second; price is $1 per million input tokens and $4 per million
  output tokens. Both are illustrative assumptions, not quotes.

## Determinism

Every random draw is seeded from the request's content, so the same request always gets the same answer, on any
machine. That's what lets the book print numbers and a test check that they still hold.

## What the mock can't tell you

How real Jev ranks, how well its probabilities are calibrated on your data, how it handles adversarial input, and
how it changes between versions. Chapter 11 shows how to measure each of those once you have a key: set
`JEVKIT_LIVE=1` and `TYPESAFE_API_KEY`, and the labs call the real service.
