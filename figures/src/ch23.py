"""Figures for Chapter 16: First calls, and the mock that makes them free."""

import json
from functools import lru_cache

import httpx2
import numpy as np
from matplotlib.ticker import FuncFormatter
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

from jevkit import MockJevTransport, soc
from jevkit.econ import JEV_PRICE_IN_PER_M
from jevkit.figs import figure, draw, C, ROOT, subplots, clean, results, summary_page, synthetic_tag

CH = "ch23"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
FAST = dict(backoff_initial=0.001, backoff_max=0.002, backoff_jitter=0.0)
ATTACK = Noul(instructions="Is this alert a real attack?")


def triage_questions():
    return {"attack": ATTACK,
            "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
            "severity": Score(criteria=soc.SEVERITY_LEVELS),
            "page": Noul(instructions="Should on-call be woken for this?")}


@lru_cache(None)
def retry_table(n=600):
    out = {}
    for q in (0.01, 0.05, 0.2):
        for r in (0, 1, 2, 3):
            t = MockJevTransport(fail_rate=q, seed=int(q * 1000) + r)
            c = TypeSafeClient(api_key="mock", transport=t, retry=RetryPolicy(max_retries=r, **FAST))
            ok = 0
            for i in range(n):
                try:
                    c.system_one(state=f"alert {i}", questions={"attack": ATTACK})
                    ok += 1
                except Exception:
                    pass
            out[(q, r)] = (ok / n, len(t.statuses) / n)
    return out


@lru_cache(None)
def error_table():
    """Provoke each failure against the mock and record what the SDK actually does."""
    def attempt(transport, api_key="mock", model=None, questions=None, retry=0):
        try:
            c = TypeSafeClient(api_key=api_key, transport=transport, retry=RetryPolicy(max_retries=retry, **FAST))
            c.system_one(state="x", questions=questions or {"a": ATTACK}, model=model)
            return "succeeds"
        except Exception as e:
            return type(e).__name__

    class Dead(httpx2.BaseTransport):
        def handle_request(self, request):
            raise httpx2.ConnectError("simulated network failure", request=request)

    return [
        ("no API key at all", "–", attempt(MockJevTransport(), api_key="")),
        ("model name that doesn’t exist", "404", attempt(MockJevTransport(), model="jev-nope")),
        ("a choice with no options", "422", attempt(MockJevTransport(), questions={"a": Choice(criteria={})})),
        ("a score with no levels", "–", attempt(MockJevTransport(), questions={"a": Score(criteria=[])})),
        ("rate limited, no retries", "429", attempt(MockJevTransport(rate_limit_every=1))),
        ("rate limited once, default retries", "429", attempt(MockJevTransport(rate_limit_every=2), retry=2)),
        ("server error, no retries", "503", attempt(MockJevTransport(fail_every=1))),
        ("server error once, default retries", "503", attempt(MockJevTransport(fail_every=2), retry=2)),
        ("network down", "–", attempt(Dead())),
    ]


@lru_cache(None)
def token_study(n=800):
    alerts = soc.load().sample(n, random_state=23)
    log = []
    c = TypeSafeClient(api_key="mock", transport=MockJevTransport(log=log))
    text, js, four_one, four_sep = [], [], [], []
    for a in alerts.itertuples():
        state_json = {"alert": a.title, "rule": a.rule, "ioc_score": a.ioc_score, "after_hours": bool(a.after_hours),
                      "asset_criticality": int(a.asset_criticality), "prior_alerts_24h": int(a.prior_alerts_24h),
                      "user": a.user, "role": a.role}
        r = c.system_one(state=a.description, questions={"attack": ATTACK})
        text.append(r.usage.input_tokens)
        r = c.system_one(state=state_json, questions={"attack": ATTACK})
        js.append(r.usage.input_tokens)
        r = c.system_one(state=a.description, questions=triage_questions())
        four_one.append(r.usage.input_tokens)
        sep = 0
        for k, q in triage_questions().items():
            sep += c.system_one(state=a.description, questions={k: q}).usage.input_tokens
        four_sep.append(sep)
    return {k: np.array(v) for k, v in dict(text=text, json=js, four_one=four_one, four_sep=four_sep).items()}


def record():
    rt = retry_table()
    tk = token_study()
    cost = lambda t: float(np.mean(t)) * JEV_PRICE_IN_PER_M       # $ per million calls
    results(CH, retry={f"{q}_{r}": v[0] for (q, r), v in rt.items()},
            ok_20_0=rt[(0.2, 0)][0], ok_20_2=rt[(0.2, 2)][0], ok_05_2=rt[(0.05, 2)][0],
            tokens_text=float(np.mean(tk["text"])), tokens_json=float(np.mean(tk["json"])),
            tokens_four_one=float(np.mean(tk["four_one"])), tokens_four_sep=float(np.mean(tk["four_sep"])),
            cost_text=cost(tk["text"]), cost_json=cost(tk["json"]), cost_four_one=cost(tk["four_one"]),
            cost_four_sep=cost(tk["four_sep"]), errors=[list(e) for e in error_table()])


@figure(CH, "layers")
def layers():
    f, ax = draw.canvas("text", 2.35)
    draw.box(ax, 0.0, 1.55, 4.7, 0.5, "your code:  client.system_one(state=…, questions=…)", kind="neutral", size=6.6,
             family="JetBrains Mono")
    draw.box(ax, 0.0, 0.9, 4.7, 0.45, "typesafe-sdk: builds the request, retries, parses typed answers", kind="jev", size=6.4)
    draw.arrow(ax, (2.35, 1.55), (2.35, 1.35))
    draw.box(ax, 0.0, 0.05, 2.25, 0.6, "MockJevTransport\nanswers locally, free,\nthe same every time", kind="data", size=6.1)
    draw.box(ax, 2.45, 0.05, 2.25, 0.6, "the real network\nPOST /v1/systemone\nneeds an API key", kind="neutral", size=6.1,
             dashed=True)
    draw.arrow(ax, (1.6, 0.9), (1.1, 0.65))
    draw.arrow(ax, (3.1, 0.9), (3.55, 0.65), dashed=True)
    draw.text(ax, 2.35, 0.77, "transport=", size=6.0, ha="center", family="JetBrains Mono", color=C["ink2"])
    return f


@figure(CH, "lifecycle")
def lifecycle():
    f, ax = draw.canvas("text", 1.9)
    steps = [("build and\nvalidate", "jev"), ("send", "neutral"), ("200?", "neutral"), ("parse into\ntyped answers", "jev")]
    xs = [0.0, 1.2, 2.3, 3.45]
    for (t, k), x in zip(steps, xs):
        draw.box(ax, x, 1.0, 0.95 if x != 2.3 else 0.75, 0.5, t, kind=k, size=6.0)
    for a, b in ((0.95, 1.2), (2.15, 2.3), (3.05, 3.45)):
        draw.arrow(ax, (a, 1.25), (b, 1.25))
    draw.arrow(ax, (2.67, 1.0), (1.67, 1.0), rad=-0.6, color=C["llm"], label="429 or 5xx: wait, retry",
               labeloffset=(0, -0.55), labelcolor=C["llm"])
    draw.box(ax, 2.3, 0.05, 1.6, 0.36, "out of retries, or 4xx:\nraise a TypeSafe error", kind="fail", size=5.6)
    draw.arrow(ax, (2.67, 1.0), (2.9, 0.41), color=C["fail"])
    draw.text(ax, 3.1, 1.66, "waits 0.5 s, 1 s, 2 s … (default policy)", size=5.6, color=C["ink2"])
    return f


@figure(CH, "retries")
def retries():
    rt = retry_table()
    f, ax = subplots(width="text", height=2.2)
    clean(ax, "y")
    rs = [0, 1, 2, 3]
    for q, col in ((0.01, C["data"]), (0.05, C["jev"]), (0.2, C["llm"])):
        ys = [rt[(q, r)][0] for r in rs]
        ax.plot(rs, ys, color=col, lw=2, ls="-", marker="o", ms=4)       # measured: solid; the formula: dotted
        ax.plot(rs, [1 - q ** (r + 1) for r in rs], color=col, lw=0.8, ls=(0, (2, 2)))
        ax.text(0.1, ys[0] - 0.006, f"{q:.0%} of requests fail", va="top", fontsize=6.1)
    ax.set_xticks(rs)
    ax.set_xticklabels(["0\n(no retries)", "1", "2\n(SDK default)", "3"])
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(0.75, 1.01)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("retries allowed per call")
    ax.set_ylabel("calls that succeed")
    synthetic_tag(f, "SIMULATED FAILURES · dotted: 1 − rate^(retries+1)")
    return f


@figure(CH, "errors")
def errors():
    rows = error_table()
    f, ax = draw.canvas("text", 3.05)
    draw.text(ax, 0.0, 2.92, "what went wrong", size=6.2, weight="semibold", color=C["ink2"])
    draw.text(ax, 2.05, 2.92, "HTTP", size=6.2, weight="semibold", color=C["ink2"])
    draw.text(ax, 2.6, 2.92, "what the SDK does (measured against the mock)", size=6.2, weight="semibold", color=C["ink2"])
    for i, (what, code, res) in enumerate(rows):
        y = 2.65 - i * 0.29
        if i % 2 == 0:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (0, y - 0.14), 4.7, 0.28, fc=C["neutral_t"], ec="none", zorder=0))
        draw.text(ax, 0.05, y, what, size=6.2)
        draw.text(ax, 2.05, y, code, size=6.2, family="JetBrains Mono")
        # success and failure differ in words and style (italic for success), not only in colour
        draw.text(ax, 2.6, y, "retries, then succeeds" if res == "succeeds" else f"raises {res}", size=5.9,
                  family="JetBrains Mono", color="#1E5E42" if res == "succeeds" else C["fail"],
                  style="italic" if res == "succeeds" else "normal", weight="bold" if res == "succeeds" else "normal")
    return f


@figure(CH, "tokens")
def tokens():
    tk = token_study()
    f, ax = subplots(width="text", height=2.0)
    clean(ax, "x")
    rows = [("alert as text, 1 question", tk["text"], C["data"]),
            ("alert as JSON fields, 1 question", tk["json"], C["data"]),
            ("text, 4 questions in one call", tk["four_one"], C["jev"]),
            ("text, 4 questions in 4 calls", tk["four_sep"], C["llm"])]
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        m = float(np.mean(v))
        ax.barh(y, m, color=col, height=0.55)
        ax.text(m + 8, y, f"{m:,.0f} tokens · ${m * JEV_PRICE_IN_PER_M:,.0f} per million calls", va="center",
                fontsize=6.0)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(0, max(np.mean(r[1]) for r in rows) * 1.9)
    ax.set_xlabel("input tokens per call (mean over 800 Kestrel alerts)")
    synthetic_tag(f, "MOCK TOKEN COUNTS · price vendor-reported")
    return f


@figure(CH, "record-replay")
def record_replay():
    f, ax = draw.canvas("text", 2.2)
    draw.text(ax, 0.0, 2.08, "ONCE, with a key", size=6.3, weight="bold", color=C["ink2"])
    draw.box(ax, 0.0, 1.3, 1.1, 0.5, "your tests", kind="neutral", size=6.2)
    draw.box(ax, 1.45, 1.3, 1.35, 0.5, "RecordingTransport", kind="jev", size=6.0, family="JetBrains Mono")
    draw.box(ax, 3.2, 1.3, 1.1, 0.5, "real API", kind="neutral", size=6.2, dashed=True)
    draw.arrow(ax, (1.1, 1.55), (1.45, 1.55))
    draw.arrow(ax, (2.8, 1.55), (3.2, 1.55))
    draw.cylinder(ax, 1.75, 0.68, 0.75, 0.42, kind="data", label="cassette\n.jsonl", size=5.6)
    draw.arrow(ax, (2.12, 1.3), (2.12, 1.1), head=3)
    draw.text(ax, 0.0, 0.78, "EVERY TEST RUN, no key", size=6.3, weight="bold", color=C["ink2"])
    draw.box(ax, 0.0, 0.0, 1.1, 0.5, "your tests", kind="neutral", size=6.2)
    draw.box(ax, 1.45, 0.0, 1.35, 0.5, "ReplayTransport", kind="data", size=6.0, family="JetBrains Mono")
    draw.arrow(ax, (1.1, 0.25), (1.45, 0.25))
    draw.arrow(ax, (2.12, 0.68), (2.12, 0.5), head=3)
    draw.text(ax, 2.95, 0.25, "same answers, byte for byte,\nfree and offline", size=5.8, color=C["ink2"])
    return f


@figure(CH, "summary")
def summary():
    record()
    rr = json.load(open(ROOT / "results" / f"{CH}.json"))
    panels = [
        dict(num=1, title="One line switches mock and real", kind="jev", h=1.5,
             body="Pass transport=MockJevTransport() to the official client. Remove it, add a key, and the same code calls the API."),
        dict(num=2, title="The mock behaves like the API", kind="data", h=1.5,
             body="Same request shape, same validation errors, same typed answers. Deterministic, free, offline."),
        dict(num=3, title="Retries are built in", kind="jev", h=1.5,
             body=(f"With 20% of requests failing, {rr['ok_20_0']:.0%} of calls succeed with no retries and "
                   f"{rr['ok_20_2']:.1%} with the default two.")),
        dict(num=4, title="Some errors should never be retried", kind="fail", h=1.5,
             body="Bad keys, unknown models and malformed questions fail fast. Fix the code, not the network."),
        dict(num=5, title="Ask together", kind="jev", h=1.5,
             body=(f"Four questions in one call: about {rr['tokens_four_one']:.0f} tokens. In four calls: "
                   f"{rr['tokens_four_sep']:.0f}. The state is sent once.")),
        dict(num=6, title="Record once, replay forever", kind="data", h=1.5,
             body="Record real answers to a file once; replay them in every test run, with no key and no cost."),
    ]
    return summary_page(CH, "First calls, and the mock that makes them free", panels,
                        footer="Next: an agent where Jev decides and the LLM reasons.")
