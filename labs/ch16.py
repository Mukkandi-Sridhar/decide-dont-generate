# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 16 · First calls, and the mock that makes them free
#
# *Decide, Don't Generate*, Chapter 16. Everything here runs against `jev-mock-synthetic` through the official
# `typesafe-sdk`. To call real Jev, set `JEVKIT_LIVE=1` and `TYPESAFE_API_KEY`, and use `jevkit.client()`.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## 1. The first call

# %%
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient
from jevkit import MockJevTransport

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
r = client.system_one(
    state="Unsigned executable launched from %TEMP% on HR-SRV-187. Threat intel score: 0.38.",
    questions={"attack": Noul(instructions="Is this alert a real attack?")},
)
print(r.nouls["attack"].noul, r.usage, r.request_id)
print(r.raw_http_response.headers["x-jevkit-synthetic"])

# %% [markdown]
# ## 2. What the SDK does when things go wrong

# %%
from typesafe_sdk import TypeSafeError

def attempt(**kw):
    transport = kw.pop("transport", MockJevTransport())
    questions = kw.pop("questions", {"a": Noul(instructions="Real attack?")})
    try:
        c = TypeSafeClient(api_key=kw.pop("api_key", "mock"), transport=transport,
                           retry=RetryPolicy(max_retries=kw.pop("retries", 0), backoff_initial=0.001))
        c.system_one(state="x", questions=questions, **kw)
        return "ok"
    except TypeSafeError as e:
        return type(e).__name__

print("unknown model:     ", attempt(model="jev-nope"))
print("empty choice:      ", attempt(questions={"a": Choice(criteria={})}))
print("503, no retries:   ", attempt(transport=MockJevTransport(fail_every=1)))
print("503 once, retries: ", attempt(transport=MockJevTransport(fail_every=2), retries=2))

# %% [markdown]
# ## 3. Retries against random failures

# %%
for retries in (0, 1, 2, 3):
    t = MockJevTransport(fail_rate=0.2, seed=retries)
    c = TypeSafeClient(api_key="mock", transport=t,
                       retry=RetryPolicy(max_retries=retries, backoff_initial=0.001, backoff_max=0.002))
    ok = 0
    for i in range(300):
        try:
            c.system_one(state=f"alert {i}", questions={"a": Noul(instructions="Real attack?")})
            ok += 1
        except TypeSafeError:
            pass
    print(f"{retries} retries: {ok / 300:.1%} succeed, {len(t.statuses) / 300:.2f} requests per call")

# %% [markdown]
# ## 4. Ask together: one call, four questions

# %%
from jevkit import soc
qs = {"attack": Noul(instructions="Is this alert a real attack?"),
      "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
      "severity": Score(criteria=soc.SEVERITY_LEVELS),
      "page": Noul(instructions="Should on-call be woken for this?")}
state = soc.load().description[118]
together = client.system_one(state=state, questions=qs).usage.input_tokens
apart = sum(client.system_one(state=state, questions={k: q}).usage.input_tokens for k, q in qs.items())
print(f"one call: {together} input tokens; four calls: {apart}")

# %% [markdown]
# ## 5. Record once, replay in tests

# %%
import os, tempfile
from jevkit.mock import RecordingTransport, ReplayTransport

path = os.path.join(tempfile.mkdtemp(), "cassette.jsonl")
rec = TypeSafeClient(api_key="mock", transport=RecordingTransport(MockJevTransport(), path))
first = rec.system_one(state=state, questions=qs)
rep = TypeSafeClient(api_key="mock", transport=ReplayTransport(path))
again = rep.system_one(state=state, questions=qs)
print(first.nouls["attack"].noul == again.nouls["attack"].noul)
