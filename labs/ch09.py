# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 9 · Inside Jev: what we know and what we don't
#
# *Decide, Don't Generate*, Chapter 9.
#
# 1. See exactly what the official SDK sends and receives (the mock records the exchange).
# 2. List the models the endpoint offers.
# 3. Do the vendor's price arithmetic yourself.
#
# Prices and demo figures are vendor-reported. Answers come from `jev-mock-synthetic`.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import json
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from jevkit import MockJevTransport

exchanges = []
client = TypeSafeClient(api_key="mock", transport=MockJevTransport(log=exchanges))
r = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "topic": Choice(criteria={"billing": "Payments, refunds, invoices",
                                  "technical": "Bugs, errors, access", "other": None}),
        "urgent": Noul(instructions="Does this need a reply today?"),
        "tone": Score(criteria=["calm", "annoyed", "angry"]),
    },
)
request_body, response_body = exchanges[0]
print(json.dumps(request_body, indent=2))
print(json.dumps(response_body, indent=2))

# %%
print([m.name for m in client.models.list().models])

# %% [markdown]
# ## Price arithmetic (vendor-reported inputs)

# %%
price_per_million_input = 0.042          # vendor-reported
tokens_per_decision = 500
per_decision = tokens_per_decision * price_per_million_input / 1e6
print(f"${per_decision:.6f} per decision; ${per_decision * 1e6:,.0f} per million decisions")

doom_per_hour, decisions_per_second = 7.0, 10   # vendor-reported demo figures
per_doom_decision = doom_per_hour / (decisions_per_second * 3600)
print(f"implied input tokens per Doom decision: {per_doom_decision / (price_per_million_input / 1e6):,.0f}")
