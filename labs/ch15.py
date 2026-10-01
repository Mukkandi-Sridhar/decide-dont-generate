# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 15 · A catalogue of decision patterns
#
# *Decide, Don't Generate*, Chapter 15. Six patterns, each small enough to read in one screen.
# Jev answers come from `jev-mock-synthetic` and the LLM is the book's `MockLLM`: synthetic throughout.
# For patterns 4 and 6 the mock's numbers show the *shape* of the call, not a meaningful judgement.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from typesafe_sdk import Choice, Noul, TypeSafeClient
from jevkit import MockJevTransport, soc, llm

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
mock = llm.MockLLM()
alerts = soc.load()
alert = alerts.iloc[118]

# %% [markdown]
# ## 1. Extract, then decide

# %%
fields = mock.extract(alert.description)          # the LLM writes...
attack = Noul(instructions="Is this alert a real attack?")
r = client.system_one(state={"alert": alert.title, **fields}, questions={"attack": attack})
print(fields)
print("P(attack) =", round(r.nouls["attack"].noul, 3))  # ...Jev decides

# %% [markdown]
# ## 2. The decide step inside a loop: one typed question per step

# %%
next_step = Choice(instructions="Which evidence should the agent gather next?",
                   criteria={"threat_intel": None, "host_history": None, "policy": None, "none": None})
a = client.system_one(state=alert.description, questions={"next": next_step}).choices["next"]
print(a.choice, {k: round(v, 2) for k, v in a.probabilities.items()})

# %% [markdown]
# ## 3. Guardrail gate: check a proposed action on trusted facts only

# %%
def gate(p_safe, low=0.3, high=0.75):
    return "allow" if p_safe >= high else ("block" if p_safe < low else "ask a person")

trusted = {k: alert[k] for k in ("rule", "ioc_score", "asset_criticality", "known_tool", "new_geo", "mfa_ok")}
trusted = {k: (v.item() if hasattr(v, "item") else v) for k, v in trusted.items()}
p = client.system_one(state=trusted, questions={"attack": attack}).nouls["attack"].noul
print("close the alert?", gate(1 - p), f"(P(safe) = {1 - p:.2f})")

# %% [markdown]
# ## 4. Check the writer (shape only with the mock)

# %%
draft = "Benign admin activity; the change ticket covers it."
check = Noul(instructions="Is every claim in the note supported by the evidence?")
r = client.system_one(state={"evidence": alert.description, "note": draft}, questions={"supported": check})
print("P(supported) =", round(r.nouls["supported"].noul, 2))

# %% [markdown]
# ## 5. Router with a fallback queue

# %%
kind = Choice(criteria={c: None for c in soc.CATEGORIES})
sample = alerts.sample(400, random_state=5)
routed = right = 0
for row in sample.itertuples():
    k = client.system_one(state=row.description, questions={"k": kind}).choices["k"]
    if max(k.probabilities.values()) >= 0.8:
        routed += 1
        right += k.choice == row.category
print(f"routed {routed / len(sample):.0%}, of which {right / routed:.0%} to the right team")

# %% [markdown]
# ## 6. Memory controller (shape only with the mock)

# %%
keep = Noul(instructions="Will this fact matter for future alerts about the same host or user?")
fact = "rhea.novak has an approved change ticket for admin tooling until Friday."
print("P(keep) =", round(client.system_one(state=fact, questions={"keep": keep}).nouls["keep"].noul, 2))
