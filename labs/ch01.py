# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 1 · What "learning" means
#
# *Decide, Don't Generate*, Chapter 1.
#
# In this lab you will:
#
# 1. meet the 20,000 synthetic alerts from Kestrel Logistics that run through the whole book,
# 2. "learn" the simplest possible model (one threshold on one number) by counting mistakes,
# 3. see why the model with the fewest mistakes can still be useless,
# 4. ask the mock Jev its first question.
#
# Everything here is synthetic. Kestrel Logistics does not exist, and no number in this notebook was measured
# on real Jev.


# %%
# Setup: installs the book's toolkit in a fresh environment. Does nothing if it's already installed.
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## 1. Meet the alerts

# %%
from jevkit import soc

alerts = soc.load()
print(len(alerts), "alerts over 28 days")
print(f"{alerts.malicious.mean():.1%} were real attacks")
alerts[["rule", "source", "ioc_score", "malicious"]].head()

# %%
# Read a few alerts the way an analyst would.
for text in alerts.description.sample(5, random_state=1):
    print("-", text, "\n")

# %% [markdown]
# ## 2. Learning, in its smallest form
#
# One number per alert: the threat-intel score. One rule: flag the alert if the score is at least `t`.
# Learning means trying every `t` and keeping the one that makes the fewest mistakes.

# %%
import numpy as np

score = alerts.ioc_score.to_numpy()
truth = alerts.malicious.to_numpy()

def mistakes(t):
    flagged = score >= t
    missed = (~flagged & (truth == 1)).sum()
    false_alarms = (flagged & (truth == 0)).sum()
    return missed + false_alarms

candidates = np.arange(0, 1.01, 0.01)
best = min(candidates, key=mistakes)
print(f"best threshold {best:.2f}: {mistakes(best):,} mistakes")
print(f"flag nothing at all: {mistakes(1.01):,} mistakes")

# %% [markdown]
# **Question.** How many real attacks does the "best" rule actually catch?

# %%
caught = ((score >= best) & (truth == 1)).sum()
print(f"caught {caught} of {truth.sum()} attacks")

# %% [markdown]
# ## 3. The answer isn't yes or no

# %%
import pandas as pd

buckets = pd.cut(alerts.ioc_score, np.linspace(0, 1, 11), include_lowest=True)
alerts.groupby(buckets, observed=True).malicious.agg(["count", "mean"]).round(3)

# %% [markdown]
# ## 4. Your first question to (mock) Jev
#
# The client below is the **official** TypeSafe SDK. Only the transport is swapped for the book's mock,
# so the answer comes from `jev-mock-synthetic`, not from Jev.

# %%
from typesafe_sdk import Noul, TypeSafeClient
from jevkit import MockJevTransport

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
alert = alerts.description[99]
r = client.system_one(
    state=alert,
    questions={"attack": Noul(instructions="Is this alert a real attack?")},
)
print(alert)
print(r.model, round(r.nouls["attack"].noul, 2))
print("was it really an attack?", bool(alerts.malicious[99]))

# %% [markdown]
# ## Try this
#
# 1. Replace `ioc_score` with `prior_alerts_24h`. Which threshold wins now? Is it more useful?
# 2. Change `mistakes` so that a missed attack counts ten times as much as a false alarm. Where does the best
#    threshold move? (Chapter 4 explains why.)
