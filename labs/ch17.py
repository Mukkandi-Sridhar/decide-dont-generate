# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 17 · A hybrid agent: Jev decides, the LLM reasons
#
# *Decide, Don't Generate*, Chapter 17. The agent harness is `jevkit.agent.SOCAgent`. Decisions go through the
# official SDK to `jev-mock-synthetic`; writing goes to the book's `MockLLM`. Timings and prices are simulated:
# the LLM's are illustrative, Jev's sit inside the vendor-reported range. **Synthetic throughout.**


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
from collections import Counter
import numpy as np
from jevkit import soc, agent

alerts = soc.load()
_, live = soc.history_and_live(alerts)
sample = live.sample(300, random_state=24)

# %% [markdown]
# ## 1. One alert, step by step

# %%
hybrid = agent.SOCAgent(always=("threat_intel",))
t = hybrid.run(live.iloc[118])
for s in t.steps:
    print(f"{s.kind:>8}  {s.name:<16} {s.latency_s:5.2f}s  {s.detail}")
print("action:", t.action)

# %% [markdown]
# ## 2. Watch what each decision step answers

# %%
first = agent.SOCAgent()                         # the Chapter 7 version
picks = Counter(s.detail.split()[1] for a in sample.itertuples()
                for s in first.run(a).steps if s.name == "next_evidence")
print(picks)                                     # which option never appears?

# %% [markdown]
# ## 3. Three agents on the same alerts

# %%
def summarise(ag, label):
    tr = [ag.run(a) for a in sample.itertuples()]
    y = sample.malicious.to_numpy()
    acts = np.array([x.action for x in tr])
    print(f"{label:>10}: {np.mean([sum(s.latency_s for s in x.steps) for x in tr]):.1f} s/alert, "
          f"reviews {np.mean(acts == 'review'):.0%}, threats auto-closed {((acts == 'act') & (y == 1)).sum()}")

summarise(first, "first")
summarise(hybrid, "hybrid")
summarise(agent.SOCAgent(always=("threat_intel",), decider="llm"), "all-LLM")
