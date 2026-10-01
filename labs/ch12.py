# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 12 · The Jevons paradox of decisions
#
# *Decide, Don't Generate*, Chapter 12.
#
# 1. Play with the constant-elasticity model: when does a price cut raise total spend?
# 2. Explore Kestrel's illustrative "day of decisions" at LLM and Jev prices.
# 3. See how review load grows under a fixed-share rule, and why a cost line doesn't.
#
# Everything in `jevkit.econ` is an **illustrative model**: every value is a stated assumption, not a measurement.
# Jev's price and latency are vendor-reported. The LLM figures are this book's illustrative assumptions.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## 1. Elasticity: spend = price x quantity

# %%
from jevkit import econ

for e in (0.5, 1.0, 1.5):
    print(f"elasticity {e}: a 100x price cut changes total spend by x{econ.elastic_spend(0.01, e):.2f}")

# %% [markdown]
# ## 2. A day of decisions at Kestrel (illustrative)

# %%
for p in econ.POOLS:
    print(f"{p.name:>22}: {p.per_day:>9,} a day, typical value ${p.median_value:g}, budget {p.budget_s:g} s")

# %%
worlds = {"LLM": econ.day(econ.LLM_PRICE, econ.LLM_LATENCY),
          "Jev (fast end)": econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[0]),
          "Jev (slow end)": econ.day(econ.JEV_PRICE, econ.JEV_LATENCY[1])}
for name, d in worlds.items():
    print(f"{name:>15}: {d['decisions']:>10,} decisions, ${d['spend']:,.0f} spent, "
          f"${d['value']:,.0f} of expected loss avoided")
    print("                 ", {k: v for k, v in d["by_pool"].items()})

# %% [markdown]
# ## 3. A new job appears
#
# Suppose cheap decisions make a new job possible: checking every file shared outside the company,
# 5 million a day, each worth about $0.00003. `econ.NEW_JOB` describes it. Add it and watch the bill.

# %%
print(econ.NEW_JOB)
for name, (price, lat) in {"LLM": (econ.LLM_PRICE, econ.LLM_LATENCY),
                           "Jev": (econ.JEV_PRICE, econ.JEV_LATENCY[0])}.items():
    before = econ.day(price, lat)
    after = econ.day(price, lat, extra=(econ.NEW_JOB,))
    print(f"{name}: {before['decisions']:,} -> {after['decisions']:,} decisions; "
          f"${before['spend']:,.0f} -> ${after['spend']:,.0f} a day")

# %% [markdown]
# ## 4. Who reviews all this?

# %%
for name, (price, lat) in {"LLM": (econ.LLM_PRICE, econ.LLM_LATENCY),
                           "Jev": (econ.JEV_PRICE, econ.JEV_LATENCY[0])}.items():
    f = econ.flags(price, lat)
    print(f"{name}: {f['decided']:,} reviewable decisions -> top 0.1% sends {f['top_share']:,}, "
          f"cost line sends {f['cost_line']:,} (capacity 240)")

# %% [markdown]
# **Try:** change a pool's `median_value` or `budget_s` and re-run. Which assumption moves the answer most?
