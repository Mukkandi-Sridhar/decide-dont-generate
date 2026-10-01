# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 22 · What changes now
#
# *Decide, Don't Generate*, Chapter 22. List the steps of a task in your own system, mark each as observe, decide,
# generate or act, and see what moving the decisions to a decision model would change. LLM timings and prices are
# the book's illustrative assumptions; Jev's are vendor-reported. Replace them with your own measurements.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
from jevkit import econ, llm

# Edit this list: (step, kind, how many times per task)
my_task = [
    ("read the ticket", "observe", 1),
    ("which tool next?", "decide", 4),
    ("is this enough?", "decide", 3),
    ("fetch records", "observe", 3),
    ("is this safe to do?", "decide", 1),
    ("do it", "act", 1),
    ("write the reply", "generate", 1),
]
tasks_per_day = 5_000

# %%
def cost(task, decider):
    seconds = dollars = 0.0
    for _, kind, n in task:
        if kind == "decide":
            if decider == "llm":
                seconds += n * llm.MockLLM.simulated_latency_s(40)
                dollars += n * llm.MockLLM.simulated_cost_usd(500, 40)
            else:
                seconds += n * econ.JEV_LATENCY[1]          # the slow end of the vendor-reported range
                dollars += n * econ.JEV_PRICE
        elif kind == "generate":
            seconds += llm.MockLLM.simulated_latency_s(150)
            dollars += llm.MockLLM.simulated_cost_usd(800, 150)
        else:
            seconds += n * 0.3
    return seconds, dollars

decisions = sum(n for _, k, n in my_task if k == "decide")
steps = sum(n for _, _, n in my_task)
print(f"{decisions} of {steps} steps are decisions ({decisions / steps:.0%})")
for who in ("llm", "decision model"):
    s, d = cost(my_task, "llm" if who == "llm" else "jev")
    print(f"{who:>15} decides: {s:5.1f} s per task, ${d * tasks_per_day:,.2f} a day")
