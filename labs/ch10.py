# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 10 · The type system: choice, score, noul
#
# *Decide, Don't Generate*, Chapter 10. Answers from `jev-mock-synthetic`.
#
# 1. Ask all three types in one call, with a typed response model.
# 2. See what happens when a choice has no right option.
# 3. Read a score's whole distribution, not just its average.
# 4. Choose an action by expected cost.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from typesafe_sdk import (Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer,
                          SystemOneResponse, TypeSafeClient)
from jevkit import MockJevTransport, soc

class Triage(SystemOneResponse):          # your answers, as typed fields
    attack: NoulAnswer
    kind: ChoiceAnswer
    severity: ScoreAnswer

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
alerts = soc.load()
questions = {
    "attack": Noul(instructions="Is this alert a real attack?",
                   criteria={"true": "A real attacker or real malware is involved",
                             "false": "Normal activity, a test, or a false positive"}),
    "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
    "severity": Score(instructions="How severe is this?", criteria=soc.SEVERITY_LEVELS),
}
t = client.system_one(state=alerts.description[118], questions=questions, response_model=Triage)
print(round(t.attack.noul, 3), t.kind.choice, round(t.kind.confidence, 3))
print({k: round(v, 3) for k, v in t.severity.probabilities.items()}, "average", round(t.severity.score, 2))

# %% [markdown]
# ## A choice with no right answer

# %%
threat_only = {c: None for c in soc.CATEGORIES if c not in ("benign", "policy_violation")}
harmless = alerts.query("malicious == 0").head(200).description
conf = [client.system_one(state=s, questions={"k": Choice(criteria=threat_only)}).choices["k"].confidence
        for s in harmless]
print(f"harmless alerts, forced into threat labels: median confidence {np.median(conf):.2f}")

# %% [markdown]
# ## Tails, not averages

# %%
p = t.severity.probabilities
print(f"P(medium or high) = {p[2] + p[3]:.2f}, although the average level is {t.severity.score:.2f}")

# %% [markdown]
# ## Expected cost across a choice

# %%
probs = np.array([t.kind.probabilities[c] for c in soc.CATEGORIES])
close_cost = np.array([2000, 3000, 5000, 4000, 1500, 300, 0])   # cost of closing, if the truth is each category
print(f"expected cost of closing this alert: ${probs @ close_cost:,.0f}")
