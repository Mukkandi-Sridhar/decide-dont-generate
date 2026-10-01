# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 4 · From probabilities to actions
#
# *Decide, Don't Generate*, Chapter 4. Illustrative costs, synthetic data.
#
# 1. The one-line threshold formula.
# 2. Check it against a brute-force cost search.
# 3. See the formula fail on a miscalibrated model.
# 4. Coverage and risk: deciding only the confident cases.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from jevkit import soc, policy as pol

alerts = soc.load()
train, calib, test = soc.split(alerts)
F = lambda d: soc.feature_matrix(d).to_numpy()
model = LogisticRegression(C=1e4, max_iter=5000).fit(F(train), train.malicious)
p, y = model.predict_proba(F(test))[:, 1], test.malicious.to_numpy()

# %%
false_alarm, miss = 200, 4_000            # wrongly blocking vs letting a threat through
t = false_alarm / (false_alarm + miss)
print(f"block when P > {t:.3f}")

def total_cost(p, t):
    block = p >= t
    return (block & (y == 0)).sum() * false_alarm + (~block & (y == 1)).sum() * miss

grid = np.linspace(0.005, 0.9, 300)
best = grid[np.argmin([total_cost(p, g) for g in grid])]
print(f"brute-force best {best:.3f}; cost ${total_cost(p, best):,}")
print(f"formula          {t:.3f}; cost ${total_cost(p, t):,}")
print(f"the famous 0.5        ; cost ${total_cost(p, 0.5):,}")

# %% [markdown]
# ## The formula needs honest probabilities

# %%
pos = train[train.malicious == 1]
neg = train[train.malicious == 0].sample(len(pos), random_state=0)
bal = pd.concat([pos, neg])
pb = LogisticRegression(C=1e4, max_iter=5000).fit(F(bal), bal.malicious).predict_proba(F(test))[:, 1]
print(f"rebalanced model at the formula line: ${total_cost(pb, t):,}")

# %% [markdown]
# ## Coverage and risk

# %%
confidence = np.maximum(p, 1 - p)
correct = ((p >= 0.5) == (y == 1)).astype(float)
coverage, risk, _ = pol.coverage_risk(confidence, correct)
for c in (0.5, 0.8, 0.9, 1.0):
    print(f"decide the surest {c:.0%}: error rate {risk[int(c * len(coverage)) - 1]:.2%}")
