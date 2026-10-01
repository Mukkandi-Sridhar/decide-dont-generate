# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 3 · Calibration: when 0.8 really means 80%
#
# *Decide, Don't Generate*, Chapter 3. Synthetic data throughout.
#
# 1. Build a reliability diagram by hand.
# 2. Compute ECE and the Brier decomposition.
# 3. Break calibration by rebalancing the training data.
# 4. Fix it with Platt, temperature and isotonic, fitted on a separate calibration set.
# 5. Check calibration per alert source.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from jevkit import soc, calibration as cal

alerts = soc.load()
train, calib, test = soc.split(alerts)
F = lambda d: soc.feature_matrix(d).to_numpy()
model = LogisticRegression(C=1e4, max_iter=5000).fit(F(train), train.malicious)
p, y = model.predict_proba(F(test))[:, 1], test.malicious.to_numpy()

# %% [markdown]
# ## A reliability diagram, by hand

# %%
bins = np.linspace(0, 1, 11)
which = np.clip(np.digitize(p, bins) - 1, 0, 9)
table = pd.DataFrame({"bin": which, "p": p, "y": y}).groupby("bin").agg(
    said=("p", "mean"), happened=("y", "mean"), n=("y", "size"))
table.round(3)

# %%
print("ECE  ", round(cal.ece(p, y), 4))
print(cal.brier_decomposition(p, y))

# %% [markdown]
# ## Break it: train on 50/50 rebalanced data

# %%
pos = train[train.malicious == 1]
neg = train[train.malicious == 0].sample(len(pos), random_state=0)
balanced = pd.concat([pos, neg])
model_b = LogisticRegression(C=1e4, max_iter=5000).fit(F(balanced), balanced.malicious)
pb = model_b.predict_proba(F(test))[:, 1]
print(f"average P: {pb.mean():.1%}   real attack rate: {y.mean():.1%}")
print(cal.summary(pb, y))

# %% [markdown]
# ## Fix it, on the calibration set (never the test set)

# %%
pb_cal = model_b.predict_proba(F(calib))[:, 1]
for name, fixer in [("Platt", cal.Platt()), ("Temperature", cal.Temperature()), ("Isotonic", cal.Isotonic())]:
    fixer.fit(pb_cal, calib.malicious)
    print(f"{name:>12}: ECE {cal.ece(fixer(pb), y):.4f}   log loss {cal.log_loss(fixer(pb), y):.4f}")

# %% [markdown]
# **Question:** why can't temperature scaling fix this model? (Hint: what does a temperature do to P = 0.5?)

# %% [markdown]
# ## Calibration per source

# %%
cols = [c for c in soc.feature_matrix(train).columns if not c.startswith("rule_")]
generic = LogisticRegression(C=1e4, max_iter=5000).fit(soc.feature_matrix(train)[cols], train.malicious)
pg = generic.predict_proba(soc.feature_matrix(test)[cols])[:, 1]
pd.DataFrame({"source": test.source, "said": pg, "happened": y}).groupby("source").mean().round(3)
