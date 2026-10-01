# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 20 · Build your own System One model
#
# *Decide, Don't Generate*, Chapter 20. TinyJev: a shared encoder and three typed heads (noul, choice, score),
# trained with log loss in NumPy + autograd, calibrated with one temperature per head, then plugged into the
# official SDK. Trained on the book's synthetic alerts. **Nothing here is TypeSafe's architecture.**


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## 1. Data: two weeks to train, one to calibrate, one to test

# %%
import numpy as np
from jevkit import soc, tinyjev as tj, calibration as cal

alerts = soc.load()
history, live = soc.history_and_live(alerts)
t = history.timestamp.astype("datetime64[ns]")
train_df = history[t < np.datetime64("2026-09-15")].reset_index(drop=True)
calib_df = history[t >= np.datetime64("2026-09-15")].reset_index(drop=True)
train = tj.features(train_df) + tj.targets(train_df)
calib = tj.features(calib_df) + tj.targets(calib_df)
print(len(train_df), "to train,", len(calib_df), "to calibrate,", len(live), "to test")

# %% [markdown]
# ## 2. Train all three heads at once

# %%
params, history_log = tj.train(*train, steps=1500, val=calib, every=250)
for row in history_log:
    print(row["step"], {k: round(v, 3) for k, v in row["val"].items()})

# %% [markdown]
# ## 3. Calibrate: one temperature per head

# %%
T = tj.fit_temperatures(params, *calib)
print("temperatures:", [round(x, 2) for x in T])
y = live.malicious.to_numpy()
for name, temps in (("as trained", (1, 1, 1)), ("with temperatures", T)):
    p_attack, p_kind, p_sev = tj.predict(params, live, temps)
    print(f"{name:>18}: AUC {cal.summary(p_attack, y)['auc']:.3f}  ECE {cal.ece(p_attack, y):.3f}")

# %% [markdown]
# ## 4. Plug it into the official SDK

# %%
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="mock", transport=tj.TinyJevTransport(params, T))
a = live.iloc[118]
state = {"alert": a.title, **{k: (a[k].item() if hasattr(a[k], "item") else a[k]) for k in soc.FEATURE_FIELDS}}
r = client.system_one(state=state, questions={
    "attack": Noul(instructions="Is this alert a real attack?"),
    "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
    "severity": Score(criteria=soc.SEVERITY_LEVELS)})
print(r.nouls["attack"].noul, r.choices["kind"].choice, r.scores["severity"].probabilities)

# %% [markdown]
# **Try:** train for 6,000 steps with `l2=0.0` and `d=64`. Watch the held-out loss. What temperature does the
# overtrained model need, and does it fix the reliability curve?
