# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 8 · System 1 and System 2
#
# *Decide, Don't Generate*, Chapter 8. Synthetic data and synthetic models.
#
# A fast System 1 (the mock decision model reading raw alert text) decides every alert.
# The cases it is least sure about go to System 2, modelled here as a careful investigation that
# learns the alert's true probability. How much of System 2's benefit do we get by sending only a few?


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from jevkit import soc, calibration as cal
from jevkit.batch import score_alerts

alerts = soc.load()
alerts["p"] = score_alerts(alerts, "text")
history, live = soc.history_and_live(alerts)
system1 = cal.Platt().fit(history.p, history.malicious)(live.p)
system2 = live.p_true.to_numpy()           # idealised careful investigation
y = live.malicious.to_numpy()

miss, false_alarm = 10_000, 400
line = false_alarm / (false_alarm + miss)
logit = lambda x: np.log(x / (1 - x))
doubt_order = np.argsort(np.abs(logit(np.clip(system1, 1e-6, 1 - 1e-6)) - logit(line)))

def daily_cost(share):
    k = int(share * len(y))
    act = (system1 >= line).astype(int)
    sent = doubt_order[:k]
    act[sent] = (system2[sent] >= line).astype(int)
    return (((act == 1) & (y == 0)).sum() * false_alarm + ((act == 0) & (y == 1)).sum() * miss) / 7

for share in (0, 0.1, 0.2, 0.5, 1.0):
    print(f"send {share:4.0%} to System 2: ${daily_cost(share):,.0f} a day")
