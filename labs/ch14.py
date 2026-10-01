# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 14 · Act, review, or escalate
#
# *Decide, Don't Generate*, Chapter 14.
#
# You will turn P(threat) from the mock Jev into a three-zone policy for the Kestrel Logistics SOC:
#
# 1. see what a single line at 0.5 does,
# 2. derive thresholds from costs,
# 3. check calibration before trusting any threshold,
# 4. respect the capacity of real people,
# 5. log every decision and fail safe,
# 6. watch a phishing campaign break last month's calibration.
#
# All numbers are synthetic: not measured on real Jev.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
from jevkit import soc, policy as pol, calibration as cal
from jevkit.batch import score_alerts

alerts = soc.load()
alerts["p"] = score_alerts(alerts)          # every alert asked through the SDK + mock
history, live = soc.history_and_live(alerts)
print(len(history), "alerts of history,", len(live), "in the live week")

# %% [markdown]
# ## 1. One line at 0.5

# %%
one_line = pol.ThreeZonePolicy(low=0.5, high=0.5)
r = pol.evaluate(one_line, live.p, live.malicious)
print(f"auto-closed per day: {r['act'] / 7:.0f}")
print(f"real threats among them per day: {r['missed_by_automation'] / 7:.1f}")

# %% [markdown]
# ## 2. Let the costs draw the lines

# %%
costs = pol.Costs(auto_close_miss=10_000, review_minutes=12, analyst_per_hour=75,
                  review_miss_rate=0.05, escalate_false_alarm=400)
low, high = pol.cost_optimal_thresholds(costs)
print(f"act below {low:.4f}; escalate at {high:.2f} and above")

# %% [markdown]
# ## 3. Is the number honest? Recalibrate on history.

# %%
platt = cal.Platt().fit(history.p, history.malicious)
history["pc"] = platt(history.p)
live["pc"] = platt(live.p)
for name, col in (("raw", "p"), ("calibrated", "pc")):
    m = live[col] < 0.02
    print(f"{name:>10}: says {live[col][m].mean():.2%}, really {live.malicious[m].mean():.2%}")

# %% [markdown]
# ## 4. Respect capacity

# %%
per_day = len(history) / 21
policy = pol.best_policy_with_capacity(history.pc, history.malicious,
                                       max_review_rate=240 / per_day,
                                       max_escalate_rate=40 / per_day, costs=costs)
r = pol.evaluate(policy, live.pc, live.malicious, costs)
print(policy)
for k in ("act", "review", "escalate", "missed_by_automation", "false_pages"):
    print(f"{k:>22}: {r[k] / 7:6.1f} per day")

# %% [markdown]
# **Try:** change 240 to 320 (eight analysts). How many fewer threats are auto-closed per day?

# %% [markdown]
# ## 5. Log every decision, and fail safe

# %%
import hashlib, json
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient, TypeSafeError
from jevkit import MockJevTransport

flaky = TypeSafeClient(api_key="mock", transport=MockJevTransport(fail_every=3),
                       retry=RetryPolicy(max_retries=0))

def decide(alert, client):
    state = alert.description
    try:
        r = client.system_one(state=state, questions={
            "attack": Noul(instructions="Is this alert a real attack?")})
        p = r.nouls["attack"].noul
        pc = float(platt([p])[0])
        action, model = policy.decide(pc), r.model
    except TypeSafeError as err:
        p = pc = None
        action, model = "review", f"unavailable ({type(err).__name__})"
    return {"alert_id": alert.alert_id,
            "state_sha": hashlib.sha256(state.encode()).hexdigest()[:12],
            "model": model, "p": p, "p_calibrated": pc,
            "policy": f"v3 low={policy.low:.4f} high={policy.high:.2f}",
            "action": action}

for i in range(3):
    print(json.dumps(decide(live.iloc[i], flaky)))

# %% [markdown]
# ## 6. A phishing campaign

# %%
week5 = soc.campaign_week()
week5["pc"] = platt(score_alerts(week5))
print(f"model expected {week5.pc.mean():.1%} threats; reality was {week5.malicious.mean():.1%}")
z = policy.decide_many(week5.pc)
print(f"reviews per day: {(z == 'review').sum() / 7:.0f} (capacity 240)")
