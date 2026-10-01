# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 18 · Case study: SOC alert triage
#
# *Decide, Don't Generate*, Chapter 18. Kestrel's SOC before and after the decision layer, simulated on the
# book's synthetic alerts. Jev answers come from `jev-mock-synthetic`. **Synthetic, not measured on real Jev.**


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from jevkit import ops

r = ops.before_after()
print("policy:", r["policy"])
for name in ("s_old", "s_new"):
    s = r[name]
    print(f"{name}: threats seen {s['threats_seen_share']:.0%}, never seen {s['threats_unseen_per_day']:.1f}/day, "
          f"within an hour {s['threats_within_hour_share']:.0%}")

# %% [markdown]
# ## Shadow mode: what the system would have done, against what was true

# %%
y = r["y"].astype(bool)
z = r["new"][0]
for zone in ("act", "review", "escalate"):
    print(f"{zone:>9}: {((z == zone) & ~y).sum() / 7:6.1f} harmless/day   {((z == zone) & y).sum() / 7:5.1f} threats/day")

# %% [markdown]
# ## The queue: first come, first served against highest probability first

# %%
t = ops.minutes(r["live"].timestamp)
p = r["live"].pc.to_numpy()
rev = z == "review"
fifo = ops.serve(t[rev])
prio = ops.serve(t[rev], priority=p[rev])
yt = y[rev]
for name, w in (("first come", fifo), ("by probability", prio)):
    print(f"{name:>15}: median wait for a real threat {np.median(w[yt & np.isfinite(w)]):.0f} min")

# %% [markdown]
# ## The campaign week, day by day

# %%
for row in ops.campaign():
    print(row["respond"], row["day"] + 1, row["reviews"], row["capacity"], round(row["missed"], 1))

# %% [markdown]
# **Try:** change `ops.PAGE_RESPONSE_MIN`, or respond on day 2 instead of day 3 (`ops.campaign(respond_day=1)`).
