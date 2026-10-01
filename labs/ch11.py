# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 11 · Testing Jev's calibration yourself
#
# *Decide, Don't Generate*, Chapter 11. The same protocol works on real Jev; here every answer
# comes from `jev-mock-synthetic` and every company is synthetic.
#
# 1. Measure calibration at two companies with different base rates.
# 2. Fix it: adjust for the base rate, or fit Platt scaling on a few hundred labels.
# 3. See how noisy ECE is with few labels.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from jevkit import soc, calibration as cal
from jevkit.batch import score_alerts

kestrel = soc.load()
harbor = soc.to_frame(soc.generate(n=8000, seed=21, logit_shift={r: -1.3 for r in soc.RULES}))
p_k, p_h = score_alerts(kestrel), score_alerts(harbor)
for name, p, df in (("Kestrel", p_k, kestrel), ("Harbor", p_h, harbor)):
    lo, hi = cal.ece_interval(p, df.malicious)
    print(f"{name:>8}: attacks {df.malicious.mean():.1%}, average P {p.mean():.1%}, "
          f"ECE {cal.ece(p, df.malicious):.3f} (range {lo:.3f}-{hi:.3f})")

# %% [markdown]
# ## Fix 1: adjust for the base rate (needs only an estimate of it)

# %%
rng = np.random.default_rng(0)
labelled = rng.choice(len(harbor), 300, replace=False)
rest = np.setdiff1d(np.arange(len(harbor)), labelled)
y = harbor.malicious.to_numpy()
estimated_rate = y[labelled].mean()
adjusted = cal.prior_shift(p_h, kestrel.malicious.mean(), estimated_rate)
print(f"base-rate adjusted ECE: {cal.ece(adjusted[rest], y[rest]):.3f}")

# %% [markdown]
# ## Fix 2: Platt scaling on the same 300 labels

# %%
platt = cal.Platt().fit(p_h[labelled], y[labelled])
print(f"Platt ECE: {cal.ece(platt(p_h[rest]), y[rest]):.3f}")

# %% [markdown]
# ## How many labels do you need?

# %%
for n in (100, 400, 1600):
    vals = [cal.ece(p_k[i], kestrel.malicious.to_numpy()[i])
            for i in (rng.choice(len(p_k), n, replace=False) for _ in range(200))]
    print(f"{n:>5} labels: measured ECE typically {np.median(vals):.3f} "
          f"(90% range {np.percentile(vals, 5):.3f}-{np.percentile(vals, 95):.3f})")
