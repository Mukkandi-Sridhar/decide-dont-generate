# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 2 · Probability, and how a machine learns it
#
# *Decide, Don't Generate*, Chapter 2. Synthetic data throughout.
#
# The sections below follow the chapter in order.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## Probability is the language of decisions

# %%
import numpy as np
from jevkit import soc

alerts = soc.load()
y = alerts.malicious
print(f"P(attack)                        = {y.mean():.3f}")
bad = alerts.ioc_score >= 0.7
print(f"P(attack | threat-intel >= 0.7)  = {y[bad].mean():.3f}")
print(f"P(threat-intel >= 0.7 | attack)  = {bad[y == 1].mean():.3f}")

# %% [markdown]
# Notice the last two lines are *different questions* with different answers.

# %% [markdown]
# ## The base-rate trap

# %%
def p_real_given_alert(base_rate, hit_rate, false_alarm_rate):
    real = base_rate * hit_rate
    false = (1 - base_rate) * false_alarm_rate
    return real / (real + false)

for base in (0.001, 0.01, 0.076, 0.3):
    print(f"base rate {base:>6.1%}: P(real | alert) = {p_real_given_alert(base, 0.99, 0.05):.1%}")

# %% [markdown]
# ## Bayes' rule: multiply the odds

# %%
def likelihood_ratio(mask):
    return mask[y == 1].mean() / mask[y == 0].mean()

odds = y.mean() / (1 - y.mean())
for name, mask in [("bad threat intel", alerts.ioc_score >= 0.7),
                   ("after hours", alerts.after_hours),
                   ("new country", alerts.new_geo)]:
    lr = likelihood_ratio(mask)
    odds *= lr
    print(f"{name:>17}: x{lr:4.1f}  ->  P = {odds / (1 + odds):.0%}")

both = (alerts.ioc_score >= 0.7) & alerts.after_hours
print(f"\nreality for the first two clues together: {y[both].mean():.0%} (n={both.sum()})")

# %% [markdown]
# ## Precision, recall, accuracy

# %%
flag = alerts.ioc_score >= 0.5
tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum()
fn = (~flag & (y == 1)).sum(); tn = (~flag & (y == 0)).sum()
print(f"precision {tp / (tp + fp):.0%}  recall {tp / (tp + fn):.0%}  accuracy {(tp + tn) / len(y):.0%}")
print(f"'never alert' accuracy: {1 - y.mean():.0%}")

# %% [markdown]
# ## Small samples wobble

# %%
rng = np.random.default_rng(0)
for n in (20, 100, 1000):
    draws = [y.sample(n, random_state=int(s)).mean() for s in rng.integers(0, 10**6, 5)]
    print(n, [f"{d:.0%}" for d in draws])

# %% [markdown]
# ## Data, loss and gradient descent

# %%
import numpy as np
from jevkit import soc

alerts = soc.load()
train, calib, test = soc.split(alerts)        # 60% / 20% / 20%
X = soc.feature_matrix(train).to_numpy()
Xt = soc.feature_matrix(test).to_numpy()
y, yt = train.malicious.to_numpy(), test.malicious.to_numpy()
print(X.shape, Xt.shape)

# %% [markdown]
# ## The model: add up evidence, squash with the S-curve

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# put every feature on a similar scale so one step size suits them all
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs, Xts = (X - mu) / sd, (Xt - mu) / sd

# %% [markdown]
# ## Gradient descent

# %%
w, b = np.zeros(X.shape[1]), 0.0
for step in range(401):
    p = sigmoid(Xs @ w + b)
    w -= 0.5 * Xs.T @ (p - y) / len(y)     # the slope, for every weight at once
    b -= 0.5 * np.mean(p - y)
    if step % 100 == 0:
        print(f"step {step:3d}  loss {log_loss(p, y):.4f}")

# %% [markdown]
# ## On alerts it never saw

# %%
pt = sigmoid(Xts @ w + b)
print(f"test log loss   {log_loss(pt, yt):.4f}")
print(f"truth log loss  {log_loss(test.p_true.to_numpy(), yt):.4f}   (the best possible)")

# %% [markdown]
# **Try:** change the step size from 0.5 to 5 and to 0.01. What happens to the printed loss?
#
# **Try:** add a feature that leaks the answer, e.g. `Xs = np.c_[Xs, y]`. What happens to the training
# loss? Why is that useless?

# %% [markdown]
# ## Overfitting: a model that memorises

# %%
from sklearn.tree import DecisionTreeClassifier
for depth in (2, 4, 8, 16):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    tr = log_loss(np.clip(t.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3), y)
    te = log_loss(np.clip(t.predict_proba(Xt)[:, 1], 1e-3, 1 - 1e-3), yt)
    print(f"depth {depth:2d}: train {tr:.3f}  test {te:.3f}")
