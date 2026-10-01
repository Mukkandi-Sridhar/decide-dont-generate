# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 19 · An applications gallery
#
# *Decide, Don't Generate*, Chapter 19. The domain table (`jevkit.gallery.DOMAINS`) is illustrative: every volume
# and cost is an assumption. The support-ticket demo uses synthetic tickets and `jev-mock-synthetic`, whose general
# engine is a simple word-matcher. **Synthetic, not measured on real Jev.**


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## 1. The same line formula, six jobs

# %%
from jevkit import gallery

for d in gallery.DOMAINS:
    who = "person decides" if d.human_final else "model decides"
    print(f"{d.name:>20}: flag above P = {gallery.line(d):.2%}   ({who}; needs '{d.none_option}')")

# %% [markdown]
# ## 2. Support tickets: route, and measure before trusting

# %%
import numpy as np
from typesafe_sdk import Choice, Noul, TypeSafeClient
from jevkit import MockJevTransport, calibration as cal

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
topic = Choice(instructions="Which team should handle this ticket?",
               criteria={"billing": "Payments, refunds, invoices, charges",
                         "technical": "Bugs, errors, crashes, the app not working",
                         "account": "Login, password, access, profile", "other": None})
urgent = Noul(instructions="Does this need a reply today?")

tickets = gallery.tickets()
top, conf, p_urgent = [], [], []
for t in tickets:
    r = client.system_one(state=t["text"], questions={"topic": topic, "urgent": urgent})
    top.append(r.choices["topic"].choice)
    conf.append(r.choices["topic"].confidence)
    p_urgent.append(r.nouls["urgent"].noul)
top, conf = np.array(top), np.array(conf)
truth = np.array([t["topic"] for t in tickets])
print(f"routed correctly: {(top == truth).mean():.0%}")
for line in (0.6, 0.8):
    m = conf >= line
    print(f"route only when confidence >= {line}: {m.mean():.0%} routed, {(top == truth)[m].mean():.0%} right")
print("urgency:", cal.summary(p_urgent, [t["urgent"] for t in tickets]))

# %% [markdown]
# **Try:** rewrite the `account` description so that login problems stop landing in `technical`.
# Does accuracy on `account` tickets improve? What happened to `technical`?
