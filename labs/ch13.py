# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 13 · The bake-off: six ways to make a decision
#
# *Decide, Don't Generate*, Chapter 13.
#
# Six methods decide whether each Kestrel alert is a real threat. All train or tune on weeks 1-3 and are scored
# on week 4. Jev answers come from `jev-mock-synthetic`; the LLM is the book's `MockLLM`. **Synthetic, not measured
# on real Jev or a real LLM.** The mock LLM was built as a noisier reader than mock Jev, so the accuracy gap between
# those two is a design choice. The other columns (calibration shape, parse failures, variance, labels) are the lesson.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import numpy as np
from jevkit import bakeoff as B

r = B.run()
y = r["y"]
print(f"history {len(r['hist']):,} alerts, live {len(y):,} alerts ({y.sum()} real threats)")
for m in B.METHODS:
    s = B.scores(r["probs"][m], y, verdict=r["rules_live"] if m == "rules" else None)
    print(f"{B.NAMES[m]:>26}: AUC {s['auc']:.3f}  ECE {s['ece']:.3f}  Brier {s['brier']:.4f}  "
          f"caught@240/day {s['caught']:.0%}  F1@0.5 {s['f1']:.2f}")

# %% [markdown]
# ## Jev on raw text instead of fields

# %%
s = B.scores(r["jev_text"], y)
print(f"Jev (text): AUC {s['auc']:.3f}  ECE {s['ece']:.3f}")

# %% [markdown]
# ## How many distinct confidences does the LLM state?

# %%
vals, counts = np.unique(np.round(r["probs"]["llm_json"], 3), return_counts=True)
print(dict(zip(vals, counts)))
print(f"parse failures: {1 - r['parse_ok'].mean():.1%}")

# %% [markdown]
# ## Stability: ask again

# %%
print(f"verdicts that change over 5 calls at temperature 0.7: "
      f"{B.flip_rate(r['live'].description[:1500].tolist()):.1%}")

# %% [markdown]
# ## Labels needed

# %%
curve = B.labels_curve(sizes=(100, 300, 1000, 3000), repeats=3)
for n, a, b in zip(curve["sizes"], curve["logistic"], curve["text_clf"]):
    print(f"{n:>6} labels: logistic {a:.3f}   text classifier {b:.3f}")

# %% [markdown]
# **Try:** write a better rule in `B.rules`. How close to logistic regression can a hand-written rule get?
