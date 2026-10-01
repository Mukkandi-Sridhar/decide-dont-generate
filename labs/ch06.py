# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 6 · How an LLM writes, and structured outputs
#
# *Decide, Don't Generate*, Chapter 6. Toy models only; `llm-mock-synthetic` is not a real LLM.
#
# The sections below follow the chapter in order.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## How an LLM writes, one token at a time

# %%
from jevkit import llm, text

notes = [" ".join(n) for n in text.notes_corpus(5000)]
tok = llm.BPE().fit(notes, merges=150)
for s in ["the ransomware was quarantined", "passw0rd reset", "2026-09-14"]:
    t = tok.tokenize(s)
    print(f"{len(t):2d} tokens  {t}")

# %% [markdown]
# ## The generation loop

# %%
lm = llm.TrigramLM().fit(text.notes_corpus(30000))
dist = lm.next_distribution("the", "phishing")
print(sorted(dist.items(), key=lambda kv: -kv[1])[:5])

for T in (0, 0.7, 1.5):
    for seed in range(2):
        words = [w for w, p in lm.generate(["the", "phishing"], temperature=T, seed=seed)]
        print(f"T={T:<3}  the phishing {' '.join(words)}")

# %% [markdown]
# ## Same question, twenty times

# %%
from jevkit import soc
alert = soc.load().description[118]
mock = llm.MockLLM(temperature=0.7)
answers = [mock.classify(alert)["label"] for _ in range(20)]
print(alert[:100], "...")
print({a: answers.count(a) for a in set(answers)})

# %% [markdown]
# **Try:** set `temperature=0` and repeat. Then try other alerts. Which ones flip the most?

# %% [markdown]
# ## Structured outputs and JSON mode

# %%
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError
from jevkit import soc, llm

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")          # unknown fields are an error
    verdict: Literal["malicious", "benign"]
    confidence: float
    rule: str
    threat_intel_score: float
    after_hours: bool
    related_alerts: int

def parse(raw: str) -> Verdict | None:
    try:
        return Verdict.model_validate_json(raw)
    except ValidationError:
        return None

alerts = soc.load()
train, calib, test = soc.split(alerts)
mock = llm.MockLLM()
results = [parse(mock.structured(t)) for t in test.description]
print(f"valid: {sum(r is not None for r in results):,} of {len(results):,}")

# %%
def ask_with_retry(text, tries=2):
    for attempt in range(tries):
        v = parse(llm.MockLLM(json_mode=attempt > 0).structured(text))
        if v is not None:
            return v
    return None                                        # caller must fail safe (send to review)

bad = [t for t, r in zip(test.description, results) if r is None][:3]
for t in bad:
    print(ask_with_retry(t))

# %% [markdown]
# ## Three confidences

# %%
import numpy as np
from jevkit import calibration as cal
from jevkit.batch import score_alerts

y = test.malicious.to_numpy()
verbal = []
for t in test.description:
    r = mock.classify(t)
    verbal.append(r["confidence"] if r["label"] == "malicious" else 1 - r["confidence"])
token = [mock.label_token_prob(t) for t in test.description]
alerts["p_text"] = score_alerts(alerts, "text")
jev = soc.split(alerts)[2].p_text          # same test alerts, same raw text

for name, p in (("stated in JSON", verbal), ("first-token prob", token), ("decision model", jev)):
    s = cal.summary(np.asarray(p), y)
    print(f"{name:>16}: AUC {s['auc']:.3f}  ECE {s['ece']:.3f}")

# %% [markdown]
# ## The typed version

# %%
from typesafe_sdk import Choice, Noul, TypeSafeClient
from jevkit import MockJevTransport

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
r = client.system_one(state=test.description.iloc[0], questions={
    "attack": Noul(instructions="Is this alert a real attack?"),
    "kind": Choice(criteria={c: None for c in soc.CATEGORIES}),
})
print(r.nouls["attack"].noul, r.choices["kind"].choice, r.choices["kind"].probabilities)
