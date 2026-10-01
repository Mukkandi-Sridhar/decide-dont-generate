# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 21 · Capstone: a production decision service
#
# *Decide, Don't Generate*, Chapter 21. `jevkit.service` in action: a versioned config, the decision record, the
# fail-safe, the daily monitor and shadow comparison. Jev answers come from `jev-mock-synthetic`. **Synthetic.**


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %%
import json, os, tempfile
import numpy as np
from typesafe_sdk import RetryPolicy, TypeSafeClient
from jevkit import soc, service as S, MockJevTransport
from jevkit.batch import score_alerts

alerts = soc.load()
alerts["p"] = score_alerts(alerts)
history, live = soc.history_and_live(alerts)

# %% [markdown]
# ## 1. Fit a config, look at its fingerprint

# %%
cfg = S.fit_config(history, version="triage-2026.10.2")
print(cfg)
print("fingerprint:", cfg.fingerprint())

# %% [markdown]
# ## 2. Decide, and read the record

# %%
log = os.path.join(tempfile.mkdtemp(), "decisions.jsonl")
client = TypeSafeClient(api_key="mock", transport=MockJevTransport(), retry=RetryPolicy(max_retries=0))
svc = S.DecisionService(cfg, client, log_path=log)
rec = svc.decide(live.iloc[118])
print(json.dumps(json.loads(rec.model_dump_json()), indent=1))

# %% [markdown]
# ## 3. Make it fail on purpose

# %%
broken = TypeSafeClient(api_key="mock", transport=MockJevTransport(fail_rate=0.3, seed=1),
                        retry=RetryPolicy(max_retries=0))
svc_b = S.DecisionService(cfg, broken)
recs = [svc_b.decide(a) for _, a in live.iloc[:200].iterrows()]
print(S.daily_report(recs))

# %% [markdown]
# ## 4. One day's monitor, with labels from what people saw

# %%
t = live.timestamp.astype("datetime64[ns]")
day = live[t < t.min() + np.timedelta64(1, "D")]
recs = [svc.decide(a) for _, a in day.iterrows()]
labels = {r.alert_id: int(m) for r, m in zip(recs, day.malicious) if r.zone != "act" or r.audit}
print(S.daily_report(recs, labels))

# %% [markdown]
# ## 5. Shadow a candidate version

# %%
old = S.fit_config(history, version="triage-2026.10.1", account_for_rules=False)
svc_old = S.DecisionService(old, client)
current = [svc_old.decide(a) for _, a in day.iterrows()]
print(S.shadow_compare(current, recs))
