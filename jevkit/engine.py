"""The deterministic decision engine behind `jev-mock-synthetic`.

This is NOT Jev and does not imitate Jev's internals (which are not public).
It only imitates Jev's *interface*: typed questions in, typed answers with
probabilities out. Every number it produces is synthetic.

How it decides
--------------
1. If the state looks like a Kestrel Logistics SOC alert, a "SOC skill" reads
   the alert's fields and scores it with a deliberately imperfect copy of the
   generator's true model (slightly overconfident, a little noisy). That makes
   calibration lessons realistic: the mock is good, but not perfect.
2. Anything else goes to a small lexical engine: it compares the words in the
   state against each option's label and description and turns the
   similarities into probabilities with a softmax.

Same input, same output, every time. Noise is seeded from a hash of the input.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

from . import soc

MODEL_NAME = "jev-mock-synthetic"
MODEL_DESCRIPTION = ("Deterministic synthetic stand-in for Jev, from the book 'Decide, Don't Generate'. "
                     "Not a real model. Every number it returns is synthetic.")
MODEL_RELEASE = "2026-09-29"

# How the mock distorts the true SOC probability. Documented in DECISIONS.md.
SOC_SHARPEN = 1.18       # >1 means overconfident at the extremes
SOC_NOISE = 0.35         # std of per-alert logit noise
SOC_TEXT_NOISE = 0.55    # extra noise when the state is raw text only
SOC_BIAS = 0.10          # small upward bias in the logit

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for is are was were be been it this that with as at by from or and not no "
            "any what which who whom how does do did should would could can will if then than so into about "
            "please i you we they he she my our your their me us them".split())

# A tiny synonym table so everyday examples behave sensibly.
_SYN = {
    "charged": "billing", "charge": "billing", "refund": "billing", "invoice": "billing", "payment": "billing",
    "paid": "billing", "price": "billing", "subscription": "billing", "card": "billing", "twice": "billing",
    "crash": "technical", "crashes": "technical", "error": "technical", "bug": "technical", "login": "technical",
    "password": "technical", "broken": "technical", "slow": "technical", "app": "technical", "install": "technical",
    "delivery": "shipping", "delivered": "shipping", "arrived": "shipping", "package": "shipping",
    "parcel": "shipping", "tracking": "shipping", "order": "shipping", "late": "shipping", "courier": "shipping",
    "urgent": "urgent", "asap": "urgent", "immediately": "urgent", "outage": "urgent", "down": "urgent",
    "today": "urgent", "soon": "urgent", "urgency": "urgent",
    "tone": "angry", "upset": "angry", "annoyed": "angry", "hostile": "angry", "rude": "toxic", "abusive": "toxic",
    "furious": "angry", "angry": "angry", "unacceptable": "angry", "terrible": "angry", "worst": "angry",
    "ridiculous": "angry", "thanks": "calm", "thank": "calm", "appreciate": "calm", "great": "happy", "love": "happy",
    "idiot": "toxic", "stupid": "toxic", "hate": "toxic", "shut": "toxic", "moron": "toxic",
    "winner": "spam", "prize": "spam", "free": "spam", "click": "spam", "offer": "spam", "lottery": "spam",
    "cancel": "cancel", "unsubscribe": "cancel", "close": "cancel",
}


def _tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
        if w in _SYN:
            out.append(_SYN[w])
    return out


def _flatten(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return " ".join(f"{k} {_flatten(v)}" for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return " ".join(_flatten(v) for v in x)
    return str(x)


def _seed(*parts) -> int:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _round_probs(p):
    """Round to 4 dp while keeping the sum at exactly 1."""
    p = np.asarray(p, float)
    r = np.round(p, 4)
    r[int(np.argmax(r))] += round(1.0 - float(r.sum()), 4)
    return [float(max(0.0, round(v, 4))) for v in r]


# ---------------------------------------------------------------------------
# SOC skill
# ---------------------------------------------------------------------------
_RULE_HINTS = [
    ("encoded_powershell", ["encoded command", "powershell"]),
    ("office_spawns_shell", ["winword.exe spawned", "office application spawned", "spawned cmd"]),
    ("lsass_access", ["lsass"]),
    ("unsigned_temp_binary", ["unsigned executable", "%temp%", "temp folder"]),
    ("suspicious_link", ["contains a link", "suspicious link"]),
    ("macro_attachment", ["macro"]),
    ("lookalike_domain", ["similar to", "looks like ours"]),
    ("impossible_travel", ["impossible travel", "two locations"]),
    ("mfa_fatigue", ["mfa push", "push requests"]),
    ("new_admin_role", ["administrator role", "admin role"]),
    ("rare_domain_beacon", ["beacon", "every 30 seconds", "every 60 seconds", "every 300 seconds"]),
    ("internal_port_scan", ["probed", "port scan"]),
    ("large_upload", ["uploaded", "large outbound"]),
    ("public_bucket", ["made public", "bucket"]),
    ("key_new_asn", ["access key"]),
    ("dlp_personal_cloud", ["personal cloud"]),
]


def looks_like_soc(state) -> bool:
    if isinstance(state, dict):
        if state.get("rule") in soc.RULES:
            return True
        if isinstance(state.get("alert"), dict):
            return looks_like_soc(state["alert"])
        state = _flatten(state)
    if isinstance(state, list):
        state = _flatten(state)
    t = str(state).lower()
    return ("threat intel score" in t) or any(h in t for _, hs in _RULE_HINTS for h in hs) and ("alert" in t or "user" in t or "host" in t or "kestrel" in t)


def _parse_soc_text(t: str) -> dict:
    low = t.lower()
    rule = next((r for r, hs in _RULE_HINTS if any(h in low for h in hs)), "suspicious_link")
    m = re.search(r"threat intel score[^0-9]*([01](?:\.\d+)?)", low)
    ioc = float(m.group(1)) if m else 0.2
    m = re.search(r"(\d+) related alert", low)
    prior = int(m.group(1)) if m else 0
    crit = 1
    for i, w in enumerate(["low", "medium", "high", "critical"]):
        if f"{w} criticality" in low or f"criticality {w}" in low:
            crit = i
    hm = re.search(r"\b(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2})", t)
    after = "outside business hours" in low or (bool(hm) and (int(hm.group(2)) < 7 or int(hm.group(2)) >= 20))
    mb = 0.3
    m = re.search(r"([\d,]+(?:\.\d+)?) mb", low)
    if m:
        mb = float(m.group(1).replace(",", ""))
    role = "service" if "svc-" in low else ("admin" if "admin" in low and rule != "new_admin_role" else "employee")
    return dict(rule=rule, asset_criticality=crit, after_hours=after, ioc_score=ioc,
                known_tool="approved it tooling" in low or "change ticket" in low,
                prior_alerts_24h=prior, new_geo="not seen for this user" in low or "never seen" in low,
                mfa_ok=("mfa completed" in low) or bool(re.search(r"\d+ approved", low)), mb_out=mb, role=role)


def soc_features(state) -> tuple[dict, bool]:
    """Return (features, text_only)."""
    if isinstance(state, dict) and isinstance(state.get("alert"), dict):
        state = state["alert"]
    if isinstance(state, dict) and state.get("rule") in soc.RULES:
        f = {k: state.get(k) for k in soc.FEATURE_FIELDS}
        defaults = dict(asset_criticality=1, after_hours=False, ioc_score=0.2, known_tool=False, prior_alerts_24h=0,
                        new_geo=False, mfa_ok=True, mb_out=0.3, role="employee")
        for k, v in defaults.items():
            if f.get(k) is None:
                f[k] = v
        return f, False
    return _parse_soc_text(_flatten(state)), True


def soc_probability(state) -> float:
    f, text_only = soc_features(state)
    z = soc.true_logit(f["rule"], int(f["asset_criticality"]), bool(f["after_hours"]), float(f["ioc_score"]),
                       bool(f["known_tool"]), int(f["prior_alerts_24h"]), bool(f["new_geo"]), bool(f["mfa_ok"]),
                       float(f["mb_out"]), str(f["role"]))
    rng = np.random.default_rng(_seed("soc", f))
    noise = rng.normal(0, SOC_NOISE) + (rng.normal(0, SOC_TEXT_NOISE) if text_only else 0.0)
    zm = SOC_SHARPEN * (z + SOC_BIAS) + noise
    return float(1 / (1 + math.exp(-zm)))


_MAL_WORDS = {"malicious", "attack", "compromise", "compromised", "threat", "intrusion", "true", "real", "hostile",
              "breach", "bad", "evil", "incident", "phish", "phishing", "malware", "attacker"}
_BEN_WORDS = {"benign", "false", "harmless", "noise", "safe", "legitimate", "expected", "fine", "normal", "clean"}
_ESC_WORDS = {"escalate", "escalation", "page", "contain", "containment", "isolate", "urgent", "immediately", "oncall"}
_SEV_WORDS = {"severity", "severe", "urgent", "urgency", "priority", "impact", "risk", "critical", "serious"}


def _bag(x) -> set[str]:
    return set(_WORD.findall(_flatten(x).lower()))


def _category_dist(f: dict, p_mal: float) -> dict[str, float]:
    cats = soc.RULES[f["rule"]]["cats"]
    d = {c: 0.0 for c in soc.CATEGORIES}
    for c in cats:
        d[c] += p_mal / len(cats)
    pol = 0.25 if f["rule"] in ("public_bucket", "dlp_personal_cloud", "new_admin_role") and not f["known_tool"] else 0.02
    d["policy_violation"] += (1 - p_mal) * pol
    d["benign"] += (1 - p_mal) * (1 - pol)
    return d


def _severity_dist(f: dict, p_mal: float, n_levels: int) -> np.ndarray:
    crit = int(f["asset_criticality"])
    # distribution on a 0..3 scale, then squeeze/stretch to n levels
    ben = np.array([0.82, 0.16, 0.02, 0.0])
    hi = 0.35 + 0.18 * crit
    mal = np.array([0.0, 0.18, 0.82 - hi * 0.6, hi * 0.6])
    mal = np.clip(mal, 0, None)
    mal = mal / mal.sum()
    d4 = (1 - p_mal) * ben + p_mal * mal
    if n_levels == 4:
        return d4 / d4.sum()
    xs = np.linspace(0, 3, n_levels)
    out = np.zeros(n_levels)
    for lvl, pr in enumerate(d4):
        j = int(np.argmin(np.abs(xs - lvl)))
        out[j] += pr
    out = out + 1e-4
    return out / out.sum()


def _label_affinity(label: str, desc) -> float:
    """+1 means 'this option is the malicious one', -1 the benign one, 0 neutral."""
    words = _bag(label) | _bag(desc)
    mal = len(words & _MAL_WORDS) + (2 if label.lower() in ("malicious", "attack", "true_positive", "yes") else 0)
    ben = len(words & _BEN_WORDS) + (2 if label.lower() in ("benign", "false_positive", "no", "noise") else 0)
    if mal == ben:
        return 0.0
    return 1.0 if mal > ben else -1.0


def _soc_answer(state, qtype: str, q: dict):
    f, _ = soc_features(state)
    p = soc_probability(state)
    instr = _bag(q.get("instructions"))
    if qtype == "noul":
        crit = q.get("criteria") or {}
        words = instr | _bag(crit.get("true"))
        if words & _ESC_WORDS:
            sev = _severity_dist(f, p, 4)
            return {"type": "noul", "noul": round(float(sev[2:].sum()), 4)}
        if (words & _BEN_WORDS) and not (words & _MAL_WORDS):
            return {"type": "noul", "noul": round(1 - p, 4)}
        return {"type": "noul", "noul": round(p, 4)}
    if qtype == "choice":
        crit = q["criteria"]
        labels = list(crit)
        norm = [re.sub(r"[^a-z_]", "", l.lower().replace(" ", "_")) for l in labels]
        if sum(n in soc.CATEGORIES for n in norm) >= max(2, len(labels) // 2):
            cd = _category_dist(f, p)
            raw = np.array([cd.get(n, 0.0) + 1e-4 for n in norm])
            probs = raw / raw.sum()
        else:
            aff = np.array([_label_affinity(l, crit[l]) for l in labels])
            if (aff != 0).any():
                # malicious-leaning options share p, benign-leaning share 1-p, neutral ones get the uncertain middle
                mid = 4 * p * (1 - p)  # peaks at p = 0.5
                w = np.where(aff > 0, p, np.where(aff < 0, 1 - p, 0.6 * mid))
                if (aff == 0).any():
                    w = np.where(aff != 0, w * (1 - 0.6 * mid), w)
                probs = (w + 1e-4) / (w + 1e-4).sum()
            else:
                return None
        pr = _round_probs(probs)
        probs_d = dict(zip(labels, pr))
        best = labels[int(np.argmax(pr))]
        return {"type": "choice", "choice": best, "confidence": probs_d[best], "probabilities": probs_d}
    if qtype == "score":
        crit = q["criteria"]
        n = len(crit)
        if (instr & _SEV_WORDS) or n in (3, 4, 5):
            d = _severity_dist(f, p, n)
            pr = _round_probs(d)
            return _score_payload(crit, pr)
    return None


def _score_payload(criteria, probs):
    n = len(criteria)
    legend = {str(i): criteria[i] for i in range(n)}
    pd_ = {str(i): probs[i] for i in range(n)}
    exp = round(sum(i * probs[i] for i in range(n)), 4)
    return {"type": "score", "score": exp, "confidence": float(max(probs)), "legend": legend, "probabilities": pd_}


# ---------------------------------------------------------------------------
# Generic lexical engine
# ---------------------------------------------------------------------------
def _sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    num = sum(min(a[k], b[k]) for k in a.keys() & b.keys())
    return num / math.sqrt(sum(a.values()) * sum(b.values()))


def _generic_answer(state, qtype: str, q: dict, salt: str):
    st = Counter(_tokens(_flatten(state)))
    instr = _flatten(q.get("instructions"))
    rng = np.random.default_rng(_seed("gen", salt, _flatten(state), q))
    if qtype == "choice":
        crit = q["criteria"]
        labels = list(crit)
        sims = []
        for l in labels:
            opt = Counter(_tokens(l.replace("_", " ") + " " + _flatten(crit[l])))
            sims.append(_sim(st, opt) + 0.8 * (1.0 if any(t in st for t in _tokens(l.replace("_", " "))) else 0.0))
        z = np.array(sims) * 4.5 + rng.normal(0, 0.15, len(labels))
        # "other"/"none" style options soak up probability when nothing matches
        for i, l in enumerate(labels):
            if l.lower() in ("other", "none", "unknown", "unclear"):
                z[i] = max(z[i], 1.2 - 3.0 * max(sims))
        pr = _round_probs(_softmax(z))
        probs_d = dict(zip(labels, pr))
        best = labels[int(np.argmax(pr))]
        return {"type": "choice", "choice": best, "confidence": probs_d[best], "probabilities": probs_d}
    if qtype == "noul":
        crit = q.get("criteria") or {}
        yes = Counter(_tokens(instr + " " + _flatten(crit.get("true"))))
        no = Counter(_tokens(_flatten(crit.get("false"))))
        s_yes = _sim(st, yes)
        s_no = _sim(st, no) if no else 0.12
        z = 6.0 * (s_yes - s_no) + rng.normal(0, 0.2)
        return {"type": "noul", "noul": round(float(1 / (1 + math.exp(-z))), 4)}
    if qtype == "score":
        crit = q["criteria"]
        n = len(crit)
        sims = np.array([_sim(st, Counter(_tokens(_flatten(c)))) for c in crit])
        qwords = set(_tokens(instr + " " + " ".join(_flatten(c) for c in crit)))
        dims = [d for d in ("urgent", "angry", "toxic") if d in qwords] or ["urgent", "angry", "toxic"]
        total = max(1, sum(st.values()))
        intensity = min(1.0, sum(st[d] for d in dims) / total * 6)
        pos = np.arange(n) / max(1, n - 1)
        z = 3.0 * sims - 4.0 * (pos - intensity) ** 2 + rng.normal(0, 0.1, n)
        return _score_payload(crit, _round_probs(_softmax(z)))
    raise ValueError(qtype)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def answer(state, questions: dict) -> dict:
    """Answer every question about `state`. Pure function: same input, same output."""
    out = {}
    is_soc = looks_like_soc(state)
    for name, q in questions.items():
        qtype = q["type"]
        a = None
        if is_soc:
            qq = dict(q)
            qq["instructions"] = _flatten(q.get("instructions")) + " " + name.replace("_", " ")
            a = _soc_answer(state, qtype, qq)
        if a is None:
            a = _generic_answer(state, qtype, q, name)
        out[name] = a
    return out


def count_tokens(obj) -> int:
    """Rough token count: about four characters per token."""
    return max(1, math.ceil(len(json.dumps(obj, default=str)) / 4))
