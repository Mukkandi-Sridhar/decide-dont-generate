"""Synthetic SOC alert generator.

Every alert in this book comes from here. Nothing is real: hosts, users,
IP addresses and the company ("Kestrel Logistics") are invented.

The generator is built so that the *true* probability that an alert is
malicious is known exactly. We sample features first, compute a known logit
from them, then draw the label from that probability. This lets the book
compare any model's probabilities against the truth, which is impossible
with real data.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field

import numpy as np

COMPANY = "Kestrel Logistics"

# ---------------------------------------------------------------------------
# Detection rules. `prior` is the rule's contribution to the logit; `noisy`
# rules fire a lot on harmless activity. `cats` lists the attack categories
# the rule usually catches.
# ---------------------------------------------------------------------------
RULES = {
    "encoded_powershell": dict(source="EDR", title="Encoded PowerShell command", prior=-0.2, weight=6, cats=["malware", "credential_abuse"]),
    "office_spawns_shell": dict(source="EDR", title="Office application spawned a shell", prior=0.6, weight=3, cats=["malware", "phishing"]),
    "lsass_access": dict(source="EDR", title="Process read LSASS memory", prior=0.9, weight=2, cats=["credential_abuse"]),
    "unsigned_temp_binary": dict(source="EDR", title="Unsigned binary ran from a temp folder", prior=-0.6, weight=8, cats=["malware"]),
    "suspicious_link": dict(source="Email", title="Suspicious link in inbound email", prior=-1.0, weight=14, cats=["phishing"]),
    "macro_attachment": dict(source="Email", title="Attachment with macros", prior=-0.8, weight=9, cats=["phishing", "malware"]),
    "lookalike_domain": dict(source="Email", title="Sender domain looks like ours", prior=0.2, weight=4, cats=["phishing"]),
    "impossible_travel": dict(source="Identity", title="Impossible travel sign-in", prior=-0.9, weight=10, cats=["credential_abuse"]),
    "mfa_fatigue": dict(source="Identity", title="Repeated MFA push denials", prior=0.3, weight=3, cats=["credential_abuse"]),
    "new_admin_role": dict(source="Identity", title="New admin role assigned", prior=-1.2, weight=5, cats=["credential_abuse", "policy_violation"]),
    "rare_domain_beacon": dict(source="Network", title="Regular beaconing to a rare domain", prior=0.1, weight=6, cats=["malware"]),
    "internal_port_scan": dict(source="Network", title="Port scan from internal host", prior=-0.7, weight=5, cats=["recon"]),
    "large_upload": dict(source="Network", title="Large outbound upload", prior=-1.1, weight=7, cats=["exfiltration"]),
    "public_bucket": dict(source="Cloud", title="Storage bucket made public", prior=-1.3, weight=4, cats=["exfiltration", "policy_violation"]),
    "key_new_asn": dict(source="Cloud", title="Access key used from a new network", prior=-0.4, weight=4, cats=["credential_abuse"]),
    "dlp_personal_cloud": dict(source="DLP", title="Sensitive file sent to personal cloud", prior=-1.4, weight=6, cats=["exfiltration", "policy_violation"]),
}

CATEGORIES = ["phishing", "malware", "credential_abuse", "exfiltration", "recon", "policy_violation", "benign"]

SEVERITY_LEVELS = [
    "Informational: no action needed",
    "Low: review within a week",
    "Medium: review today",
    "High: act within the hour",
]

DEPARTMENTS = ["finance", "engineering", "operations", "sales", "hr", "it", "legal", "warehouse"]
ROLES = ["employee", "contractor", "admin", "service"]
_FIRST = ["asha", "ravi", "mei", "tomas", "leila", "sam", "noor", "diego", "ana", "kofi", "yuki", "arjun",
          "fatima", "lars", "priya", "omar", "grace", "ivan", "zara", "ben", "chen", "sofia", "malik", "rhea"]
_LAST = ["rao", "okafor", "lindqvist", "haddad", "tan", "moreau", "kim", "silva", "nair", "fischer", "abe",
         "mensah", "kowalski", "reyes", "iyer", "hassan", "brennan", "novak", "das", "cohen"]

# True-model weights. Changing these changes every number in the book,
# so they are pinned and covered by tests.
TRUE_WEIGHTS = dict(
    intercept=-4.6,
    ioc=5.2,              # threat-intel reputation of the destination/sender (0..1)
    after_hours=1.0,
    criticality=0.45,     # per level, 0..3
    known_tool=-2.4,      # activity matches a known IT/admin tool or change ticket
    prior_alerts=0.45,    # per related alert in 24h (capped at 6)
    new_geo=1.1,
    mfa_ok=-1.3,
    admin=0.4,
    service=-0.5,
    bytes_log=0.22,       # log1p(MB out)
)


@dataclass
class Alert:
    alert_id: str
    timestamp: str
    source: str
    rule: str
    title: str
    host: str
    user: str
    department: str
    role: str
    asset_criticality: int
    after_hours: bool
    ioc_score: float
    known_tool: bool
    prior_alerts_24h: int
    new_geo: bool
    mfa_ok: bool
    mb_out: float
    description: str
    # ground truth (never shown to models)
    malicious: int = 0
    p_true: float = 0.0
    category: str = "benign"
    severity: int = 0
    extra: dict = field(default_factory=dict)

    def features(self) -> dict:
        """Structured fields a detector or an extractor could see."""
        return {k: getattr(self, k) for k in FEATURE_FIELDS}

    def state(self, mode: str = "structured") -> dict | str:
        """What we send to a decision model. `text` mode is the raw alert only."""
        if mode == "text":
            return self.description
        s = {"alert": self.title, "source": self.source, "rule": self.rule, "host": self.host,
             "user": self.user, "department": self.department}
        s.update({k: getattr(self, k) for k in FEATURE_FIELDS if k not in ("rule",)})
        s["description"] = self.description
        return s

    def truth(self) -> dict:
        return dict(malicious=self.malicious, p_true=self.p_true, category=self.category, severity=self.severity)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


FEATURE_FIELDS = ["rule", "asset_criticality", "after_hours", "ioc_score", "known_tool",
                  "prior_alerts_24h", "new_geo", "mfa_ok", "mb_out", "role"]


def true_logit(rule: str, crit: int, after_hours: bool, ioc: float, known_tool: bool, prior: int,
               new_geo: bool, mfa_ok: bool, mb_out: float, role: str) -> float:
    w = TRUE_WEIGHTS
    z = w["intercept"] + RULES[rule]["prior"]
    z += w["ioc"] * ioc + w["after_hours"] * after_hours + w["criticality"] * crit
    z += w["known_tool"] * known_tool + w["prior_alerts"] * min(prior, 6)
    z += w["new_geo"] * new_geo + w["mfa_ok"] * mfa_ok
    z += w["admin"] * (role == "admin") + w["service"] * (role == "service")
    z += w["bytes_log"] * math.log1p(mb_out)
    # a small interaction: bad reputation *and* off-hours is worse than either
    z += 1.6 * ioc * after_hours
    return z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ip(rng, bad: bool) -> str:
    if bad:
        return f"{rng.choice([45, 91, 103, 185, 193])}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"
    return f"10.{rng.integers(0, 40)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"


_DOMAINS_BAD = ["invoice-share.top", "cdn-update-check.net", "secure-docs-login.co", "files-transfer.xyz",
                "m365-verify.info", "parcel-notice.click"]
_DOMAINS_OK = ["sharepoint.com", "github.com", "windowsupdate.com", "salesforce.com", "zoom.us", "okta.com"]
_MAIL_OK = ["docusign.net", "mailchimp.com", "partners.kestrel-freight.com", "invoices.acme-supply.com", "zoom.us"]
_LOOKALIKE_BAD = ["kestre1-logistics.com", "kestrel-logistlcs.com", "kestrellogistics-hr.com"]
_LOOKALIKE_OK = ["kestrel-logistics.co.uk", "kestrelfreight.com"]


def _describe(rng, a: dict) -> str:
    """Write the alert the way a SIEM would show it: terse, inconsistent, a bit messy."""
    r = a["rule"]
    host, user, dept = a["host"], a["user"], a["department"]
    t = a["time"]
    crit_word = ["low", "medium", "high", "critical"][a["asset_criticality"]]
    ioc = a["ioc_score"]
    bad_dest = ioc > 0.55
    if r == "lookalike_domain":
        dom = rng.choice(_LOOKALIKE_BAD if bad_dest else _LOOKALIKE_OK)
    elif r in ("suspicious_link", "macro_attachment"):
        dom = rng.choice(_DOMAINS_BAD if bad_dest else _MAIL_OK)
    else:
        dom = rng.choice(_DOMAINS_BAD if bad_dest else _DOMAINS_OK)
    ip = _ip(rng, bad_dest)
    tool = " Matches approved IT tooling (change ticket on file)." if a["known_tool"] else ""
    geo = " Sign-in from a country not seen for this user." if a["new_geo"] else ""
    mfa = " MFA completed." if a["mfa_ok"] else " MFA not completed."
    prior = a["prior_alerts_24h"]
    related = f" {prior} related alert{'s' if prior != 1 else ''} in the last 24h." if prior else " No related alerts in 24h."
    intel = f" Threat intel score: {ioc:.2f}."
    size = a["mb_out"]
    parts = {
        "encoded_powershell": f"powershell.exe ran with an encoded command on {host} ({dept}, {crit_word} criticality) as {user} at {t}. Outbound connection to {ip}.",
        "office_spawns_shell": f"winword.exe spawned cmd.exe on {host} ({dept}) under {user} at {t}. Child process contacted {dom}.",
        "lsass_access": f"Process on {host} opened a handle to lsass.exe with read access. User {user}, {dept}, asset criticality {crit_word}. Time {t}.",
        "unsigned_temp_binary": f"Unsigned executable launched from %TEMP% on {host} by {user} at {t}. Network call to {ip}.",
        "suspicious_link": f"Inbound email to {user} ({dept}) contains a link to {dom}. Received {t}. User clicked: {'yes' if rng.random() < 0.35 else 'unknown'}.",
        "macro_attachment": f"Email to {user} with a macro-enabled attachment from {dom}. Received {t}.",
        "lookalike_domain": f"Email from {dom} to {user} ({dept}); sender domain is similar to {COMPANY.lower().replace(' ', '')}.com. Received {t}.",
        "impossible_travel": f"{user} signed in from two locations 4,800 km apart within 50 minutes. Latest sign-in {t} from {ip}.",
        "mfa_fatigue": f"{user} received {rng.integers(6, 25)} MFA push requests in 10 minutes; " + (f"{rng.integers(1, 3)} approved" if a["mfa_ok"] else f"none approved{'' if rng.integers(1, 3) else ''}") + f". Time {t}.",
        "new_admin_role": f"{user} was assigned the Global Administrator role at {t} by an account in {dept}.",
        "rare_domain_beacon": f"{host} is connecting to {dom} every {rng.choice([30, 60, 300])} seconds. First seen {t}. User {user}.",
        "internal_port_scan": f"{host} ({dept}) probed {rng.integers(40, 900)} internal ports on {rng.integers(3, 60)} hosts starting {t}.",
        "large_upload": f"{host} uploaded {size:,.0f} MB to {ip} ({dom}) starting {t}. User {user}, {dept}.",
        "public_bucket": f"Storage bucket 'kl-{dept}-{rng.integers(100, 999)}' was made public by {user} at {t}. Bucket holds {size:,.0f} MB.",
        "key_new_asn": f"Access key owned by {user} ({a['role']}) used from a network never seen before ({ip}) at {t}.",
        "dlp_personal_cloud": f"{user} ({dept}) uploaded a file labelled Confidential ({size:,.0f} MB) to a personal cloud drive at {t}.",
    }
    base = parts[r]
    extra = intel + related + (geo if r in ("impossible_travel", "key_new_asn", "mfa_fatigue", "new_admin_role") else "")
    if r in ("impossible_travel", "key_new_asn", "new_admin_role"):
        extra += mfa
    extra += tool
    if a["after_hours"] and rng.random() < 0.5:
        extra += " Outside business hours."
    return (base + extra).strip()


def _category(rng, rule: str, malicious: int, known_tool: bool) -> str:
    if not malicious:
        if rule in ("public_bucket", "dlp_personal_cloud", "new_admin_role") and not known_tool and rng.random() < 0.35:
            return "policy_violation"
        return "benign"
    return str(rng.choice(RULES[rule]["cats"]))


def _severity(rng, malicious: int, category: str, crit: int, p: float) -> int:
    if not malicious:
        return 1 if category == "policy_violation" else int(rng.random() < 0.15)
    base = 2 if category in ("credential_abuse", "exfiltration", "malware") else 1
    base += int(crit >= 2 and rng.random() < 0.7)
    return int(min(3, base))


def generate(n: int = 20000, seed: int = 7, days: int = 28, start: str = "2026-09-01T00:00",
             rule_boost: dict | None = None, logit_shift: dict | None = None) -> list[Alert]:
    """Generate `n` alerts over `days` days. Deterministic for a given set of arguments.

    `rule_boost` multiplies how often a rule fires; `logit_shift` adds to a rule's true logit.
    Together they simulate a campaign: more alerts of one kind, and more of them real.
    """
    rng = np.random.default_rng(seed)
    rules = list(RULES)
    rw = np.array([RULES[r]["weight"] * (rule_boost or {}).get(r, 1.0) for r in rules], float)
    rw /= rw.sum()
    out: list[Alert] = []
    start = np.datetime64(start)
    for i in range(n):
        rule = str(rng.choice(rules, p=rw))
        role = str(rng.choice(ROLES, p=[0.72, 0.12, 0.08, 0.08]))
        dept = "it" if role == "admin" and rng.random() < 0.7 else str(rng.choice(DEPARTMENTS))
        crit = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.38, 0.22, 0.10]))
        minute = int(rng.integers(0, 60 * 24 * days))
        ts = start + np.timedelta64(minute, "m")
        hour = (minute // 60) % 24
        after_hours = bool(hour < 7 or hour >= 20)
        # threat-intel reputation: mostly low, with a long tail
        ioc = float(np.clip(rng.beta(1.1, 6.0) if rng.random() < 0.9 else rng.beta(4, 2.2), 0, 1))
        known_tool = bool(rng.random() < (0.45 if role in ("admin", "service") else 0.12))
        prior = int(min(rng.poisson(0.6), 9))
        new_geo = bool(rng.random() < 0.12)
        mfa_ok = bool(rng.random() < 0.7)
        mb_out = float(np.round(rng.lognormal(2.0, 1.4), 1)) if rule in ("large_upload", "public_bucket", "dlp_personal_cloud") else float(np.round(rng.lognormal(-1.0, 1.0), 2))
        z = true_logit(rule, crit, after_hours, ioc, known_tool, prior, new_geo, mfa_ok, mb_out, role)
        z += (logit_shift or {}).get(rule, 0.0)
        p = float(sigmoid(z))
        mal = int(rng.random() < p)
        cat = _category(rng, rule, mal, known_tool)
        sev = _severity(rng, mal, cat, crit, p)
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        user = f"{first}.{last}" if role != "service" else f"svc-{rng.choice(['backup', 'deploy', 'etl', 'scanner', 'mdm'])}"
        host = f"{dept[:3].upper()}-{rng.choice(['LT', 'WS', 'SRV', 'VM'])}-{rng.integers(1, 400):03d}"
        a = dict(rule=rule, host=host, user=user, department=dept, role=role, asset_criticality=crit,
                 after_hours=after_hours, ioc_score=round(ioc, 2), known_tool=known_tool, prior_alerts_24h=prior,
                 new_geo=new_geo, mfa_ok=mfa_ok, mb_out=mb_out, time=str(ts).replace("T", " ")[:16])
        desc = _describe(rng, a)
        aid = "KL-" + hashlib.sha1(f"{seed}-{i}".encode()).hexdigest()[:8].upper()
        out.append(Alert(alert_id=aid, timestamp=a["time"], source=RULES[rule]["source"], rule=rule,
                         title=RULES[rule]["title"], host=host, user=user, department=dept, role=role,
                         asset_criticality=crit, after_hours=after_hours, ioc_score=a["ioc_score"],
                         known_tool=known_tool, prior_alerts_24h=prior, new_geo=new_geo, mfa_ok=mfa_ok,
                         mb_out=mb_out, description=desc, malicious=mal, p_true=p, category=cat, severity=sev))
    return out


def to_frame(alerts: list[Alert]):
    import pandas as pd
    return pd.DataFrame([a.to_row() for a in alerts])


def feature_matrix(df):
    """Numeric features for classic models (logistic regression and friends)."""
    import pandas as pd
    X = pd.DataFrame({
        "ioc_score": df["ioc_score"].astype(float),
        "after_hours": df["after_hours"].astype(float),
        "asset_criticality": df["asset_criticality"].astype(float),
        "known_tool": df["known_tool"].astype(float),
        "prior_alerts_24h": df["prior_alerts_24h"].clip(upper=6).astype(float),
        "new_geo": df["new_geo"].astype(float),
        "mfa_ok": df["mfa_ok"].astype(float),
        "role_admin": (df["role"] == "admin").astype(float),
        "role_service": (df["role"] == "service").astype(float),
        "log_mb_out": np.log1p(df["mb_out"].astype(float)),
    })
    for r in RULES:
        X[f"rule_{r}"] = (df["rule"] == r).astype(float)
    return X


def split(df, seed: int = 0, frac=(0.6, 0.2, 0.2)):
    """Train / calibration / test split, deterministic."""
    idx = np.random.default_rng(seed).permutation(len(df))
    a = int(frac[0] * len(df))
    b = a + int(frac[1] * len(df))
    return df.iloc[idx[:a]].reset_index(drop=True), df.iloc[idx[a:b]].reset_index(drop=True), df.iloc[idx[b:]].reset_index(drop=True)


_CACHE: dict = {}


def load(n: int = 20000, seed: int = 7):
    """The book's standard dataset as a DataFrame (cached)."""
    key = (n, seed)
    if key not in _CACHE:
        _CACHE[key] = to_frame(generate(n, seed))
    return _CACHE[key].copy()


def history_and_live(df=None):
    """Weeks 1-3 are history (where we set thresholds); week 4 is 'live' (where we check them)."""
    df = load() if df is None else df
    ts = df.timestamp.astype("datetime64[ns]")
    cut = np.datetime64("2026-09-22T00:00")
    return df[ts < cut].reset_index(drop=True), df[ts >= cut].reset_index(drop=True)


def campaign_week(seed: int = 11):
    """A fifth week with a phishing campaign: link alerts fire 3x as often and far more of them are real."""
    alerts = generate(n=6200, seed=seed, days=7, start="2026-09-29T00:00",
                      rule_boost={"suspicious_link": 3.0, "lookalike_domain": 2.0},
                      logit_shift={"suspicious_link": 1.6, "lookalike_domain": 1.2})
    return to_frame(alerts)
