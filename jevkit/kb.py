"""Kestrel Logistics' (synthetic) security knowledge base, for the RAG chapter.

Each document is a short policy or runbook. Each fact has questions written in different words
from the document, so retrieval has to work for its living. Some questions have no answer in
the knowledge base at all: a good system should notice and say "I don't know".
"""

from __future__ import annotations

import re

DOCS = {
    "approved-tools": ("Approved IT tooling",
        "The IT team may run PowerShell with encoded commands only from the management servers MGT-SRV-001 and MGT-SRV-002. "
        "Remote administration uses the approved RMM agent; any other remote access tool is prohibited. "
        "Scheduled maintenance scripts must reference a change ticket in the ServiceNow queue. "
        "Unsigned binaries may run from temporary folders only during software packaging on build machines."),
    "change-policy": ("Change management policy",
        "Every production change needs a change ticket approved by the change advisory board. "
        "Emergency changes may start without approval but must be ticketed within four hours. "
        "Changes to domain administrator groups always need two approvers from the identity team."),
    "service-accounts": ("Service account inventory",
        "svc-backup copies file-server snapshots to the offsite storage bucket kl-backup-offsite every night between 01:00 and 04:00. "
        "svc-etl moves shipment data to the analytics warehouse hourly. "
        "svc-scanner runs vulnerability scans against internal subnets on Sundays. "
        "Service accounts must never sign in interactively; an interactive sign-in by a service account is always investigated."),
    "mfa-policy": ("Multi-factor authentication policy",
        "All staff use push-based multi-factor authentication through the authenticator app. "
        "Users who receive unexpected push prompts must deny them and report to the SOC. "
        "Number matching is enabled for all push prompts since March 2026. "
        "Accounts with more than ten denied pushes in an hour are locked automatically."),
    "travel-policy": ("Travel and remote access policy",
        "Staff travelling abroad must register the trip in the travel portal before leaving. "
        "Sign-ins from countries on the restricted list are blocked regardless of registration. "
        "Contractors may not access production systems from outside the country of their contract."),
    "phishing-runbook": ("Phishing response runbook",
        "When a user reports a phishing email, the analyst pulls all copies from mailboxes with the purge tool. "
        "If a user clicked the link, reset their password and revoke active sessions immediately. "
        "Lookalike domains are submitted to the takedown vendor within one business day. "
        "Invoice-themed phishing is escalated to finance leadership because of wire fraud risk."),
    "malware-runbook": ("Malware response runbook",
        "Isolate the endpoint with the EDR console before collecting any evidence. "
        "Do not reboot an infected machine; memory evidence is lost on reboot. "
        "Ransomware indicators on any server trigger the major incident process and a page to the on-call lead."),
    "data-handling": ("Data classification and handling",
        "Files labelled Confidential may not be stored on personal cloud drives. "
        "Shipment manifests are Internal; customer contracts and payroll exports are Confidential. "
        "Storage buckets holding Confidential data must never be public; making one public requires a security exception."),
    "escalation": ("SOC escalation matrix",
        "Page the on-call responder for any confirmed credential compromise of an administrator. "
        "Escalate to the CISO within one hour for suspected data exfiltration over 1 GB. "
        "Analysts may close alerts on their own when they match an approved tool and a valid change ticket."),
    "asset-criticality": ("Asset criticality register",
        "Domain controllers, the payroll database and the shipment tracking platform are rated critical. "
        "Finance laptops are rated high because they approve payments. "
        "Warehouse scanners are rated low and are isolated on their own network segment."),
    "logging": ("Logging and retention standard",
        "Security logs are kept for 400 days in the SIEM. "
        "EDR telemetry is kept for 30 days on the vendor platform. "
        "Email gateway logs are kept for 90 days."),
    "vendors": ("Third-party access standard",
        "Vendors connect through the privileged access gateway, never through direct VPN accounts. "
        "Vendor access is granted for a maximum of 30 days and must be renewed with a new request. "
        "The freight partner Kestrel Freight has read-only access to shipment status through the partner API."),
}

# (doc, question). Questions deliberately avoid the documents' exact wording where possible.
QUESTIONS = [
    ("approved-tools", "Which machines are allowed to run encoded PowerShell?"),
    ("approved-tools", "Can an admin use TeamViewer for remote support?"),
    ("approved-tools", "Is an unsigned program in a temp directory ever fine?"),
    ("change-policy", "How long do we have to file a ticket for an urgent change?"),
    ("change-policy", "Who has to approve adding someone to domain admins?"),
    ("service-accounts", "Is svc-backup supposed to upload data at night?"),
    ("service-accounts", "What does the ETL service account do?"),
    ("service-accounts", "A service account logged in interactively. Is that normal?"),
    ("service-accounts", "When does the vulnerability scanner run?"),
    ("mfa-policy", "What should someone do if they get MFA prompts they didn't expect?"),
    ("mfa-policy", "Do we use number matching on push notifications?"),
    ("mfa-policy", "When does an account get locked for push spam?"),
    ("travel-policy", "Can a contractor log in to production while abroad?"),
    ("travel-policy", "Does registering a trip allow sign-ins from restricted countries?"),
    ("phishing-runbook", "A user clicked a phishing link. What are the first steps?"),
    ("phishing-runbook", "What happens to lookalike domains we find?"),
    ("phishing-runbook", "Why are fake invoice emails escalated to finance?"),
    ("malware-runbook", "Should we restart a laptop that has malware on it?"),
    ("malware-runbook", "What's the first action for an infected endpoint?"),
    ("malware-runbook", "Who gets paged when ransomware shows up on a server?"),
    ("data-handling", "Can confidential files go to someone's personal Dropbox?"),
    ("data-handling", "Is a payroll export confidential or internal?"),
    ("data-handling", "Is it ok to make a bucket with confidential data public?"),
    ("escalation", "When do we page on-call for a compromised account?"),
    ("escalation", "How quickly must the CISO hear about a large data theft?"),
    ("escalation", "Can an analyst close an alert without a second opinion?"),
    ("asset-criticality", "How critical are finance laptops?"),
    ("asset-criticality", "Which systems count as critical assets?"),
    ("logging", "How long do we keep SIEM logs?"),
    ("logging", "For how many days is EDR data retained?"),
    ("vendors", "How do suppliers get access to our network?"),
    ("vendors", "How long can a vendor keep their access?"),
]

UNANSWERABLE = [
    "What is the office wifi password?",
    "Who won the company football tournament?",
    "What is our cyber insurance deductible?",
    "Which firewall brand do we use at the Rotterdam depot?",
    "How many parking spaces does headquarters have?",
    "What's the policy on USB drives in the warehouse?",
    "When is the next penetration test scheduled?",
    "Which antivirus did we use before 2024?",
    "What is the SLA for the helpdesk printer queue?",
    "Who is the data protection officer?",
    "What's the budget for security awareness training?",
    "Do we allow personal phones on the warehouse floor?",
]


ANSWER_KEY = {'Which machines are allowed to run encoded PowerShell?': 'MGT-SRV-001', 'Can an admin use TeamViewer for remote support?': 'approved RMM agent', 'Is an unsigned program in a temp directory ever fine?': 'software packaging', 'How long do we have to file a ticket for an urgent change?': 'four hours', 'Who has to approve adding someone to domain admins?': 'two approvers', 'Is svc-backup supposed to upload data at night?': 'kl-backup-offsite', 'What does the ETL service account do?': 'analytics warehouse', 'A service account logged in interactively. Is that normal?': 'never sign in interactively', 'When does the vulnerability scanner run?': 'on Sundays', "What should someone do if they get MFA prompts they didn't expect?": 'deny them', 'Do we use number matching on push notifications?': 'Number matching', 'When does an account get locked for push spam?': 'locked automatically', 'Can a contractor log in to production while abroad?': 'Contractors may not', 'Does registering a trip allow sign-ins from restricted countries?': 'restricted list', 'A user clicked a phishing link. What are the first steps?': 'revoke active sessions', 'What happens to lookalike domains we find?': 'takedown vendor', 'Why are fake invoice emails escalated to finance?': 'wire fraud', 'Should we restart a laptop that has malware on it?': 'Do not reboot', "What's the first action for an infected endpoint?": 'Isolate the endpoint', 'Who gets paged when ransomware shows up on a server?': 'major incident', "Can confidential files go to someone's personal Dropbox?": 'personal cloud drives', 'Is a payroll export confidential or internal?': 'payroll exports are Confidential', 'Is it ok to make a bucket with confidential data public?': 'security exception', 'When do we page on-call for a compromised account?': 'credential compromise', 'How quickly must the CISO hear about a large data theft?': 'within one hour', 'Can an analyst close an alert without a second opinion?': 'close alerts on their own', 'How critical are finance laptops?': 'approve payments', 'Which systems count as critical assets?': 'Domain controllers', 'How long do we keep SIEM logs?': '400 days', 'For how many days is EDR data retained?': '30 days on the vendor', 'How do suppliers get access to our network?': 'privileged access gateway', 'How long can a vendor keep their access?': 'maximum of 30 days'}


def distractors(n: int = 300, seed: int = 0):
    """Old alert descriptions: same vocabulary as the questions, but no answers in them."""
    from . import soc
    df = soc.load()
    return [("alert-archive", t) for t in df.description.sample(n, random_state=seed)]


def corpus(size_words: int = 30, with_distractors: bool = True):
    ch = chunks(size_words, overlap=max(0, size_words // 4)) if size_words else sentences()
    return ch + (distractors() if with_distractors else [])


def chunks(size_words: int = 40, overlap: int = 10):
    """Split every document into overlapping word windows. Returns [(doc_id, text)]."""
    out = []
    for doc, (title, body) in DOCS.items():
        words = (title + ". " + body).split()
        step = max(1, size_words - overlap)
        for i in range(0, max(1, len(words) - overlap), step):
            out.append((doc, " ".join(words[i:i + size_words])))
            if i + size_words >= len(words):
                break
    return out


def sentences():
    out = []
    for doc, (title, body) in DOCS.items():
        for s in re.split(r"(?<=\.)\s+", body):
            out.append((doc, f"{title}: {s}"))
    return out


class Retriever:
    """TF-IDF retrieval over word and/or character n-grams. mode: 'word', 'char' or 'hybrid'."""

    def __init__(self, texts, mode: str = "hybrid"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.texts = list(texts)
        self.mode = mode
        self.vw = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(self.texts)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(self.texts)
        self.Dw, self.Dc = self.vw.transform(self.texts), self.vc.transform(self.texts)

    def scores(self, queries):
        sw = (self.vw.transform(queries) @ self.Dw.T).toarray()
        sc = (self.vc.transform(queries) @ self.Dc.T).toarray()
        return {"word": sw, "char": sc, "hybrid": (sw + sc) / 2}[self.mode]

    def search(self, query: str, k: int = 3):
        s = self.scores([query])[0]
        top = s.argsort()[::-1][:k]
        return [(float(s[i]), self.texts[i]) for i in top]
