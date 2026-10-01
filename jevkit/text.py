"""Small text tools for Part II: cleaning alert text, word co-occurrence embeddings, alert embeddings."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

STOP = set("a an the of to in on at by for from and or is was were be been has have had with as it its this that "
           "user time first seen last related alerts alert no one approved we our".split())


def clean(text: str) -> list[str]:
    t = text.lower()
    t = re.sub(r"\b[a-z]+\.[a-z]+\b(?!\.exe)", " ", t)        # usernames like asha.rao
    t = re.sub(r"\b(svc-[a-z]+)\b", " serviceaccount ", t)
    t = re.sub(r"[a-z0-9.-]+\.(com|net|top|co|xyz|info|click|uk|us)\b", " domain ", t)
    t = re.sub(r"\b[a-z]{3}-(lt|ws|srv|vm)-\d+\b", " host ", t)
    t = re.sub(r"kl-[a-z]+-\d+", " bucketname ", t)
    t = t.replace(".exe", "exe").replace("%temp%", "tempfolder")
    words = re.findall(r"[a-z]+", t)
    return [w for w in words if w not in STOP and len(w) > 1]


@lru_cache(None)
def word_vectors(dim: int = 50, min_count: int = 40):
    """PPMI co-occurrence (whole alert as the window), compressed with SVD. Returns (vocab, vectors)."""
    from . import soc
    docs = [clean(d) for d in soc.load().description]
    from collections import Counter
    cnt = Counter(w for d in docs for w in set(d))
    vocab = sorted(w for w, c in cnt.items() if c >= min_count)
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(vocab), len(vocab)))
    for d in docs:
        ids = sorted({idx[w] for w in d if w in idx})
        for a in ids:
            M[a, ids] += 1
    np.fill_diagonal(M, 0)
    total = M.sum()
    row = M.sum(1, keepdims=True)
    col = M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row * col))
    ppmi = np.nan_to_num(np.maximum(pmi, 0), posinf=0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    vec = U[:, :dim] * np.sqrt(S[:dim])
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
    return vocab, vec, M


def nearest(word: str, k: int = 5):
    vocab, vec, _ = word_vectors()
    i = vocab.index(word)
    sims = vec @ vec[i]
    order = np.argsort(-sims)
    return [(vocab[j], float(sims[j])) for j in order[1:k + 1]]


def alert_embedder(train_texts, dim: int = 64, seed: int = 0):
    """TF-IDF over cleaned words, compressed to `dim` numbers (latent semantic analysis). Returns a function."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(analyzer=clean, min_df=5, sublinear_tf=True)
    X = tf.fit_transform(train_texts)
    svd = TruncatedSVD(dim, random_state=seed).fit(X)

    def embed(texts):
        Z = svd.transform(tf.transform(texts))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return embed


# ---------------------------------------------------------------------------
# A synthetic corpus of short analyst notes, for learning word embeddings.
# Words in the same class are interchangeable in every template, but the
# embedding method is never told the classes: it has to discover them.
# ---------------------------------------------------------------------------
WORD_CLASSES = {
    "lure": ["phishing", "spoofed", "lookalike", "fake", "impersonation"],
    "malware": ["malware", "trojan", "ransomware", "backdoor", "dropper"],
    "secret": ["password", "credentials", "token", "passcode", "login"],
    "money": ["invoice", "payment", "refund", "bill", "transfer"],
    "exfil": ["upload", "exfiltration", "download", "copy", "leak"],
    "office": ["printer", "meeting", "lunch", "holiday", "coffee"],
    "action": ["blocked", "quarantined", "isolated", "contained", "disabled"],
    "role": ["analyst", "manager", "engineer", "contractor", "admin"],
    "device": ["laptop", "server", "workstation", "phone", "tablet"],
}

TEMPLATES = [
    "the {lure} email asked the {role} for their {secret}",
    "a {lure} message about an overdue {money}",
    "user reported a {lure} {money} request",
    "the {malware} was {action} on the {device}",
    "{malware} tried to {exfil} files from the {device}",
    "large {exfil} from a {device} to an unknown server",
    "the {role} reset the {secret} after the {lure} attempt",
    "{role} asked about the {office} schedule",
    "reminder about the {office} on friday",
    "the {office} invite came from the {role}",
    "the {role} {action} the {device} after the {malware} alert",
    "suspicious {exfil} of {money} records by a {role}",
    "the {office} was moved by the {role}",
    "stolen {secret} used from a new {device}",
    "the {malware} spread from one {device} to another",
    "{money} approved by the {role} after a {lure} call",
    "please bring your {device} to the {office}",
    "{secret} expired so the {role} could not sign in",
    "the {device} was {action} and the {secret} rotated",
    "no {exfil} seen after the {malware} was {action}",
    "download the {office} slides before friday",
    "the admin {action} the {device} for patching",
    "copy of the {office} notes sent to the {role}",
    "the {role} missed the {office} because of the {malware} cleanup",
    "{money} for the team {office} was approved",
]


def notes_corpus(n: int = 30000, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
        def fill(m):
            cls = m.group(1)
            if rng.random() < 0.08:                        # people don't write to templates: some noise
                cls = str(rng.choice(list(WORD_CLASSES)))
            return str(rng.choice(WORD_CLASSES[cls]))
        sent = re.sub(r"\{(\w+)\}", fill, t)
        out.append(sent.split())
    return out


@lru_cache(None)
def note_vectors(dim: int = 20, window: int = 2):
    """Word vectors from the notes corpus: PPMI over a +/-2 word window, compressed with SVD."""
    docs = notes_corpus()
    from collections import Counter
    cnt = Counter(w for d in docs for w in d)
    vocab = sorted(cnt)
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(vocab), len(vocab)))
    for d in docs:
        ids = [idx[w] for w in d]
        for i, a in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    M[a, ids[j]] += 1
    total = M.sum()
    row = M.sum(1, keepdims=True)
    col = M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (row * col))
    ppmi = np.nan_to_num(np.maximum(pmi, 0), posinf=0, neginf=0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    vec = U[:, :dim] * np.sqrt(S[:dim])
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
    return vocab, vec, M


def note_nearest(word: str, k: int = 4):
    vocab, vec, _ = note_vectors()
    i = vocab.index(word)
    sims = vec @ vec[i]
    order = np.argsort(-sims)
    return [(vocab[j], round(float(sims[j]), 2)) for j in order[1:k + 1]]
