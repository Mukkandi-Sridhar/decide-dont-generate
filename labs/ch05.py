# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Lab 5 · Deep learning in one chapter
#
# *Decide, Don't Generate*, Chapter 5.
#
# The sections below follow the chapter in order.


# %%
import importlib.util, subprocess, sys
if importlib.util.find_spec("jevkit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.2", "autograd",
                    "git+https://github.com/Mukkandi-Sridhar/decide-dont-generate"], check=True)

# %% [markdown]
# ## Neurons to networks

# %%
import numpy as np
from jevkit.learn import sigmoid, log_loss, fit_logistic

rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, (700, 2))
far = np.sqrt(((X - [0.1, 0.05]) ** 2).sum(1))
y = (rng.random(700) < sigmoid(12 * (far - 0.92))).astype(float)   # suspicious when far from normal

w, b, _ = fit_logistic(X, y, lr=1, steps=3000)
print(f"one neuron: log loss {log_loss(sigmoid(X @ w + b), y):.3f}")

# %% [markdown]
# ## A hidden layer, from scratch

# %%
H = 12
W1 = rng.normal(0, 1, (2, H)); b1 = np.zeros(H)
W2 = rng.normal(0, 0.3, H);    b2 = 0.0
lr = 0.3
for step in range(3001):
    h_in = X @ W1 + b1
    h = np.maximum(0, h_in)                      # ReLU hinges
    p = sigmoid(h @ W2 + b2)                     # forward pass
    d_out = (p - y) / len(y)                     # blame at the output
    d_h = np.outer(d_out, W2) * (h_in > 0)       # blame flows back through the hinges
    W2 -= lr * h.T @ d_out; b2 -= lr * d_out.sum()
    W1 -= lr * X.T @ d_h;   b1 -= lr * d_h.sum(0)
    if step % 1000 == 0:
        print(f"step {step:4d}  loss {log_loss(p, y):.3f}")

# %% [markdown]
# **Try:** delete the ReLU (use `h = h_in`). What happens to the loss, and why?

# %% [markdown]
# ## On Kestrel's alerts

# %%
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from jevkit import soc, calibration as cal
from jevkit.learn import Standardizer

alerts = soc.load()
train, calib, test = soc.split(alerts)
F = lambda d: soc.feature_matrix(d).to_numpy()
st = Standardizer().fit(F(train))
models = {
    "logistic regression": LogisticRegression(C=1e4, max_iter=5000),
    "network, 16 units": MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, random_state=0),
    "network, 64+64 units": MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=400, random_state=0),
}
for name, m in models.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(st(F(train)), train.malicious)
    s = cal.summary(m.predict_proba(st(F(test)))[:, 1], test.malicious)
    print(f"{name:>22}: AUC {s['auc']:.3f}  log loss {s['log_loss']:.3f}  ECE {s['ece']:.3f}")

# %% [markdown]
# ## Embeddings: meaning as geometry

# %%
import numpy as np
from jevkit import text

notes = text.notes_corpus()
print(len(notes), "notes, e.g.:")
for n in notes[:4]:
    print("  ", " ".join(n))

# %%
vocab, vectors, counts = text.note_vectors()
def cosine(a, b):
    va, vb = vectors[vocab.index(a)], vectors[vocab.index(b)]
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

for a, b in [("invoice", "bill"), ("invoice", "trojan"), ("upload", "download"), ("phishing", "spoofed")]:
    print(f"{a:>9} vs {b:<9} {cosine(a, b):+.2f}")
print(text.note_nearest("ransomware"))

# %% [markdown]
# ## Alerts as vectors

# %%
from sklearn.neighbors import NearestNeighbors
from jevkit import soc, calibration as cal

alerts = soc.load()
train, calib, test = soc.split(alerts)
embed = text.alert_embedder(train.description.tolist())
Z_train = embed(train.description.tolist())
Z_test = embed(test.description.tolist())

nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Z_train)
dist, idx = nn.kneighbors(Z_test[:1])
print("QUERY:", test.description.iloc[0][:110])
for d, i in zip(dist[0], idx[0]):
    print(f"  sim {1 - d:.2f}  {train.description.iloc[i][:100]}")

# %% [markdown]
# ## Among the 50 most similar past alerts, how many were attacks?

# %%
nn50 = NearestNeighbors(n_neighbors=50, metric="cosine").fit(Z_train)
_, idx = nn50.kneighbors(Z_test)
p_knn = np.clip(train.malicious.to_numpy()[idx].mean(1), 0.005, 0.995)
print(cal.summary(p_knn, test.malicious))

# %% [markdown]
# ## Attention and transformers, visually

# %%
import numpy as np

def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

def self_attention(H, Wq, Wk, Wv):
    Q, K, V = H @ Wq, H @ Wk, H @ Wv            # questions, labels, contents
    weights = softmax(Q @ K.T / np.sqrt(K.shape[1]))
    return weights @ V, weights                  # each word: a weighted mix of the values

rng = np.random.default_rng(0)
H = rng.normal(size=(3, 8))                      # three words, 8 numbers each
out, w = self_attention(H, *(rng.normal(size=(8, 8)) for _ in range(3)))
print(np.round(w, 2))                            # every row sums to 1

# %% [markdown]
# ## The toy task

# %%
from jevkit import attention as at
X, y, notes = at.make_data(4000, 0)
X_test, y_test, _ = at.make_data(1000, 1)
for n, label in list(zip(notes, y))[:5]:
    print(f"{' '.join(n):<40} threat={int(label)}")

# %%
from sklearn.linear_model import LogisticRegression
def bag(M):
    B = np.zeros((len(M), len(at.VOCAB)))
    for i, row in enumerate(M):
        for t in row:
            B[i, t] += 1
    return B[:, 1:]
bow = LogisticRegression(max_iter=2000).fit(bag(X), y)
print(f"bag of words accuracy: {bow.score(bag(X_test), y_test):.1%}")

# %% [markdown]
# ## Train one attention layer (about 20 seconds)

# %%
params = at.train(X, y, steps=4000, lr=0.01, seed=1)
acc = ((at.forward(params, X_test) > 0.5) == y_test).mean()
print(f"attention accuracy: {acc:.1%}")

for words in (["invoice", "not", "phishing"], ["phishing", "not", "invoice"]):
    p, A = at.forward(params, at.encode(words), return_attn=True)
    print(words, "P(threat) =", round(float(p[0]), 3))
    print(np.round(A[0, :3, :3], 2))

# %% [markdown]
# ## Training at scale, and why big models are overconfident

# %%
from collections import Counter, defaultdict
import numpy as np
from jevkit import text

notes = text.notes_corpus(40000)
train_notes, test_notes = notes[:30000], notes[30000:33000]
V = len({w for n in notes for w in n}) + 2

def trigram_loss(n_train):
    counts = defaultdict(Counter)
    for s in train_notes[:n_train]:
        t = ["<s>", "<s>"] + s + ["</s>"]
        for a, b, c in zip(t, t[1:], t[2:]):
            counts[(a, b)][c] += 1
    total, n = 0.0, 0
    for s in test_notes:
        t = ["<s>", "<s>"] + s + ["</s>"]
        for a, b, c in zip(t, t[1:], t[2:]):
            ctx = counts[(a, b)]
            total -= np.log((ctx[c] + 0.05) / (sum(ctx.values()) + 0.05 * V))
            n += 1
    return total / n

for n in (30, 300, 3000, 30000):
    print(f"{n:>6} notes: next-word log loss {trigram_loss(n):.3f}")

# %% [markdown]
# ## Overtraining

# %%
import warnings
from sklearn.neural_network import MLPClassifier
from jevkit import soc, calibration as cal
from jevkit.learn import Standardizer

alerts = soc.load()
train, calib, test = soc.split(alerts)
F = lambda d: soc.feature_matrix(d).to_numpy()
st = Standardizer().fit(F(train))
net = MLPClassifier(hidden_layer_sizes=(128, 128), learning_rate_init=0.002, random_state=0, batch_size=256)
for epoch in range(1, 201):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.partial_fit(st(F(train)), train.malicious, classes=[0, 1])
    if epoch in (3, 20, 60, 200):
        p = net.predict_proba(st(F(test)))[:, 1]
        acc = ((p > 0.5) == test.malicious).mean()
        s = cal.summary(p, test.malicious)
        print(f"epoch {epoch:3d}: accuracy {acc:.3f}  log loss {s['log_loss']:.3f}  ECE {s['ece']:.3f}")

# %% [markdown]
# ## Fixing it after the fact

# %%
p_cal = net.predict_proba(st(F(calib)))[:, 1]
for name, fix in (("temperature", cal.Temperature()), ("Platt", cal.Platt())):
    fix.fit(p_cal, calib.malicious)
    print(f"{name:>11}: ECE {cal.ece(fix(p), test.malicious):.3f}  log loss {cal.log_loss(fix(p), test.malicious):.3f}")
