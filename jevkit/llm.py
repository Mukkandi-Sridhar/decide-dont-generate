"""Toy language-model machinery for Part II, all synthetic.

- `BPE`: a tiny byte-pair-encoding tokenizer you can train in a second.
- `TrigramLM`: a next-token generator trained on the synthetic analyst notes. It writes
  one token at a time, exactly like an LLM does, just with a far simpler model.
- `MockLLM`: a deterministic stand-in for an LLM API used for SOC triage (classification,
  structured extraction, judging). Its behaviour is *designed* to show the failure modes the
  book discusses (overconfident stated confidence, occasional invalid JSON, run-to-run variance
  when sampling). It is not a model of any real product. Every number it produces is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from . import engine, soc

# ---------------------------------------------------------------------------
# Byte-pair encoding, the smallest version that still teaches the idea
# ---------------------------------------------------------------------------


class BPE:
    """Start from characters; repeatedly merge the most frequent adjacent pair."""

    def fit(self, texts, merges: int = 200):
        words = Counter()
        for t in texts:
            for w in t.lower().split():
                words[" ".join(list(w)) + " </w>"] += 1
        self.merges = []
        for _ in range(merges):
            pairs = Counter()
            for w, c in words.items():
                s = w.split()
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            self.merges.append((a, b))
            pat = re.compile(r"(?<!\S)" + re.escape(a + " " + b) + r"(?!\S)")
            words = Counter({pat.sub(a + b, w): c for w, c in words.items()})
        return self

    def tokenize(self, text: str) -> list[str]:
        out = []
        for w in text.lower().split():
            s = list(w) + ["</w>"]
            for a, b in self.merges:
                i = 0
                while i < len(s) - 1:
                    if s[i] == a and s[i + 1] == b:
                        s[i:i + 2] = [a + b]
                    else:
                        i += 1
            out += [t.replace("</w>", "") for t in s if t != "</w>"]
        return out


# ---------------------------------------------------------------------------
# A trigram "LLM": the generation loop, without the billions of parameters
# ---------------------------------------------------------------------------


class TrigramLM:
    def fit(self, sentences):
        self.counts = defaultdict(Counter)
        self.vocab = set()
        for s in sentences:
            toks = ["<s>", "<s>"] + list(s) + ["</s>"]
            self.vocab.update(toks)
            for a, b, c in zip(toks, toks[1:], toks[2:]):
                self.counts[(a, b)][c] += 1
        self.vocab = sorted(self.vocab)
        return self

    def next_distribution(self, a: str, b: str, alpha: float = 0.01) -> dict[str, float]:
        ctx = self.counts.get((a, b), Counter())
        tot = sum(ctx.values()) + alpha * len(self.vocab)
        return {w: (ctx[w] + alpha) / tot for w in self.vocab if w != "<s>"}

    def generate(self, prompt: list[str], temperature: float = 1.0, seed: int = 0, max_tokens: int = 15):
        rng = np.random.default_rng(seed)
        toks = ["<s>", "<s>"] + list(prompt)
        steps = []
        for _ in range(max_tokens):
            dist = self.next_distribution(toks[-2], toks[-1])
            words = list(dist)
            logp = np.log(np.array([dist[w] for w in words]))
            if temperature == 0:
                i = int(np.argmax(logp))
            else:
                z = logp / temperature
                pr = np.exp(z - z.max())
                pr /= pr.sum()
                i = int(rng.choice(len(words), p=pr))
            w = words[i]
            steps.append((w, float(dist[w])))
            if w == "</s>":
                break
            toks.append(w)
        return steps


# ---------------------------------------------------------------------------
# Mock LLM for SOC triage
# ---------------------------------------------------------------------------

MOCK_LLM_NAME = "llm-mock-synthetic"

# Illustrative serving assumptions (NOT measurements of any product). Used for latency/cost charts.
LLM_TIME_TO_FIRST_TOKEN_S = 0.45
LLM_TOKENS_PER_SECOND = 60.0
LLM_PRICE_IN_PER_M = 1.00       # $ per million input tokens (illustrative mid-range)
LLM_PRICE_OUT_PER_M = 4.00      # $ per million output tokens (illustrative)

LABEL_SET = ("malicious", "benign")


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()[:8], "big")


class MockLLM:
    """A synthetic chat model specialised (for the book) in reading SOC alerts.

    Design, documented in DECISIONS.md:
    - It *reads* well: its internal belief is the mock-Jev probability from raw text plus extra noise.
    - Its token probability for the label is sharpened (temperature 0.45): overconfident.
    - Its stated ("verbalized") confidence snaps to a few round numbers, mostly 0.9-0.99.
    - With temperature > 0 it samples, so repeated calls can disagree.
    - About 2% of structured-output responses are broken JSON; 1.5% add an invented field.
    """

    def __init__(self, temperature: float = 0.0, seed: int = 0, json_mode: bool = False):
        self.temperature = temperature
        self.seed = seed
        self.json_mode = json_mode
        self.calls = 0

    # -- internals -------------------------------------------------------
    def _belief(self, text: str) -> float:
        p = engine.soc_probability(text)
        z = math.log(max(p, 1e-6) / max(1 - p, 1e-6))
        rng = np.random.default_rng(_h("llm-belief", text))
        z += rng.normal(0, 0.45)                        # a different reader, a different error
        return 1 / (1 + math.exp(-z))

    def label_token_prob(self, text: str) -> float:
        """P('malicious' as the first answer token), as an API's logprobs would show it."""
        p = self._belief(text)
        z = math.log(p / (1 - p)) / 0.45
        return 1 / (1 + math.exp(-z))

    def _sample_label(self, text: str) -> str:
        q = self.label_token_prob(text)
        if self.temperature == 0:
            return "malicious" if q >= 0.5 else "benign"
        self.calls += 1
        rng = np.random.default_rng(_h("llm-sample", text, self.seed, self.calls))
        qt = 1 / (1 + math.exp(-math.log(q / (1 - q)) / self.temperature))
        return "malicious" if rng.random() < qt else "benign"

    def verbalized_confidence(self, text: str, label: str) -> float:
        q = self.label_token_prob(text)
        c = q if label == "malicious" else 1 - q
        rng = np.random.default_rng(_h("llm-verbal", text))
        if c > 0.8 or rng.random() < 0.7:
            return float(rng.choice([0.9, 0.95, 0.95, 0.99]))
        return float(rng.choice([0.7, 0.8, 0.85]))

    # -- public API ------------------------------------------------------
    def classify(self, text: str) -> dict:
        label = self._sample_label(text)
        return {"label": label, "confidence": self.verbalized_confidence(text, label)}

    def structured(self, text: str) -> str:
        """Return the raw string the model 'wrote' when asked for JSON: usually valid."""
        label = self._sample_label(text)
        conf = self.verbalized_confidence(text, label)
        f, _ = engine.soc_features(text)
        obj = {"verdict": label, "confidence": conf, "rule": f["rule"], "threat_intel_score": f["ioc_score"],
               "after_hours": bool(f["after_hours"]), "related_alerts": int(f["prior_alerts_24h"])}
        r = np.random.default_rng(_h("llm-json", text)).random()
        if not self.json_mode and r < 0.02:
            return "Sure! Here is the JSON:\n```json\n" + json.dumps(obj) + "\n```"
        if not self.json_mode and r < 0.035:
            return json.dumps(obj)[:-1]                     # truncated: invalid JSON
        if r > 0.985:
            obj["attacker_country"] = "unknown"             # a field nobody asked for
        return json.dumps(obj)

    def extract(self, text: str) -> dict:
        """Structured fields pulled from raw alert text (for 'LLM extracts, Jev decides')."""
        f, _ = engine.soc_features(text)
        out = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in f.items()}
        rng = np.random.default_rng(_h("llm-extract", text))
        if rng.random() < 0.03:                             # occasional misread number
            out["ioc_score"] = round(float(np.clip(out["ioc_score"] + rng.normal(0, 0.25), 0, 1)), 2)
        return out

    def judge(self, text: str, proposed: str) -> int:
        """LLM-as-judge: rate a proposed verdict 1-10."""
        q = self.label_token_prob(text)
        agree = q if proposed == "malicious" else 1 - q
        rng = np.random.default_rng(_h("llm-judge", text, proposed, self.seed, self.calls if self.temperature else 0))
        base = 2 + 7.5 * agree + rng.normal(0, 1.3 if self.temperature else 0.6)
        return int(np.clip(round(base), 1, 10))

    @staticmethod
    def simulated_latency_s(output_tokens: int) -> float:
        return LLM_TIME_TO_FIRST_TOKEN_S + output_tokens / LLM_TOKENS_PER_SECOND

    @staticmethod
    def simulated_cost_usd(input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * LLM_PRICE_IN_PER_M + output_tokens * LLM_PRICE_OUT_PER_M) / 1e6
