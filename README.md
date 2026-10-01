# Decide, Don't Generate: code and labs

This is the companion repository for the book ***Decide, Don't Generate: Jev, System One Models, and the Decision
Layer of Agentic AI*** by Sridhar Mukkandi.

Most of what an AI agent does isn't writing. It's deciding: is this alert real, which team gets this ticket, is
this action safe? The book is about building that decision layer properly. That means probabilities you can check,
thresholds set by what mistakes cost, and agents that know when to hand a case to a person. This repository has
everything you need to run it yourself:

- **22 labs**, one for each chapter, as Jupyter notebooks
- **`jevkit`**, the book's toolkit: a free local mock of the Jev API, the synthetic security team's alerts, and the
  calibration, policy and agent code the chapters build
- **four browser tools** for playing with the book's numbers, live at
  **[mukkandi-sridhar.github.io/decide-dont-generate](https://mukkandi-sridhar.github.io/decide-dont-generate/)**
- **the code behind every figure** in the book

No API key, no GPU and no cloud account are needed. Everything runs on a laptop.

## Start in two minutes

```bash
git clone https://github.com/Mukkandi-Sridhar/decide-dont-generate
cd decide-dont-generate
pip install -e .
pip install jupyter
jupyter notebook labs/ch01.ipynb
```

Or install just the toolkit, which is what the first cell of every lab does:

```bash
pip install git+https://github.com/Mukkandi-Sridhar/decide-dont-generate
```

Python 3.10 or newer. A lab can also run in Google Colab: upload the notebook and run its first cell, which installs
the toolkit.

## The labs

| Part | Chapter | Lab |
|---|---|---|
| I. Probability and decisions | 1. What "learning" means | [`labs/ch01.ipynb`](labs/ch01.ipynb) |
| | 2. Probability, and how a machine learns it | [`labs/ch02.ipynb`](labs/ch02.ipynb) |
| | 3. Calibration: when 0.8 really means 80% | [`labs/ch03.ipynb`](labs/ch03.ipynb) |
| | 4. From probabilities to actions | [`labs/ch04.ipynb`](labs/ch04.ipynb) |
| II. Networks, language models and agents | 5. Deep learning in one chapter | [`labs/ch05.ipynb`](labs/ch05.ipynb) |
| | 6. How an LLM writes, and structured outputs | [`labs/ch06.ipynb`](labs/ch06.ipynb) |
| | 7. RAG, agents, and where they break | [`labs/ch07.ipynb`](labs/ch07.ipynb) |
| III. Jev in depth | 8. System 1 and System 2 | [`labs/ch08.ipynb`](labs/ch08.ipynb) |
| | 9. Inside Jev: what we know and what we don't | [`labs/ch09.ipynb`](labs/ch09.ipynb) |
| | 10. The type system: choice, score, noul | [`labs/ch10.ipynb`](labs/ch10.ipynb) |
| | 11. Testing Jev's calibration yourself | [`labs/ch11.ipynb`](labs/ch11.ipynb) |
| | 12. The Jevons paradox of decisions | [`labs/ch12.ipynb`](labs/ch12.ipynb) |
| IV. The decision layer | 13. The bake-off: six ways to make a decision | [`labs/ch13.ipynb`](labs/ch13.ipynb) |
| | 14. Act, review, or escalate | [`labs/ch14.ipynb`](labs/ch14.ipynb) |
| | 15. A catalogue of decision patterns | [`labs/ch15.ipynb`](labs/ch15.ipynb) |
| V. Building | 16. First calls, and the mock that makes them free | [`labs/ch16.ipynb`](labs/ch16.ipynb) |
| | 17. A hybrid agent: Jev decides, the LLM reasons | [`labs/ch17.ipynb`](labs/ch17.ipynb) |
| | 18. Case study: SOC alert triage | [`labs/ch18.ipynb`](labs/ch18.ipynb) |
| | 19. An applications gallery | [`labs/ch19.ipynb`](labs/ch19.ipynb) |
| | 20. Build your own System One model | [`labs/ch20.ipynb`](labs/ch20.ipynb) |
| | 21. Capstone: a production decision service | [`labs/ch21.ipynb`](labs/ch21.ipynb) |
| VI. What changes now | 22. What changes now | [`labs/ch22.ipynb`](labs/ch22.ipynb) |

Each notebook has a matching `.py` file (the same lab in [jupytext](https://jupytext.readthedocs.io) format), which
is easier to read in a diff.

## The mock, and the real API

Every Jev answer in the book comes from `jev-mock-synthetic`, a mock that runs on your machine. You call it through
TypeSafe's official Python library, `typesafe-sdk`, exactly as you would call the real service:

```python
from typesafe_sdk import Noul, TypeSafeClient
from jevkit import MockJevTransport

client = TypeSafeClient(api_key="mock", transport=MockJevTransport())
r = client.system_one(state="chen.fischer received 21 MFA push requests in 10 minutes; 1 approved.",
                      questions={"attack": Noul(instructions="Is this a real attack?")})
print(r.nouls["attack"].noul)
```

To use the real API instead, remove the `transport` argument and pass your key, or set `JEVKIT_LIVE=1` and
`TYPESAFE_API_KEY` and every lab will use it. The mock copies the API's interface, not its model: its answers are
synthetic. [`docs/mock-design.md`](docs/mock-design.md) explains exactly how it's built, and
[`docs/jev-facts.md`](docs/jev-facts.md) lists every claim the book makes about Jev and where it came from.

## The browser tools

Four small pages let you change the book's numbers and watch what happens:

| Tool | Chapters |
|---|---|
| [Threshold simulator](https://mukkandi-sridhar.github.io/decide-dont-generate/threshold.html) | 14 |
| [Calibration tool](https://mukkandi-sridhar.github.io/decide-dont-generate/calibration.html) | 3 and 11 |
| [Bake-off explorer](https://mukkandi-sridhar.github.io/decide-dont-generate/bakeoff.html) | 13 |
| [Decision cost calculator](https://mukkandi-sridhar.github.io/decide-dont-generate/cost.html) | 9, 12 and 16 |

They run entirely in your browser. The source is in [`site/`](site/); open `site/index.html` locally if you prefer.

## What's where

```
jevkit/         the toolkit: mock API, synthetic alerts, calibration, policies, agents, the decision service
labs/           one lab per chapter (.ipynb and .py)
site/           the four browser tools
figures/src/    the code that draws every figure in the book (python tools/build_figures.py all)
results/        every number the book quotes, as the figure code computes it
docs/           how the mock works, and the facts ledger for Jev
tests/          unit tests, and a weekly check against the real API when a key is configured
assets/fonts/   the fonts the figures use (SIL Open Font License)
```

## Checks

Every push runs the unit tests, executes all 22 labs against the mock and rebuilds every figure
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). A separate weekly job calls the real API, if a key is
configured, to check that it still has the shape the book teaches.

```bash
pip install -e ".[dev,figs]"
pytest -q                          # unit tests
python tools/run_labs.py           # run every lab in mock mode
python tools/build_figures.py all  # redraw every figure into figures/<chapter>/
```

## Please note

- **Everything here is synthetic.** Kestrel Logistics, the security team in the book, doesn't exist, and neither do
  its alerts. Numbers from the mock are examples of a method, not measurements of real Jev. Test on your own data.
- **This is an independent project,** not affiliated with or endorsed by TypeSafe AI. Product names belong to their
  owners.
- **The code is MIT-licensed** (see [`LICENSE`](LICENSE)). The book's text and illustrations are not part of this
  repository and remain all rights reserved.

## The book

*Decide, Don't Generate* by Sridhar Mukkandi, 2026, in paperback and on Kindle. The link will be added here when
the book is live on Amazon.

Found a bug in a lab, or something in the book that doesn't match the code? Please
[open an issue](https://github.com/Mukkandi-Sridhar/decide-dont-generate/issues).
