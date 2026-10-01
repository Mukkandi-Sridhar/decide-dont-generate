"""The book's structure, in one place. Used by the build tools.

Chapter files use the book's numbering (chapters/ch01.qmd … ch22.qmd). Figure and result files keep the ids of the
first draft (figures/chNN, results/chNN.json), which the chapters reference directly; see DECISIONS.md D-58.
"""

PARTS = [
    ("I", "Prerequisites: probability and decisions", [
        (1, "ch01", "What “learning” means"),
        (2, "ch02", "Probability, and how a machine learns it"),
        (3, "ch03", "Calibration: when 0.8 really means 80%"),
        (4, "ch04", "From probabilities to actions"),
    ]),
    ("II", "Prerequisites: networks, language models and agents", [
        (5, "ch05", "Deep learning in one chapter"),
        (6, "ch06", "How an LLM writes, and structured outputs"),
        (7, "ch07", "RAG, agents, and where they break"),
    ]),
    ("III", "Jev in depth", [
        (8, "ch08", "System 1 and System 2"),
        (9, "ch09", "Inside Jev: what we know and what we don’t"),
        (10, "ch10", "The type system: choice, score, noul"),
        (11, "ch11", "Testing Jev’s calibration yourself"),
        (12, "ch12", "The Jevons paradox of decisions"),
    ]),
    ("IV", "The decision layer", [
        (13, "ch13", "The bake-off: six ways to make a decision"),
        (14, "ch14", "Act, review, or escalate"),
        (15, "ch15", "A catalogue of decision patterns"),
    ]),
    ("V", "Building", [
        (16, "ch16", "First calls, and the mock that makes them free"),
        (17, "ch17", "A hybrid agent: Jev decides, the LLM reasons"),
        (18, "ch18", "Case study: SOC alert triage"),
        (19, "ch19", "An applications gallery"),
        (20, "ch20", "Build your own System One model"),
        (21, "ch21", "Capstone: a production decision service"),
    ]),
    ("VI", "What changes now", [
        (22, "ch22", "What changes now"),
    ]),
]

CHAPTERS = {c[1]: (c[0], c[2], p[0], p[1]) for p in PARTS for c in p[2]}
