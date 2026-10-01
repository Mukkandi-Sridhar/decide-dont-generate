# Jev facts ledger

Every statement the book makes about Jev or TypeSafe, where it came from, and how far it has been checked. The
book sorts each claim into three piles: **verified** (read in something anyone can inspect, mainly the public
`typesafe-sdk` library), **vendor-reported** (TypeSafe said it, or an article quoting TypeSafe did) and **unknown**.
When the book was written, web access was limited to search-result summaries, so the claims below that aren't marked
verified have not been checked against a primary source. If you can confirm or correct one, please open an issue.

| Claim | Status | Source seen (summary only) |
|---|---|---|
| TypeSafe AI, San Francisco, founded 2024 | unverified | Wikipedia summary |
| Founders: Diogo Almeida (ex-OpenAI, instruction-following research), Erik Gafni, Sasha Sheng | unverified | search summaries |
| Released in limited early access, 15 Sept 2026; $40M seed led by DCVC | unverified | Wikipedia, TechCrunch summaries |
| Self-serve sign-ups opened 20 Sept with $5 credit; paused 22 Sept under demand | unverified | runtimewire / aggregator summary |
| "System One model": typed decisions + calibrated probabilities, no text generation | vendor-described | TypeSafe blog summary |
| Named after William Stanley Jevons; Almeida: cheaper intelligence -> wider deployment | reported | Wikipedia summary |
| "Parallel sampler": all outputs in a single query, not autoregressive | vendor-reported | TypeSafe blog summary |
| Training method "Reinforcement Learning for Calibrated Decisions" (RLCD) | vendor-reported | TypeSafe blog summary |
| "Two orders of magnitude faster"; "40-200x faster" than frontier LLMs; up to "400x lower cost" on classification | vendor-reported | blog / press summaries |
| 70-500 ms end-to-end latency | vendor-reported | press summary |
| $0.042 per million input tokens, output free (OpenRouter listing "Jev 1.13") | vendor-reported | OpenRouter summary |
| Doom demo: plays from structured text game state ~10x/s, ~$7/hour | vendor-reported | The Register summary |
| API: POST /v1/systemone {state, model, questions}; types choice / score / noul; default model `jev-latest` | **verified** | typesafe-sdk 0.7.2 source on PyPI (read in full) |
| Answer shapes (noul has only `noul`; choice has `confidence`, `probabilities`; score has `score`, `confidence`, `legend`, `probabilities`) | **verified** | SDK generated OpenAPI models |
| Output tokens "currently free of charge" | **verified (docstring)** | SDK schema description |
| Critique: calibration depends on the user's data distribution; treat outputs as scores and recalibrate (Alex Molas, 23 Sept 2026) | reported | blog summary |
| arXiv papers: Jev-Mem (2609.23986), JEV-as-a-Judge (2609.26550), radiology factuality (2609.27607), crash narratives (2609.24052), pentest harnesses (2609.28940), edge orchestration (2609.22753), "Jev in the Wild" (2609.30216), "JEV vs LLMs as Rubric Judges" (2609.29769), "Just Ask Jev" (2609.29429) | titles only | arXiv search summaries |
