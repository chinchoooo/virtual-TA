---
url: https://tds.s-anand.net/2026-05/labs/week-4/ragas-evaluation-dashboard/
term: 2026-05
origin: main:2026-05/labs/week-4/ragas-evaluation-dashboard.md
---

# Lab — RAGAS Evaluation Dashboard

## Objective
Build an interactive Streamlit dashboard that runs RAGAS evaluation across multiple RAG retrieval strategies side-by-side.

## Requirements
1. Implement three RAG strategies: Naive RAG (dense only), Hybrid RAG (dense + BM25 + RRF), and Contextual RAG (using Anthropic's chunk context injection).
2. Generate a set of test questions and ground-truth answers using an LLM.
3. Evaluate the retrieval strategies using RAGAS metrics (Faithfulness, Context Recall, Factual Correctness, Response Relevancy).
4. Build a Streamlit dashboard showing metrics comparison tables, strategy charts, and per-question breakdowns.

## Deliverables
- Source code.
- Screenshot of the Streamlit dashboard comparing all three strategies.
- Downloadable evaluation results JSON file.

---
