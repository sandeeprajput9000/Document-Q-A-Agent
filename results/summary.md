# Experiment summary

## Current smoke run

The downloader fetched 25 PDFs containing 908 pages. The baseline configuration built an index with 1,258 chunks. A warm offline query about the Longformer paper returned a citation to `longformer.pdf`, page 10, in 0.090 seconds using the extractive FakeLLM fallback. This single query is not a benchmark score.

## Benchmark status

No experiment grid has run. `data/benchmark/questions.json` contains no manually verified questions, and Watsonx credentials are not configured. Accuracy, p95 latency, and the LLM judge are therefore unreported. After verifying 50 source-linked questions and configuring Watsonx, run `python scripts/run_experiments.py`; the scripts will write observed results here, to `experiments.csv`, and to the accuracy plots.

