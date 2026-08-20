# Agent Evaluation Results

> **WARNING:** every LLM-mode case fell back to the offline
> deterministic path, so the LLM column below is a copy of the offline
> column, NOT a measurement of the LLM. Do not quote it as an LLM result.
> Fix the API key / model id and re-run `python eval/run_agent_eval.py`.

| Metric | Offline | LLM (NOT RUN - all fell back) |
|---|---|---|
| Root-cause accuracy | 24/25 (96.0%) | 24/25 (96.0%) |
| Decision accuracy | 24/25 (96.0%) | 24/25 (96.0%) |
| Hallucination rate | 0/25 (0.0%) | 0/25 (0.0%) |
| Schema validity | 25/25 (100.0%) | 25/25 (100.0%) |
