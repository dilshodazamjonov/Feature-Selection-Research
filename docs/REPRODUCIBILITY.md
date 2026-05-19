# Reproducibility

This repository is a research framework. Reproducibility is handled through experiment manifests, config hashing, saved artifacts, and explicit leakage controls.

## What Is Tracked

Each completed run records:
- exact effective config hash
- data fingerprint based on visible source files
- run manifest with status and timestamps
- git commit hash when available
- fold-level CV outputs
- final OOT outputs
- selected feature tables
- feature stability summaries
- runtime summaries
- saved model bundle for reuse

## LLM Reproducibility

Reusable LLM cache lives under:
- `artifacts/llm_cache/`

Per-run LLM artifacts live under each run directory:
- `llm_responses/`
- `feature_rankings/`
- `selected_feature_sets/`

The run artifacts preserve:
- prompt text
- prompt hash
- metadata signature
- selected feature payload
- raw response payload
- final selected feature set

## Behavior Preservation

The refactor was intended to preserve experiment behavior rather than redesign the methodology.

Intentional changes during refactor:
- Home Credit raw files now live under `data/homecredit/raw`
- description metadata now lives under `data/<dataset>/metadata/`
- reusable LLM cache default moved to `artifacts/llm_cache`
- run directories now duplicate LLM artifacts into research-oriented folders

Non-goals:
- no calibration changes
- no stacking changes
- no deployment logic
- no new modeling strategy beyond existing evaluation vehicles
