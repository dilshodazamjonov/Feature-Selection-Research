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
- peak RAM and peak GPU memory where available
- saved model bundle for reuse

New runs are isolated under
`results/runs/<dataset>/<YYYY-MM-DD_selector_model>/` and registered in
`results/run_index.csv`. The common single-run and matrix runners initialize
the active layout before writing and use deterministic collision suffixes
instead of reusing an existing run directory. Each `manifest.json` declares
which standard artifacts are applicable and present.

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
- selector outputs use `selected_features_` with a compatibility property for
  the former spelling
- `boruta` and `boruta_rfe` now resolve to Boruta-only and Boruta → RFE,
  respectively, for future runs
- the deterministic Home Credit domain-rule script is no longer named or
  described as an LLM implementation

These source/configuration corrections apply to future execution only. No
historical experiment was rerun and no saved metric or split definition was
changed.

Historical finalized outputs live in a separate immutable bundle. They are
optional to active repository validation and can be validated read-only by
passing its repository root to `validate_repository_state.py` with
`--legacy-repository-root`.

## Finalized two-dataset statistical review

The current canonical Home Credit and LendingClub v2 review is reproducible
from authenticated persisted predictions and result metadata; it does not
require raw research tables or model execution. The binding protocol is
`configs/protocols/prompt_14_two_dataset_analysis_v1/analysis_protocol_lock.json`.
Exact outputs and claim wording are indexed by:

- `results/finalized_research/canonical_artifact_manifest.json`
- `results/final_research_package_v2/final_research_report.md`
- `results/final_research_package_v2/final_results_tables.csv`
- `results/final_research_package_v2/claims_and_evidence.csv`
- `cleanup/audits/prompt_14_two_dataset_oot_review_v3/results_digest.json`

Metric reconciliation uses an absolute tolerance of `1e-10`. AUC inference
uses paired DeLong; registered intervals use 2,000 target-stratified paired
bootstrap draws with seed `20260721`; Holm adjustment retains every member of
all 36 original families. The third-dataset protocol remains frozen and
unexecuted.

Non-goals:
- no calibration changes
- no stacking changes
- no deployment logic
- no new modeling strategy beyond existing evaluation vehicles
