# Final Report Notebook Protocol

## Purpose

Final dataset notebooks are narrative research reports, not coding workspaces. Their job is to present the dataset-specific review clearly enough for professor or research supervision, using already-generated experiment artifacts rather than exploratory scratch code.

## Required Design Rule

- Notebook logic must stay thin.
- Functions live in `src/credit_risk_fs/reporting/notebook_report.py`.
- Notebooks must not define helper functions.
- Notebooks must not manually read many CSV files in scattered cells.
- Notebooks should only import reporting helpers, load prepared tables and plots, display them, and add concise interpretation.

## Reproducibility Contract

- Final dataset reports should be reproducible from aggregate CSV artifacts under `results/<dataset>/`.
- The primary inputs are:
  - `final_comparison_table.csv`
  - `feature_stability_table.csv`
  - `feature_drift_table.csv`
  - `semantic_coverage_table.csv`
  - `paired_fold_comparisons.csv`
  - `llm_call_summary.csv`
  - `matrix_runs.csv`
  - `failed_runs.csv`
- If an aggregate table is missing, the reporting wrapper may fall back to per-run artifacts when available, but it must not invent values.
- If a section is incomplete, the notebook must surface that clearly.

## Mandatory Report Content

Each final dataset notebook should keep the same high-level structure:

1. Setup and imports
2. Research question and dataset role
3. Dataset snapshot
4. DEV/OOT split rationale
5. Experiment matrix overview
6. Topline performance comparison
7. Stability review
8. Drift and robustness review
9. Semantic coverage review
10. Efficiency tradeoff
11. Best runs deep dive
12. Failure cases and surprises
13. Conclusions for the dataset
14. Future CLIP-style validation placeholder
15. Next actions

## DEV/OOT Split Rule

The DEV/OOT split rationale is mandatory in every final notebook.

The report must state:

- The split is time-based, not random.
- DEV is the older window used for CV, feature selection, and model training.
- OOT is the newer holdout used only for final evaluation.
- Window choice should be justified by both observation counts and target-rate behavior over time.
- OOT metrics and OOT bad-rate behavior must not be used to tune selectors or hyperparameters.

## Reporting Tone

- Be concrete and analytical.
- Avoid scratchpad code and debugging cells.
- Avoid repeating the same conclusion in every section.
- Do not over-claim marginal gains.
- Do not say LLM replaces statistical selectors unless the evidence clearly supports that claim.
- Prefer disciplined wording such as: `LLM screening is useful as a first-stage helper`.

## Future CLIP-Style Placeholder

The final notebooks should reserve a future-work section for CLIP-style semantic-statistical validation, but they should not implement that method yet.

That placeholder should explain:

- This is not image CLIP.
- The intended future method is a dual-encoder or contrastive alignment between feature text or metadata and empirical feature behavior.
- The method must be evaluated under the same DEV/OOT protocol.
- OOT metrics must not be used for CLIP-style training or feature selection.
- The comparison target is an alternative first-stage screener, not a replacement for the whole pipeline.
