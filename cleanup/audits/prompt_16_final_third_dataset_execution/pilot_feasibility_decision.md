# Prompt 16 authenticated pilot feasibility decision

## Decision

The resource pilot passes with the lock-declared resource-infeasibility rule applied. All 27 selector fits and all 30 evaluation cells are authenticated and visible. Twenty-eight evaluations completed; the two full-feature cells (LR and CatBoost) exceeded the frozen 24 GiB owned-process cap and remain sealed as unavailable rather than being dropped, approximated, or retried under altered settings.

DEV is authorized only after this gate is committed. OOT remains closed until complete authenticated five-fold DEV and the separate DEV-freeze commit.

## Authenticated scope and accounting

- Matrix: 1,526,659 rows, 1,959 predictors, 31 Parquet parts, and zero depth-2 files opened.
- Raw inventory: 19 authenticated included records; all declared depth-2 files remained excluded.
- Pilot fold: 200,661 training rows and 204,567 validation rows with zero case-ID overlap.
- Checkpoints: 27/27 selector seals and 30/30 evaluation seals; 28 complete and 2 resource-infeasible.
- Predictions and metrics: all 28 completed prediction files match the frozen validation identity and all saved metrics recalculate within 1e-12 absolute/relative tolerance.
- Leakage control: every completed evaluation records validation_target_used_for_fit=false; selection encoding is training-only and natural support is never padded.

## Resource feasibility

The authenticated matrix build took 1.08 active hours and peaked at 1.72 GiB. The pilot's reconciled active work took 7.64 hours. The two full-feature attempts produced the overall 25.76 GiB sampled peak and are explicitly unavailable. Cells 003-030 completed in the final session with a 9.25 GiB peak.

With row-count scaling and the required 20% operational allowance, five-fold DEV is forecast at 137.8 active hours and OOT at 55.5 active hours. The largest DEV-fold bound is 46.2 hours versus the frozen 168-hour fold limit; OOT is below its 168-hour limit. Forecast incremental DEV plus OOT output is 0.82 GiB, while 213.5 GiB is currently free and the recalculated launch floor is 80.0 GiB.

These are resource forecasts, not performance conclusions. No pilot metric was used to change a method, model, budget, seed, split, fold, or comparison.

## Required next step

Commit this second Prompt-16 gate, validate the exact fold-scoped DEV command, and run folds 1 through 5 sequentially with checkpoint resume. Preserve full-feature resource-infeasibility in every fold where it recurs. Do not open OOT or the prior two-dataset numeric findings.
