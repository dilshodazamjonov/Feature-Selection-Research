# Full baseline v1

Status: **frozen and implementation-tested; real execution not started**

## Scope and scientific boundary

Prompt 10 authenticated the Prompt 9 implementation commits and all six real
DEV-fold-1 pilot artifacts. The pilot audit/review path read no OOT dataset,
prediction, or metric.
The full-baseline configuration is frozen before the manual execution command can
reach a dataset worker. During a future full run, OOT is used only for the locked
final evaluation defined by the existing experiment pipeline; it is never an input
to selector fitting, feature-budget decisions, configuration changes, or retries.
Configuration adaptation after an OOT result is forbidden.

The frozen voting protocol is not part of this matrix and remains unchanged.

## Prompt 9 evidence review

All six artifacts authenticate against Prompt 9 configuration SHA-256
`a694d26d43d563d20ca1c8657770c298581a930792bf7d2fa2aee49806507f8a`.
Every cell completed with `oot_accessed=false`, `oot_evaluated=false`, no controlled
stop, four estimator threads, and the laptop-safe resource policy.

| Dataset | Selector | Runtime | Peak process RAM | Minimum available RAM | Result |
|---|---:|---:|---:|---:|---|
| Home Credit | CatBoost-SHAP | 170.2 s | 2.10 GiB | 20.74 GiB | 40/40 |
| LendingClub v2 | CatBoost-SHAP | 135.6 s | 6.75 GiB | 15.86 GiB | 40/40 |
| Home Credit | Boruta RF | 225.0 s | 2.10 GiB | 20.38 GiB | 26 confirmed; short of 40 |
| LendingClub v2 | Boruta RF | 508.1 s | 6.75 GiB | 15.76 GiB | 161 confirmed; top 40 used |
| Home Credit | CatBoost RFE | 297.1 s | 2.10 GiB | 20.43 GiB | 40/40; 13 fits |
| LendingClub v2 | CatBoost RFE | 331.9 s | 6.75 GiB | 15.71 GiB | 40/40; 14 fits |

These are configuration/capacity findings on the earliest DEV fold, not predictive
performance findings and not full-run runtime measurements.

## Home Credit Boruta decision

The 26-versus-40 result is retained as a valid
`infeasible_natural_support` outcome. The final mode is `confirmed_top_k` with a
confirmed-only natural support. The 25 tentative features are reported but are not
promoted; the 478 rejected features are never padding candidates.

Consequences:

- Home Credit + LR requests 20 and may receive up to 20 confirmed features.
- Home Credit + CatBoost requests 40 but may receive fewer; fold 1 produced 26.
- LendingClub fold 1 had 161 confirmed features, so both 20- and 40-feature budgets
  were feasible there.
- Every fold and full-DEV fit records the requested budget, natural support,
  selected count, and budget status. A shortfall is not silently repaired and is
  not a pipeline failure.

## Frozen matrix

The complete matrix is 9 methods × 2 datasets × 2 final models × seed 42 =
**36 cells**. Each cell uses five expanding grouped-time DEV folds with a one-group
gap, followed by the existing locked final OOT evaluation.

Method order is cheapest-first by family:

1. `full_features`
2. `random_k`
3. `iv_woe`
4. `mrmr_mutual_information`
5. `lasso_l1_logistic`
6. `legacy_rf_relevance_corr`
7. `catboost_shap`
8. `boruta_random_forest`
9. `rfe_catboost`

For each method the order is Home Credit/LR, Home Credit/CatBoost,
LendingClub v2/LR, LendingClub v2/CatBoost. LR receives a 20-feature request and
CatBoost a 40-feature request; `full_features` deliberately ignores the budget.

The final LR and CatBoost configurations are copied explicitly into the frozen
file. CatBoost uses the established 1,500-iteration configuration rather than the
model class's constructor default.

## Frozen selector and resource decisions

- IV: 10 quantile bins, 0.5 zero-count smoothing.
- MI-mRMR: 10 bins, MID objective.
- LASSO: `C=0.05`, liblinear, 2,000 iterations, no zero-coefficient fill.
- Random-k/full features: deterministic local seed 42 controls.
- Legacy RF relevance/correlation: preserved under its accurate canonical ID.
- CatBoost-SHAP: 500 iterations, depth 6, learning rate 0.05, native regular
  mean-absolute SHAP, 10,000 training-row explanation sample.
- Boruta RF: 500-tree/depth-6 forest, engine `auto`, 10 iterations,
  `confirmed_top_k`, no tentative/rejected padding.
- CatBoost RFE: 500 iterations, depth 6, learning rate 0.05, fractional step 0.20.
- CPU only; seed 42; four estimator threads; one cell, one fold, and zero data
  loader subprocesses at a time; no nested parallelism.
- RAM, disk, graceful-stop, cleanup, and inter-run readiness limits come from
  `configs/execution/local_laptop_safe_v1.yaml`.
- Per-cell wall limits: 3 h light/SHAP, 6 h Boruta, 8 h RFE.

## Resumption and integrity

Every cell has an immutable deterministic `fbv1-NNN-...` run ID. The runner:

- authenticates the repository, frozen config, and all six Prompt 9 artifacts;
- performs machine/resource preflight before dataset access;
- skips only completed runs whose manifest, checkpoint, required files, and
  declared artifact hashes authenticate;
- resumes only a registered non-completed checkpoint with matching run, data,
  protocol, configuration, and artifact identity;
- quarantines untracked partial artifacts through the existing checkpoint layer;
- stops after the first manual, resource, wall-clock, or unexpected failure;
- keeps completed runs immutable; and
- uses the same command for the initial run and any later resume.

## Exact PowerShell commands

Run from `D:\python projects\Research`:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py --status
.\.venv\Scripts\python.exe scripts\run_full_baseline.py --audit-pilots
.\.venv\Scripts\python.exe scripts\run_full_baseline.py
```

If the command stops or the machine reboots, run the same execution command again:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py
```

Optional live log view in another PowerShell window:

```powershell
Get-Content logs\runs.log -Wait -Tail 40
```

Do not add flags or edit the frozen YAML between attempts. `--status` and
`--audit-pilots` are read-only and do not start a real baseline worker.

## Expected elapsed time on the pilot machine

The six pilots measured only the smallest/earliest DEV fold. Scaling their heavy
selector work across five expanding folds, a full-DEV selector fit, both model
budgets, and both datasets gives roughly 20–25 hours for the heavy selectors
alone. Repeated data preparation, the six lighter methods, and 108 final CatBoost
fold/full-DEV fits add substantial time.

A practical uninterrupted expectation is **about 30–45 hours** on this machine
(roughly 1.5 days, with two days a sensible scheduling window). This is an
engineering estimate, not a deadline. Resource stops, other machine load, disk
speed, and resume attempts can extend it; the configured wall limits are safety
ceilings per cell, not the expected duration.
