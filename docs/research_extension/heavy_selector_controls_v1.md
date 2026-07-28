# Heavy selector controls v1

Roadmap stage: Prompt 8 — heavy selector implementation and registry integration
Contract: `lightweight_selector_contract_v1` (shared with Prompt 7)
Status: **implemented and tested**

**Prompt 8 ran no real-data pilot and no OOT evaluation.** No Home Credit or
LendingClub file, no fold, no selector pilot, no baseline matrix, no model
comparison. Every number in this stage comes from a 500-row deterministic
synthetic fixture. Nothing here changes or reinterprets the Prompt 6 evidence, and
no performance, stability, drift, runtime-superiority, or materiality claim is
created.

## Methods

| Registry ID | Display label | Implementation ID | Cost |
|---|---|---|---|
| `rfe_catboost` | RFE (CatBoost) | `rfe_catboost_fractional_step_v1` | heavy |
| `boruta_random_forest` | Boruta (random forest) | `boruta_random_forest_confirmed_tentative_v1` | heavy |
| `catboost_shap` | CatBoost-SHAP | `catboost_native_shap_regular_mean_abs_train_sample_v1` | heavy |

All three share the Prompt 7 selector contract and registry. They are declared
`allowed_in_frozen_voting = False`; the frozen protocol keeps exactly
`("rf_corr_mrmr", "boruta")`.

## Estimator identities and fit boundaries

Every supervised fit — imputation, sampling, estimator training, importance
calculation, ranking — uses only the rows and labels passed to `fit`, which the
fold runner restricts to the training portion of a DEV fold. `training_identity_sha256`
hashes those exact rows and the target, so a leaked validation or OOT row is
detectable after the fact.

| Method | Estimator | Material parameters |
|---|---|---|
| `rfe_catboost` | `catboost.CatBoostClassifier` | 500 iterations, depth 6, lr 0.05, CPU, `allow_writing_files=False`, `verbose=False`, explicit `random_seed` and `thread_count` |
| `boruta_random_forest` | `sklearn.ensemble.RandomForestClassifier` via `boruta.BorutaPy` | forest 500 trees / depth 6 / `class_weight=None`; engine `n_estimators="auto"`, `max_iter=10`, `perc=100`, `alpha=0.05`, `two_step=True`, `verbose=0`; explicit `random_state` and `n_jobs` |
| `catboost_shap` | `catboost.CatBoostClassifier` | 500 iterations, depth 6, lr 0.05, CPU, `allow_writing_files=False`, `verbose=False`, explicit `random_seed`, `thread_count`, `task_type` |

## RFE: fractional step and fit-count semantics

Removal per iteration is `max(1, int(step_fraction * len(surviving)))`, capped so
the budget is never overshot. The default `step_fraction` is **0.20**.

The historical `rfe` route is different and stays that way: it uses
`sklearn.feature_selection.RFE` with an **integer** `step=10`. Both readings are
preserved rather than reconciled; every `rfe_catboost` result carries a
`legacy_counterpart` block recording the difference.

Recorded per fit: the configured fraction, the requested and realized removals per
iteration, the removed feature names, the exact estimator fit count, and the final
estimator importances. `k == universe` performs **zero** fits.

The published score is rank-derived. Importances from different refits are not one
comparable scale, so the raw per-step values live in `elimination_history` instead
of being presented as a global score.

## Boruta: confirmed / tentative / rejected

The three states partition the candidate universe exactly once. Confirmed wins when
a feature appears in both `support_` and `support_weak_`.

**Natural support is confirmed-only, in every mode.** The historical
`BorutaSelector` reads only `support_` and discards the tentative state entirely;
this method preserves it.

| Mode | `k` | Behaviour | Padding policy |
|---|---|---|---|
| `natural_confirmed` | ignored (recorded) | all confirmed | none |
| `confirmed_top_k` | required | top `k` confirmed | never pads; short of `k` → `infeasible_natural_support` |
| `confirmed_then_tentative` | required | confirmed then tentative | tentative allowed; **rejected never**; short of `k` → `infeasible_natural_support` |

`confirmed_then_tentative` is labelled a matched-budget adaptation in its warnings
and never presented as natural Boruta support. No mode fills from rejected
features; a rejected-feature ranking would be a separate methodological decision
and a separate implementation identity.

The engine requires a finite numeric matrix — a constraint the legacy selector
shares. The selector **refuses** non-finite input at `design_matrix_validation`
rather than imputing, because imputing silently would change the method's
preprocessing.

## CatBoost-SHAP: calculation and aggregation

| Element | Value |
|---|---|
| Feature importance type | `EFstrType.ShapValues` (native) |
| SHAP calculation type | `Regular` |
| Model output scale | native raw model output |
| Aggregation | mean absolute SHAP over explanation rows |
| Expected-value column | trailing column, excluded from the ranking |
| Fallback | none permitted |

Verified against catboost 1.2.10. Any Exact/Approximate, interventional,
reference-data, sampled-background, probability-output, or multiclass variant
requires a new implementation ID.

### Explanation-sample boundary and hashing

The sample is drawn **only** from the selector's training partition using a local
seeded generator, stratified on the target with proportional allocation and at
least one row per class. Recorded: requested size, realized size, training row
count, positive and negative counts, training and sample positive rates, seed, and
an ordered row-identity SHA-256. A requested size above the available rows uses
every training row and records that. `explanation_sample_size=None` means no
sampling at all.

## Seed and thread contracts

| Method | Seed fields | Thread field |
|---|---|---|
| `rfe_catboost` | `random_state` → `random_seed` | `thread_count` |
| `boruta_random_forest` | `random_state` → forest and engine | `n_jobs` |
| `catboost_shap` | `random_state` → `random_seed`; `explanation_sample_seed` | `thread_count` |

Thread counts are validated positive. Global NumPy RNG state is never read or
written by any heavy method.

## Deterministic tie policies

| Method | Tie rule |
|---|---|
| `rfe_catboost` | least important eliminated first; an exact importance tie eliminates the candidate appearing **later** in the authenticated order |
| `boruta_random_forest` | within each state, `(engine ranking_, authenticated candidate order)` |
| `catboost_shap` | descending mean absolute SHAP, then authenticated candidate order |

No rule depends on incidental container or DataFrame column order.

## Budget infeasibility

A request above the eligible universe returns the whole universe as
`clipped_to_universe`, never padded or duplicated — the pre-existing
`resolve_feature_budget` clamp. A model-chosen support short of the budget is
`infeasible_natural_support`. An empty universe is `empty_universe` with an empty
selection and no exception.

`k=None` is unsupported for `rfe_catboost` and `catboost_shap` and fails at
`budget_validation` before any estimator is constructed; neither fabricates a
natural support.

## Controlled-failure policy

`ControlledSelectorFailure` carries method, stage, cause, and configuration. There
is no fallback path: no parameter, seed, thread count, step size, iteration count,
sample, estimator, algorithm, or budget is changed to get past a failure. Stages:
`budget_validation`, `candidate_universe_validation`, `target_validation`,
`design_matrix_validation`, `estimator_fit`, `importance_extraction`, `engine_fit`,
`support_extraction`, `shap_calculation`, `shap_shape_validation`,
`shap_value_validation`.

## Historical compatibility

- `RFESelector` and `BorutaSelector` are unchanged; their constructor defaults are
  asserted.
- `rfe`, `boruta`, `boruta_rfe`, `mrmr`, `pca`, `domain_rule_baseline`, and `none`
  resolve exactly as before, with unchanged budget wiring.
- The frozen voting protocol is unchanged and its `boruta` voter still resolves to
  `BorutaSelector`.
- Prompt 7 identities and aliases are unchanged.
- The two new contract fields are optional, so Prompt 5/6/7 artifacts load with no
  migration and no rewrite.

## Configuration still to be frozen by Prompt 9

The synthetic tests use deliberately tiny profiles — CatBoost 30 iterations /
depth 3, forest 40 trees / depth 4, Boruta `max_iter` 8–10. **These are
synthetic-test profiles, not research settings.** Prompt 9 must measure real
single-fold cost and freeze:

1. CatBoost iterations, depth, and learning rate for `rfe_catboost` and
   `catboost_shap`.
2. `step_fraction` for `rfe_catboost` (default 0.20) and the resulting fit count at
   the real candidate-universe sizes (529 / 675).
3. Boruta `max_iter`, forest `n_estimators`, and `max_depth`, plus the real
   confirmed/tentative counts.
4. `explanation_sample_size` for `catboost_shap` (the 10,000 default is a
   placeholder).
5. Thread counts per method under the repository's resource policy.
6. Which Boruta selection mode the comparison family uses, and whether a
   tentative-inclusive mode is required at all.
7. The `n_bins = 10` MI-mRMR discretization carried over from Prompt 7.

## Handoff

Prompt 9 owns bounded single-fold runtime/resource pilots on Home Credit and
LendingClub and the launch-configuration freeze. Later prompts own the full matrix
and selector combinations. Prompt 8 created no real-data result to reinterpret.
