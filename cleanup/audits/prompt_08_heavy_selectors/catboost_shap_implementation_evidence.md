# `catboost_shap` implementation evidence

Implementation: `src/credit_risk_fs/selectors/heavy/catboost_shap.py`
Identity: `catboost_shap` / "CatBoost-SHAP" /
`catboost_native_shap_regular_mean_abs_train_sample_v1`

## Installed API, verified before implementing

| Check | Result |
|---|---|
| catboost version | **1.2.10** |
| `EFstrType.ShapValues` | present |
| `get_feature_importance(shap_calc_type=...)` | present, default `'Regular'` |
| Returned shape | `(rows, n_features + 1)` |
| Trailing column | constant across rows — the expected/base value |

No substitution was needed, so no `NEEDS USER ACTION` arose.

## The variant, fully recorded

| Element | Value |
|---|---|
| Feature importance type | `EFstrType.ShapValues` (native CatBoost) |
| SHAP calculation type | `Regular` |
| Model output scale | native raw model output |
| Aggregation | mean absolute SHAP over explanation rows |
| Expected-value column | trailing column, **excluded** from the ranking |
| Fallback importance | **none permitted** |

All of it is encoded in the implementation ID and repeated in every result's
`configuration` and `heavy_metadata`. Any Exact/Approximate, interventional,
reference-data, sampled-background, probability-output, or multiclass variant
requires a new implementation ID.

The repository had **no** prior SHAP path: `CatBoostModel.get_feature_importance()`
takes no arguments and therefore returns PredictionValuesChange. Asserted by
`test_repository_had_no_prior_shap_path`, which inspects the source.

## Independent oracle

`test_scores_equal_a_direct_native_shap_calculation` refits CatBoost from the
result's own recorded `estimator_params`, calls
`get_feature_importance(Pool(...), type=EFstrType.ShapValues,
shap_calc_type="Regular")` directly, drops the trailing column, and takes
`mean(|shap|)` per feature.

- Tolerance: **1e-12**, far tighter than any meaningful SHAP difference. CatBoost
  fitting is deterministic for a fixed seed, thread count, and CPU task type, so
  the only expected difference is float64 accumulation order in the mean.
- `explanation_sample_size=None` is used so the sample is unambiguously "all
  training rows" and the oracle needs **no copy of the sampling logic** to
  reproduce it.
- Result: every feature matches within tolerance.

`test_expected_value_column_is_excluded` additionally computes the base column's
own mean absolute value and asserts no produced score equals it — so a leaked base
column would be caught by value, not only by array width.

## Explanation-sample boundary

| Property | Behaviour |
|---|---|
| Source | **selector training partition only**; outer validation and OOT rows are structurally unreachable |
| RNG | local `numpy.random.default_rng(explanation_sample_seed)`; global state untouched |
| Rule | deterministic stratified without replacement, proportional allocation with at least one row per class |
| Recorded | requested size, realized size, training row count, positive/negative counts, training and sample positive rates, seed, ordered row-identity SHA-256, scope |
| Smaller partition than requested | uses every training row and records `used_all_training_rows` plus a warning |
| No sampling requested | `explanation_sample_size=None` → all rows, nothing drawn |

Verified: the same configuration reproduces the row-identity hash and the scores
exactly; a different `explanation_sample_seed` changes the hash but **not** the
method identity; stratified prevalence tracks the training rate to within 0.02;
corrupting labels on rows outside the supplied partition leaves the hash, the
scores, and the selection unchanged.

Prompt 9 freezes the real explanation-sample size after runtime measurement. The
default of 10,000 is a placeholder, not a frozen value.

## Budget behaviour

| Case | Behaviour |
|---|---|
| `k=None` | controlled failure at `budget_validation`, before CatBoost is constructed |
| `k<=0` | controlled failure at `budget_validation` |
| `0 < k <= universe` | top `k` unique features, `satisfied` |
| `k > universe` | `clipped_to_universe` |
| natural support | `None`; `supports_natural_support = False` |

Zero SHAP values are valid scores. If **every** score is zero, the ranking falls to
the authenticated candidate order, the requested feasible `k` is still returned,
and an `all_scores_zero` warning records that the result carries no evidence of
informativeness.

## Controlled failures

| Trigger | Stage |
|---|---|
| `k=None` / `k<=0` | `budget_validation` |
| CatBoost `fit` raises | `estimator_fit` |
| native SHAP call raises | `shap_calculation` |
| array shape ≠ `(rows, features+1)` | `shap_shape_validation` |
| any NaN or Inf in the SHAP array | `shap_value_validation` |
| single-class target | `target_validation` |
| excluded column offered | `candidate_universe_validation` |

`test_shap_failure_uses_no_fallback_importance` is the decisive one: the mocked
model raises for the SHAP call but would happily return a plain importance vector.
The selector fails instead of using it.

## Logging

`START` / `DONE` events for both `catboost_fit` and `native_shap_values` through
the existing module logger, asserted by
`test_catboost_progress_is_suppressed_and_stages_are_logged`. No CatBoost iteration
output appears; `allow_writing_files=False` and `verbose=False` are forced.

Tests: `tests/selectors/test_heavy_catboost_shap.py` — **25 passed**.
