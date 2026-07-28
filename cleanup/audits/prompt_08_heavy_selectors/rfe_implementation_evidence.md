# `rfe_catboost` implementation evidence

Implementation: `src/credit_risk_fs/selectors/heavy/rfe_catboost.py`
Identity: `rfe_catboost` / "RFE (CatBoost)" / `rfe_catboost_fractional_step_v1`

## Estimator identity — roadmap confirmed in code

The roadmap records the RFE baseline as CatBoost-backed. Verified: `RFESelector`
constructs `catboost.CatBoostClassifier` and hands it to sklearn's `RFE`. The
identity therefore accurately says CatBoost, and the same estimator profile is
reused: 500 iterations, depth 6, learning rate 0.05, CPU,
`allow_writing_files=False`, `verbose=False`.

No estimator was substituted to match the roadmap, because none needed to be.

## Recorded discrepancy: step semantics

| | Legacy `rfe` | New `rfe_catboost` |
|---|---|---|
| Mechanism | `sklearn.feature_selection.RFE` | explicit elimination loop |
| Step | `step=10`, **integer** features per iteration | `step_fraction=0.20`, **fraction of surviving features** |
| Fit count recorded | no | yes |
| Per-iteration removals recorded | no | yes |

The roadmap asks for a fractional step defaulting to 0.20 for the standalone
method. The legacy path already freezes an integer step of 10 and is used by the
frozen voting pipeline's downstream RFE stage, so it keeps that value. Both
readings are preserved rather than reconciled: the discrepancy is recorded in the
descriptor notes, in `legacy_counterpart` inside every result's configuration, and
asserted by `test_legacy_rfe_selector_keeps_its_integer_step`.

A wrapper around `RFESelector` was rejected because sklearn's `RFE` exposes neither
a fit count nor realized per-iteration removals, both of which Prompt 8 requires as
evidence.

## Semantics

- Fit boundary: rows and labels passed to `fit`; no `eval_set`; every refit uses
  the same rows.
- Removal per iteration: `max(1, int(step_fraction * len(surviving)))`, then
  capped so the budget is never overshot.
- Tie rule: least important first; an exact importance tie eliminates the candidate
  appearing **later** in the authenticated order, so the surviving set never
  depends on container order.
- Final ordering: one last fit on the survivors, ordered by descending importance
  with the authenticated order as tie break.
- Full ranking: survivors first, then eliminated features latest-first, which
  reproduces elimination order exactly.
- Published score: rank-derived. Per-iteration CatBoost importances come from
  different refits and are **not** one comparable scale; the raw values are
  retained in `elimination_history` and `final_estimator_importances` instead.
- Natural support: `None`. `supports_natural_support = False`.

## Budget behaviour

| Case | Behaviour |
|---|---|
| `k=None` | controlled failure at `budget_validation`, before CatBoost is constructed |
| `k<=0` | controlled failure at `budget_validation` |
| `0 < k < universe` | exactly `k` unique features, `satisfied` |
| `k == universe` | full universe, **0 estimator fits**, `elimination_skipped=True`, explicit warning |
| `k > universe` | `clipped_to_universe`, never padded |

## Observed evidence

Step honoured, from `test_fractional_step_is_honored_and_history_is_complete`
(6 candidates, `k=1`, `step_fraction=0.5`):

| iteration | surviving_before | requested_removals | realized_removals |
|---|---|---|---|
| 1 | 6 | 3 | 3 |
| 2 | 3 | 1 | 1 |
| 3 | 2 | 1 | 1 |

Fit count = 3 elimination fits + 1 final ordering fit = **4**. Removed features
plus selected features reconstitute the universe exactly, with no duplicates.

Overshoot protection, from `test_realized_removals_never_overshoot_the_budget`
(`k=5`, `step_fraction=0.9`): requested 5, realized **1**.

Signal recovery: on a 500-row fixture with a linear signal, a CatBoost-friendly
nonlinear signal and four noise columns, `k=2` selects exactly
`{linear_signal, nonlinear_signal}` and both outrank every noise column.

## Controlled failures

| Trigger | Stage |
|---|---|
| `k=None` | `budget_validation` |
| `k<=0` | `budget_validation` |
| CatBoost `fit` raises | `estimator_fit` |
| importance length ≠ feature count | `importance_extraction` |
| excluded column offered as a candidate | `candidate_universe_validation` |

No parameter, sample, step, or estimator is changed after a failure.

## Determinism and isolation

- Same seed and configuration reproduce selection, ranking, fit count, and the
  elimination-history DataFrame exactly.
- Reversing the input column order leaves the selected set unchanged.
- Global NumPy RNG state is untouched.
- Corrupting labels on rows outside the supplied partition changes nothing, and
  `training_identity_sha256` matches.
- `allow_writing_files=False` and `verbose=False`: no `catboost_info/`, no
  iteration table on stdout.

Tests: `tests/selectors/test_heavy_rfe.py` — **20 passed**.
