# Prompt 8 heavy-selector audit

What existed before Prompt 8, what each path actually computes, and the action
taken. No existing selector was modified.

## Direct answers to the required questions

| Question | Answer |
|---|---|
| Is existing RFE CatBoost-backed as the roadmap records? | **Yes.** `RFESelector` builds `catboost.CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, task_type="CPU")` and passes it to `sklearn.feature_selection.RFE`. |
| Does it remove one feature at a time or use a fractional step? | **Neither.** `step=10` is an **integer**, so sklearn removes exactly 10 features per iteration. It is not one-at-a-time and not fractional. |
| Does existing Boruta expose confirmed and tentative separately? | **No.** `BorutaSelector` reads only `support_` and `ranking_`. `support_weak_` is never accessed, so the tentative state is discarded entirely and a caller cannot distinguish tentative from rejected. |
| Does any existing path silently pad Boruta support? | **No.** `BorutaSelector` clamps a requested count *down* via `resolve_feature_budget(n_features, len(confirmed))`, so it can return fewer than requested but never pads. |
| Does any existing SHAP code use native CatBoost ShapValues, generic `shap`, PredictionValuesChange, or ordinary importance? | **There is no SHAP code at all.** `CatBoostModel.get_feature_importance()` is called with no arguments, which is CatBoost's default **PredictionValuesChange**. The `shap` package is a declared dependency but is not imported anywhere in `src/`. |
| Does any existing heavy method read validation data or labels outside its supplied fit boundary? | **No.** `process_fold` calls `selector.fit(X_train, y_train)`; both legacy selectors use only the arrays handed to `fit`. `RFESelector` passes no `eval_set`. |
| Can the old selectors be composed behind the Prompt 7 protocol without changing their code or historical routing? | **Partly.** Their *estimator profiles* were reused, but neither class can be wrapped to satisfy the new evidence requirements (fit counts, per-iteration removals, three-state support) without changing them. Prompt 8 therefore adds separate implementations and leaves both classes byte-for-byte intact. No `NEEDS USER ACTION` was required: nothing had to change in either legacy class. |

## Per-path detail

### `RFESelector` — `credit_risk_fs/selectors/rfe.py`

| Field | Evidence |
|---|---|
| Registry/config ID | `rfe`; also inside `boruta_rfe` via `rfe_kwargs` |
| Display label | class name only; no descriptor before Prompt 8 |
| Implementation | `sklearn.feature_selection.RFE` wrapping `CatBoostClassifier` |
| Estimator | CatBoost: 500 iterations, depth 6, lr 0.05, CPU, `allow_writing_files=False`, `verbose=False`, `thread_count` configurable |
| Fit boundary | rows/labels passed to `fit`; no `eval_set` |
| Ranking | sklearn `ranking_` (1 = selected, higher = eliminated earlier); no per-iteration importances retained |
| Natural support | none — a wrapper method |
| Fixed-budget behaviour | `n_features` required; **raises `ValueError` if `n_features > X.shape[1]`**; returns full universe if equal; enforces an exact-budget contract post-fit |
| Seed/thread contract | `random_state` → CatBoost `random_state`; `thread_count` validated positive |
| Logging/resource control | module `logging.getLogger(__name__)`, two INFO lines; no heartbeats |
| Serialization | `selected_features_`, `selection_trace_` DataFrame, `effective_estimator_config_` dict |
| Historical consumers | `rfe` and `boruta_rfe` registry routes; the downstream RFE stage of the frozen voting pipeline |
| Compatibility risk | step semantics, exact-budget error behaviour, estimator profile |
| Prompt 8 action | **preserved unchanged**; new separate `rfe_catboost` added |

### `BorutaSelector` — `credit_risk_fs/selectors/boruta.py`

| Field | Evidence |
|---|---|
| Registry/config ID | `boruta`; inside `boruta_rfe`; inside `llm_then_boruta`; the frozen voting voter `boruta` |
| Implementation | `boruta.BorutaPy` wrapping `sklearn.ensemble.RandomForestClassifier` |
| Estimator | forest: 500 trees, depth 6, `n_jobs` configurable, `random_state`; BorutaPy: `n_estimators="auto"`, `max_iter=10`, `verbose=0` |
| Fit boundary | `X.to_numpy()` / `y.to_numpy()` from `fit` only |
| Ranking | `feature_ranking_`, ordered by `(engine rank, name)` |
| Natural support | **confirmed only**; tentative discarded |
| Fixed-budget behaviour | clamps down to the confirmed count; never pads |
| Seed/thread contract | `random_state` to both forest and engine; `n_jobs` validated positive |
| Serialization | `selected_features_`, `feature_ranking_` list |
| Historical consumers | the `boruta` voter of `cross_dataset_rank_voting_v1` — all 16 runs |
| Compatibility risk | **highest in the repository**; the frozen voting protocol depends on it |
| Prompt 8 action | **preserved unchanged**; new separate `boruta_random_forest` added; `allowed_in_frozen_voting=False` on every new descriptor |

### CatBoost importance / SHAP

| Field | Evidence |
|---|---|
| `CatBoostModel.get_feature_importance()` | no arguments → PredictionValuesChange, **not SHAP** |
| `evaluation/_feature_utils.py:59` | same no-argument call for reporting |
| Native SHAP anywhere in `src/` | **absent before Prompt 8** |
| Installed API verified | catboost 1.2.10; `EFstrType.ShapValues` present; `get_feature_importance(..., shap_calc_type=...)` present with default `'Regular'`; returns `(rows, n_features + 1)`, trailing column constant |
| Prompt 8 action | new `catboost_shap`, native `EFstrType.ShapValues` + `shap_calc_type='Regular'`, no fallback permitted |

### Untouched

`boruta_then_rfe.py`, `pca.py`, `llm_screening.py`, `llm_then_stat.py`,
`domain_rule_baseline.py`, `stable_core_llm_fill.py`, `fixed_rank_then_mrmr.py`,
`mrmr.py`, and every Prompt 7 module's behaviour.

## Prompt 7 assumptions that needed correcting

Two places assumed *every registered method is cheap*, which stopped being true:

1. `scripts/verify_lightweight_selectors.py` iterated every registry ID, so it
   would have run heavy CatBoost and Boruta fits inside the light fixture.
2. `tests/selectors/test_lightweight_integration.py` did the same and asserted
   light-only properties.

Both now filter on the declared cost class via the new
`method_ids_by_cost_class("light")`. This **narrows scope rather than weakening
assertions**: the same five light methods are still asserted in full, and the
heavy methods are covered by `test_heavy_integration.py` plus their own focused
suites. A hard-coded name list was deliberately avoided so a future method cannot
silently escape either fixture.

No assumption was found that every selector always returns exactly `k`: the
Prompt 7 contract already modelled `clipped_to_universe` and
`infeasible_natural_support`, and `guarantees_exact_k` is now declared per
descriptor.
