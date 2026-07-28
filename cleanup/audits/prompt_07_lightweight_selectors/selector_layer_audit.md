# Prompt 7 selector-layer audit

What existed before Prompt 7, what each path actually computes, and the action
taken. No second selector framework was created: the pre-existing
`FeatureSelector` protocol in `src/credit_risk_fs/selectors/base.py` is the shared
interface, and Prompt 7 extends it rather than replacing it.

## Existing shared machinery reused unchanged

| Component | Location | Reused for |
|---|---|---|
| `FeatureSelector` protocol | `selectors/base.py` | `fit` / `transform` / `fit_transform` contract |
| `SelectedFeaturesMixin` | `selectors/base.py` | `selected_features_` plus the legacy alias |
| `validate_feature_frame` | `selectors/base.py` | duplicate-name and type rejection |
| `select_feature_frame` | `selectors/base.py` | transform-time schema errors |
| `resolve_feature_budget` | `selectors/base.py` | the pre-existing `min(requested, available)` clamp policy |
| `get_selector` | `selectors/registry.py` | method-ID resolution |
| `apply_feature_budget_to_selector_kwargs` | `experiments/config.py` | final-budget wiring |
| `write_json_atomic` / `write_csv_atomic` | `experiments/atomic_io.py` | artifact publication |
| `reject_historical_write` | `experiments/result_paths.py` | legacy-root write barrier |
| `sha256_text` | `utils/hashing.py` | universe and fit-boundary hashes |
| `process_fold` consumption | `models/_fold.py` | fold-local `selector.fit(X_train, y_train)` |

`models/_fold.py` calls `selector.fit(X_train, y_train)` and reads results through
`get_selected_features`. Every Prompt 7 selector satisfies that shape, so the fold
runner required **no change**.

## Per-path findings

### `RandomForestRelevanceMRMRSelector` — the legacy "mRMR"

| Field | Content |
|---|---|
| Current method ID | `mrmr` (registry), `rf_corr_mrmr` (voting voter ID) |
| Display label before | class already named `RandomForestRelevanceMRMRSelector` |
| Actual algorithm | relevance = mean RandomForest impurity importance; redundancy = mean absolute Pearson correlation with already-selected features, floored at 0.05; greedy score = relevance / redundancy |
| Canonical MI-mRMR? | **No.** Neither term is mutual information |
| Supervised | Yes |
| Fit boundary | rows passed to `fit`; `get_mrmr_features` subsamples to 10,000 rows via `RandomState(random_state)` for the correlation matrix |
| Ranking available | Yes — `rf_importances_` (higher is better) plus `selection_trace_` |
| Fixed-budget support | Yes — `k` |
| Seeded | `random_state` into `RandomForestClassifier` and the correlation subsample |
| Serialization | `selected_features_` list; trace as a DataFrame |
| Historical use | **All** artifacts naming `mrmr` / `rf_corr_mrmr`, including the `rf_corr_mrmr` voter and the reference arm of all 16 `cross_dataset_rank_voting_v1` runs |
| Action taken | **Preserved byte-for-byte.** Given the accurate canonical ID `legacy_rf_relevance_corr`; `mrmr` retained as a compatibility alias resolving to the same class |

The class already declared `algorithm_name = "rf_relevance_correlation_redundancy"`
and `canonical_mrmr = False`, with a source comment reserving a separate entry for
a future canonical implementation. Prompt 7 completed that intent at the registry
level, which was the remaining misleading surface: the *ID* still read `mrmr`
even though the *class* was honest.

### `none` / `""` — the no-selection route

| Field | Content |
|---|---|
| Current method ID | `none`, `""` |
| Actual behaviour | `get_selector` returns `(None, {})`; `process_fold` then uses every supplied column |
| Supervised | No |
| Action taken | **Left exactly as-is.** A new `full_features` control names the same semantics explicitly, with a ranking, a universe hash, and an audit record |

### `IVWOEFilter` (third-party, `iv_woe_filter`)

| Field | Content |
|---|---|
| Location | installed dependency, used only inside `selectors/llm_screening.py` as a pre-filter |
| Actual algorithm | optimal binning → WOE/IV → **threshold** selection on `min_iv`, and WOE **encoding** during `transform` |
| Fold-local | Yes, `fit` uses only the supplied `X`/`y` |
| Budget support | **No** — threshold-based, not top-k |
| Tie rule | undocumented; inherits `sort_values` order |
| Smoothing | fixed `eps=1e-12` added to distribution *shares* |
| Action taken | **Not wrapped.** It cannot express a value-preserving, budget-matched, deterministically-ordered selector. IV was implemented natively; `test_lightweight_iv.py` cross-checks the two against each other on a fixture where the definitions coincide, and they agree to `1e-9` |

### Other existing selectors (untouched)

| Method ID | Class | Action |
|---|---|---|
| `boruta` | `BorutaSelector` | untouched — Prompt 8 scope |
| `rfe` | `RFESelector` | untouched — Prompt 8 scope |
| `boruta_rfe` | `BorutaThenRFESelector` | untouched — Prompt 8 scope |
| `pca` | `PCASelector` | untouched — out of scope; it produces components, not feature subsets |
| `llm`, `llm_then_stat`, `llm_then_mrmr`, `llm_then_boruta` | LLM selectors | untouched; no ranking regenerated |
| `domain_rule_baseline` | `DomainRuleBaselineSelector` | untouched — deterministic domain rules, **not** an LLM call |
| `stable_core_llm_fill` | `StableCoreLLMFillSelector` | untouched |

## Frozen artifact schema that must NOT be reused

`experiments/rank_voting.build_long_voter_ranking_frame` is hard-wired to
`ELIGIBLE_VOTERS = ("rf_corr_mrmr", "boruta")` and validates an exact
`2 * universe_size` row count. It is part of a frozen protocol. Prompt 7 does not
call, extend, or modify it; the new long frame borrows its **field vocabulary**
(`candidate_universe_sha256`, `fit_scope`, `score_direction` → `score_orientation`,
`seed`) so the two read as one repository without coupling to the frozen
validator.

## Gap the contract had to close

The pre-existing protocol carries only `selected_features_`. It has no room for
method identity, implementation version, complete ranking, score orientation,
natural-versus-matched-budget distinction, budget feasibility, tie rule,
candidate-universe hash, fit-boundary hash, or controlled-failure reason. Those
are precisely the fields Prompt 7 needs as evidence, so
`selectors/lightweight/contract.py` adds them in a `SelectionResult` that travels
*alongside* the existing attribute rather than replacing it.
