# Lightweight selector contract v1

Contract version: `lightweight_selector_contract_v1`
Defined in: `src/credit_risk_fs/selectors/lightweight/contract.py`

Every Prompt 7 selector still implements the pre-existing `FeatureSelector`
protocol. This document describes the **evidence record** layered on top of it.

## Fields

| Field | Meaning |
|---|---|
| `method_id` | canonical registry ID, one per algorithm |
| `display_label` | accurate human-readable name |
| `implementation_id` | versioned implementation identity; changing the estimator changes this |
| `selection_mode` | `natural`, `matched_budget`, `full_control`, `random_control`, `coefficient_ranking` |
| `supervised` | whether the target was made available at all |
| `fit_scope` | `dev_fold_training_only` (shares the frozen voting protocol's value) |
| `seed` | seed recorded even when the algorithm is RNG-free |
| `configuration` | full configuration dict, recorded verbatim |
| `candidate_universe` | exact ordered eligible universe |
| `candidate_universe_sha256` | ordered-name hash, byte-compatible with `rank_voting._ordered_name_hash` |
| `requested_budget` | requested final feature count, or `None` |
| `selected_features` | ordered selected subset |
| `actual_selected_count` | derived length |
| `budget_status` | `satisfied`, `clipped_to_universe`, `not_applicable`, `infeasible_natural_support`, `empty_universe` |
| `ranking` | complete ordered ranking when the algorithm supplies one |
| `raw_scores` | per-feature score |
| `score_orientation` | `higher_is_better`, `lower_is_better`, `rank_1_is_best`, `not_applicable` |
| `natural_selected` | algorithm-chosen subset, separate from the budget-matched one |
| `tie_rule` | documented deterministic tie policy |
| `training_row_count` | rows the selector actually received |
| `training_identity_sha256` | hash of the exact training rows *and* target |
| `fit_seconds` | wall-clock fit duration |
| `warnings` | non-fatal disclosures |
| `failure_reason` | controlled-failure cause |

## Invariants enforced on construction

Checked in `SelectionResult.__post_init__`, so an invalid result cannot reach an
artifact:

1. `selection_mode`, `budget_status`, and `score_orientation` come from closed vocabularies.
2. The candidate universe has no duplicate names.
3. Selected features are unique.
4. Every selected feature belongs to the candidate universe.
5. Every ranked feature belongs to the candidate universe.
6. Every selected feature carries a rank — a selected feature absent from the ranking would make the published rank column silently incomplete.
7. The natural support, when present, lies inside the candidate universe.
8. `requested_budget` is non-negative.
9. `budget_status == "satisfied"` requires `actual_selected_count == requested_budget`.
10. `budget_status == "not_applicable"` requires `requested_budget is None`.

## Tie rule

`descending_score_then_ascending_feature_name` — sort by score descending, break
ties on the ascending feature **name**. Because the key is a pure function of the
score and the name, reordering the input DataFrame's columns cannot reorder the
output. `full_features` instead uses `candidate_universe_order_preserved`, since it
produces no comparable score.

Verified by `test_rank_by_score_is_independent_of_mapping_order`,
`test_equal_information_value_breaks_ties_on_the_feature_name`,
`test_exact_ties_resolve_on_the_ascending_feature_name`, and
`test_selection_is_deterministic_and_order_independent`.

## Over-budget policy

**Preserves the pre-existing repository policy** in
`selectors.base.resolve_feature_budget`, which clamps to `min(requested, available)`.
A request larger than the eligible universe returns every eligible feature and
records `budget_status = "clipped_to_universe"` plus a warning. The budget is
never padded, never duplicated, and never redefined.

A separate case is a *model-chosen* support smaller than the budget: LASSO records
`infeasible_natural_support` and returns only its natural support. Reaching the
budget by appending zero-coefficient features is possible **only** through the
explicitly named `coefficient_ranking` mode, gated behind
`allow_zero_coefficient_fill=True`, so a padded subset can never be read as an L1
support.

## Zero-feature and empty-universe outcomes

An empty candidate universe yields `budget_status = "empty_universe"`, an empty
selection, a warning, and `selected_features_ == []` — a controlled, testable
outcome rather than an exception. An L1 fit that shrinks every coefficient to zero
yields an empty natural support with an explicit warning and no substitute
selector.

## Controlled failure

`ControlledSelectorFailure` carries `method_id`, `stage`, `cause`, and
`configuration`. There is no fallback path anywhere in the package: a selector
that cannot run says so. Stages currently raised:
`candidate_universe_validation`, `target_validation`, `binning`,
`design_matrix_validation`.

## Fit-boundary evidence

`training_identity_sha256` hashes the ordered row index **and** the target values
the selector received. Two selectors handed different row sets cannot produce the
same hash, so a leaked validation or OOT row is detectable after the fact. The two
unsupervised controls receive no target at all, so their hashes are disjoint from
every supervised hash — asserted by
`test_supervised_methods_see_the_target_and_controls_do_not`.

## Serialization

`to_dict` / `to_json` / `from_dict` / `from_json` round-trip method identity,
implementation identity, selection mode, ranking order, selection order, scores,
counts, budget status, natural support, seed, and both hashes. Reload rejects an
unknown `contract_version`. `to_long_frame` emits the common artifact schema
(`LONG_FRAME_COLUMNS`), one row per ranked feature, with dense one-based ranks.
