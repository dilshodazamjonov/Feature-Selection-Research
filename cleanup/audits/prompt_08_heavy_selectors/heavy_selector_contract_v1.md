# Heavy-selector contract v1

Heavy methods use the **same** contract as Prompt 7's light methods —
`lightweight_selector_contract_v1` in
`src/credit_risk_fs/selectors/lightweight/contract.py` — and are declared in the
**same** `MethodDescriptor` registry. No second framework was created. What Prompt
8 added is the minimum required to represent a real estimator honestly.

## Contract extensions

Two optional `SelectionResult` fields:

| Field | Purpose |
|---|---|
| `estimator_config_sha256` | hash of the estimator configuration actually handed to the fitted model — scientific evidence, not plumbing |
| `heavy_metadata` | one nested mapping for fit counts, elimination history, Boruta support states, explanation-sample identity, SHAP calculation type, thread counts, and resource observations |

Both default to `None` and are read with `.get` on reload, so **Prompt 7 artifacts
load with no migration and no rewrite** — asserted by
`test_prompt_07_payloads_still_load_without_the_new_fields`. `CONTRACT_VERSION` is
therefore unchanged at `lightweight_selector_contract_v1`; a nested mapping was
chosen over a dozen new top-level fields precisely to keep the version stable.

One base-class hook:

| Hook | Purpose |
|---|---|
| `_validate_configuration(eligible_count)` | called at the top of `fit`, **before** target validation and before any estimator is constructed, so an unsupported budget fails without cost |

Three new `SELECTION_MODES` for Boruta: `natural_confirmed`, `confirmed_top_k`,
`confirmed_then_tentative`.

## New descriptor fields

Every method now declares:

| Field | Meaning |
|---|---|
| `cost_class` | `light` or `heavy`; drives fixture scoping via `method_ids_by_cost_class` |
| `estimator_family` | the estimator actually used, or `None` for pure controls |
| `guarantees_exact_k` | whether a successful fit always returns exactly `k` |
| `score_name` | what the published score is |
| `serialization_version` | the contract the results conform to |
| `allowed_in_frozen_voting` | `False` for every Prompt 8 method |
| `controlled_failure_conditions` | the conditions under which it raises |

## Per-method contract

| | `rfe_catboost` | `boruta_random_forest` | `catboost_shap` |
|---|---|---|---|
| Cost class | heavy | heavy | heavy |
| Estimator | CatBoostClassifier | RandomForestClassifier via BorutaPy | CatBoostClassifier |
| Supervised | yes | yes | yes |
| Ranking | complete (elimination order) | complete (state then engine rank) | complete (mean abs SHAP) |
| Score orientation | `rank_1_is_best` | `rank_1_is_best` | `higher_is_better` |
| Natural support | **absent** — wrapper method | **confirmed only** | **absent** — no defensible threshold |
| `k` required | **yes** | only in fixed-budget modes | **yes** |
| Exact `k` guaranteed | yes | no | yes |
| Selection modes | `matched_budget` | `natural_confirmed`, `confirmed_top_k`, `confirmed_then_tentative` | `matched_budget` |
| Seed field | `random_state` → `random_seed` | `random_state` → forest and engine | `random_state` → `random_seed`, plus `explanation_sample_seed` |
| Thread field | `thread_count` | `n_jobs` | `thread_count` |
| Sampling | none | none | `explanation_sample_size` |
| Serialization | contract v1 + heavy metadata | same | same |

## Pre-fit validation

Failures that occur before any estimator exists:

- `rfe_catboost`: `k=None` (no natural stopping point), `k<=0`
- `catboost_shap`: `k=None` (no defensible natural SHAP threshold), `k<=0`
- `boruta_random_forest`: `k=None` or `k<=0` in a fixed-budget mode; unknown
  `selection_mode` is rejected at construction
- every method: identity/target/split/time/excluded columns offered as candidates

`test_k_none_fails_before_catboost_is_called` and
`test_fixed_budget_modes_require_k_before_the_engine_runs` prove this by
monkeypatching the estimator to raise if constructed.

## Budget policy

Unchanged from Prompt 7: a request exceeding the eligible universe returns the
whole universe as `clipped_to_universe`, never padded or duplicated. `k` equal to
the universe is `satisfied` — and for RFE it performs **zero** estimator fits
rather than eliminating pointlessly.

A model-chosen support short of the budget is `infeasible_natural_support`. For
Boruta this never pads from tentative (in `confirmed_top_k`) or from rejected (in
any mode).

## Controlled failure

`ControlledSelectorFailure` carries method, stage, cause, and configuration. There
is no fallback anywhere. Stages: `budget_validation`,
`candidate_universe_validation`, `target_validation`,
`design_matrix_validation`, `estimator_fit`, `importance_extraction`,
`engine_fit`, `support_extraction`, `shap_calculation`,
`shap_shape_validation`, `shap_value_validation`.

## Stage logging

`selectors/heavy/_support.heavy_stage` emits `START` / `DONE` / `ERROR` through the
existing module logger, so the repository's handlers decide where each line lands
and full tracebacks go to the debug logger rather than the run log. The wrapper
does not seed, sample, reorder, or retry anything, so it cannot influence a
result. Opaque library calls get stage boundaries rather than invented iteration
progress. `allow_writing_files=False` and `verbose=False` are forced on every
CatBoost fit, so no `catboost_info/` directory and no iteration table appear —
asserted by `test_catboost_output_is_suppressed_and_writes_no_files`.
