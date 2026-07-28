# Prompt 8 final implementation report

Status: **PROMPT_08_HEAVY_SELECTOR_IMPLEMENTATION_PASS**

## Completion gate

| Requirement | Met | Evidence |
|---|---|---|
| Prompt 7 baseline and commit authenticated | yes | HEAD `4847085`, clean tree, `git show --stat` matches the handoff exactly |
| Stale run-directory assertion resolved by exact 16-run identity proof | yes | lock file + disk + run index + Prompt 6 inventory all set-equal; `run_count_assertion_resolution.md` |
| CatBoost RFE has an accurate descriptor and common-protocol implementation | yes | `rfe_catboost` / `rfe_catboost_fractional_step_v1` |
| RFE uses fractional removal, records fit/elimination history, requires a budget, fakes no natural support | yes | `step_fraction=0.20`; history table; `k=None` fails pre-fit; `natural_selected is None` |
| Boruta preserves confirmed/tentative/rejected | yes | three states partition the universe; asserted against a deterministic stub |
| Boruta natural support is confirmed-only | yes | in every mode, including the tentative-inclusive one |
| Tentative-inclusive mode separately named, never called natural | yes | `confirmed_then_tentative` + explicit warning |
| Frozen `BorutaSelector`, `RFESelector`, aliases, voting unchanged | yes | signatures asserted; `ELIGIBLE_VOTERS == ("rf_corr_mrmr", "boruta")` |
| CatBoost-SHAP uses native recorded SHAP with mean-absolute aggregation | yes | `EFstrType.ShapValues`, `shap_calc_type='Regular'`, verified on catboost 1.2.10 |
| Expected-value column excluded correctly | yes | width check plus a by-value check that no score equals the base mean |
| Explanation sample deterministic and training-only | yes | local seeded RNG, stratified, row-identity SHA-256 recorded |
| Explicit seed/thread, tie, budget, failure, serialization behaviour per method | yes | `heavy_selector_contract_v1.md` |
| Common leakage/ordering/edge-case/determinism/compatibility tests pass | yes | 46 in `test_heavy_common.py` |
| Independent selector-specific oracle tests pass | yes | native-SHAP oracle at 1e-12; RFE step/fit-count arithmetic; Boruta stub states |
| Synthetic heavy fixture passes | yes | 4 cases, 0 failures, 3.3 s |
| Full suite has no failure and no weakened assertion | yes | **870 passed, 0 failed, 31 skipped** |
| Prompt 6 41-file package byte-identical | yes | 41/41, 0 changed |
| Validator authenticates expected artifacts/runs | yes | passed, 388 artifacts, 20 runs |
| No real workload or OOT evaluation ran | yes | `real_dataset_loaded`/`oot_data_loaded`/`real_fold_executed` all false |
| No scientific result, run index, prediction, model, or historical artifact changed | yes | validator + hash comparison + `results/` file-set equality |
| Complete audit package exists | yes | 14 files, 31 manifested deliverables, 0 missing |
| Committed locally, not pushed | yes | see below |

## Registered methods

| Registry ID | Display label | Implementation ID |
|---|---|---|
| `rfe_catboost` | RFE (CatBoost) | `rfe_catboost_fractional_step_v1` |
| `boruta_random_forest` | Boruta (random forest) | `boruta_random_forest_confirmed_tentative_v1` |
| `catboost_shap` | CatBoost-SHAP | `catboost_native_shap_regular_mean_abs_train_sample_v1` |

Registry: 6 methods → 9. Nothing removed, no existing identity changed.

## Audit answers

**RFE is CatBoost-backed**, as the roadmap records — confirmed in code, no estimator
substituted. But it uses an **integer** `step=10`, not a fractional step. The new
standalone method uses fractional 0.20; the legacy path keeps 10; the discrepancy is
recorded in the descriptor and in every result's `legacy_counterpart` block.

**Existing Boruta does not expose tentative.** `BorutaSelector` reads only
`support_` and `ranking_`; `support_weak_` is never accessed, so tentative and
rejected are indistinguishable. It does not pad — it clamps down.

**There was no SHAP code at all.** `CatBoostModel.get_feature_importance()` takes no
arguments and returns PredictionValuesChange. The `shap` package is a declared
dependency but is imported nowhere in `src/`.

**No legacy selector had to change**, so no `NEEDS USER ACTION` arose. Their
estimator profiles were reused; their classes were not touched.

## Latent issue found and fixed

Putting heavy methods in the shared Prompt 7 registry exposed two places that
assumed every registered method is cheap: `scripts/verify_lightweight_selectors.py`
and `tests/selectors/test_lightweight_integration.py` both iterated every ID and
would have run heavy CatBoost and Boruta fits inside the light fixture. Both now
filter on the declared cost class via the new `method_ids_by_cost_class("light")` —
a scope narrowing, not a weakened assertion, with heavy coverage provided by
`test_heavy_integration.py` and 113 focused tests.

## Test counts

| Scope | Result |
|---|---|
| `test_heavy_rfe.py` | 20 passed |
| `test_heavy_boruta.py` | 22 passed |
| `test_heavy_catboost_shap.py` | 25 passed |
| `test_heavy_common.py` | 46 passed |
| `test_heavy_integration.py` | 9 passed |
| Selector suite | **279 passed** (13.5 s) |
| Full repository suite | **870 passed, 0 failed, 31 skipped** (205.84 s) |
| Prompt 7 baseline | 745 passed, 1 failed, 31 skipped |
| Delta | **+125 passing, the pre-existing failure resolved, 0 new skips** |

## Synthetic fixture

500 rows, 7 ordered candidates (linear signal, nonlinear signal, near-duplicate,
two noise, constant, 30 % missing), 4 metadata/excluded columns outside the
universe. Total 3.3 s.

| Case | Mode | Selected | Status |
|---|---|---|---|
| `rfe_catboost_feasible_budget` | `matched_budget` | 3/3 | `satisfied` |
| `boruta_natural_confirmed` | `natural_confirmed` | 4/— | `not_applicable` |
| `boruta_insufficient_confirmed` | `confirmed_top_k` | **4/7** | `infeasible_natural_support` |
| `catboost_shap_feasible_budget` | `matched_budget` | 3/3 | `satisfied` |

The third row is the point: 7 requested, 4 returned, nothing padded.

## Changed and committed paths

31 deliverables. Source: `src/credit_risk_fs/selectors/heavy/` (5 files),
`selectors/lightweight/{contract,registry}.py`, `selectors/registry.py`,
`experiments/config.py`, `scripts/verify_heavy_selectors.py`,
`scripts/verify_lightweight_selectors.py`, 5 heavy test files,
`tests/selectors/test_lightweight_integration.py`,
`tests/test_manual_research_orchestration.py`,
`docs/research_extension/heavy_selector_controls_v1.md`, plus 2 regenerated Prompt
7 audit JSONs. Generated: 4 audit artifacts. Documentation: 8 audit markdown files.

**Separately identified:** `tests/test_manual_research_orchestration.py` is the
narrow authenticated 14-to-16 preservation-test correction authorized by Phase 1. It
is included in the Prompt 8 commit and flagged in the manifest under
`separately_identified_changes`.

Excluded: `src/credit_risk_fs/clip/checkpointing.py` (untouched, empty diff at
Phase 0), `results/`, `logs/`, scratch fixture output, `__pycache__`.

## Known pre-existing issues

1. **No linter configured.** `pyproject.toml` has no ruff/mypy config or
   dependency, so gate 1 is *not applicable* and is not claimed as passed.
2. **`clip/checkpointing.py`** was committed in `e48a848` before Prompt 7; its
   Phase 0 diff was empty and Prompt 8 did not touch it.
3. **`n_bins = 10`** for MI-mRMR is still an unfrozen Prompt 7 limitation carried
   into Prompt 9's freeze list.

## Blockers

None. No `NEEDS USER ACTION` condition was met.

## Recommendation for Prompt 9

Run **one bounded single-fold pilot per dataset per heavy method** — six runs — on
DEV fold 1 only, at the real candidate-universe sizes (Home Credit 529,
LendingClub v2 675), under the existing resource policy with hard wall-clock and RSS
stops. Do not touch OOT.

Order them cheapest-first so a stop costs the least information:
`catboost_shap` (one fit + one SHAP pass) → `boruta_random_forest` (bounded by
`max_iter`) → `rfe_catboost` (the most fits by far).

Measure and freeze, in this order of risk:

1. **`rfe_catboost` fit count.** At `step_fraction=0.20`, going from 675 to 40
   features takes roughly `ceil(log(40/675)/log(0.8)) ≈ 13` elimination rounds plus
   one ordering fit — about 14 CatBoost fits at 500 iterations each, on the full
   fold. This is the single largest cost in the stage and the most likely to need a
   larger `step_fraction`. Record the realized count from
   `heavy_metadata["estimator_fit_count"]` rather than estimating it.
2. **`catboost_shap` explanation-sample size.** The 10,000 default is a
   placeholder. SHAP cost scales with rows × trees; measure at 10,000 and at the
   full fold before freezing.
3. **Boruta `max_iter` and forest size**, plus the real confirmed/tentative counts
   — those decide whether a tentative-inclusive mode is needed at all, and if
   confirmed counts land well below the final budgets, `confirmed_top_k` will
   report infeasible on every cell, which is a design decision to surface early.
4. Thread counts per method, and the MI-mRMR `n_bins` carried from Prompt 7.

Every heavy result already carries `estimator_fit_count`, elimination history,
support-state counts, explanation-sample identity, `peak_process_rss_bytes`, and
`minimum_available_ram_bytes`, so the pilot needs no new instrumentation — only a
runner that records them.

---

**No real research workload or OOT evaluation was executed in Prompt 8.**
**Prompt 9 has not started.**
