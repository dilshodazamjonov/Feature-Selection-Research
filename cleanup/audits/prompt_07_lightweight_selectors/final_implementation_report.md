# Prompt 7 final implementation report

Status: **PROMPT_07_LIGHTWEIGHT_SELECTOR_IMPLEMENTATION_PASS**

## Completion gate

| Requirement | Met | Evidence |
|---|---|---|
| Prompt 6 has an authenticated, separable provenance checkpoint | yes | commit `07777f3`; `baseline_and_preservation.md` |
| Standalone IV registered, documented, independently tested | yes | `iv_woe`; hand-calculated WOE/IV oracle |
| Canonical MI-mRMR registered, documented, distinct from legacy | yes | `mrmr_mutual_information`; unreachable from the `mrmr` alias |
| Legacy RF-relevance/correlation preserved under an accurate identity | yes | `legacy_rf_relevance_corr`; implementation byte-for-byte unchanged |
| L1-logistic LASSO exposes natural and matched-budget without conflating | yes | `lasso_l1_logistic`; `infeasible_natural_support` and the separate `coefficient_ranking` mode |
| Deterministic random-k registered and reproducible | yes | `random_k`; local-generator oracle test |
| Full candidate features exposed through the existing no-selection path | yes | `full_features`; `none` route unchanged |
| Every method uses the shared protocol and output schema | yes | all subclass `LightweightSelector`; `LONG_FRAME_COLUMNS` |
| Leakage / ordering / budget / missing-column / duplicate-name / zero-feature / serialization / registry / seed tests pass | yes | 155 focused tests |
| One deterministic synthetic tiny-fixture integration pass succeeds | yes | `tiny_fixture_results.json` — PASS, 5 selectors, 0 failures |
| No real dataset, fold, pilot, matrix, comparison, or OOT evaluation ran | yes | `fixture.real_dataset_loaded = false`, `oot_data_loaded = false` |
| No existing artifact, result, run index, or Prompt 6 hash changed | yes | 41/41 Prompt 6 package hashes identical; validator passed |
| Documentation and audit artifacts complete | yes | 27 deliverables, 0 missing |
| Prompt 7 committed separately; unrelated user changes untouched | yes | see the commit section |

## Method identities

| Registry ID | Display label | Implementation ID |
|---|---|---|
| `iv_woe` | Information Value (WOE binning) | `iv_woe_quantile_binned_v1` |
| `mrmr_mutual_information` | mRMR (mutual information) | `mrmr_mutual_information_discrete_plugin_v1` |
| `lasso_l1_logistic` | LASSO (L1-penalized logistic regression) | `lasso_l1_logistic_v1` |
| `random_k` | Random-k control | `random_k_local_generator_v1` |
| `full_features` | Full candidate features | `full_candidate_features_v1` |
| `legacy_rf_relevance_corr` | Legacy RF relevance / correlation redundancy | `rf_relevance_correlation_redundancy` |

## The legacy mRMR finding

The method registered as `mrmr` computes relevance as **mean RandomForest impurity
importance** and redundancy as the **mean absolute Pearson correlation** with
already-selected features (floored at 0.05), selecting greedily on
`relevance / redundancy`. Neither term is mutual information, so it is **not**
canonical mRMR.

The class was already honest — named `RandomForestRelevanceMRMRSelector`, with
`algorithm_name = "rf_relevance_correlation_redundancy"` and
`canonical_mrmr = False`, plus a source comment reserving a separate entry for a
canonical implementation. The remaining misleading surface was the registry **ID**.

Treatment: the implementation is preserved byte-for-byte; `legacy_rf_relevance_corr`
is now its accurate canonical ID; `mrmr` is retained as a compatibility alias
resolving to the same class. No historical artifact was rewritten. The descriptor
records that all 16 `cross_dataset_rank_voting_v1` runs and the `rf_corr_mrmr`
voter used this algorithm.

## IV definition and independent numeric result

`WOE = ln(dist_good/dist_bad)`, `IV = Σ (dist_good − dist_bad)·WOE`, distributions
taken within class, event = class 1 = default.

Oracle: 300 rows across three grades with 10 / 30 / 60 bads.

| Grade | dist_bad | dist_good | WOE | IV contribution |
|---|---|---|---|---|
| A | 0.10 | 0.45 | 1.5040774 | 0.5264271 |
| B | 0.30 | 0.35 | 0.1541507 | 0.0077075 |
| C | 0.60 | 0.20 | −1.0986123 | 0.4394449 |
| **Total** | | | | **0.9735795** |

The implementation matches this hand-calculated total to `1e-12`, and matches the
independently installed `iv_woe_filter.IVWOEFilter` to `1e-9` on the same fixture.

## Canonical mRMR estimator decision

Relevance and redundancy both use `sklearn.metrics.mutual_info_score` (discrete
plug-in, nats) over training-partition quantile discretization with an explicit
missing code. **No dependency was installed.** The estimator is deterministic, so
no seed is needed and results are bit-reproducible.

The Kraskov k-NN estimator behind `mutual_info_classif` was considered and
rejected on the record: it injects `random_state` tie-breaking noise and estimates
continuous–continuous MI by a different construction. Adopting it would require a
new `implementation_id`, not a silent redefinition.

Because of that, the discretization rule is treated as part of the algorithm's
identity and is recorded in every result's configuration. A declared
zero-relevance policy prevents a constant column from displacing a predictor under
the difference objective.

**No NEEDS USER ACTION was raised for this**, because the chosen path required no
new dependency, no native extension, and no ambiguity: one estimator, one
discretization rule, both recorded and versioned. The single judgment call —
`n_bins = 10` — is flagged in the limitations as needing to be frozen before any
real run.

## LASSO natural vs matched-budget semantics

- `k=None` → natural support, `budget_status = not_applicable`
- `k` given, support ≥ k → top-k of the absolute-coefficient ranking; the natural
  support is published alongside
- `k` given, support < k → `infeasible_natural_support`; only the natural support
  is returned; **the budget is not padded**
- `allow_zero_coefficient_fill=True` → `coefficient_ranking` mode, the only route
  past the natural support, named so it can never be read as an L1 support

Total shrinkage to an empty support is a valid recorded outcome. Convergence
failure is reported with no solver, tolerance, penalty, sample, or feature change.

## Random-k and full-control behaviour

`random_k` draws from `numpy.random.default_rng(random_state)`, never touching
global RNG state (asserted), never receiving the target (asserted — identical
results under flipped and absent labels), and publishing the full random priority
order so the subset is reproducible from the artifact alone. Different seeds do not
change the method identity.

`full_features` returns exactly the eligible candidate universe in its
authenticated order, equals the historical `none` route, ignores a fixed budget by
design and records the ignore, and never receives the target.

## Test counts

| Scope | Result |
|---|---|
| Focused selector suite | **155 passed** |
| Full repository suite | **1 failed, 745 passed, 31 skipped** (162.74 s) |
| Before Prompt 7 | 1 failed, 604 passed, 31 skipped |
| Delta | **+141 passing, 0 new failures, 0 new skips** |
| Warnings | 107, all pre-existing `PerformanceWarning`s in the LendingClub feature builder |

## Tiny-fixture results

600 rows, 6 ordered candidates (strong signal, near-duplicate, weak signal, noise,
constant, 35 % missing), 4 metadata/excluded columns held outside the candidate
frame, feature budget 4.

| Method | Mode | Selected | Budget status | Deterministic | Serialization exact |
|---|---|---|---|---|---|
| `iv_woe` | matched_budget | 4/4 | satisfied | yes | yes |
| `mrmr_mutual_information` | matched_budget | 4/4 | satisfied | yes | yes |
| `lasso_l1_logistic` | matched_budget | 1/4 | infeasible_natural_support | yes | yes |
| `random_k` | random_control | 4/4 | satisfied | yes | yes |
| `full_features` | full_control | 6/— | not_applicable | yes | yes |

The LASSO row is the informative one: at `C = 0.05` on this fixture the L1 support
holds a single feature, and the selector reports the budget as **unmet** rather
than padding it.

## Preservation evidence

- Prompt 6 package: **41/41 files byte-identical**; 0 changed, 0 added, 0 removed.
- Prompt 6 audit JSONs: unchanged.
- Repository validator: `active_results.status = passed`, 388 artifacts verified,
  20 registered runs, 12 removed paths verified absent.
- Frozen voting protocol still declares exactly `("rf_corr_mrmr", "boruta")`.
- `boruta`, `boruta_rfe`, `rfe`, `pca`, `domain_rule_baseline`, `none` all resolve
  unchanged.
- No write reached `results/` or `D:/ResearchFindings/results`.

## Known pre-existing issues

1. **The 14-versus-16 assertion.** `tests/test_manual_research_orchestration.py::test_existing_pilots_and_isolated_capacity_evidence_remain_unchanged`
   expects 14 `cdv1-0NN` run directories; there are 16 since Prompt 5 completed
   runs 015/016. Failing before and after Prompt 7. **Not edited** — it is a Prompt
   5 preservation guard, and quietly relaxing it is exactly what this audit trail
   exists to prevent. Needs an explicit decision outside Prompt 7.
2. **`clip/checkpointing.py`.** The handoff described this as an uncommitted
   unrelated user change to exclude. It was already committed in `e48a848` before
   Prompt 7 began, at the user's explicit instruction. Prompt 7 did not touch it
   and its diff was empty at Phase 0.
3. **No linter configured.** The repository has no ruff/mypy configuration or
   dependency, so gate 1 is not applicable rather than passing.

## Blockers

None. No NEEDS USER ACTION condition was met.

## Recommendation for Prompt 8

Register RFE, Boruta, and CatBoost-SHAP as `MethodDescriptor` entries and subclass
`LightweightSelector`. Its budget resolution, invariant checking, hashing, tie
handling, and leakage guarding are method-agnostic, so a new selector needs only
`_compute` returning `(ranking, raw_scores, natural_support)`.

Three specifics:

- **Boruta has a natural support** (confirmed / rejected / tentative). Model it the
  way LASSO models its L1 support: `natural` and `matched_budget` as distinct
  modes, with `infeasible_natural_support` when confirmed features fall short of
  the budget. Do not pad with tentative features except through a separately named
  predeclared mode.
- **The existing `BorutaSelector` and `RFESelector` must be preserved unchanged**,
  exactly as the legacy mRMR selector was — the `boruta` voter is part of the
  frozen voting protocol and all 16 runs depend on it.
- **CatBoost-SHAP needs an explicit thread and seed contract** and a declared
  policy for whether SHAP values come from training rows only. Record the estimator
  and SHAP variant in `implementation_id`, as MI-mRMR records its estimator.

Prompt 8 should also decide the 14-versus-16 assertion explicitly rather than
inheriting it a third time.

---

**No real research workload or OOT evaluation was executed in Prompt 7.**
**Prompt 8 has not started.**
