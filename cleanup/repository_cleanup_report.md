# Repository cleanup report

## Scope and baseline

- Repository instructions: no `AGENTS.md` was present.
- Protected legacy boundary: `results/finalized_research`,
  `results/final_research_package_v2`, `results/research_summary`, and all saved
  artifacts under `results/`/`artifacts/`. No separately configured
  extension-results root was found; active configurations use `results/<run>`.
- Pre-existing dirty state: historical result/report files were already marked
  deleted and nine archived report copies were untracked. That migration was
  treated as user-owned and was not changed.
- Baseline tests: 344 passed, 31 failed. Every failure required missing saved
  CLIP inputs under `results/corrected_homecredit_clip`, `results/clip`, or
  `results/clip_v2`.
- Baseline repository validator: failed before validation because
  `results/research_summary/run_index.csv` was already absent.
- Baseline audit findings: `boruta` and `boruta_rfe` were duplicate aliases;
  RFE and sequential orchestration lived in `boruta.py`; selector result
  attributes were inconsistent; two selector modules were empty forwarders;
  the standalone deterministic rule script was named as an LLM implementation;
  and `MRMR` implemented RF relevance rather than canonical mutual-information
  relevance.

## Rename and ownership map

| Previous path/symbol | Canonical path/symbol | Reason and compatibility |
|---|---|---|
| `selectors/boruta.py::RFESelector` | `selectors/rfe.py::RFESelector` | RFE is a separate responsibility. The old import resolves lazily with a deprecation warning. |
| `selectors/boruta.py::BorutaRFESelector` | `selectors/boruta_then_rfe.py::BorutaThenRFESelector` | Sequential orchestration belongs in a combination module. The old class name and old-module import remain compatibility shims. |
| `MRMR` | `RandomForestRelevanceMRMRSelector` | The new name states the implemented relevance definition. `MRMR` and registry key `mrmr` remain aliases for historical callers/artifacts. |
| `llm_then_stat.py::DomainRuleBaselineSelector` | `domain_rule_baseline.py::DomainRuleBaselineSelector` | Removed a same-named empty forwarder and gave the standalone selector direct ownership. |
| `llm_then_stat.py::StableCoreLLMFillSelector` | `stable_core_llm_fill.py::StableCoreLLMFillSelector` | Removed a same-named empty forwarder and isolated the combination. |
| `scripts/generate_homecredit_llm_feature_ranking.py` implementation | `scripts/generate_homecredit_domain_rule_ranking.py` | The implementation is deterministic and makes no LLM call. The old filename is a small visible-deprecation wrapper. Historical ranking artifacts were not renamed or edited. |

## API, registry, and configuration changes

- `selected_features_` is the canonical fitted selector attribute.
  `SelectedFeaturesMixin.selected_features` is a documented bidirectional
  compatibility property.
- Shared helpers now validate duplicate/missing feature names, clamp
  non-negative budgets, select zero columns safely, and adapt legacy selectors.
- `boruta` now resolves directly to `BorutaSelector` (confirmed Boruta features
  only, with an optional cap and no feature backfill).
- `boruta_rfe` now resolves to `BorutaThenRFESelector` with `use_rfe=True`.
  `configs/selectors/boruta.yaml` records both distinct configurations.
- `llm_then_boruta` now composes the Boruta-only selector, not the former
  RFE-disabled orchestration wrapper.
- RF-importance ties and greedy custom-mRMR score ties use feature name as a
  deterministic secondary key. Registered and nested random seeds remain
  explicit.
- PCA now reports its fitted component names through `selected_features_`.
- No dependency, metric definition, DEV/OOT split definition, model
  hyperparameter, or historical result was changed.

## Implemented mRMR definition

The executable registered method is **not canonical mutual-information mRMR**.
It averages impurity-based relevance from 128-tree random forests across
`n_iter`, samples at most 10,000 training rows for redundancy calculations,
computes mean absolute configured correlation to the selected set, floors
redundancy at 0.05, and greedily maximizes relevance divided by redundancy.
Existing paper/report use of the unqualified term “mRMR” therefore does not
fully match the executable algorithm. The `mrmr` registry name is retained only
for backward compatibility; a canonical implementation must be added later
under a distinct class/registry entry.

## Duplicate and dead logic removed

- Removed duplicate selected-feature getter/setter logic from
  `llm_then_stat.py` and pipeline fallbacks; the compatibility adapter in
  `selectors/base.py` is canonical.
- Removed the empty forwarding behavior from `domain_rule_baseline.py` and
  `stable_core_llm_fill.py` by moving each implementation to its named module.
- Removed four unreferenced private artifact helpers from
  `evaluation/_feature_utils.py`:
  `_save_selected_features`, `_save_feature_statistics`,
  `_save_correlation_matrix`, and `_save_stagewise_selection`, plus the
  duplicate `_safe_get_selected_features`. Repository-wide code/config/test/doc
  searches found no callers or dynamic references. Current fold result writing
  remains in `models/_fold.py` and `pipelines/common.py`.
- Files deleted: none. The historical script filename remains as a compatibility
  entry point, and protected artifacts were not removed.

## Tests added or updated

- Added selector protocol, compatibility property/adapter, duplicate/missing
  name, zero-result, and pickle serialization tests.
- Added mocked Boruta/RFE tests for deterministic ties, oversized budgets,
  zero-feature behavior, and legacy imports without fitting expensive models.
- Added registry tests proving distinct Boruta-only and Boruta → RFE classes and
  configurations.
- Added custom-mRMR tests for deterministic RF ties and per-iteration seed
  propagation.
- Added deterministic domain-rule ranking and deprecated-script delegation
  tests, plus a source scan for stale pre-refactor imports.
- Updated the existing feature-budget test to match Boruta-only configuration.

## Final validation

| Check | Result |
|---|---|
| Focused selector/package tests | 50 passed |
| Non-CLIP suite | 259 passed |
| Reverse-transfer selector tests | 45 passed |
| Full `pytest tests -q` | 361 passed, 31 failed; the same 31 missing-result failures as baseline |
| `python -m compileall -q src scripts` | passed |
| `git diff --check` | passed (line-ending conversion warnings only) |
| Configured formatter/linter/type checker | none configured in `pyproject.toml`/repository |
| Repository validator | same pre-existing `results/research_summary/run_index.csv` missing-file failure |

The passing count increased by 17 new tests. No test that passed at baseline
regressed. No training run, feature-selection experiment, prediction,
embedding, or artifact generation command was executed.

## Protected-artifact verification

The final `git status -- results reports artifacts` matches the captured
pre-edit baseline: only the pre-existing historical deletions and nine
untracked `reports/archive` copies are listed, with no new protected-path change.
No command in this cleanup wrote to `results/`, `reports/`, or `artifacts/`.

## Remaining issues

- Restore or remap the externally migrated historical result bundle before the
  repository validator and 31 fixture-dependent CLIP tests can pass; do not
  regenerate those artifacts as part of source cleanup.
- Implement canonical mutual-information mRMR as a new method if scientifically
  required; do not change the preserved custom algorithm behind existing
  historical labels.
- Decide whether standalone RFE and `boruta_rfe` should enter a future matrix;
  they are executable but were not added to finalized experiment definitions.
- LASSO, CatBoost–SHAP, statistical voting, and normalized average-rank voting
  remain planned/missing.
- Establish a dedicated extension-results root before new experiments if future
  outputs must be physically isolated from restored legacy results.
