# Credit-risk feature-selection research

This repository contains executable Home Credit and LendingClub v2
feature-selection pipelines. New work should use `src/credit_risk_fs`,
`scripts`, the active configurations, and the active-results contract in
`results/README.md`; saved metrics, predictions, rankings, manifests,
checkpoints, embeddings, and splits are evidence, not regenerable source files.

Useful entry points:

- `docs/current_pipeline.md` — active dataset and pipeline boundary
- `docs/experiment_protocol.md` — methods, metrics, and non-goals
- `cleanup/audit/cleanup_summary.md` — prior artifact cleanup boundary
- `reports/archive/` — retained historical narrative reports

Historical finalized results have been moved to a separate immutable bundle and
are not part of the active `results/` tree. Its filesystem location is not
configured in this repository. The repository validator checks active results
by default; an explicitly supplied read-only legacy repository can be checked
with `--legacy-repository-root`.

## Feature-selection methods

`implemented` means the method is executable and wired to a registry, CLI, or
pipeline. `partial` means a component exists without a standalone/default
experiment path. `missing` means no executable implementation was found.

| Method | Canonical implementation | Status | Behavior |
|---|---|---|---|
| Information Value (IV) | third-party `IVWOEFilter`, used by `LLMSelector` | partial | Optional prefilter inside LLM and LLM-hybrid paths; no standalone registry entry. |
| RF-relevance/correlation-redundancy mRMR-like selection | `RandomForestRelevanceMRMRSelector` (`MRMR` compatibility alias) in `src/credit_risk_fs/selectors/mrmr.py` | implemented | Uses mean random-forest impurity importance for relevance and mean absolute Pearson correlation (floored at 0.05) for redundancy, then greedily maximizes relevance/redundancy. This is not canonical mutual-information mRMR. |
| Random-forest importance top-k | `RandomForestRelevanceMRMRSelector(method="rf")` | partial | Deterministic RF-importance ordering is implemented, but it has no separate registry/matrix entry. |
| LASSO | — | missing | No estimator, coefficient selector, configuration, or pipeline wiring exists. |
| RFE (CatBoost) | `RFESelector` in `src/credit_risk_fs/selectors/rfe.py` | partial | Standalone implementation exists for composition, but it is not a matrix selector. |
| Boruta | `BorutaSelector` in `src/credit_risk_fs/selectors/boruta.py` | implemented | The `boruta` registry entry keeps confirmed Boruta features only and may cap them to the model budget; it does not run RFE or backfill rejected features. |
| Boruta → RFE | `BorutaThenRFESelector` in `src/credit_risk_fs/selectors/boruta_then_rfe.py` | implemented, not in matrix | The `boruta_rfe` registry entry explicitly sets `use_rfe=True` and runs RFE only on Boruta-confirmed features. |
| CatBoost–SHAP | — (`shap` dependency only) | missing | No explainer, score calculation, selector, or pipeline wiring exists. |
| Fold-local LLM ranking | `LLMSelector` in `src/credit_risk_fs/selectors/llm_screening.py` | implemented | Applies training-slice missingness/optional IV filtering and obtains or reuses a cached OpenAI ranking. |
| Home Credit deterministic domain-rule ranking | `build_ranking` in `scripts/generate_homecredit_domain_rule_ranking.py` | partial | Applies checked-in rules to metadata and makes no LLM/network call. The historical `generate_homecredit_llm_feature_ranking.py` command is a deprecated wrapper. |
| LLM → custom mRMR-like / LLM → Boruta | `LLMThenStatSelector` in `src/credit_risk_fs/selectors/llm_then_stat.py` | implemented | Screens a broad fold-local LLM candidate pool before the configured statistical selector; the Boruta variant uses Boruta only. |
| Stable-core + LLM fill | `StableCoreLLMFillSelector` in `src/credit_risk_fs/selectors/stable_core_llm_fill.py` | implemented | Builds a bootstrap custom-mRMR-like stable core and fills the remaining budget from the LLM ranking. |
| Domain-rule baseline | `DomainRuleBaselineSelector` in `src/credit_risk_fs/selectors/domain_rule_baseline.py` | implemented | Ranks training-slice metadata by fixed semantic-group priority, missingness, coverage, and name. |
| PCA baseline | `PCASelector` in `src/credit_risk_fs/selectors/pca.py` | implemented | Produces named principal components rather than selecting original columns. |
| CLIP text-only semantic baseline | `build_text_baseline` in `src/credit_risk_fs/clip/text_baseline.py` | partial | Creates frozen text-embedding ranking artifacts; it is not a matrix selector. |
| Corrected Home Credit CLIP → custom mRMR-like | `FixedRankThenMRMRSelector` and `scripts/run_corrected_homecredit_clip_pipelines.py` | implemented | Freezes a corrected CLIP candidate ranking and applies the DEV-only custom selector. |
| Reverse-transfer CLIP → custom mRMR-like | `FixedRankThenMRMRSelector` and `src/credit_risk_fs/pipelines/reverse_transfer.py` | implemented | Freezes aligned seed consensus/candidate pools and applies the DEV-only custom selector. |
| Full-feature control | `get_selector("none")` | implemented | Keeps every post-preprocessing feature; it is not a matrix entry. |
| Random-k control | — | missing | No selector samples k columns at random. |
| Statistical majority voting | — | missing | No cross-selector vote aggregator exists. |
| Normalized average-rank voting | — | missing | Existing CLIP seed-score averaging and one rule ranker's normalized rank are not average-rank voting. |

The executable algorithm historically called `mrmr` is therefore mRMR-like,
not canonical mRMR. Existing paper/report terminology does not fully describe
the implementation: it should be read as the registered historical method
label, while the code now exposes the accurate internal class name. A future
mutual-information implementation must receive a separate class and registry
entry rather than silently replacing this algorithm.

## Selector interface and compatibility

All public selectors expose fitted output names through `selected_features_`.
The former `selected_features` spelling remains as a documented compatibility
property, and old imports of `RFESelector`/`BorutaRFESelector` from
`selectors.boruta` resolve lazily with a deprecation warning. The old
`BorutaRFESelector` class name is retained; canonical new code should import
`BorutaThenRFESelector` from `selectors.boruta_then_rfe`.

## Planned combinations

- Available for future opt-in use, but not part of a finalized matrix:
  `boruta_rfe` (Boruta → RFE).
- Planned only: IV → Boruta, IV → custom mRMR-like, IV → RFE, and Boruta →
  custom mRMR-like.
- Planned only: statistical majority voting and normalized average-rank voting.

No LASSO or CatBoost–SHAP selector was added during this cleanup.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe cleanup/tools/validate_repository_state.py --root .
.\.venv\Scripts\python.exe -m compileall src scripts
git diff --check
```

The repository validator requires `results/run_index.csv` and the active
top-level results directories. It does not require or recreate the former
`results/research_summary/` hierarchy.
