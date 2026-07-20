# Credit-risk feature-selection research

The repository contains completed Home Credit and LendingClub v2 experiments
covering statistical, LLM-assisted, corrected contrastive, and directional
transfer feature-selection pipelines.

Start with:

- `results/finalized_research/README.md` for the canonical research index
- `results/finalized_research/STATUS.md` for completed and pending work
- `results/final_research_package_v2/final_research_report.md` for the report
- `results/research_summary/results_access_guide.md` for registry access

The scientific outputs are immutable saved artifacts. Do not rerun training,
feature selection, predictions, embeddings, checkpoints, or data splits to
reproduce the report.

## Feature-Selection Methods

This inventory reflects executable repository code, not method names found only
in reports or saved artifacts. `implemented` means the method is executable and
wired to a selector registry, CLI, or pipeline; `partial` means a component or
ranker exists but is not a complete standalone/default path; `notebook-only`
means executable code exists only in a notebook; and `missing` means no
executable implementation was found. No notebook-only implementations or
notebooks were found in the audited tree.

| Method | Class/function name | Source file | Status | One-sentence description |
|---|---|---|---|---|
| Information Value (IV) | `IVWOEFilter` (third-party), called by `LLMSelector.fit` | `src/credit_risk_fs/selectors/llm_screening.py` | partial | IV/WOE is an optional prefilter inside the LLM and LLM-hybrid paths, but it has no standalone repository selector or registry entry. |
| mRMR | `MRMR` | `src/credit_risk_fs/selectors/mrmr.py` | implemented | Uses random-forest importance for relevance and correlation for redundancy, with registered model-specific top-k budgets. |
| Random-forest importance top-k | `MRMR(method="rf")` | `src/credit_risk_fs/selectors/mrmr.py` | partial | The mRMR class can return the top-k random-forest importances, but this mode has no separate registry or matrix entry. |
| LASSO | — | — | missing | No LASSO estimator, coefficient-based selector, configuration, or pipeline wiring was found. |
| RFE (CatBoost) | `RFESelector` | `src/credit_risk_fs/selectors/boruta.py` | partial | A CatBoost-backed RFE class exists, but it is internal, unregistered as a standalone method, and disabled in default runs. |
| Boruta | `BorutaSelector`; registered through `BorutaRFESelector` | `src/credit_risk_fs/selectors/boruta.py` | implemented | Runs random-forest Boruta and, by default, caps Boruta's ordered support/ranking to the configured feature budget without RFE. |
| Boruta → RFE | `BorutaRFESelector(use_rfe=True)` | `src/credit_risk_fs/selectors/boruta.py` | partial | The sequential path is executable as an opt-in, but `use_rfe` is false in both registry defaults and `configs/selectors/boruta.yaml`, so it is absent from the research matrix. |
| CatBoost–SHAP | — | `pyproject.toml` (dependency only) | missing | The `shap` package is declared, but no SHAP import, explainer, score calculation, selector, or pipeline was found. |
| Fold-local LLM ranking | `LLMSelector` | `src/credit_risk_fs/selectors/llm_screening.py` | implemented | Applies missingness and optional IV filtering, obtains or reuses a cached OpenAI ranking, and selects the configured top feature budget. |
| Home Credit “LLM semantic” rule ranking | `build_ranking` | `scripts/generate_homecredit_llm_feature_ranking.py` | partial | Despite its filename and output terminology, this is a deterministic hard-coded 529-feature rule ranker with no LLM call or selector integration. |
| LLM → mRMR / LLM → Boruta | `LLMThenStatSelector` | `src/credit_risk_fs/selectors/llm_then_stat.py` | implemented | Screens a broad LLM candidate pool before a fold-local statistical selector; the Boruta variant uses the default RFE-disabled Boruta wrapper. |
| Stable-core + LLM fill | `StableCoreLLMFillSelector` | `src/credit_risk_fs/selectors/llm_then_stat.py` | implemented | Builds a bootstrap-mRMR stable core and fills the remaining budget from the LLM ranking. |
| Domain-rule baseline | `DomainRuleBaselineSelector` | `src/credit_risk_fs/selectors/llm_then_stat.py` | implemented | Ranks metadata by fixed semantic-group priorities, missingness, coverage, and name, and is included in the statistical matrix. |
| PCA baseline | `PCASelector` | `src/credit_risk_fs/selectors/pca.py` | implemented | Produces principal components rather than selecting original columns and is included in the statistical matrix. |
| CLIP text-only semantic baseline | `build_text_baseline` | `src/credit_risk_fs/clip/text_baseline.py` | partial | Creates anchor-similarity ranking artifacts from frozen text embeddings, but explicitly does not provide a matrix-integrated selector or final feature subset. |
| Corrected Home Credit CLIP → mRMR | `FixedRankThenMRMRSelector`; `main` | `src/credit_risk_fs/selectors/fixed_rank_then_mrmr.py`<br>`scripts/run_corrected_homecredit_clip_pipelines.py` | implemented | Freezes a corrected CLIP consensus candidate ranking, optionally intersects it with an LLM-approved pool, and applies DEV-only mRMR. |
| Reverse-transfer CLIP → mRMR | `aggregate_seed_embeddings`; `fixed_candidate_pool`; `FixedRankThenMRMRSelector` | `src/credit_risk_fs/clip/reverse_transfer.py`<br>`src/credit_risk_fs/pipelines/reverse_transfer.py` | implemented | Aligns and mean-aggregates five frozen CLIP seed scores, freezes model-specific candidate pools, and applies Home Credit DEV-only mRMR. |
| Full-feature/no-selection control | `get_selector("none")` | `src/credit_risk_fs/selectors/registry.py`<br>`src/credit_risk_fs/pipelines/common.py` | implemented | The `none` registry option returns no selector, causing the pipeline to retain every post-preprocessing feature, although it is not a research-matrix entry. |
| Fixed-k budget controls | `resolve_feature_budget`; `apply_feature_budget_to_selector_kwargs` | `src/credit_risk_fs/experiments/config.py` | implemented | LR and CatBoost feature budgets are propagated into supported selectors, with separate LLM/CLIP candidate-pool sizes where configured. |
| Random-k control | — | — | missing | No selector that samples k feature columns at random was found. |
| Statistical majority voting | — | — | missing | No cross-selector majority-vote feature aggregator was found; stable-core bootstrap frequency is a single-method stability procedure, not this vote. |
| Normalized average-rank voting | — | — | missing | Normalized ranks are emitted by one rule-ranking script, while CLIP averages seed scores before ranking; neither implements normalized average-rank voting. |

### Planned Combinations

- **Already present, opt-in:** Boruta → RFE is available through
  `BorutaRFESelector(use_rfe=True)`, but all shipped defaults disable RFE and the
  `boruta_rfe` registry alias does not enable it.
- **Planned only:** IV → Boruta, IV → mRMR, IV → RFE, and Boruta → mRMR have no
  direct executable pipeline; the existing IV → LLM → statistical paths are not
  equivalent to these combinations.
- **Planned only:** statistical majority voting and normalized average-rank
  voting have no executable aggregator. The five-seed CLIP consensus is an
  arithmetic mean of scores followed by ranking, not average-rank voting.

Implementation naming is not fully uniform: Boruta classes expose
`selected_features` while mRMR and the common selector protocol use
`selected_features_`; `boruta` and `boruta_rfe` currently resolve to identical
RFE-disabled defaults; and `DomainRuleBaselineSelector` and
`StableCoreLLMFillSelector` are implemented in `llm_then_stat.py` but re-exported
from same-named modules. The standalone 529-feature “LLM semantic” script is a
separate deterministic rule implementation, not a duplicate of the API-backed
`LLMSelector`.

Validate the current repository state:

```powershell
.\.venv\Scripts\python.exe cleanup/tools/validate_repository_state.py --root .
.\.venv\Scripts\python.exe -m pytest tests -q
```

See `results/finalized_research/reproduction/` for the complete validation and
saved-artifact report-build boundary.
