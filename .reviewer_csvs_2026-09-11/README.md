# Existing repository CSVs for the reviewer requests

Prepared 11 September 2026; revised to exclude all observation-level data at the user's request. This package contains **10 CSV files and this README**: aggregate results, DEV-fold AUCs, feature rankings, feature definitions and provenance. The ten observation-level HO/DEV score files and their exporter were deleted. The accompanying ZIP was rebuilt from the reduced package. No model was rerun and no missing result was invented.

HO in the request corresponds to **OOT** in the repository. CSVs use UTF-8. There are no borrower/observation records, IDs, labels, dates or individual probabilities in this package. Counts are aggregate counts; saved thresholds are model-level values. Empty metadata fields mean unavailable, not zero. Unavailable AUCs have no placeholder rows in the HO table.

**Files and shared uses**

| File | Contents | Requests served |
| --- | --- | --- |
| [ho_auc_by_run.csv](ho_auc_by_run.csv) | 118 existing numerical HO results, selector/backbone/configuration identity and saved model-level thresholds where available | 2, B1, C2, C4, D/Table 7 |
| [dev_fold_auc.csv](dev_fold_auc.csv) | 603 actual DEV validation-fold AUCs; excludes appended `mean`/`std` summary rows | B1 |
| [llm_cached_rankings.csv](llm_cached_rankings.csv) | Ranked feature names for six Home Credit, 24 LendingClub v2 and one Stability cache | B3 |
| [llm_ranking_inventory.csv](llm_ranking_inventory.csv) | One row per cache: dataset, fold/scope, requested budget, returned length, candidate count, prompt identity and source | B3 interpretation and provenance |
| [stability_feature_dictionary.csv](stability_feature_dictionary.csv) | Exact copy of the existing 1,959-row feature-semantics file | C8 |
| [stability_llm_final_selections.csv](stability_llm_final_selections.csv) | Final pure-LLM and stable-core/fill selections for both backbones, with core/fill membership | C2 explanation |
| [manuscript_auc_reconciliation.csv](manuscript_auc_reconciliation.csv) | Six updated manuscript-versus-run comparisons (five identified runs, one unmatched claim), plus six retained prior comparison entries | 2, B1 and C2 discrepancy context |
| [source_manifest.csv](source_manifest.csv) | Repository source paths, byte sizes and SHA-256 hashes | Traceability |
| [file_manifest.csv](file_manifest.csv) | Exported CSV row/column counts, sizes and hashes; excludes itself | Integrity |
| [validation_checks.csv](validation_checks.csv) | Checks of the retained aggregate/feature files and exclusion of observation-level data | Quality checks |

**2 — Full results matrix**

`ho_auc_by_run.csv` contains 24 recorded configurations per backbone for Home Credit and LendingClub v2, and 11 completed configurations per backbone for Stability: **118 rows total**. There are 22 distinct repository selector names across these groups. The repository contains several experiment generations and budget variants; an exact manuscript list defining the requested fourteen selectors was not provided or located. Consequently, this is an available-results handoff, not a padded 84-row manuscript table.

The table distinguishes `canonical_matrix` historical runs, `classical_extension` runs, and `stability_final_amended` runs. IV→Boruta P=100/200/300 are different configurations. Historical `mrmr`, later `legacy_rf_relevance_corr`, and `mrmr_mutual_information` retain their actual identities; they must not be collapsed by taking the largest HO AUC. Full-feature controls are explicitly labelled. Separate CLIP experiments, voting sensitivities, pilots, failed attempts and duplicate aggregate tables are outside this export's main-selector scope.

The historical 32 run summaries are available inside the earlier repository evidence package. Their original sources point to external backups, but those backups were not read for this handoff. The corresponding historical predictions are not present inside this repository. In particular, the historical Home Credit/LendingClub `llm` results are labelled as **archived screened-run evidence**, distinct from the claimed full-pool Pure LLM method. They cannot authenticate the latter's performance.

`manuscript_auc_reconciliation.csv` incorporates the latest six-row user update: five named saved runs and the unmatched Home Credit LR mRMR claim at 0.7699. Six other entries from the earlier comparison table are retained with `update_scope=retained_prior_claim_not_in_latest_user_table`. The five named runs already agree with `ho_auc_by_run.csv` within 1e-12; the authoritative saved AUCs and all DEV-fold results remain unchanged. The comparison CSV records the user-supplied saved AUC precision, run ID, linked result ID, exact manuscript-minus-saved difference, four-decimal difference, reported gap and previous manuscript value.

The updated manuscript AUCs are 0.8344 and 0.8784 for Stability LR/CatBoost Pure LLM; 0.7707 for LendingClub CatBoost LLM then mRMR; 0.7402 for LendingClub LR Pure LLM; 0.7935 for Home Credit CatBoost Pure LLM; and 0.7699 for Home Credit LR mRMR. These remain manuscript claims rather than replacements for saved-run measurements. Home Credit LR mRMR has no assigned matching run or invented gap for the 0.7699 claim. Its separately recorded historical `lr_statistical_mrmr_53a793cb32fe` result remains 0.74568891636559542.

For LendingClub LR, the current values give 0.7402 - 0.6926566628426155 = 0.0475433371573845, which rounds to **+0.0475**. The user's reported **+0.0476** is preserved in `user_reported_gap`; it is not substituted for the recalculated difference. Other supplied four-decimal gaps agree with the displayed inputs. Leader margins were not calculated because the exact fourteen-method mapping remains unresolved.

**B1 — DEV selection and paired testing**

The DEV file contains **240 Home Credit, 240 LendingClub v2, and 123 Stability** validation AUCs. Its grain is `(record_id, fold_id)`. Fold IDs are only 1–5. Existing CV summary rows labelled `mean` or `std` were removed. Each row retains its source and run identity.

Of Stability's 170 registered DEV cells, 123 have saved numerical results and 47 are unavailable. The export includes **22 completed DEV cells belonging to configurations without a completed HO result**, identified by `record_id=no_ho_cNNN`; these IDs intentionally have no HO-table row. It does not manufacture their missing folds or final scores.

Per-observation HO predictions were removed at the user's request. The retained aggregate HO AUCs and DEV-fold AUCs support descriptive comparisons and DEV leader selection; this package cannot supply a paired DeLong/bootstrap test without the excluded prediction records. Lookup columns pointing to the deleted files were removed as well.

A possible downstream DEV rule is: first freeze the eligible selector/configuration set and any LLM-versus-classical comparison groups; require all five authenticated DEV folds; rank by unweighted mean validation AUC; break exact ties lexicographically by selector, configuration and execution run ID. Resolve eligibility and protocol comparability before inspecting the resulting HO comparison. Do not silently rank a two-fold run against a five-fold run. Selection limited to configurations with available HO scores would itself need to be disclosed.

For all-pairs testing, define the multiplicity family in advance, for example every eligible pair within a dataset/backbone, and adjust the entire family. For a selected-pair design, state the selection rule and the family across cases. This handoff runs **no DeLong/bootstrap tests** and certifies no new confirmatory claim. Historical HO inspection and unmatched manuscript values cannot be undone by a retrospective selection rule. A complete fourteen-selector confirmatory analysis remains unsupported where matching DEV/HO evidence is missing.

NumPy and pandas are already installed in the repository's `.venv`; nothing was installed. All retained files are ordinary CSVs. The former observation-level exporter and its example score lookup were removed.

**B3 — Cached rankings for Table 7**

The export contains exactly **6 / 24 / 1 lists**. Home Credit has five DEV-fold requests plus a final-DEV request. LendingClub has four requested-budget series (20, 40, 60, 100), each with five DEV folds plus final DEV. Three additional LendingClub K20 fold-1/2/3 caches have different prompt hashes and are outside that complete 24-list grid; other earlier LendingClub cache variants are also omitted. The three omitted cache filenames start `fold_fold_1_dce42c`, `fold_fold_2_8513a6`, and `fold_fold_3_ab3d5b`. They are different requests, not byte-identical duplicates.

Rank is one-based, best first. Actual response length is preserved; requested budget and returned length are separate. The first 20/40 entries can be used without padding. The older caches are the saved **screened implementations**, not authenticated full-pool Pure LLM requests. Their candidate pools can change by fold. Do not pool all 24 LendingClub lists as identical repeated experiments: group the intended prompt/budget series and state the feature universe used for Nogueira. Final DEV is not a sixth validation fold.

Stability's single 100-entry sealed order was reused for K=20/40. There is only **one independent ranking**, so between-request pairwise Jaccard/Nogueira cannot be estimated. Identical fold selections caused by reusing that order do not establish robustness across fresh LLM requests. No new stability-statistic table was generated.

**B4 - Recent HO months, optional**

Observation-level scores and dates are excluded at the user's request. No recent-month AUC table was generated, so this optional analysis is not supplied. The original data uses loan issue months for LendingClub and decision dates for Stability; Home Credit lacks authenticated calendar decision dates. Stability's saved HO window is 2020-02-26 through 2020-10-05; 2024 is the dataset name.

**C2 and D — Stability LLM hybrids and the Table 7 gap**

Neither LLM→mRMR nor LLM→Boruta has an executed Stability result in the final registry, for either backbone. The [supplement amendment](../configs/protocols/homecredit_model_stability_2024_v2/prompt_16_llm_supplement_amendment.json) adds only `llm` and `stable_core_llm_fill` to the 30 classical configurations and prohibits additional LLM hybrids/sensitivities. This supports the sentence: **“LLM→Boruta was not evaluated on Stability 2024 under the final registered experiment; no corresponding performance or selection-stability row is available.”** The same execution gap applies to LLM→mRMR. No empty hybrid CSV was added.

The saved Stability LR results are **0.681477094750896** for stable core + LLM fill and **0.698422219778291** for pure LLM, a difference of approximately **−0.0169451**. The quoted **0.8344** is the manuscript claim. The latest user update identifies `p16v2-c031-llm-lr` as the corresponding saved run, whose AUC is 0.6984222197782909; it does not establish a saved run scoring 0.8344. The analogous CatBoost claim is 0.8784 versus 0.7663857376854197 for `p16v2-c032-llm-catboost`. Comparing 0.6815 against 0.8344 mixes manuscript and measured evidence.

The saved LR hybrid takes 15 stable-core features and five LLM additions; only nine of its 20 features overlap pure LLM's top 20. CatBoost takes 34 core features and six additions, with 14 of 40 overlapping pure LLM. These selection differences are documented in `stability_llm_final_selections.csv`. They explain why the methods are not equivalent, but they do not establish a causal explanation for the performance difference. No repository-authenticated explanation establishes the claimed 0.6815-versus-0.8344 gap.

Twelve other Stability HO cells are explicitly unavailable: both backbones for full features, LASSO, Boruta RF, normalized statistical average rank, Boruta→MI mRMR, and Boruta→CatBoost RFE. Full-feature failures inherited all five DEV resource stops; the other cases record `selector_resource_infeasible`. Their absent AUCs are not zero and are not supplied as result rows.

**C3 - Score PSI inputs, optional**

All per-observation DEV final-refit scores were deleted at the user's request. No replacement score vectors, OOF scores or PSI calculations are included. This optional recomputation is not supported by the reduced package.

**C4 - Frozen thresholds, optional**

`ho_auc_by_run.csv` retains 90 saved **model-level threshold values**. These are not borrower-level probabilities. All individual HO scores and labels were deleted, so TPR/FPR/precision cannot be recomputed from this reduced package. No threshold-metric table or default threshold was invented.

**C8 — Stability feature dictionary**

`stability_feature_dictionary.csv` is a byte-for-byte copy of `outputs/prompt_16_homecredit_model_stability_2024/clip_preparation_v1/metadata/feature_semantics.csv`. Its 1,959 unique feature rows include `feature_name`, `source_or_formula`, `source_table` (source family), `aggregation`, source-feature identity and descriptions. It contains 12 source families and 18 recorded aggregation operations, so counts by those fields are supported directly.

A complete normalized **window** field does not exist in this dictionary. Some names/descriptions contain time horizons; others do not. No horizons were inferred from names or filled with invented defaults. Thus C8 is partial: source/aggregation summaries and a lineage paragraph are supported, while a complete count-by-window table needs an authenticated window mapping.

**D — Institutional emails**

No institutional author email addresses were found in the inspected project/front-matter documentation. No contact CSV or invented address was added. Author-supplied addresses are still needed.

**Deduplication and verification**

The package keeps one copy of each aggregate result, feature dictionary and ranking list. C2 and D share one explanation; model-level thresholds stay in the HO table. Distinct selector implementations and budget variants retain their identities. No observation-level CSV or probability exporter remains, and the ZIP contains only the reduced package.

The retained 118 HO result rows and 603 DEV-fold rows were checked for their expected grain. DEV rows contain only folds 1-5, with no duplicate configuration/fold keys. Rankings retain the requested 6/24/1 lists and the dictionary contains 1,959 unique feature names. `validation_checks.csv` records checks for this reduced package. `file_manifest.csv` lists current CSV hashes and dimensions, excluding itself; `source_manifest.csv` retains provenance for aggregate and feature evidence. Paths and hashes are metadata, not copies of source observations. The latest comparison update comes from the user message dated 11 September 2026, recorded in the comparison CSV; it is not represented as an original repository experiment source.
