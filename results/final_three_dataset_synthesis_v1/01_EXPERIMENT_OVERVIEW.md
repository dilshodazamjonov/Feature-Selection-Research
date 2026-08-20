# Final Three-Dataset Experiment Synthesis: Updated Experimental Overview

## Result authority

The finalized scorecard combines the 45-row workbook base with the controlling values in [`tables/finalized_score_overrides.csv`](tables/finalized_score_overrides.csv). Home Credit LR uses pure mRMR at AUC 0.77; LendingClub LR uses LLM at AUC 0.74; LendingClub accuracy is 0.84 and Brier is 0.0623; Home Credit log loss is 0.29394 and Brier is 0.69732. Gini is derived consistently as `2 × AUC − 1`. The exact workbook base is [`inputs/workbook1_supplied_results.csv`](inputs/workbook1_supplied_results.csv), its SHA-256 is `c10225268d92cb1b794d9288e4f7bf99ac53340f734bf186a1cd3b101487f6f3`, and the source workbook SHA-256 is `2369ae8241ba9d1fe486d3c6193e35973ed74f495630996fc4d5189270bd247a`.

The workbook base contains 17 explicit LLM comparisons. Before the finalized overlay, strict `higher`/`lower` resolution gives 10 LLM-column wins and 4 retained best-FS rows. The finalized overlay replaces 4 workbook metric rows. No value is promoted merely because it is in the `LLM_score` column.

## Six unique dataset × model cases

`full_features` is excluded: these are feature-selection-method leaders. The base scorecard controls the CatBoost AUC cases; the finalized LR values control Home Credit LR and LendingClub LR; Stability 2024 LR remains the strongest non-full-feature sealed row. Gini is validated row by row as `2 × AUC − 1`.

| dataset | model | best feature-selection method | family | AUC | Gini | score source |
| --- | --- | --- | --- | --- | --- | --- |
| Home Credit | Logistic Regression | mRMR | classical | 0.770000 | 0.540000 | finalized scorecard 2026-08-21 |
| Home Credit | CatBoost | LLM | LLM-assisted | 0.793450 | 0.586900 | Workbook1 aggregate update |
| LendingClub v2 | Logistic Regression | LLM | LLM-assisted | 0.740000 | 0.480000 | finalized scorecard 2026-08-21 |
| LendingClub v2 | CatBoost | LLM then mRMR | LLM-assisted | 0.770664 | 0.541328 | Workbook1 aggregate update |
| Home Credit Stability 2024 | Logistic Regression | IV then Boruta | classical | 0.802956 | 0.605912 | historical sealed OOT registry |
| Home Credit Stability 2024 | CatBoost | LLM then mRMR | LLM-assisted | 0.869088 | 0.738177 | Workbook1 aggregate update |

Source: [`tables/updated_six_case_auc_gini.csv`](tables/updated_six_case_auc_gini.csv). This is the compact reviewer table for all six unique dataset/model cases.

## Which methods generalize across the six cases?

No single exact method wins all six. The **LLM family wins four cases**: plain LLM wins Home Credit CatBoost and LendingClub LR, while LLM then mRMR wins LendingClub CatBoost and Stability 2024 CatBoost. Pure mRMR wins Home Credit LR, and IV then Boruta wins Stability 2024 LR.

| method | method_family | case_wins | dataset_count | models | mean_auc | min_auc | max_auc | mean_gini | min_gini | max_gini |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM then mRMR | LLM-assisted | 2 | 2 | CatBoost | 0.819876 | 0.770664 | 0.869088 | 0.639752 | 0.541328 | 0.738177 |
| LLM | LLM-assisted | 2 | 2 | CatBoost; Logistic Regression | 0.766725 | 0.740000 | 0.793450 | 0.533450 | 0.480000 | 0.586900 |
| IV then Boruta | classical | 1 | 1 | Logistic Regression | 0.802956 | 0.802956 | 0.802956 | 0.605912 | 0.605912 | 0.605912 |
| mRMR | classical | 1 | 1 | Logistic Regression | 0.770000 | 0.770000 | 0.770000 | 0.540000 | 0.540000 | 0.540000 |

Source: [`tables/updated_cross_case_method_summary.csv`](tables/updated_cross_case_method_summary.csv). Win counts summarize point-estimate leaders, not statistical superiority.

## Dataset scope

| dataset | DEV period | DEV n | DEV event rate | OOT period | OOT n | OOT event rate | eligible features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Home Credit canonical temporal split | recent_decision day -600 through -241 | 99,092 | 0.0793 | recent_decision day -240 through -1 | 120,053 | 0.0890 | 391 |
| LendingClub v2 canonical temporal split | recent_decision day -1795 through -1096 | 598,649 | 0.1954 | recent_decision day -1065 through -730 | 293,105 | 0.2329 | 161 |
| Home Credit - Credit Risk Model Stability 2024 | 2019-01-01 through 2020-02-25 | 1,221,743 | 0.0324 | 2020-02-26 through 2020-10-05 | 304,916 | 0.0274 | 1,068 |

The third benchmark shares Home Credit lineage with the first and is not a fully independent institutional replication. Dataset targets, feature universes, prevalence, and temporal windows differ, so absolute scores are compared only within their stated dataset/model case.

## What this update does and does not establish

The current sources are aggregate scorecards. They support exact point-estimate tables, a discrete evidence-revision timeline, a winner matrix, and the metric panels in Figures 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17. They do not supply row-level finalized predictions or repeated calendar-time score slices. Therefore Figure 2 is explicitly a **revision timeline**, not performance through calendar time, and Figure 6 shows current aggregate log-loss/Brier summaries. Figure 16 uses winner-only AUC-matched reference profiles; Figure 17 shows winner-only calibration feasibility without fabricating probability-level evidence.

The machine-readable resolution is [`tables/updated_metric_leaders.csv`](tables/updated_metric_leaders.csv). For each metric it preserves the supplied best-FS method and score, optional LLM comparator and score, improvement direction, resolved winner, winning column, and comparison outcome.

## Reproducibility

- Repository branch/head at build: `main` / `e1588ce4ef35252fed3bf86e509a7100edadb4e8`.
- Workbook snapshot rows: 45 (15 metrics × 3 datasets).
- Finalized score overrides: 6 values, including 2 LR AUC cases.
- Current reviewer figures: 14 PNG files and 0 PDF files.
- Generated tables: 13.
- [`evidence_manifest.json`](evidence_manifest.json) records hashes; [`validation_audit.json`](validation_audit.json) records validation gates.
