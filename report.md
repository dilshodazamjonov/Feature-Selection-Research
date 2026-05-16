# Research Experiment Report

Generated from `results_full_run/` on 2026-05-16.

## Scope

This report summarizes the completed full experiment matrix under `results_full_run/`.

- Completed runs: `16`
- Failed runs: `0`
- Models: `lr`, `catboost`
- Selector families: `statistical`, `llm`, `hybrid`
- Candidate feature pool: `529` engineered features
- Feature budgets:
  - `lr`: `20`
  - `catboost`: `40`

The full run matrix and failure log exist locally as:

- `results_full_run/matrix_runs.csv`
- `results_full_run/failed_runs.csv`

This report is also self-contained now: the important CSV content is embedded later in a portable appendix as raw `csv` blocks, so you can paste this file into another ChatGPT session without losing the result tables.

## Paper Artifact Index

### Main CSVs

- `final_comparison_table.csv`
  - Main cross-run comparison table.
  - Use this for the headline performance/stability/drift table in the paper.
- `paired_fold_comparisons.csv`
  - Fold-paired comparison against the `mrmr` baseline with mean deltas and 95% CIs.
  - Use this for the “is the method actually better than the baseline?” section.
- `llm_call_summary.csv`
  - LLM usage, cache reuse, and token accounting.
  - Use this for the cost/reproducibility discussion.
- `plot_reports/all/oot_metric_table.csv`
  - Compact OOT metrics table aligned with the plotting outputs.
- `plot_reports/all/stability_metric_table.csv`
  - Compact stability table aligned with the stability plots.
- `plot_reports/all/monthly_metric_table.csv`
  - Fold/month-level table used by the temporal trend plots.

### Figures

Local plot image files exist under `results_full_run/plot_reports/all/`:

- `oot_performance_comparison.png`
- `stability_comparison.png`
- `performance_vs_stability.png`
- `feature_count_vs_gini.png`
- `selected_feature_psi_comparison.png`
- `model_score_psi_comparison.png`
- `lift_at_10_comparison.png`
- `monthly_gini_trend.png`
- `monthly_psi_trend.png`
- `monthly_lift_trend.png`

These image files are not embedded visually here because pasted markdown in ChatGPT web will not render local PNG paths. Their meaning is described in the plot sections below.

## Executive Summary

The strongest overall baseline in this run is still `mrmr`, especially when the question is fold-level consistency and not just the final OOT snapshot. The best final OOT metrics in both model families came from `stable_core_llm_fill`, but the paired fold comparison against `mrmr` shows:

- `catboost`: `stable_core_llm_fill` is effectively tied with `mrmr`; the confidence interval crosses zero.
- `lr`: `stable_core_llm_fill` is slightly better on the final OOT snapshot, but worse than `mrmr` on paired CV folds; the confidence interval is below zero.

That means the safe publication claim is not “LLM beats feature selection.” The defensible claim is:

`LLM-based metadata screening is competitive, produces low-drift feature sets, and hybridization can slightly improve final OOT performance in some settings, while mRMR remains the strongest and most consistent baseline.`

Pure `llm` is competitive but not the best overall method. `llm_then_boruta` is weak and should not be a headline method. `pca` and `domain_rule_baseline` can look artificially strong on stability because their selected representation is fixed across folds; they should be treated as context baselines, not the main comparison target for raw feature selection quality.

## Recommended Paper Tables

### Table 1. Main OOT Results

Source CSVs:

- embedded below as `final_comparison_table_compact.csv`
- embedded below as `oot_metric_table.csv`

Suggested primary ranking criterion: `oot_gini`, with `oot_auc`, `oot_ks`, `lift_at_10`, `selected_feature_psi_mean`, and `model_score_psi` as supporting metrics.

### CatBoost Results

| selector | type | budget | OOT AUC | OOT Gini | OOT KS | Lift@10 | feature PSI mean | score PSI | Nogueira | mean Jaccard | runtime sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stable_core_llm_fill | hybrid | 40 | 0.769498 | 0.538996 | 0.404988 | 3.441042 | 0.012256 | 0.003348 | 0.707914 | 0.576283 | 3632.844800 |
| mrmr | statistical | 40 | 0.768669 | 0.537337 | 0.402893 | 3.435428 | 0.018889 | 0.005355 | 0.729550 | 0.603699 | 850.482371 |
| llm_then_boruta | hybrid | 40 | 0.764485 | 0.528970 | 0.394944 | 3.389585 | 0.011055 | 0.003399 | 0.459100 | 0.336871 | 547.880246 |
| llm_then_mrmr | hybrid | 40 | 0.762283 | 0.524567 | 0.391540 | 3.368067 | 0.010593 | 0.003025 | 0.418533 | 0.303317 | 451.450253 |
| llm | llm | 40 | 0.761863 | 0.523725 | 0.390337 | 3.346549 | 0.003640 | 0.003853 | 0.461805 | 0.342279 | 324.385257 |
| domain_rule_baseline | statistical | 40 | 0.734483 | 0.468966 | 0.345606 | 3.073361 | 0.003293 | 0.005877 | 1.000000 | 1.000000 | 270.931609 |
| pca | statistical | 40 | 0.707112 | 0.414223 | 0.305578 | 2.630835 | 0.029159 | 0.014564 | 1.000000 | 1.000000 | 312.651617 |
| boruta | statistical | 40 | 0.688705 | 0.377411 | 0.277538 | 2.552246 | 0.006240 | 0.001138 | 0.564576 | 0.452871 | 1141.664843 |

Interpretation:

- `stable_core_llm_fill` is the best final OOT CatBoost configuration by Gini, AUC, KS, and Lift@10.
- The absolute gain over `mrmr` is small:
  - Gini: `0.538996` vs `0.537337`
  - AUC: `0.769498` vs `0.768669`
- `mrmr` is still the strongest CatBoost baseline because it is slightly more stable and much cheaper than `stable_core_llm_fill`.
- Pure `llm` is competitive and has lower feature drift than `mrmr`, but it does not beat `mrmr` on performance.
- `llm_then_boruta` and `boruta` are clearly weaker than `mrmr`.

### Logistic Regression Results

| selector | type | budget | OOT AUC | OOT Gini | OOT KS | Lift@10 | feature PSI mean | score PSI | Nogueira | mean Jaccard | runtime sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stable_core_llm_fill | hybrid | 20 | 0.748857 | 0.497715 | 0.368780 | 3.103300 | 0.012572 | 0.005034 | 0.729784 | 0.593849 | 1451.738020 |
| mrmr | statistical | 20 | 0.745689 | 0.491378 | 0.361827 | 3.095815 | 0.013276 | 0.006494 | 0.771356 | 0.641672 | 328.610049 |
| llm | llm | 20 | 0.738122 | 0.476243 | 0.354482 | 3.003193 | 0.001614 | 0.007079 | 0.636248 | 0.495259 | 836.553030 |
| llm_then_mrmr | hybrid | 20 | 0.733895 | 0.467790 | 0.347336 | 2.937703 | 0.007912 | 0.003841 | 0.490747 | 0.352022 | 86.321099 |
| domain_rule_baseline | statistical | 20 | 0.724896 | 0.449792 | 0.334621 | 2.871277 | 0.003022 | 0.011185 | 1.000000 | 1.000000 | 17.838322 |
| pca | statistical | 20 | 0.672903 | 0.345805 | 0.259738 | 2.269703 | 0.035985 | 0.029073 | 1.000000 | 1.000000 | 16.719139 |
| llm_then_boruta | hybrid | 20 | 0.650827 | 0.301654 | 0.218360 | 2.190179 | 0.007856 | 0.001253 | 0.506336 | 0.362974 | 240.702781 |
| boruta | statistical | 20 | 0.630620 | 0.261241 | 0.187376 | 2.035809 | 0.004064 | 0.002392 | 0.521925 | 0.409820 | 957.607561 |

Interpretation:

- `stable_core_llm_fill` is the best final OOT LR configuration by Gini, AUC, KS, and Lift@10.
- The absolute gain over `mrmr` is again small:
  - Gini: `0.497715` vs `0.491378`
  - AUC: `0.748857` vs `0.745689`
- Pure `llm` is better than `llm_then_mrmr` for LR in this run.
- `llm_then_boruta` is poor and should not be emphasized.
- `boruta` is the weakest LR baseline by a wide margin.

## Recommended Paper Table 2. Stability Results

Source CSV:

- embedded below as `stability_metric_table.csv`

Key interpretation:

- Among comparable selectors, `mrmr` is the most stable method in both model families.
- `stable_core_llm_fill` is the strongest LLM-assisted method on stability.
- Pure `llm` is more stable than `llm_then_mrmr` and `llm_then_boruta`.
- `pca` and `domain_rule_baseline` have perfect stability because their representation or rule set is fixed; that does not mean they are better raw-feature selectors.

Headline stability takeaways:

- `catboost + mrmr`: Nogueira `0.729550`, Jaccard `0.603699`
- `catboost + stable_core_llm_fill`: Nogueira `0.707914`, Jaccard `0.576283`
- `lr + mrmr`: Nogueira `0.771356`, Jaccard `0.641672`
- `lr + stable_core_llm_fill`: Nogueira `0.729784`, Jaccard `0.593849`

## Recommended Paper Table 3. Paired Fold Comparison Against `mrmr`

Source CSV:

- embedded below as `paired_fold_comparisons.csv`

This is the most important “do the improvements hold up?” table. The baseline is `mrmr` within each model family.

### LLM-Family Comparisons Only

| model | candidate | metric | mean delta vs `mrmr` | 95% CI lower | 95% CI upper | interpretation |
|---|---|---|---|---|---|---|
| catboost | llm | auc | -0.006140 | -0.015337 | 0.003058 | not clearly different from `mrmr` |
| catboost | llm | gini | -0.012279 | -0.030674 | 0.006116 | not clearly different from `mrmr` |
| catboost | llm_then_mrmr | auc | -0.001722 | -0.010787 | 0.007343 | essentially tied with `mrmr` |
| catboost | llm_then_mrmr | gini | -0.003444 | -0.021575 | 0.014686 | essentially tied with `mrmr` |
| catboost | stable_core_llm_fill | auc | 0.000651 | -0.004154 | 0.005456 | essentially tied with `mrmr` |
| catboost | stable_core_llm_fill | gini | 0.001301 | -0.008309 | 0.010911 | essentially tied with `mrmr` |
| catboost | llm_then_boruta | auc | -0.008793 | -0.014668 | -0.002918 | worse than `mrmr` |
| catboost | llm_then_boruta | gini | -0.017586 | -0.029336 | -0.005836 | worse than `mrmr` |
| lr | llm | auc | -0.008368 | -0.011767 | -0.004968 | worse than `mrmr` |
| lr | llm | gini | -0.016735 | -0.023535 | -0.009936 | worse than `mrmr` |
| lr | llm_then_mrmr | auc | -0.006155 | -0.011862 | -0.000448 | worse than `mrmr` |
| lr | llm_then_mrmr | gini | -0.012310 | -0.023724 | -0.000896 | worse than `mrmr` |
| lr | stable_core_llm_fill | auc | -0.004208 | -0.007142 | -0.001274 | worse than `mrmr` |
| lr | stable_core_llm_fill | gini | -0.008416 | -0.014285 | -0.002547 | worse than `mrmr` |
| lr | llm_then_boruta | auc | -0.013547 | -0.023424 | -0.003670 | worse than `mrmr` |
| lr | llm_then_boruta | gini | -0.027093 | -0.046847 | -0.007339 | worse than `mrmr` |

Interpretation:

- `catboost`: the best LLM-assisted method is `stable_core_llm_fill`, but it is a tie with `mrmr`, not a clear win.
- `lr`: every LLM-family method underperforms `mrmr` in the paired fold analysis.
- This is why the publication claim should stay conservative.

## Recommended Paper Table 4. Drift Results

Source CSVs:

- embedded below as `final_comparison_table_compact.csv`
- summarized below as `monthly_metric_summary.csv`

Key drift observations:

- Pure `llm` produces the lowest selected feature PSI in both model families among the competitive methods.
- `mrmr` is stronger on predictive performance but tends to use slightly higher-drift features than pure `llm`.
- `stable_core_llm_fill` gives a better performance/stability trade-off than pure `llm` in this run.

### Feature Drift Summary by Model

#### CatBoost

Lowest feature PSI means:

- `domain_rule_baseline`: `0.003293`
- `llm`: `0.003640`
- `boruta`: `0.006240`
- `llm_then_mrmr`: `0.010593`
- `llm_then_boruta`: `0.011055`
- `stable_core_llm_fill`: `0.012256`
- `mrmr`: `0.018889`
- `pca`: `0.029159`

#### Logistic Regression

Lowest feature PSI means:

- `llm`: `0.001614`
- `domain_rule_baseline`: `0.003022`
- `boruta`: `0.004064`
- `llm_then_boruta`: `0.007856`
- `llm_then_mrmr`: `0.007912`
- `stable_core_llm_fill`: `0.012572`
- `mrmr`: `0.013276`
- `pca`: `0.035985`

Interpretation:

- If your story is “LLM is a first screening helper that favors semantically stable variables,” the drift numbers support that framing better than the fold-level superiority claim.

## Recommended Paper Table 5. Runtime and LLM Cost

Source CSV:

- embedded below as `llm_call_summary.csv`

### LLM Usage Summary

- Actual LLM ranking calls made: `6`
- Cache hits: `42`
- Prompt tokens: `302,333`
- Completion tokens: `29,775`
- Total tokens: `332,108`

Important note:

- Only the `lr + llm` run made real LLM calls in this execution.
- All remaining LLM and hybrid runs reused the cached fold/dev rankings.
- This is efficient and reproducible, but it also means the paper should explicitly state that the shared ranking cache was enabled.

### Runtime Summary

- Total wall-clock runtime across all completed runs: `3.186` hours
- CatBoost family total: `2.092` hours
- LR family total: `1.093` hours

Observations:

- `lr + llm_then_mrmr` was very fast: `86.321` seconds
- `catboost + llm` was faster than `catboost + mrmr`: `324.385` vs `850.482` seconds
- `stable_core_llm_fill` was the slowest competitive method in both families

This supports a secondary practical claim:

`LLM screening can reduce search cost in some settings, but the most stable hybrid may still be computationally expensive.`

## Plot Descriptions

### 1. OOT Performance Comparison

Local plot file:

- `results_full_run/plot_reports/all/oot_performance_comparison.png`

Supporting CSV:

- embedded below as `oot_metric_table.csv`

Description:

- This figure compares final OOT AUC, Gini, KS, precision, recall, F1, accuracy, and Lift@10 across all 16 runs.
- It should be used as the main “who performed best on the held-out period?” chart.

Takeaway:

- `stable_core_llm_fill` is the top final OOT method in both `lr` and `catboost`.
- `mrmr` is a very close second and remains the strongest baseline.
- `llm_then_boruta`, `boruta`, and `pca` are visibly weaker.

### 2. Stability Comparison

Local plot file:

- `results_full_run/plot_reports/all/stability_comparison.png`

Supporting CSV:

- embedded below as `stability_metric_table.csv`

Description:

- This figure compares Nogueira stability, Kuncheva stability, pairwise Jaccard, and stable-feature ratios across runs.
- It is the main figure for the “selector robustness across temporal folds” discussion.

Takeaway:

- Among comparable methods, `mrmr` is the most stable.
- `stable_core_llm_fill` is the strongest LLM-assisted method on stability.
- `pca` and `domain_rule_baseline` are perfect by construction and should be discussed with caution.

### 3. Performance vs Stability

Local plot file:

- `results_full_run/plot_reports/all/performance_vs_stability.png`

Description:

- This scatter plot positions each run by OOT performance and stability.
- It is useful for arguing about trade-offs rather than single-metric wins.

Takeaway:

- `mrmr` and `stable_core_llm_fill` occupy the best trade-off region.
- Pure `llm` is reasonably competitive but sits below them on performance.
- `llm_then_boruta` is dominated.

### 4. Feature Count vs Gini

Local plot file:

- `results_full_run/plot_reports/all/feature_count_vs_gini.png`

Description:

- This plot relates selected feature count to final OOT Gini.
- In this run the budgets are fixed within model family, so the figure is mostly useful for showing `20`-feature LR vs `40`-feature CatBoost families rather than internal selector differences.

Takeaway:

- The LR family operates on `20` selected features and the CatBoost family on `40`.
- Within each family, differences are due to selector quality rather than feature-count variation.

### 5. Selected Feature PSI Comparison

Local plot file:

- `results_full_run/plot_reports/all/selected_feature_psi_comparison.png`

Description:

- This figure compares the average distribution shift of the selected features between DEV and OOT.
- It is the most useful plot for supporting the “LLM screening chooses stable semantics” argument.

Takeaway:

- Pure `llm` has very low feature drift in both model families.
- `mrmr` performs better predictively but uses slightly higher-drift features.
- `stable_core_llm_fill` gives a balanced compromise.

### 6. Model Score PSI Comparison

Local plot file:

- `results_full_run/plot_reports/all/model_score_psi_comparison.png`

Description:

- This figure compares PSI on the model score distribution itself.
- It measures whether the full model output distribution shifts from DEV to OOT.

Takeaway:

- Competitive methods generally keep model score PSI low.
- `pca` is notably worse here, especially for LR.
- The LLM-based methods do not show an obvious instability penalty on score shift.

### 7. Lift@10 Comparison

Local plot file:

- `results_full_run/plot_reports/all/lift_at_10_comparison.png`

Description:

- This figure compares top-decile ranking quality.
- It is useful if the paper wants a credit-risk operations angle instead of only AUC/Gini.

Takeaway:

- `catboost + stable_core_llm_fill` and `catboost + mrmr` are the strongest top-decile rankers.
- `lr + stable_core_llm_fill` and `lr + mrmr` are also very close.

### 8. Monthly Gini Trend

Local plot file:

- `results_full_run/plot_reports/all/monthly_gini_trend.png`

Supporting CSV:

- summarized below as `monthly_metric_summary.csv`

Description:

- This figure plots fold/month-level Gini over time windows.
- It shows whether a method’s performance is stable across successive temporal slices rather than only on the final OOT block.

Takeaway:

- `catboost + stable_core_llm_fill` and `catboost + mrmr` remain strongest across folds.
- `lr + mrmr` has the strongest mean fold-level Gini among LR methods.
- This trend figure agrees with the paired fold comparison more than with the final OOT snapshot.

### 9. Monthly PSI Trend

Local plot file:

- `results_full_run/plot_reports/all/monthly_psi_trend.png`

Description:

- This figure tracks fold-level model PSI over time windows.
- It highlights whether a method becomes less reliable in later temporal buckets.

Takeaway:

- Competitive methods stay within a fairly low score-shift range.
- `pca` shows worse shift behavior than the stronger selectors.

### 10. Monthly Lift Trend

Local plot file:

- `results_full_run/plot_reports/all/monthly_lift_trend.png`

Description:

- This figure tracks top-decile lift across temporal folds.
- It supports the operational ranking story across time.

Takeaway:

- `catboost + mrmr` and `catboost + stable_core_llm_fill` are the strongest month-to-month top-decile performers.
- `lr + mrmr` and `lr + stable_core_llm_fill` are again the strongest LR methods.

## Best Publication Narrative

Based on this run alone, the strongest paper narrative is:

1. `mrmr` is the strongest and most consistent classical baseline.
2. Pure `llm` is competitive and produces low-drift feature sets, supporting its role as a metadata-based screening helper.
3. `stable_core_llm_fill` is the most promising hybrid because it can slightly improve final OOT performance while remaining close to `mrmr` on stability.
4. The evidence does not support a strong claim that pure LLM selection universally outperforms statistical feature selection.

The most defensible wording is:

`LLM-assisted metadata screening is a practical first-stage selector that can produce competitive and drift-aware feature sets, while hybridization with statistical methods offers the best chance of modest OOT gains.`

The wording to avoid is:

`LLM is better than standard feature selection methods.`

## What Is Strong Enough for the Paper

Strong points from this run:

- Full matrix completed with `16` successful runs and `0` failures.
- Clear baseline winner: `mrmr`.
- Clear hybrid candidate: `stable_core_llm_fill`.
- Clear cost/reuse story: `6` actual LLM calls, `42` cache hits.
- Clear drift story: pure `llm` has low feature PSI.
- Clear negative controls: `boruta`, `llm_then_boruta`, and `pca` are weaker.

What is still weak for publication:

- This is still a single-dataset study.
- The paired fold analysis does not show a robust win over `mrmr`.
- The LLM story is stronger as a screening/stability helper than as a performance winner.

## Suggested Wording for the Results Section

Suggested concise wording:

`Across both LR and CatBoost model families, mRMR remained the strongest and most consistent statistical baseline. Pure LLM screening produced competitive but not leading predictive performance, while selecting notably low-drift feature sets. Among the hybrid methods, stable_core_llm_fill achieved the best final OOT discrimination in both model families, although paired fold comparisons indicated that these gains were small and not consistently superior to mRMR. Taken together, the results support the use of LLMs as metadata-driven screening helpers rather than as standalone replacements for statistical feature selection.`

## Recommended Artifacts to Cite in the Paper

Main tables:

- already embedded below as `final_comparison_table_compact.csv`
- already embedded below as `paired_fold_comparisons.csv`
- already embedded below as `oot_metric_table.csv`
- already embedded below as `stability_metric_table.csv`
- already embedded below as `monthly_metric_summary.csv`

Main figures:

- described above under `Plot Descriptions`
- local plot files are under `results_full_run/plot_reports/all/`

## Portable CSV Appendix

These are compact embedded CSV versions of the key result files. They are included so this report still carries the underlying evidence if you move it to another chat or tool that cannot resolve local file links.

### matrix_runs.csv

```csv
run_id,model,selector,experiment_type,status,config_hash,output_folder
lr_statistical_mrmr_f433a879ae5c,lr,mrmr,statistical,completed,f433a879ae5cd1767d75146abfd664dde39584abb6bb150892cde393c9d45d3b,results_full_run\lr\statistical\lr_statistical_mrmr_f433a879ae5c
lr_statistical_boruta_8ece4b7f81c9,lr,boruta,statistical,completed,8ece4b7f81c926a400f3d845f3a2727e1164262c8519beb692bdf882ec01a916,results_full_run\lr\statistical\lr_statistical_boruta_8ece4b7f81c9
lr_statistical_pca_b757784da58a,lr,pca,statistical,completed,b757784da58a4a9bccf0977ede860548bb70eb18fc9fecfcf452979833dbd225,results_full_run\lr\statistical\lr_statistical_pca_b757784da58a
lr_statistical_domain_rule_baseline_44b66f0307b2,lr,domain_rule_baseline,statistical,completed,44b66f0307b2c4b28d5119cdaa6b28665d87ac4cf53111b5a4b390a6b7dd8445,results_full_run\lr\statistical\lr_statistical_domain_rule_baseline_44b66f0307b2
lr_llm_llm_480b58e35e02,lr,llm,llm,completed,480b58e35e0263dd33f4bb5c43b327efb283a07553d91c1cc6fe93ce56737427,results_full_run\lr\llm\lr_llm_llm_480b58e35e02
lr_hybrid_llm_then_mrmr_c6524b5e90bc,lr,llm_then_mrmr,hybrid,completed,c6524b5e90bc9d4da8708f348d00f73fde95910036c88e767a4c8a9a2d947b7b,results_full_run\lr\hybrid_mrmr\lr_hybrid_llm_then_mrmr_c6524b5e90bc
lr_hybrid_llm_then_boruta_3a1ea75b8bd4,lr,llm_then_boruta,hybrid,completed,3a1ea75b8bd4c77752f903d59768277c843cdb67b4bf4088940f0bc7aeb43191,results_full_run\lr\hybrid_boruta\lr_hybrid_llm_then_boruta_3a1ea75b8bd4
lr_hybrid_stable_core_llm_fill_fe23ffe71d2b,lr,stable_core_llm_fill,hybrid,completed,fe23ffe71d2bfc5135fd7725a63cd36d66ead848da35f860bb4cfccb0966051c,results_full_run\lr\hybrid_stable_core_llm_fill\lr_hybrid_stable_core_llm_fill_fe23ffe71d2b
catboost_statistical_mrmr_36313976914b,catboost,mrmr,statistical,completed,36313976914bafa16ea6653b0c0824400106ea112fb72cf26e1a4d1956fef7c4,results_full_run\catboost\statistical\catboost_statistical_mrmr_36313976914b
catboost_statistical_boruta_0b1f882d8b42,catboost,boruta,statistical,completed,0b1f882d8b4253739c02f855878fb4bb300d184592e83f6853802fc7a41ccd8c,results_full_run\catboost\statistical\catboost_statistical_boruta_0b1f882d8b42
catboost_statistical_pca_b8df1ad7d122,catboost,pca,statistical,completed,b8df1ad7d12281d48e510364a3d1c608baebad996f471341226e349aff8a761a,results_full_run\catboost\statistical\catboost_statistical_pca_b8df1ad7d122
catboost_statistical_domain_rule_baseline_70a31ece5ab9,catboost,domain_rule_baseline,statistical,completed,70a31ece5ab9f22c70eb9ae2e005a8076de1051f759975d956dd50e6a171f5c1,results_full_run\catboost\statistical\catboost_statistical_domain_rule_baseline_70a31ece5ab9
catboost_llm_llm_5a7c774fc737,catboost,llm,llm,completed,5a7c774fc737543eba5fdc0f522b532044a295a4e49771aa00982c6de5f3b5ab,results_full_run\catboost\llm\catboost_llm_llm_5a7c774fc737
catboost_hybrid_llm_then_mrmr_f81d8df7913d,catboost,llm_then_mrmr,hybrid,completed,f81d8df7913da58c1b374a137216be5a98a9309dff6fc1c233a3e54e7e17761e,results_full_run\catboost\hybrid_mrmr\catboost_hybrid_llm_then_mrmr_f81d8df7913d
catboost_hybrid_llm_then_boruta_6c3d83667ec0,catboost,llm_then_boruta,hybrid,completed,6c3d83667ec0bede8f5a8d4b62f0ff51d4476aa889e11e72494e989fc7e61372,results_full_run\catboost\hybrid_boruta\catboost_hybrid_llm_then_boruta_6c3d83667ec0
catboost_hybrid_stable_core_llm_fill_fb0c8b692cda,catboost,stable_core_llm_fill,hybrid,completed,fb0c8b692cdacd60a9bbcda03c83f175c2d3380f8fb1d3e0b0e25e8039a8cda7,results_full_run\catboost\hybrid_stable_core_llm_fill\catboost_hybrid_stable_core_llm_fill_fb0c8b692cda
```

### failed_runs.csv

```csv
run_id,model,selector,experiment_type,status,error,failed_at,output_folder
```

### final_comparison_table_compact.csv

```csv
model,selector,experiment_type,feature_budget,oot_auc,oot_gini,oot_ks,lift_at_10,selected_feature_psi_mean,model_score_psi,nogueira_stability,mean_pairwise_jaccard,runtime_seconds
catboost,stable_core_llm_fill,hybrid,40,0.769498,0.538996,0.404988,3.441042,0.012256,0.003348,0.707914,0.576283,3632.844800
catboost,mrmr,statistical,40,0.768669,0.537337,0.402893,3.435428,0.018889,0.005355,0.729550,0.603699,850.482371
catboost,llm_then_boruta,hybrid,40,0.764485,0.528970,0.394944,3.389585,0.011055,0.003399,0.459100,0.336871,547.880246
catboost,llm_then_mrmr,hybrid,40,0.762283,0.524567,0.391540,3.368067,0.010593,0.003025,0.418533,0.303317,451.450253
catboost,llm,llm,40,0.761863,0.523725,0.390337,3.346549,0.003640,0.003853,0.461805,0.342279,324.385257
catboost,domain_rule_baseline,statistical,40,0.734483,0.468966,0.345606,3.073361,0.003293,0.005877,1.000000,1.000000,270.931609
catboost,pca,statistical,40,0.707112,0.414223,0.305578,2.630835,0.029159,0.014564,1.000000,1.000000,312.651617
catboost,boruta,statistical,40,0.688705,0.377411,0.277538,2.552246,0.006240,0.001138,0.564576,0.452871,1141.664843
lr,stable_core_llm_fill,hybrid,20,0.748857,0.497715,0.368780,3.103300,0.012572,0.005034,0.729784,0.593849,1451.738020
lr,mrmr,statistical,20,0.745689,0.491378,0.361827,3.095815,0.013276,0.006494,0.771356,0.641672,328.610049
lr,llm,llm,20,0.738122,0.476243,0.354482,3.003193,0.001614,0.007079,0.636248,0.495259,836.553030
lr,llm_then_mrmr,hybrid,20,0.733895,0.467790,0.347336,2.937703,0.007912,0.003841,0.490747,0.352022,86.321099
lr,domain_rule_baseline,statistical,20,0.724896,0.449792,0.334621,2.871277,0.003022,0.011185,1.000000,1.000000,17.838322
lr,pca,statistical,20,0.672903,0.345805,0.259738,2.269703,0.035985,0.029073,1.000000,1.000000,16.719139
lr,llm_then_boruta,hybrid,20,0.650827,0.301654,0.218360,2.190179,0.007856,0.001253,0.506336,0.362974,240.702781
lr,boruta,statistical,20,0.630620,0.261241,0.187376,2.035809,0.004064,0.002392,0.521925,0.409820,957.607561
```

### paired_fold_comparisons.csv

```csv
model,candidate_selector,metric,mean_delta_candidate_minus_baseline,ci95_lower,ci95_upper
catboost,boruta,auc,-0.021719,-0.042930,-0.000508
catboost,domain_rule_baseline,auc,-0.040485,-0.047632,-0.033337
catboost,llm,auc,-0.006140,-0.015337,0.003058
catboost,llm_then_boruta,auc,-0.008793,-0.014668,-0.002918
catboost,llm_then_mrmr,auc,-0.001722,-0.010787,0.007343
catboost,pca,auc,-0.056171,-0.061416,-0.050925
catboost,stable_core_llm_fill,auc,0.000651,-0.004154,0.005456
catboost,boruta,gini,-0.043438,-0.085861,-0.001016
catboost,domain_rule_baseline,gini,-0.080969,-0.095264,-0.066674
catboost,llm,gini,-0.012279,-0.030674,0.006116
catboost,llm_then_boruta,gini,-0.017586,-0.029336,-0.005836
catboost,llm_then_mrmr,gini,-0.003444,-0.021575,0.014686
catboost,pca,gini,-0.112341,-0.122832,-0.101851
catboost,stable_core_llm_fill,gini,0.001301,-0.008309,0.010911
lr,boruta,auc,-0.094530,-0.142653,-0.046406
lr,domain_rule_baseline,auc,-0.022592,-0.026015,-0.019169
lr,llm,auc,-0.008368,-0.011767,-0.004968
lr,llm_then_boruta,auc,-0.013547,-0.023424,-0.003670
lr,llm_then_mrmr,auc,-0.006155,-0.011862,-0.000448
lr,pca,auc,-0.079332,-0.087390,-0.071274
lr,stable_core_llm_fill,auc,-0.004208,-0.007142,-0.001274
lr,boruta,gini,-0.189059,-0.285306,-0.092813
lr,domain_rule_baseline,gini,-0.045184,-0.052031,-0.038338
lr,llm,gini,-0.016735,-0.023535,-0.009936
lr,llm_then_boruta,gini,-0.027093,-0.046847,-0.007339
lr,llm_then_mrmr,gini,-0.012310,-0.023724,-0.000896
lr,pca,gini,-0.158664,-0.174779,-0.142549
lr,stable_core_llm_fill,gini,-0.008416,-0.014285,-0.002547
```

### llm_call_summary.csv

```csv
run_id,model,selector,experiment_type,llm_calls_actually_made,llm_cache_hits,llm_prompt_tokens,llm_completion_tokens,llm_total_tokens
catboost_hybrid_llm_then_boruta_6c3d83667ec0,catboost,llm_then_boruta,hybrid,0,6,0,0,0
catboost_hybrid_llm_then_mrmr_f81d8df7913d,catboost,llm_then_mrmr,hybrid,0,6,0,0,0
catboost_hybrid_stable_core_llm_fill_fb0c8b692cda,catboost,stable_core_llm_fill,hybrid,0,6,0,0,0
catboost_llm_llm_5a7c774fc737,catboost,llm,llm,0,6,0,0,0
catboost_statistical_boruta_0b1f882d8b42,catboost,boruta,statistical,0,0,0,0,0
catboost_statistical_domain_rule_baseline_70a31ece5ab9,catboost,domain_rule_baseline,statistical,0,0,0,0,0
catboost_statistical_mrmr_36313976914b,catboost,mrmr,statistical,0,0,0,0,0
catboost_statistical_pca_b8df1ad7d122,catboost,pca,statistical,0,0,0,0,0
lr_hybrid_llm_then_boruta_3a1ea75b8bd4,lr,llm_then_boruta,hybrid,0,6,0,0,0
lr_hybrid_llm_then_mrmr_c6524b5e90bc,lr,llm_then_mrmr,hybrid,0,6,0,0,0
lr_hybrid_stable_core_llm_fill_fe23ffe71d2b,lr,stable_core_llm_fill,hybrid,0,6,0,0,0
lr_llm_llm_480b58e35e02,lr,llm,llm,6,0,302333,29775,332108
lr_statistical_boruta_8ece4b7f81c9,lr,boruta,statistical,0,0,0,0,0
lr_statistical_domain_rule_baseline_44b66f0307b2,lr,domain_rule_baseline,statistical,0,0,0,0,0
lr_statistical_mrmr_f433a879ae5c,lr,mrmr,statistical,0,0,0,0,0
lr_statistical_pca_b757784da58a,lr,pca,statistical,0,0,0,0,0
```

### oot_metric_table.csv

```csv
label,oot_auc,oot_gini,oot_ks,oot_precision,oot_recall,oot_f1,oot_accuracy,oot_lift_at_10,oot_bad_rate_capture_at_10
catboost_hybrid_llm_then_boruta,0.764485,0.528970,0.394944,0.206998,0.613305,0.309527,0.756399,3.389585,0.338978
catboost_hybrid_llm_then_mrmr,0.762283,0.524567,0.391540,0.212829,0.585797,0.312223,0.770235,3.368067,0.336826
catboost_hybrid_stable_core_llm_fill,0.769498,0.538996,0.404988,0.215735,0.604978,0.318052,0.769035,3.441042,0.344124
catboost_llm_llm,0.761863,0.523725,0.390337,0.209530,0.600674,0.310685,0.762705,3.346549,0.334674
catboost_statistical_boruta,0.688705,0.377411,0.277538,0.160210,0.541729,0.247288,0.706396,2.552246,0.255240
catboost_statistical_domain_rule_baseline,0.734483,0.468966,0.345606,0.164196,0.686190,0.264985,0.661100,3.073361,0.307354
catboost_statistical_mrmr,0.768669,0.537337,0.402893,0.218870,0.600112,0.320756,0.773725,3.435428,0.343563
catboost_statistical_pca,0.707112,0.414223,0.305578,0.205189,0.410647,0.273645,0.805919,2.630835,0.263099
lr_hybrid_llm_then_boruta,0.650827,0.301654,0.218360,0.128777,0.636040,0.214188,0.584509,2.190179,0.219031
lr_hybrid_llm_then_mrmr,0.733895,0.467790,0.347336,0.155551,0.724925,0.256141,0.625149,2.937703,0.293787
lr_hybrid_stable_core_llm_fill,0.748857,0.497715,0.368780,0.166341,0.716411,0.269993,0.655102,3.103300,0.310348
lr_llm_llm,0.738122,0.476243,0.354482,0.158739,0.719124,0.260071,0.635703,3.003193,0.300337
lr_statistical_boruta,0.630620,0.261241,0.187376,0.122029,0.629678,0.204438,0.563701,2.035809,0.203593
lr_statistical_domain_rule_baseline,0.724896,0.449792,0.334621,0.153587,0.707148,0.252362,0.626981,2.871277,0.287144
lr_statistical_mrmr,0.745689,0.491378,0.361827,0.164038,0.714165,0.266795,0.650538,3.095815,0.309600
lr_statistical_pca,0.672903,0.345805,0.259738,0.130846,0.706213,0.220786,0.556213,2.269703,0.226984
```

### stability_metric_table.csv

```csv
model,selector,experiment_type,nogueira_stability,kuncheva_stability,mean_pairwise_jaccard,stable_feature_count_80,stable_feature_ratio_80
catboost,stable_core_llm_fill,hybrid,0.707914,0.707914,0.576283,28.0,0.700000
catboost,mrmr,statistical,0.729550,0.729550,0.603699,29.0,0.725000
catboost,llm_then_boruta,hybrid,0.459100,0.459100,0.336871,17.0,0.425000
catboost,llm_then_mrmr,hybrid,0.418533,0.418533,0.303317,15.0,0.375000
catboost,llm,llm,0.461805,0.461805,0.342279,17.0,0.425000
catboost,domain_rule_baseline,statistical,1.000000,1.000000,1.000000,40.0,1.000000
catboost,pca,statistical,1.000000,1.000000,1.000000,40.0,1.000000
catboost,boruta,statistical,0.564576,0.564576,0.452871,25.0,0.625000
lr,stable_core_llm_fill,hybrid,0.729784,0.729784,0.593849,15.0,0.750000
lr,mrmr,statistical,0.771356,0.771356,0.641672,15.0,0.750000
lr,llm,llm,0.636248,0.636248,0.495259,11.0,0.550000
lr,llm_then_mrmr,hybrid,0.490747,0.490747,0.352022,8.0,0.400000
lr,domain_rule_baseline,statistical,1.000000,1.000000,1.000000,20.0,1.000000
lr,pca,statistical,1.000000,1.000000,1.000000,20.0,1.000000
lr,llm_then_boruta,hybrid,0.506336,0.506336,0.362974,8.0,0.400000
lr,boruta,statistical,0.521925,0.521925,0.409820,10.0,0.500000
```

### monthly_metric_summary.csv

This is a compact embedded summary derived from `plot_reports/all/monthly_metric_table.csv`.

```csv
label,gini_mean,gini_std,psi_model_mean,lift_at_10_mean
catboost_hybrid_stable_core_llm_fill,0.509986,0.023789,0.012495,3.320051
catboost_statistical_mrmr,0.508685,0.027046,0.012992,3.336481
catboost_hybrid_llm_then_mrmr,0.505240,0.010285,0.013551,3.283014
catboost_llm_llm,0.496405,0.035494,0.011520,3.179901
catboost_hybrid_llm_then_boruta,0.491099,0.022299,0.016256,3.212219
lr_statistical_mrmr,0.472307,0.021723,0.001212,2.981133
catboost_statistical_boruta,0.465246,0.045681,0.011380,3.025727
lr_hybrid_stable_core_llm_fill,0.463891,0.027646,0.001326,2.918698
lr_hybrid_llm_then_mrmr,0.459998,0.032982,0.001830,2.893270
lr_llm_llm,0.455572,0.022478,0.002002,2.916008
lr_hybrid_llm_then_boruta,0.445214,0.035170,0.002734,2.803955
catboost_statistical_domain_rule_baseline,0.427715,0.019064,0.004207,2.812974
lr_statistical_domain_rule_baseline,0.427123,0.016503,0.002300,2.802220
catboost_statistical_pca,0.396343,0.026407,0.035494,2.570459
lr_statistical_pca,0.313643,0.025869,0.007192,2.113680
lr_statistical_boruta,0.283248,0.088484,0.001493,2.122753
```

## Best Reusable Trained Bundles

If you want to reuse the strongest trained models for later external evaluation, the most relevant bundles are:

- CatBoost best final OOT:
  - [final_model_bundle.joblib](results_full_run/catboost/hybrid_stable_core_llm_fill/catboost_hybrid_stable_core_llm_fill_fb0c8b692cda/models/final_model_bundle.joblib)
- CatBoost strongest baseline:
  - [final_model_bundle.joblib](results_full_run/catboost/statistical/catboost_statistical_mrmr_36313976914b/models/final_model_bundle.joblib)
- LR best final OOT:
  - [final_model_bundle.joblib](results_full_run/lr/hybrid_stable_core_llm_fill/lr_hybrid_stable_core_llm_fill_fe23ffe71d2b/models/final_model_bundle.joblib)
- LR strongest baseline:
  - [final_model_bundle.joblib](results_full_run/lr/statistical/lr_statistical_mrmr_f433a879ae5c/models/final_model_bundle.joblib)

## Final Recommendation

If you submit using only this experimental run, position the contribution as:

- an LLM-assisted metadata screening framework,
- evaluated against standard selectors,
- with evidence that hybridization can be competitive and drift-aware,
- but without claiming universal superiority over `mrmr`.

If you want the paper to be clearly stronger, the next experimental addition should be a second dataset or at least a second temporal split regime.
