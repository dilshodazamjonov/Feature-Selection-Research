# Finalized Credit-Risk Feature-Selection Metrics and Winner Curves

## Technical summary

- The finalized six-case AUC leaders are fixed at the values below. Gini is derived exactly as `2 × AUC − 1`.
- LendingClub accuracy is **0.840000** and its Brier winner is **0.062300**. Home Credit log loss is **0.293940** and its Brier winner is **0.697320**.
- The ROC image contains exactly six winner profiles—one for each dataset × model case—and every plotted trapezoidal area equals the reported AUC.
- AUC does not identify calibration. Finalized row-level probabilities were not provided, so empirical reliability-bin coordinates cannot be reconstructed. Home Credit and Stability also contain same-method log-loss/Brier pairs that violate the necessary bound `log loss ≥ 2 × Brier`; those pairs cannot be produced by one binary probability vector.

## The six finalized AUC winners

| dataset | model | winning FS method | family | AUC | Gini |
| --- | --- | --- | --- | --- | --- |
| Home Credit | Logistic Regression | mRMR | classical | 0.770000 | 0.540000 |
| Home Credit | CatBoost | LLM | LLM-assisted | 0.793450 | 0.586900 |
| LendingClub v2 | Logistic Regression | LLM | LLM-assisted | 0.740000 | 0.480000 |
| LendingClub v2 | CatBoost | LLM then mRMR | LLM-assisted | 0.770664 | 0.541328 |
| Home Credit Stability 2024 | Logistic Regression | IV then Boruta | classical | 0.802956 | 0.605912 |
| Home Credit Stability 2024 | CatBoost | LLM then mRMR | LLM-assisted | 0.869088 | 0.738177 |

The plot contains one line per panel and no losing method. The smooth line is a disclosed AUC-matched reference profile, not an empirical threshold trace.

![Finalized winner-only ROC profiles](plots/winner_roc_curves.png)

## Finalized score changes

| dataset | model | metric | method | finalized value | authority |
| --- | --- | --- | --- | --- | --- |
| Home Credit | Logistic Regression | auc | mRMR | 0.770000 | finalized scorecard 2026-08-21 |
| LendingClub v2 | Logistic Regression | auc | LLM | 0.740000 | finalized scorecard 2026-08-21 |
| LendingClub v2 | CatBoost | accuracy | LLM | 0.840000 | finalized scorecard 2026-08-21 |
| LendingClub v2 | CatBoost | brier | LLM then mRMR | 0.062300 | finalized scorecard 2026-08-21 |
| Home Credit | CatBoost | log_loss | LLM | 0.293940 | finalized scorecard 2026-08-21 |
| Home Credit | CatBoost | brier | LLM | 0.697320 | finalized scorecard 2026-08-21 |

These values are the controlling point estimates throughout this report. The older workbook values remain only as the preserved base snapshot in `results/final_three_dataset_synthesis_v1/inputs/workbook1_supplied_results.csv`.

The threshold-metric figure includes finalized LendingClub accuracy `0.84` alongside the other threshold-dependent winners.

![Finalized threshold-dependent metric winners](plots/threshold_metric_winners.png)

The calibration-error figure includes LendingClub Brier `0.0623`, Home Credit log loss `0.29394`, and Home Credit Brier `0.69732`.

![Finalized log-loss and Brier winners](plots/calibration_error_metrics.png)

## Methods that win across the six AUC cases

| method | family | AUC case wins | datasets | models | mean AUC | minimum AUC | maximum AUC | mean Gini |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM then mRMR | LLM-assisted | 2 | 2 | CatBoost | 0.819876 | 0.770664 | 0.869088 | 0.639752 |
| LLM | LLM-assisted | 2 | 2 | CatBoost; Logistic Regression | 0.766725 | 0.740000 | 0.793450 | 0.533450 |
| IV then Boruta | classical | 1 | 1 | Logistic Regression | 0.802956 | 0.802956 | 0.802956 | 0.605912 |
| mRMR | classical | 1 | 1 | Logistic Regression | 0.770000 | 0.770000 | 0.770000 | 0.540000 |

LLM-assisted methods win four cases. Plain LLM wins Home Credit CatBoost and LendingClub Logistic Regression; LLM then mRMR wins LendingClub CatBoost and Stability CatBoost. Pure mRMR wins Home Credit Logistic Regression, and IV then Boruta wins Stability Logistic Regression.

## Complete finalized 45-metric scorecard

| dataset | metric | direction | winning method | model | finalized score | resolution |
| --- | --- | --- | --- | --- | --- | --- |
| Home Credit | auc | higher | LLM | catboost | 0.7934500 | only supplied winner |
| Home Credit | gini | higher | LLM | catboost | 0.5869000 | only supplied winner |
| Home Credit | ks | higher | LLM | catboost | 0.4505434 | LLM_score wins by metric direction |
| Home Credit | precision | higher | Stable core + LLM fill | catboost | 0.2463853 | only supplied winner |
| Home Credit | recall | higher | LLM then mRMR | lr | 0.7230539 | only supplied winner |
| Home Credit | f1 | higher | Stable core + LLM fill | catboost | 0.3326340 | only supplied winner |
| Home Credit | accuracy | higher | PCA | catboost | 0.8503994 | only supplied winner |
| Home Credit | log_loss | lower | LLM | catboost | 0.2939400 | finalized scorecard value |
| Home Credit | brier | lower | LLM | catboost | 0.6973200 | finalized scorecard value |
| Home Credit | lift_at_10 | higher | RFE CatBoost | catboost | 3.5804424 | only supplied winner |
| Home Credit | bad_rate_capture_at_10 | higher | RFE CatBoost | catboost | 0.3580651 | only supplied winner |
| Home Credit | score_psi | lower | LLM then Boruta | lr | 0.0008732 | only supplied winner |
| Home Credit | feature_psi_mean | lower | Boruta RF | lr | 0.0013655 | only supplied winner |
| Home Credit | feature_psi_median | lower | LLM | catboost;lr | 0.0000000 | only supplied winner |
| Home Credit | feature_psi_max | lower | Boruta (legacy) | lr | 0.0118607 | only supplied winner |
| LendingClub v2 | auc | higher | LLM then mRMR | catboost | 0.7706640 | only supplied winner |
| LendingClub v2 | gini | higher | LLM then mRMR | catboost | 0.5413280 | only supplied winner |
| LendingClub v2 | ks | higher | LLM then mRMR | catboost | 0.3943100 | LLM_score wins by metric direction |
| LendingClub v2 | precision | higher | LLM | catboost | 0.3851499 | only supplied winner |
| LendingClub v2 | recall | higher | PCA | lr | 0.6467796 | only supplied winner |
| LendingClub v2 | f1 | higher | LLM | catboost | 0.4649128 | only supplied winner |
| LendingClub v2 | accuracy | higher | LLM | catboost | 0.8400000 | finalized scorecard value |
| LendingClub v2 | log_loss | lower | LLM then mRMR | catboost | 0.1324000 | LLM_score wins by metric direction |
| LendingClub v2 | brier | lower | LLM then mRMR | catboost | 0.0623000 | finalized scorecard value |
| LendingClub v2 | lift_at_10 | higher | IV then Boruta | catboost | 2.2684663 | only supplied winner |
| LendingClub v2 | bad_rate_capture_at_10 | higher | IV then Boruta | catboost | 0.2268505 | only supplied winner |
| LendingClub v2 | score_psi | lower | Random K | catboost | 0.0005986 | best_fs_method retained |
| LendingClub v2 | feature_psi_mean | lower | Domain rules | lr | 0.0000573 | best_fs_method retained |
| LendingClub v2 | feature_psi_median | lower | Boruta (legacy); Domain rules; LLM | lr;catboost | 0.0000000 | best_fs_method retained |
| LendingClub v2 | feature_psi_max | lower | Domain rules | lr | 0.0011126 | best_fs_method retained |
| Home Credit Stability 2024 | auc | higher | LLM then mRMR | catboost | 0.8690884 | only supplied winner |
| Home Credit Stability 2024 | gini | higher | LLM then mRMR | catboost | 0.7381768 | only supplied winner |
| Home Credit Stability 2024 | ks | higher | LLM then mRMR | catboost | 0.5934300 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | precision | higher | IV then Boruta | catboost | 0.0842774 | only supplied winner |
| Home Credit Stability 2024 | recall | higher | CatBoost SHAP | lr | 0.8599832 | only supplied winner |
| Home Credit Stability 2024 | f1 | higher | IV then Boruta | catboost | 0.1515696 | only supplied winner |
| Home Credit Stability 2024 | accuracy | higher | IV then Boruta | catboost | 0.7694611 | only supplied winner |
| Home Credit Stability 2024 | log_loss | lower | LLM then mRMR | catboost | 0.2300000 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | brier | lower | LLM then mRMR | catboost | 0.1200000 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | lift_at_10 | higher | RFE CatBoost | catboost | 5.0759904 | only supplied winner |
| Home Credit Stability 2024 | bad_rate_capture_at_10 | higher | RFE CatBoost | catboost | 0.5076057 | only supplied winner |
| Home Credit Stability 2024 | score_psi | lower | LLM then mRMR | catboost | 0.0002100 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | feature_psi_mean | lower | LLM then mRMR | lr | 0.0068000 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | feature_psi_median | lower | LLM then mRMR | lr | 0.0004500 | LLM_score wins by metric direction |
| Home Credit Stability 2024 | feature_psi_max | lower | LLM then mRMR | lr | 0.0930000 | LLM_score wins by metric direction |

The base comparison rule is direction-aware: use the LLM comparison only when it improves on the current score under the metric's `higher` or `lower` direction. Finalized replacements then control the affected rows.

## Winner-only calibration evidence

The six panels contain only the finalized AUC winners. A diagonal ideal-calibration reference is shown, but no empirical winner curve is invented from aggregate metrics. Each panel states whether matching Brier/log-loss evidence exists, whether it passes the necessary inequality, and whether row-level probabilities are still required.

![Finalized winner-only calibration feasibility](plots/winner_calibration_curves.png)

### Probability-metric consistency checks

| dataset | Brier winner | Brier | log-loss winner | log loss | 2 × Brier | pair result | constant Brier baseline | constant log-loss baseline | accuracy winner | accuracy | accuracy-bound comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Home Credit | LLM (catboost) | 0.697320 | LLM (catboost) | 0.293940 | 1.394640 | fails: no common binary probability predictions can reproduce both | 0.081101 | 0.300282 | PCA (catboost) | 0.850399 | not applicable: different winning method |
| LendingClub v2 | LLM then mRMR (catboost) | 0.062300 | LLM then mRMR (catboost) | 0.132400 | 0.124600 | passes necessary log-loss/Brier bound | 0.178635 | 0.542707 | LLM (catboost) | 0.840000 | not applicable: different winning method |
| Home Credit Stability 2024 | LLM then mRMR (catboost) | 0.120000 | LLM then mRMR (catboost) | 0.230000 | 0.240000 | fails: no common binary probability predictions can reproduce both | 0.026632 | 0.125518 | IV then Boruta (catboost) | 0.769461 | not applicable: different winning method |

The accuracy bounds below apply only when accuracy, Brier, and log loss come from the same probability vector and accuracy uses threshold 0.5. The current per-metric winners differ, so cross-winner accuracy comparisons are not mathematical validation tests.

## Metric computation methodology

### ROC-AUC and Gini

ROC points are `(FPR(t), TPR(t))` over score thresholds `t`, where `FPR = FP/(FP+TN)` and `TPR = TP/(TP+FN)`. Empirical ROC-AUC is the area under that curve and is equivalent to the probability that a randomly selected event receives a higher score than a randomly selected non-event, with standard tie handling.

For the finalized six-panel ROC image, row-level finalized predictions were unavailable. Each reference profile uses `TPR = FPR^α`; `α` is solved deterministically so numerical trapezoidal integration equals the finalized table AUC to `1e-12`. Therefore:

```text
ROC curve AUC = reported table AUC
Gini = 2 × AUC − 1
```

The reference shape supports score-consistent presentation only. It does not recover thresholds, empirical uncertainty, or the original score distribution.

### KS and the frozen decision threshold

```text
KS = max_t(TPR(t) − FPR(t))
```

Historical evaluation selected a KS-maximizing threshold on fitting-partition scores. For final OOT evaluation, the threshold was selected on full-DEV training scores and then held fixed; OOT targets did not select the threshold. Finalized aggregate threshold metrics do not include the row-level confusion matrices needed for independent reproduction.

### Accuracy, precision, recall and F1

At a fixed threshold, `TP`, `FP`, `TN`, and `FN` are computed from the binary prediction. Then:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

Undefined precision or recall divisions are handled as zero in the historical implementation. The finalized aggregate update does not provide confusion-matrix counts.

### Log loss, Brier score and calibration

For binary outcomes `y_i ∈ {0,1}` and predicted event probabilities `p_i`:

```text
Log loss = −mean(y_i × ln(p_i) + (1 − y_i) × ln(1 − p_i))
Brier    = mean((y_i − p_i)^2)
```

Reliability curves group probabilities into bins and plot mean predicted probability against observed event rate. The historical implementation uses ten quantile bins. Aggregate AUC, Brier and log loss do not uniquely determine those bin coordinates.

Necessary per-prediction checks include:

```text
Log loss ≥ 2 × Brier              # natural logarithm
Accuracy ≥ 1 − 4 × Brier          # threshold = 0.5
Accuracy ≥ 1 − log_loss / ln(2)   # threshold = 0.5
```

The accuracy inequalities require the same rows, probabilities and method for all metrics. The log-loss/Brier bound does not depend on a classification threshold.

### Score PSI

Score PSI compares the DEV out-of-fold probability distribution with locked OOT probabilities. The implemented procedure is:

1. Validate that all reference and comparison scores are finite probabilities in `[0,1]`.
2. Fit ten candidate quantile bins on DEV OOF scores only.
3. Collapse duplicate quantile edges; force the outer bounds to `0` and `1`.
4. Apply those frozen edges unchanged to OOT scores.
5. Compute DEV and OOT proportions for every effective bin.
6. Add smoothing epsilon `1e-6` to each proportion and use the natural logarithm.

```text
PSI = Σ_i ((OOT_i + ε) − (DEV_i + ε)) × ln((OOT_i + ε) / (DEV_i + ε))
ε = 1e−6
```

Lower PSI means less distribution shift. The inherited `0.10` and `0.25` bands are monitoring descriptors, not hypothesis-test thresholds. Finalized PSI scores are aggregate values and are not independently recomputed here because the finalized probability vectors and frozen bin evidence were not provided.

### Selected-feature PSI

Numeric features use DEV-derived quantile edges with infinite outer bounds and an explicit missing-value state. Duplicate numeric edges collapse. Categorical features use DEV levels, an explicit missing state, and a single unseen-OOT state. Both use epsilon `1e-6` and the same natural-log PSI contribution formula. Mean, median and maximum feature PSI summarize the selected original features; lower is better.

### Lift and bad-rate capture at 10%

Sort cases from highest to lowest predicted event risk and select the top 10% of rows:

```text
capture@10 = events in top decile / all events
lift@10    = event rate in top decile / overall event rate
Lift@10 ≈ capture@10 / 0.10
```

The approximation becomes exact when the selected population is exactly 10% and the denominator conventions are identical.

## Limitations and robustness status

- The finalized scores are accepted as the reporting point estimates, but no finalized row-level predictions or fold results were provided for independent recomputation.
- The winner-only ROC shapes are deterministic AUC-matched references. They must not be described as empirical ROC curves or used to select an operating threshold.
- A genuine calibration curve cannot be reconstructed from aggregate metrics. Home Credit and Stability additionally fail the necessary same-prediction log-loss/Brier inequality, so no calibration curve can reproduce all those finalized values simultaneously.
- No new confidence intervals, DeLong tests, bootstrap intervals or significance claims are created from the finalized aggregate update.

## Recommended next evidence step

Preserve one finalized prediction file per winning dataset × model case with target, predicted probability, immutable row identifier, evaluation scope and model/method identity. That single evidence layer would allow empirical ROC curves, calibration bins, Brier/log-loss reconciliation, frozen-threshold confusion matrices and exact independent verification without changing the finalized score table.

## Further evidence needed

Home Credit Brier `0.69732` remains the finalized reported value. Reconciliation requires either the row-level probability file that produced it or an explicit statement that it and natural-log loss `0.29394` use different populations, probability definitions, or evaluation scopes; they cannot describe one common binary probability vector as currently defined.
