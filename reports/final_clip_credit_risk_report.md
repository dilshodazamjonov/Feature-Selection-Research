# Final CLIP Credit-Risk Feature-Selection Report

## 1. Bottom-line verdict

Prompt 8 completes the final analysis and reporting layer from saved artifacts only. The scientific conclusion is conservative: CLIP-style representation learning is a valid architectural screening experiment, but the saved downstream OOT evidence does not support replacing the frozen LLM or mRMR workflows.

The current contrastive encoder aligns semantic feature metadata with a limited DEV statistical view, primarily reflecting missingness behavior. It is an architectural and screening experiment rather than a comprehensive statistical feature-quality representation.

## 2. Objective and research design

The study compares original statistical selectors, original LLM-assisted selectors, and two frozen CLIP selector extensions: `clip` and `clip_then_mrmr`. Home Credit is the CLIP training dataset. LendingClub v2 is external validation only. Legacy LendingClub is not part of the CLIP training, integration, evaluation, plots, or conclusions.

## 3. Data, temporal validation, and leakage controls

The final analysis uses saved Prompt 7 OOT predictions and aggregate tables. Home Credit has 120,053 OOT rows; LendingClub v2 has 293,105 OOT rows. Run manifests record checkpoint, anchor, feature-set, config, prediction, metric, and source hashes. Per-run leakage audits report passed status, and the analysis does not rerun feature selection, model fitting, or prediction generation.

## 4. Main OOT predictive results

Best fixed-method result by dataset/model:

- homecredit catboost: mrmr AUC 0.7668, Gini 0.5337, PSI 0.0102
- homecredit lr: mrmr AUC 0.7457, Gini 0.4914, PSI 0.0065
- lendingclub_v2 catboost: llm AUC 0.7137, Gini 0.4275, PSI 0.0058
- lendingclub_v2 lr: llm AUC 0.6927, Gini 0.3853, PSI 0.0065

Best CLIP-family result by dataset/model:

- homecredit catboost: best CLIP-family selector `clip_then_mrmr` AUC 0.7035
- homecredit lr: best CLIP-family selector `clip_then_mrmr` AUC 0.6632
- lendingclub_v2 catboost: best CLIP-family selector `clip_then_mrmr` AUC 0.6999
- lendingclub_v2 lr: best CLIP-family selector `clip_then_mrmr` AUC 0.6842

Across the four dataset/model panels, `clip_then_mrmr` is consistently stronger than direct `clip`, but it generally trails the strongest frozen mRMR or LLM-assisted baselines. This supports CLIP as an experimental selector, not a replacement.

## 5. Score drift, semantic coverage, and redundancy

Model score PSI is taken from saved Prompt 7 run artifacts. DEV score vectors are not persisted, so PSI was not independently recomputed. The interpretation thresholds are low drift below 0.10, moderate drift from 0.10 to below 0.25, and high drift at or above 0.25. Most CLIP-family runs have low score PSI; Home Credit LR direct `clip` is moderate.

Semantic coverage and redundancy are descriptive. Broader semantic coverage does not imply predictive superiority. Home Credit direct CLIP has high repeated-family share, while `clip_then_mrmr` reduces redundancy in the LR panel. LendingClub v2 CLIP selections show low repeated-family share in the saved artifacts.

## 6. Representation learning and seed robustness

Prompt 5 retained five seeds and selected seed 55 by the prespecified lowest Home Credit validation loss rule. LendingClub v2 did not influence seed or checkpoint selection. This is representation-level evidence only; downstream multi-seed predictions were not materialized.

## 7. LendingClub v2 external validation

LendingClub v2 was external-only. CLIP-family results transfer enough to run valid OOT comparisons, but they do not consistently outperform LLM or mRMR baselines. The external validation finding is therefore weak for replacement and useful mainly as a boundary check.

## 8. Limitations

The main limitations are recorded in `results/clip/final_analysis/limitations_register.csv`. The most important are the missingness-only statistical view, non-fold-local CLIP preparation, fixed feature budgets, limited dataset count, limited seed count, and unavailable DEV score vectors for independent PSI recomputation. No fairness, causal, production-readiness, or operational-cost claim is made.

## 9. Conclusion

CLIP is scientifically usable as an experimental representation and screening extension in this repository. The final OOT evidence does not justify replacing the LLM workflow or the strongest mRMR baselines. The recommended interpretation is: keep CLIP as a documented research extension, preserve LLM and mRMR baselines, and treat future CLIP work as requiring richer DEV statistical views and broader external validation.
