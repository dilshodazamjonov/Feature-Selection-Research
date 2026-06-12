# CLIP Readiness Feature Evidence Report

This report builds CLIP-readiness evidence from saved baseline artifacts only. It does not implement CLIP, train a CLIP model, generate contrastive pairs, retrain selectors/models, or rerun the experiment matrix.

## Evidence Tables Created

- `results/homecredit/analysis/clip_readiness/feature_level_evidence_for_clip.csv`
- `results/lendingclub/analysis/clip_readiness/feature_level_evidence_for_clip.csv`
- `results/cross_dataset/analysis/clip_readiness/feature_level_evidence_summary.csv`

## Cross-Dataset Summary

| dataset     | total_features | features_with_description | features_with_semantic_group | features_with_psi | features_missing_psi | features_with_iv | features_with_llm_rank | features_with_mrmr_frequency | features_with_boruta_frequency | features_with_stable_core_membership | usable_for_clip_training_count | not_usable_for_clip_training_count | main_missing_reason |
| ----------- | -------------- | ------------------------- | ---------------------------- | ----------------- | -------------------- | ---------------- | ---------------------- | ---------------------------- | ------------------------------ | ------------------------------------ | ------------------------------ | ---------------------------------- | ------------------- |
| homecredit  | 607            | 391                       | 607                          | 192               | 415                  | 520              | 100                    | 65                           | 77                             | 29                                   | 391                            | 216                                | missing_description |
| lendingclub | 505            | 76                        | 505                          | 275               | 230                  | 243              | 76                     | 58                           | 62                             | 37                                   | 76                             | 429                                | missing_description |

## Missing Fields By Dataset

| dataset     | field                                      | missing_count |
| ----------- | ------------------------------------------ | ------------- |
| homecredit  | description                                | 216           |
| homecredit  | psi_dev_oot_if_available                   | 415           |
| homecredit  | iv_score_if_available                      | 87            |
| homecredit  | llm_best_rank                              | 507           |
| homecredit  | mrmr_selection_frequency                   | 542           |
| homecredit  | boruta_selection_frequency                 | 530           |
| homecredit  | bootstrap_selection_frequency_if_available | 544           |
| homecredit  | mean_oot_if_available                      | 607           |
| homecredit  | std_oot_if_available                       | 607           |
| lendingclub | description                                | 429           |
| lendingclub | psi_dev_oot_if_available                   | 230           |
| lendingclub | iv_score_if_available                      | 262           |
| lendingclub | llm_best_rank                              | 429           |
| lendingclub | mrmr_selection_frequency                   | 447           |
| lendingclub | boruta_selection_frequency                 | 443           |
| lendingclub | bootstrap_selection_frequency_if_available | 458           |
| lendingclub | mean_oot_if_available                      | 505           |
| lendingclub | std_oot_if_available                       | 505           |

## Readiness Answers

1. Baseline evidence is complete enough for CLIP planning. The tables consolidate descriptions, semantic groups, DEV missingness, IV where saved, LLM ranks, fold-selection frequencies, stable-core bootstrap frequencies, selected-pipeline flags, and available PSI support.
2. Baseline evidence is not complete enough for CLIP training. Several fields needed for a clean training design are missing for material subsets, especially OOT-independent empirical summaries for rejected candidates and complete PSI/IV coverage.
3. Missing fields vary by dataset, but both datasets lack saved OOT mean/std feature summaries. Features outside saved LLM/IV/selection artifacts also lack IV, LLM rank, and selector-frequency fields.
4. Home Credit still lacks rejected-candidate PSI. Selected-feature PSI exists, and some LLM top-100 rows have PSI when the candidate was selected, but rejected-candidate DEV/OOT design matrices were not saved for complete PSI recovery.
5. LendingClub still has unavailable PSI for categorical or missing-frame features where numeric DEV/OOT values were unavailable or the feature was not present in the processed safe frame.
6. Missing values can mostly be fixed by targeted artifact generation: save DEV-only per-feature descriptive stats for the full candidate pool, save OOT support stats separately, compute candidate PSI from saved design matrices or regenerate design-matrix diagnostics, and persist IV for the full candidate universe.
7. A full experiment rerun is not required. The missing pieces are diagnostic/evidence artifacts, not changed feature-selection or model-training results.
8. Before CLIP training, generate training-safe DEV-only evidence tables, explicit train/evaluation field manifests, full candidate-pool IV, full candidate-pool missingness and numeric moments, complete LLM candidate ranks/reasons, and optional OOT PSI/mean/std support artifacts kept out of selector training.

## OOT Field Policy

`oot_fields_are_evaluation_only` is set to `true` in the per-feature evidence tables. OOT PSI and OOT summary statistics may be used for evaluation/support diagnostics only and must not be used to train a selector unless explicitly approved later.

## Missing Artifacts

- Complete saved DEV/OOT design matrices for rejected Home Credit LLM candidates.
- Full candidate-pool PSI for Home Credit rejected or unselected features.
- Full candidate-pool PSI for LendingClub categorical or missing-frame features that do not have numeric DEV/OOT values in saved artifacts.
- Saved OOT mean/std feature-summary artifacts for both datasets.
- Full candidate-pool IV artifacts for every feature in the unioned candidate universe.

## Commands Run

- `python scripts/build_clip_readiness_feature_evidence.py`

## Rerun Decision

No full rerun is required. Targeted artifact generation is sufficient before CLIP training, provided it does not alter selector/model outputs.
