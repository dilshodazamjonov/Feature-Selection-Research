# CLIP Readiness Feature Evidence Report

This report builds CLIP-readiness evidence from saved baseline artifacts only. It does not implement CLIP, train a CLIP model, generate contrastive pairs, retrain selectors/models, or rerun the experiment matrix.

## Evidence Tables Created

- `results/homecredit/analysis/clip_readiness/feature_level_evidence_for_clip.csv`
- `results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
- `results/lendingclub_v2/analysis/clip_readiness/feature_level_evidence_for_clip.csv`
- `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
- `results/cross_dataset_v2/analysis/clip_readiness/feature_level_evidence_summary.csv`
- `results/cross_dataset_v2/analysis/clip_readiness/dev_only_clip_training_evidence_summary.csv`

## Cross-Dataset Summary

| dataset        | total_features | features_with_description | features_with_semantic_group | features_with_psi | features_missing_psi | features_with_iv | features_with_llm_rank | features_with_mrmr_frequency | features_with_boruta_frequency | features_with_stable_core_membership | usable_for_clip_training_count | not_usable_for_clip_training_count | dev_only_training_allowed_count | dev_only_training_blocked_count | main_missing_reason          |
| -------------- | -------------- | ------------------------- | ---------------------------- | ----------------- | -------------------- | ---------------- | ---------------------- | ---------------------------- | ------------------------------ | ------------------------------------ | ------------------------------ | ---------------------------------- | ------------------------------- | ------------------------------- | ---------------------------- |
| homecredit     | 607            | 436                       | 607                          | 192               | 415                  | 520              | 100                    | 65                           | 77                             | 29                                   | 436                            | 171                                | 436                             | 171                             | missing_description          |
| lendingclub_v2 | 796            | 675                       | 796                          | 0                 | 796                  | 576              | 149                    | 61                           | 61                             | 37                                   | 576                            | 220                                | 576                             | 220                             | no_dev_training_signal_saved |

## Missing Fields By Dataset

| dataset        | field                                      | missing_count |
| -------------- | ------------------------------------------ | ------------- |
| homecredit     | description                                | 171           |
| homecredit     | psi_dev_oot_if_available                   | 415           |
| homecredit     | iv_score_if_available                      | 87            |
| homecredit     | llm_best_rank                              | 507           |
| homecredit     | mrmr_selection_frequency                   | 542           |
| homecredit     | boruta_selection_frequency                 | 530           |
| homecredit     | bootstrap_selection_frequency_if_available | 544           |
| homecredit     | mean_oot_if_available                      | 607           |
| homecredit     | std_oot_if_available                       | 607           |
| lendingclub_v2 | description                                | 121           |
| lendingclub_v2 | psi_dev_oot_if_available                   | 796           |
| lendingclub_v2 | iv_score_if_available                      | 220           |
| lendingclub_v2 | llm_best_rank                              | 647           |
| lendingclub_v2 | mrmr_selection_frequency                   | 735           |
| lendingclub_v2 | boruta_selection_frequency                 | 735           |
| lendingclub_v2 | bootstrap_selection_frequency_if_available | 746           |
| lendingclub_v2 | mean_oot_if_available                      | 796           |
| lendingclub_v2 | std_oot_if_available                       | 796           |

## Readiness Answers

1. Baseline evidence is complete enough for CLIP planning across Home Credit and LendingClub v2. The tables consolidate descriptions, semantic groups, DEV missingness, IV where saved, LLM ranks, fold-selection frequencies, stable-core bootstrap frequencies, selected-pipeline flags, and available PSI support.
2. The generated `dev_only_clip_training_evidence.csv` files are the only CLIP-training candidate evidence tables from this script. They exclude OOT/PSI fields and include explicit leakage-review decisions and blocking reasons.
3. Missing fields vary by dataset, but both datasets lack saved OOT mean/std feature summaries. Features outside saved LLM/IV/selection artifacts also lack IV, LLM rank, and selector-frequency fields.
4. Home Credit still lacks rejected-candidate PSI. Selected-feature PSI exists, and some LLM top-100 rows have PSI when the candidate was selected, but rejected-candidate DEV/OOT design matrices were not saved for complete PSI recovery.
5. LendingClub v2 still has unavailable PSI for categorical or missing-frame features where numeric DEV/OOT values were unavailable or the feature was not present in the processed safe frame.
6. Missing values can mostly be fixed by targeted artifact generation: save DEV-only per-feature descriptive stats for the full candidate pool, save OOT support stats separately, compute candidate PSI from saved design matrices or regenerate design-matrix diagnostics, and persist IV for the full candidate universe.
7. A full experiment rerun is not required. The missing pieces are diagnostic/evidence artifacts, not changed feature-selection or model-training results.
8. Before any actual CLIP model training, use only the DEV-only evidence tables as training inputs and keep OOT PSI/mean/std support artifacts out of selector training.

## DEV-Only Training Leakage Policy

Train CLIP selector evidence only from application-time feature metadata, DEV missingness/statistical summaries, DEV IV, DEV fold-selection frequencies, and DEV LLM ranks. Do not use OOT labels, OOT feature summaries, PSI, target columns, split columns, IDs, post-origination outcomes, payment/settlement/hardship fields, or any feature marked non-safe in dataset leakage review.

`oot_fields_are_evaluation_only` is set to `true` in the per-feature evidence tables. OOT PSI and OOT summary statistics may be used for evaluation/support diagnostics only and must not be used to train a selector unless explicitly approved later. The DEV-only training evidence files intentionally omit the OOT/PSI columns.

## Missing Artifacts

- Complete saved DEV/OOT design matrices for rejected Home Credit LLM candidates.
- Full candidate-pool PSI for Home Credit rejected or unselected features.
- Full candidate-pool PSI for LendingClub v2 categorical or missing-frame features that do not have numeric DEV/OOT values in saved artifacts.
- Saved OOT mean/std feature-summary artifacts for both datasets.
- Full candidate-pool IV artifacts for every feature in the unioned candidate universe.

## Commands Run

- `python scripts/build_clip_readiness_feature_evidence.py`

## Rerun Decision

No full rerun is required. Targeted artifact generation is sufficient before CLIP training, provided it does not alter selector/model outputs.
