# todo_done fill log

Generated 2026-09-20T14:49:25Z after 196.3 minutes.

## CSV fill summary

| file | filled cells | total value cells | status |
|---|---|---|---|
| 01_obfuscation.csv | 24 | 24 | complete |
| 06_leakage.csv | 15 | 15 | complete |
| 08_boruta.csv | 0 | 108 | excluded (--steps) |
| 09_psi.csv | 32 | 48 | partial |
| 10_stability.csv | 0 | 82 | excluded (--steps) |
| 11_subsets.csv | 0 | 360 | excluded (--steps) |
| 12_overlap.csv | 0 | 4 | excluded (--steps) |
| 15_cohort.csv | 2 | 6 | copied verbatim (computationally expensive; not recomputed) |
| 16_l1.csv | 12 | 12 | copied verbatim (computationally expensive; not recomputed) |
| 17_mrmr.csv | 2 | 2 | complete |
| 21_brier.csv | 0 | 84 | copied verbatim (computationally expensive; not recomputed) |

## Skipped or blank cells

- **09** {'dataset': 'homecredit_model_stability_2024', 'backbone': 'lr', 'selector': 'Pure LLM', 'K': '20'}: full-DEV selection unavailable on this machine
- **09** {'dataset': 'homecredit_model_stability_2024', 'backbone': 'lr', 'selector': 'RFE CatBoost', 'K': '20'}: full-DEV selection unavailable on this machine
- **09** {'dataset': 'homecredit_model_stability_2024', 'backbone': 'catboost', 'selector': 'Pure LLM', 'K': '40'}: full-DEV selection unavailable on this machine
- **09** {'dataset': 'homecredit_model_stability_2024', 'backbone': 'catboost', 'selector': 'CatBoost SHAP', 'K': '40'}: full-DEV selection unavailable on this machine

## Notes

- **copy**: 15_cohort.csv copied verbatim from todo/ (2/6 value cells were already filled)
- **copy**: 16_l1.csv copied verbatim from todo/ (12/12 value cells were already filled)
- **copy**: 21_brier.csv copied verbatim from todo/ (0/84 value cells were already filled)
- **plan**: 12 selector fits are not covered by frozen evidence and will be recomputed where feasible
- **plan**: 4 third-dataset selector refits skipped (--third-refits skip, --third-matrix auto); their cells stay blank
- **09**: {'dataset': 'homecredit', 'backbone': 'lr', 'selector': 'Stable core + LLM fill', 'K': '20'}: archive report psi_mean=0.0125 psi_median=0.0053 psi_max=0.0774 vs recomputed 0.0125/0.0053/0.0774
- **09**: {'dataset': 'homecredit', 'backbone': 'lr', 'selector': 'mRMR', 'K': '20'}: archive report psi_mean=0.0133 psi_median=0.0057 psi_max=0.0774 vs recomputed 0.0133/0.0057/0.0774
- **09**: {'dataset': 'homecredit', 'backbone': 'catboost', 'selector': 'RFE CatBoost', 'K': '40'}: frozen full_baseline_v1 PSI available for 37 shared columns; max abs difference 9.96e-17
- **09**: {'dataset': 'lendingclub_v2', 'backbone': 'lr', 'selector': 'Pure LLM', 'K': '20'}: archive report psi_mean=0.0044 psi_median=None psi_max=None vs recomputed 0.0125/0.0087/0.0357
- **06**: domain-rule name screen flagged 5 of 529 Home Credit candidates (markers TARGET/LABEL/BAD_RATE/OUTCOME/FUTURE); candidate-set membership uses the union of the cached fold-local LLM candidate lists (fold1: 374, fold2: 369, fold3: 365, fold4: 374, fold5: 372, full_dev: 391); the manuscript's 373 is not reproducible from any file on disk (TODO item 22a)
- **01**: candidate sets per partition (cache): {fold1: 374, fold2: 369, fold3: 365, fold4: 374, fold5: 372, full_dev: 391}
- **01**: wrote 01_obfuscation.csv, obfuscated_lists_armA.csv, obfuscated_lists_armB.csv, homecredit_named_rankings.csv, 01_obfuscation_named_control.csv, 01_obfuscation_pairwise.csv, 01_obfuscation_settings.txt

## Cell sources

- recomputed: 63 cells
