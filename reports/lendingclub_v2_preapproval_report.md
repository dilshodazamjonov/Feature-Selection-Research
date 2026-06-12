# LendingClub v2 Preapproval Report

This report covers metadata and feature-preparation readiness only. No CatBoost/LR/Boruta/mRMR experiment matrix was run.

## Answers

1. Final candidate features in `data/lendingclub_v2/processed/application_train.csv`: `675`.
2. Features with descriptions: `675`.
3. Description coverage is 100%: `yes`.
4. Semantic groups: `19`.
5. Dominant semantic groups are listed in `data/lendingclub_v2/metadata/semantic_group_distribution.csv`; the largest groups are shown in the inventory report.
6. Features marked `needs_manual_review`: `0`.
7. Leakage-risk columns excluded: `39` explicit columns/patterns are listed in `data/lendingclub_v2/metadata/leakage_review.csv`.
8. The feature space is richer than v1: v2 has `675` candidate features versus the v1 report's approximately `300` engineered candidates and 76 described LLM features.
9. The v2 design is more comparable to Home Credit in count and metadata coverage, while remaining LendingClub-specific and leakage-screened.
10. It is approval-ready for human inspection if the reviewer accepts the generated feature families and leakage review.
11. A full LendingClub v2 matrix rerun should not be run until human approval.
12. After approval only, run:

```bash
python scripts/run_matrix.py --dataset lendingclub_v2
python scripts/aggregate_results.py --dataset lendingclub_v2
python scripts/make_plots.py --dataset lendingclub_v2
```

## Rerun Decision

- Matrix run performed now: no.
- Full rerun required before inspection: no.
- Targeted artifact generation completed: v2 processed table and metadata inspection artifacts.