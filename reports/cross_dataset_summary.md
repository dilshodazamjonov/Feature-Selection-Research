# Cross-Dataset Summary

## Main Cross-Dataset Conclusion

Across both datasets, LLM screening is competitive and consistently low-drift. mRMR remains the strongest exact-stability reference, especially when the question is repeatable feature identity rather than semantic coverage or drift. Home Credit favors `stable_core_llm_fill`, while LendingClub favors pure `llm`. The contribution is not universal dominance; it is LLM-assisted first-stage screening as a useful, drift-aware candidate generator.

## Cross-Dataset Comparison Table

| dataset     | best LR selector     | best LR OOT AUC | best CatBoost selector | best CatBoost OOT AUC | strongest non-LLM baseline | mRMR OOT AUC | best LLM-family delta vs mRMR | LLM-family mean feature PSI | non-LLM mean feature PSI | best exact-stability selector | key caveat                                                                                                                                                                    |
| ----------- | -------------------- | --------------- | ---------------------- | --------------------- | -------------------------- | ------------ | ----------------------------- | --------------------------- | ------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| homecredit  | stable_core_llm_fill | 0.7489          | stable_core_llm_fill   | 0.7683                | mrmr                       | 0.7668       | 0.0015                        | 0.0086                      | 0.0142                   | mrmr                          | Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation. |
| lendingclub | llm                  | 0.6982          | llm                    | 0.7165                | mrmr                       | 0.706        | 0.0105                        | 0.0054                      | 0.3069                   | mrmr                          | LendingClub uses the processed leakage-audited path; OOT has a higher bad rate than DEV and is a harder validation period.                                                    |

## Performance Pattern

LLM-family selectors sit near the top of the OOT leaderboard on both datasets, but the margins over mRMR are small. The safest interpretation is that LLM screening is useful as a first-stage helper, not that it replaces mRMR or universally dominates statistical selectors.

## Exact Stability Pattern

mRMR and deterministic baselines remain important exact-stability references. Exact feature stability does not by itself settle the research question, because a perfectly repeatable selector can still be semantically narrow, higher drift, or weaker on OOT discrimination.

## Drift Pattern

The post-run PSI evidence supports the lower-drift part of the LLM claim more strongly than the performance-dominance claim. LLM-family selected pools generally avoid high average selected-feature PSI, while PCA is the recurring drift and performance caution case.

## Semantic Coverage Pattern

Semantic coverage is dataset and metadata-rule dependent. Home Credit has clearer source-table and business-concept separation. LendingClub previously overused `other`; the revised mapping makes the coverage evidence more interpretable but should still be treated as report-layer relabeling rather than changed selection results.

## Dataset-Specific Behavior

Home Credit supports the stable-core hybrid most clearly. LendingClub supports the pure LLM selector more clearly and also provides a leakage-audited external validation setting with a harder OOT period because OOT bad rate is higher than DEV.

## Final Claim Wording

Use this wording: LLM screening is useful as a first-stage helper. Do not say LLM replaces mRMR. Do not say LLM universally dominates statistical selectors.

## Caveats

- Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation.
- Paired fold tests do not strongly support many OOT gains; small AUC gaps should be described cautiously.
- The LendingClub semantic grouping improvement is metadata/report relabeling only and does not change selected features or model results.
