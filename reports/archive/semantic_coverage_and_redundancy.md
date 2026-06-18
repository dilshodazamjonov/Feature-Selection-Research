# Semantic Coverage and Redundancy

## LendingClub Mapping Update

LendingClub semantic grouping was too coarse. The report-layer mapping reduced selected-feature `other` labels from 210 to 32 rows in the reviewed selected-feature artifacts. This does not change feature selection results or model metrics.

## Updated LendingClub Redundancy Table

| dataset     | model    | selector             | selected feature count | number of semantic groups | semantic group entropy if easy | largest group share | average within-group absolute correlation | max within-group absolute correlation | redundancy risk flag |
| ----------- | -------- | -------------------- | ---------------------- | ------------------------- | ------------------------------ | ------------------- | ----------------------------------------- | ------------------------------------- | -------------------- |
| lendingclub | catboost | llm                  | 40                     | 11                        | 2.1739036711627757             | 0.2                 | 0.351260866089172                         | 0.9820282606588904                    | high_max_correlation |
| lendingclub | catboost | llm_then_mrmr        | 40                     | 9                         | 1.93388039870999               | 0.35                | 0.7501241652778082                        | 0.9999999322151233                    | high_max_correlation |
| lendingclub | catboost | mrmr                 | 40                     | 10                        | 2.0233657590084295             | 0.25                | 0.6732254056571326                        | 0.9999999322151235                    | high_max_correlation |
| lendingclub | catboost | stable_core_llm_fill | 40                     | 8                         | 1.9433779257532877             | 0.25                | 0.6732254056571325                        | 0.9999999322151233                    | high_max_correlation |
| lendingclub | lr       | llm                  | 20                     | 9                         | 2.0126926438092823             | 0.25                | 0.3966709648679842                        | 0.75828048598988                      | moderate             |
| lendingclub | lr       | llm_then_mrmr        | 20                     | 8                         | 1.834371970281624              | 0.3                 | 0.4492799180853292                        | 0.7582804859898801                    | moderate             |
| lendingclub | lr       | mrmr                 | 20                     | 5                         | 1.4877983800016508             | 0.4                 | 0.9999999322151235                        | 0.9999999322151235                    | high_max_correlation |
| lendingclub | lr       | stable_core_llm_fill | 20                     | 6                         | 1.6385064445042257             | 0.35                | 0.8791402091025017                        | 0.9999999322151233                    | high_max_correlation |

## Interpretation

LLM pipelines can support business-concept coverage, but the strength of that claim is dataset and metadata-rule dependent. The LendingClub update makes concepts such as FICO score, income capacity, revolving utilization, bankcard capacity, credit-history length, recent inquiries, account-opening activity, mortgage history, derogatory events, loan terms, and exposure amount visible in the report. If AUC differences are small, this semantic interpretability can be a defensible secondary advantage, but it should not be overstated as universal superiority.
