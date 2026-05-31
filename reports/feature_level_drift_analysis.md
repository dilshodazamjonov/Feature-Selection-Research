# Feature-Level Drift Analysis

## Updated LLM Top-100 Candidate Evidence

This report now separates LLM nomination from downstream statistical selection. `llm_top100_candidate_psi.csv` lists saved final-dev LLM top-100 candidates and flags whether each candidate survived into the final selected set. No model retraining or selector rerun was performed.

For LendingClub, top-100 candidate PSI was computed from the processed safe DEV/OOT frame where numeric candidate columns were available. For Home Credit, the full DEV/OOT design matrix for rejected LLM candidates is not present in saved artifacts, so rejected-candidate PSI is explicitly marked missing; selected-feature PSI remains available from per-run artifacts.

## Interpretation

- Did LLM nominate high-drift features? LendingClub candidate PSI shows the LLM top pool is mostly low-drift where PSI could be computed. Home Credit cannot fully answer this for rejected candidates without a saved DEV/OOT design matrix.
- Did mRMR/Boruta keep or reject high-drift LLM candidates? The updated breakdown keeps `in_final_selected_set` and `selected_by_downstream_stat_selector`; LendingClub can evaluate this directly, while Home Credit can only evaluate final selected features.
- Are high-PSI features caused by LLM nomination or downstream statistical selection? Current selected-feature evidence does not support a broad claim that LLM nomination itself causes high PSI. Where PSI is missing for rejected candidates, the report marks the limitation rather than inventing values.
- Dataset difference: LendingClub has enough processed safe artifact coverage for a stronger top-pool PSI audit; Home Credit requires targeted artifact generation to audit rejected candidates.

## homecredit

| selector             | psi_flag    | candidate_count |
| -------------------- | ----------- | --------------- |
| llm                  | low         | 50              |
| llm                  | unavailable | 150             |
| llm_then_boruta      | low         | 56              |
| llm_then_boruta      | unavailable | 144             |
| llm_then_mrmr        | low         | 57              |
| llm_then_mrmr        | unavailable | 143             |
| stable_core_llm_fill | low         | 39              |
| stable_core_llm_fill | unavailable | 161             |

Missing PSI reasons:

| missing_from_dev_oot_reason                                                                                                 | candidate_count |
| --------------------------------------------------------------------------------------------------------------------------- | --------------- |
| DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features | 598             |

## lendingclub

| selector             | psi_flag    | candidate_count |
| -------------------- | ----------- | --------------- |
| llm                  | low         | 74              |
| llm                  | unavailable | 78              |
| llm_then_boruta      | low         | 65              |
| llm_then_boruta      | unavailable | 87              |
| llm_then_mrmr        | low         | 71              |
| llm_then_mrmr        | unavailable | 81              |
| stable_core_llm_fill | low         | 71              |
| stable_core_llm_fill | unavailable | 81              |

Missing PSI reasons:

| missing_from_dev_oot_reason         | candidate_count |
| ----------------------------------- | --------------- |
| feature_not_in_processed_safe_frame | 287             |
| numeric_dev_oot_values_unavailable  | 40              |
