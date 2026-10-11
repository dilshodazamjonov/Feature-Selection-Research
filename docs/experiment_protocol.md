# Experiment protocol

The study compares classical feature selection with LLM-assisted pre-screening on three credit portfolios, Home
Credit, LendingClub v2 and Stability 2024, under one protocol run by `scripts/run_study.py` (configuration:
`configs/study/`). The top-level `README.md` gives the full description; this file lists the fixed elements.

## Research questions

- **RQ1.** Does an LLM pre-screened subset discriminate at least as well as a classically selected one on a held-out
  population scored once by a frozen pipeline, under a fixed budget?
- **RQ2.** Are the selected subsets reproducible across development folds?

## Fixed elements

- **Split.** DEV and HO along each dataset's ordering index; five expanding-window folds within DEV with one index unit
  between training and validation partitions.
- **Candidate set.** Features with at most 90 % missing values on the training partition of the first fold; the same
  set for every selector.
- **Strategies.** Classical: Domain rules, Random K, PCA, IV/WOE, mRMR (mutual information), Boruta RF, IV then Boruta,
  RFE CatBoost, CatBoost SHAP. LLM-assisted: Pure LLM, LLM then mRMR, LLM then Boruta, Stable core + LLM fill. Reference:
  Full features. Supplementary: the classical rank-voting family (Home Credit, LendingClub v2).
- **Backbones and budgets.** Logistic regression with K = 20 original features, CatBoost with K = 40; one fixed
  configuration each.
- **LLM stage.** GPT-4.1 mini (`gpt-4.1-mini-2025-04-14`), temperature 0, one prompt template, definition-only records;
  rankings cached and reused (`docs/llm_prompt_protocol.md`).
- **Outcomes.** HO ROC-AUC (headline); KS, Brier, log loss, Capture@10, Lift@10; type-aware feature PSI; Nogueira
  stability and pairwise Jaccard over the five fold subsets; runtime and peak memory; LLM token usage.
- **Comparisons.** Family leaders under the HO, DEV and mixed rules; paired HO AUC differences with 95 % paired
  bootstrap intervals (2,000 resamples, seed 20260721); materiality bar 0.010 AUC; DeLong with Holm only for the voting
  family.
- **Robustness.** Native categorical handling, L1 selector, family cap and depth-0 controls, response sampling, name
  obfuscation, descriptions rebuilt from lineage, sub-populations, feature-budget sweep, matched mRMR timing.

## Non-goals

No calibration study, no profit or fairness evaluation, no deployment or production governance claim.
