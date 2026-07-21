# Experiment Protocol

## Main Hypothesis

LLM-based metadata screening can be useful as a first-stage helper before downstream statistical feature selection in credit scoring.

## Datasets

- Home Credit: main development dataset
- LendingClub: external validation dataset

## Selectors

Compared selectors include:
- `mrmr`
- `boruta`
- `pca`
- `domain_rule_baseline`
- `llm`
- `llm_then_mrmr`
- `llm_then_boruta`
- `stable_core_llm_fill`

The registered `mrmr` method is the preserved historical label for
`RandomForestRelevanceMRMRSelector`: random-forest impurity importance supplies
relevance, mean absolute correlation supplies redundancy, and selection is
greedy by their ratio. It is mRMR-like rather than canonical mutual-information
mRMR. `boruta` is Boruta-only. The separate `boruta_rfe` registry entry runs
Boruta → RFE with RFE enabled, but it is not part of the finalized matrix above.

## Models

Models are evaluation vehicles only:
- Logistic Regression
- CatBoost

## Preserved Metrics

The framework preserves or supports:
- OOT AUC
- OOT Gini
- OOT KS
- Lift@10
- selected feature PSI
- model score PSI
- Nogueira stability
- pairwise Jaccard stability
- semantic coverage by group
- redundancy analysis
- runtime
- feature-selection, training, prediction, evaluation, and total run time
- peak RAM and peak GPU memory where available
- LLM call and cache summaries

Active outputs follow the contract in `results/README.md`. Runs are registered
in `results/run_index.csv` and never overwrite an existing run directory.

## Explicit Non-Goals

- no calibration study
- no stacking study
- no deployment claims
- no production PD governance claim
