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
- LLM call and cache summaries

## Explicit Non-Goals

- no calibration study
- no stacking study
- no deployment claims
- no production PD governance claim
