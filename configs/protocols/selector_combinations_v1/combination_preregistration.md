# Selector-combination preregistration v1

Status: frozen before any Prompt 10 predictive metric or prediction value was opened.

## Decision boundary

This protocol is derived only from the pre-existing roadmap in `README.md`, the
Prompt 7–10 selector/artifact contracts, the frozen Prompt 10 matrix, and the
existing candidate-pool, budget, seed, fold, and resource policies. Structural
metadata was permitted; observed AUC, Gini, KS, Lift, PSI, targets, and prediction
values were not inspected. The repository has no tracked `PLAN.md`; `README.md`
is the available pre-existing roadmap and is bound by hash in the lock.

The only retained combinations are:

1. `iv_then_boruta`: fixed IV top-100/top-200/top-300 intermediate pool, then
   canonical random-forest Boruta confirmed-only natural support. Top 200 is
   primary; 100 and 300 are bounded sensitivities. No padding is permitted.
2. `boruta_then_rfe_catboost`: Boruta-confirmed pool, then unchanged CatBoost RFE
   to LR 20 or CatBoost 40 when feasible.
3. `boruta_then_mrmr_mutual_information`: Boruta-confirmed pool, then canonical
   MI-mRMR (`n_bins=10`, `objective=mid`, seed 42) to LR 20 or CatBoost 40 when
   feasible. The legacy RF/correlation method is expressly not interchangeable.
4. `statistical_normalized_average_rank`: exactly IV, LASSO, CatBoost RFE,
   random-forest Boruta, and CatBoost-SHAP, each weighted 1/5.

For both Boruta-first refiners, support below the final budget is
`infeasible_natural_support`; support equal to it is `no_refinement_possible`;
only larger support is refined. Tentative or rejected Boruta features are never
used as filler.

## Statistical voter

All five components must match the exact ordered candidate universe and training
partition. Authentic scores/ranks are converted to tied midranks. With `p > 1`,
quality is `q = 1 - (r - 1)/(p - 1)`; with `p = 1`, `q = 1`. LASSO ranks absolute
coefficient magnitudes and retains the zero-coefficient tie block. Boruta uses
confirmed, tentative, and rejected state blocks; absent an authentic within-state
ranking, every feature in a block receives its block midrank. Legitimately
unsupported features get the declared worst quality, but missing or corrupted
component evidence invalidates the cell. The aggregate is the arithmetic mean of
five qualities. Aggregate ties resolve only by canonical universe order, and the
model-specific top 20/40 never expands across a tie.

## Pilot and gates

The pilot covers Home Credit and LendingClub v2, DEV fold 1 only, in cheapest-first
combination order: statistical voter, IV→Boruta, Boruta→MI-mRMR, Boruta→RFE. It
contains 24 evaluation cells backed by 18 unique selection identities; reuse is
allowed only for an exact authenticated model-independent selection identity.
Prompt 11 does not execute it. Full DEV stays blocked until a later authenticated
pilot-review approval lock exists; OOT additionally requires complete retained DEV
and the Prompt 10 baseline completion dependency.

## Comparisons and claims

The finite baseline comparison families are frozen in
`combination_comparison_registry.json`: each principal selector versus matched full
features and matched random-k within dataset and evaluation model. Paired DeLong,
paired target-stratified 2,000-repetition bootstrap (seed 20260721), and Holm
correction within named dataset/model/reference families are used only on aligned
saved predictions. Combination-versus-component comparisons are registered now
but remain pending.

No outcome may change membership, order, budgets, weights, adapters, missing-rank
rules, tie rules, or inclusion after this freeze.
