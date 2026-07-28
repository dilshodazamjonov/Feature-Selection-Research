# Lightweight selector controls v1

Roadmap stage: Prompt 7 — lightweight selector implementation and selector-layer repair
Contract: `lightweight_selector_contract_v1`
Status: **implemented and tested**

**Prompt 7 ran no real dataset experiment.** No Home Credit or LendingClub file,
no fold, no selector pilot, no model fit, no OOT evaluation. Every result in this
stage comes from a 600-row deterministic synthetic fixture. Nothing here changes
or reinterprets the Prompt 6 evidence.

## Methods

| Registry ID | Display label | Algorithm | Supervised |
|---|---|---|---|
| `iv_woe` | Information Value (WOE binning) | total IV over WOE bins | yes |
| `mrmr_mutual_information` | mRMR (mutual information) | sequential max-relevance / min-redundancy on discrete plug-in MI | yes |
| `lasso_l1_logistic` | LASSO (L1-penalized logistic regression) | support of an L1 logistic fit | yes |
| `random_k` | Random-k control | uniform random subset, target never inspected | no |
| `full_features` | Full candidate features | no selection over eligible candidates | no |
| `legacy_rf_relevance_corr` | Legacy RF relevance / correlation redundancy | RF impurity relevance / mean absolute correlation | yes |

## Method definitions

### Information Value (`iv_woe`)

`WOE = ln(dist_good / dist_bad)` per bin, `IV = Σ (dist_good − dist_bad) · WOE`,
where the distributions are shares *within* each class. The event is class 1 =
default, matching the authenticated positive-class orientation. Numeric features
are binned by training-partition quantiles (10 bins by default); categorical
features use one bin per observed level. Missing values form an explicit
`__MISSING__` bin rather than being dropped, so informative missingness is
measured rather than discarded.

Zero-count smoothing is explicit and configurable (`zero_count_smoothing`,
default 0.5), added to the good and bad count of every bin before the
distributions are formed, keeping the totals consistent with the cells. With
smoothing at 0 and no empty class cell this reduces exactly to the classical
definition — which is what makes the hand-calculated oracle test meaningful. A
bin whose unsmoothed WOE is non-finite contributes zero rather than poisoning the
feature total with an infinity.

Bin-level evidence (`bin_table_`) carries counts, class distributions, WOE, per-bin
IV contribution, and numeric bounds, so any feature's IV can be recalculated by
hand from the artifact.

### Canonical mRMR (`mrmr_mutual_information`)

Greedy: at each step append the feature maximizing
`relevance − mean_redundancy` (the `mid` objective; `miq` uses the quotient form).
Relevance is `I(feature; target)`. Redundancy is the mean `I(feature; s)` over
already-selected `s`. Both use the discrete plug-in estimator
`sklearn.metrics.mutual_info_score` in nats, over variables discretized inside the
training partition (quantile bins, explicit missing code).

**Estimator decision.** No new dependency was installed and no randomness enters,
so the result is bit-reproducible. The k-nearest-neighbour (Kraskov) estimator
behind `mutual_info_classif` was considered and rejected: it adds `random_state`
tie-breaking noise and estimates continuous–continuous MI by a different
construction. Adopting it would be a different `implementation_id`, not a silent
redefinition of this one. Consequently the **discretization rule is part of the
algorithm's identity**, not a tuning knob, and is recorded in every result.

**Zero-relevance policy.** A feature with `I(feature; target) ≤ relevance_floor`
carries no information about default, yet under the difference objective it scores
`0 − 0 = 0` and would outrank a genuinely relevant feature that merely overlaps an
earlier pick — letting a constant column displace a predictor. Zero-relevance
features are therefore held back and appended only once the informative pool is
exhausted. They still appear in the ranking; the policy is declared in the result
configuration.

### LASSO (`lasso_l1_logistic`)

"LASSO" here means L1-penalized **logistic** regression, never least-squares
Lasso. Imputation (median), scaling (`StandardScaler`), and the penalized fit are
all estimated on the rows passed to `fit` and nothing else.

Explicit and recorded: solver (`liblinear` by default, deterministic;
`saga` permitted and stochastic), `C`, `max_iter`, `tol`, `class_weight`,
`coefficient_tolerance`, seed, and thread count. Signed and absolute coefficients
are both emitted. Non-numeric candidates raise a controlled failure rather than
being implicitly one-hot encoded, because that would change the candidate-universe
identity and therefore its hash.

Convergence failure is reported, never repaired — the solver, tolerance, penalty,
sample, and feature set are not changed behind the caller's back. Total shrinkage
to an empty support is a valid recorded outcome, not permission to substitute
another selector.

### Random-k (`random_k`)

Chance-performance control. Randomness comes from a local
`numpy.random.default_rng(random_state)`; global NumPy RNG state is never read or
written. The published ranking is the full random priority order, so the subset is
reproducible from the artifact alone. Different seeds are supported without
changing the method's identity. The target is never received.

### Full candidate features (`full_features`)

"Full features" means every candidate surviving the frozen leakage and metadata
exclusions — **not** every raw dataset column. It reuses the semantics of the
pre-existing `none` route, which stays available and continues to resolve to no
selector object. A fixed-`k` request is ignored by design and the ignore is
recorded.

## Natural versus matched-budget semantics

Only LASSO has a natural support, and the distinction is the point:

- **natural** (`k=None`): the L1 fit chooses its own subset size.
  `budget_status = not_applicable`.
- **matched_budget** (`k` given, natural support ≥ k): the top-k prefix of the
  absolute-coefficient ranking. The natural support is still published alongside.
- **matched_budget with an insufficient support**: `budget_status =
  infeasible_natural_support`. Only the natural support is returned. **The budget
  is not padded.**
- **coefficient_ranking**: the only way to extend past the natural support,
  reachable exclusively via `allow_zero_coefficient_fill=True`. Named separately so
  a padded subset can never be read as an L1 support.

## Fixed-budget behaviour

A budget larger than the eligible universe returns every eligible feature and
records `clipped_to_universe` with a warning — preserving the pre-existing
`selectors.base.resolve_feature_budget` clamp rather than inventing a new policy.
Budgets are never padded, duplicated, or redefined. An empty universe yields
`empty_universe` with an empty selection, a warning, and no exception.

## Seed and tie policies

One tie rule for every ranking selector:
`descending_score_then_ascending_feature_name`. Because the sort key is a pure
function of score and name, reordering the input DataFrame's columns cannot
reorder the output. `full_features` uses
`candidate_universe_order_preserved` instead, having no comparable score.

Seeds are recorded even where the algorithm is RNG-free (MI-mRMR is deterministic
without one, and says so via `deterministic_without_rng`).

## Legacy compatibility

`mrmr` has always meant the random-forest relevance / absolute-correlation
redundancy selector. It keeps resolving there and is now also reachable under the
accurate ID `legacy_rf_relevance_corr`. The implementation is preserved
byte-for-byte; the class already declared `canonical_mrmr = False`.

Canonical MI-mRMR is a separate ID and is **unreachable** from the legacy alias.
No historical artifact was rewritten to appear as if it had used canonical
MI-mRMR. All 16 `cross_dataset_rank_voting_v1` runs, and the `rf_corr_mrmr` voter,
used the legacy algorithm; that is recorded in the descriptor's `historical_use`.

## Limitations

1. **No performance evidence.** Prompt 7 produces infrastructure evidence only.
   Nothing here says any method is better, more stable, or more material than any
   other. That requires the Prompt 9 pilots.
2. **Fixture-scale only.** Every number was produced on 600 synthetic rows and 6
   synthetic candidates. Runtime and memory figures are not cost estimates for the
   real feature universes (529 / 675 candidates).
3. **MI-mRMR cost is quadratic in the budget.** Each step evaluates MI against
   every already-selected feature. Pair MI is cached, but a large budget on a
   full-size universe has not been profiled.
4. **The discretization rule is an algorithmic choice.** Changing `n_bins` changes
   the estimator, not a hyperparameter. It must be frozen before any real run.
5. **LASSO requires a numeric design matrix.** Categorical candidates must be
   encoded upstream; the selector refuses rather than encoding implicitly.
6. **IV binning was not tuned.** The 10-bin quantile default was not selected
   against any validation or OOT metric, and must not be.
7. **`n_jobs` no longer reaches LogisticRegression.** scikit-learn 1.8 removed its
   effect. The configured value is recorded but not forwarded.

## Handoff

Prompt 8 adds RFE, Boruta, and CatBoost-SHAP. It should register them through this
same contract and reuse `LightweightSelector`, whose budget resolution, invariant
checking, hashing, and leakage guarding are method-agnostic. Prompt 8 must not
reopen the frozen protocol, the comparison family, the bootstrap design, or any
completed run directory.

Real-data selector pilots and the compute freeze belong to Prompt 9.
