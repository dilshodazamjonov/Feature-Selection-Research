# `boruta_random_forest` implementation evidence

Implementation: `src/credit_risk_fs/selectors/heavy/boruta_rf.py`
Identity: `boruta_random_forest` / "Boruta (random forest)" /
`boruta_random_forest_confirmed_tentative_v1`

## Engine and configuration

Reuses the installed dependency and the authenticated estimator path. **Nothing
was installed or replaced.**

| Element | Value |
|---|---|
| Engine | `boruta.BorutaPy` |
| Forest | `sklearn.ensemble.RandomForestClassifier` |
| `n_estimators` (forest) | 500 |
| `max_depth` | 6 |
| `class_weight` | `None` |
| `n_estimators` (engine) | `"auto"` |
| `max_iter` | 10 |
| `perc` | 100 |
| `alpha` | 0.05 |
| `two_step` | `True` |
| `verbose` | 0 |
| Seed | `random_state` → both forest and engine |
| Threads | `n_jobs`, validated positive |

The forest profile is copied from the legacy `BorutaSelector` so the two remain
comparable. `stop_reason_` is captured when the engine provides it, otherwise a
recorded `engine_completed_max_iter_N` marker.

**The tiny profile used by the synthetic tests (40 trees, depth 4, `max_iter` 8–10)
is not the research configuration.** Every result records that caveat in
`configuration["profile_note"]`. Prompt 9 measures real single-fold cost and
freezes production.

## The audit finding this method exists to fix

`BorutaSelector` reads only `support_` and `ranking_`. `support_weak_` is never
accessed, so **the tentative state is discarded entirely** and a caller cannot
distinguish a tentative feature from a rejected one. Asserted by
`test_legacy_boruta_selector_is_unchanged_and_lacks_tentative`.

This implementation preserves all three states. Confirmed wins when a feature
appears in both `support_` and `support_weak_`, so the states stay disjoint and
partition the candidate universe exactly once.

## Selection modes

| Mode | `k` | Behaviour | Padding |
|---|---|---|---|
| `natural_confirmed` (default) | ignored | all confirmed features; tentative and rejected reported separately | none; ignoring `k` is recorded as a warning |
| `confirmed_top_k` | required | top `k` confirmed by engine rank | **never**; short of `k` → `infeasible_natural_support` |
| `confirmed_then_tentative` | required | confirmed first, then tentative | tentative allowed; **rejected never**; short of `k` → `infeasible_natural_support` |

`natural_selected` is **confirmed-only in every mode**, including
`confirmed_then_tentative`, whose result carries the explicit warning
"confirmed_then_tentative is a matched-budget adaptation, not natural Boruta
support". No mode can fill from rejected features; a full rejected-feature ranking
would be a separate methodological decision and a separate implementation identity.

## Ordering

Within each state, features are ordered by `(engine ranking_, authenticated
candidate order)`. The published ranking is confirmed → tentative → rejected, so
the state ordering is explicit in the artifact and no mode has to invent one.
Verified with a deliberate engine-rank tie in
`test_confirmed_ordering_uses_engine_rank_then_candidate_order`.

## Observed evidence (deterministic stub)

Stub: confirmed `{alpha, bravo}`, tentative `{charlie, delta}`, rejected
`{echo, foxtrot}`.

| Mode | `k` | Selected | Status |
|---|---|---|---|
| `natural_confirmed` | — | `alpha, bravo` | `not_applicable` |
| `confirmed_top_k` | 4 | `alpha, bravo` | `infeasible_natural_support` |
| `confirmed_then_tentative` | 4 | `alpha, bravo, charlie, delta` | `satisfied` |
| `confirmed_then_tentative` | 6 | `alpha, bravo, charlie, delta` | `infeasible_natural_support` |

The last row is the important one: 6 was requested, 4 were returned, and the two
rejected features were **not** used to close the gap.

## Non-finite input is refused, not imputed

`BorutaPy` delegates to a scikit-learn forest, which rejects NaN — a constraint the
**legacy `BorutaSelector` shares**, since it also passes `X.to_numpy()` straight
through. This implementation checks up front and raises at
`design_matrix_validation` naming the offending columns, rather than letting an
opaque sklearn `ValueError` surface as a generic engine failure.

It deliberately does **not** impute: doing so would be a hidden preprocessing
change. The synthetic fixture median-imputes in the harness and records
`harness_median_imputation_applied: true`.

## Controlled failures

| Trigger | Stage |
|---|---|
| `k=None` in a fixed-budget mode | `budget_validation` (engine never constructed) |
| `k<=0` in a fixed-budget mode | `budget_validation` |
| unknown `selection_mode` | `ValueError` at construction |
| non-finite candidate values | `design_matrix_validation` |
| engine `fit` raises | `engine_fit` |
| `support_`/`support_weak_`/`ranking_` length mismatch | `support_extraction` |
| excluded column offered | `candidate_universe_validation` |

Empty confirmed support is a **valid natural outcome**, not a failure: it returns
an empty selection with an explicit warning and promotes nothing.

## Determinism, isolation, compatibility

- Same configuration reproduces selection, ranking, and estimator config hash.
- Global NumPy RNG state untouched.
- Corrupting labels outside the supplied partition changes nothing.
- Serialization preserves all three support states and the full
  `support_states` mapping.
- The frozen voting protocol still resolves its `boruta` voter to
  `BorutaSelector`; `allowed_in_frozen_voting = False` on this descriptor.
- A tiny real-engine test covers wiring and status only — it asserts that the three
  state counts sum to the candidate count and that a stop reason exists, **not**
  which feature was confirmed, because real Boruta is stochastic. Support-state
  policy is asserted against the deterministic stub instead.

Tests: `tests/selectors/test_heavy_boruta.py` — **22 passed**.
