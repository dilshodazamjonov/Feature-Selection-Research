# Prompt 8 baseline and preservation record

Authenticated before any Prompt 8 code was written.

## Phase 0 git state — handoff matches the repository exactly

| Item | Expected (handoff) | Observed | Match |
|---|---|---|---|
| Branch | `main` | `main` | yes |
| HEAD | `4847085` | `4847085a6be9fb22345113767e936ebd552085ee` | yes |
| Prompt 7 start | `e48a848` | `e48a848` (HEAD~1) | yes |
| Prompt 6 checkpoint | `07777f3` | `07777f3` (HEAD~2) | yes |
| Prompt 7 pushed | no | `main...origin/main [ahead 1]` | yes, local only |
| `git status --short` | — | empty | clean |
| `git diff --stat` | — | empty | clean |
| `git diff -- clip/checkpointing.py` | empty | empty | yes |

`git show --stat 4847085` confirms the reported Prompt 7 content: 28 files,
5,518 insertions, 5 deletions — the `selectors/lightweight/` package, both
scripts, seven test files, the `prompt_07_lightweight_selectors/` audit root, and
`docs/research_extension/lightweight_selector_controls_v1.md`.

**No material starting fact disagreed with the handoff.** No discrepancy record
was required.

`src/credit_risk_fs/clip/checkpointing.py` was confirmed to have no diff and was
not modified, restored, stashed, or staged at any point in Prompt 8.

## Pre-edit gates

| Gate | Result |
|---|---|
| `pytest tests/selectors -q` | 155 passed (matches the handoff) |
| `pytest tests/test_voting_evidence_provenance.py -q` | 10 passed |
| Prompt 6 package hash snapshot | 41 files hashed, all matching the Prompt 7 baseline |

## Snapshotted legacy behaviour

Recorded before any change, and asserted by tests afterwards:

- `RFESelector.__init__` defaults: `n_features=50`, `step=10` (**integer**),
  `random_state=42`, `thread_count=1`; estimator
  `catboost.CatBoostClassifier`, 500 iterations, depth 6, lr 0.05, CPU.
- `BorutaSelector.__init__` defaults: `max_iter=10`, `random_state=42`,
  `n_features=None`, `n_jobs=1`; forest 500 trees, depth 6.
- `get_selector("rfe")` → `RFESelector`; `get_selector("boruta")` →
  `BorutaSelector`; `get_selector("mrmr")` →
  `RandomForestRelevanceMRMRSelector`; `get_selector("none")` → `(None, {})`.
- `rank_voting.ELIGIBLE_VOTERS == ("rf_corr_mrmr", "boruta")`.
- Prompt 7 identities: `iv_woe_quantile_binned_v1`,
  `mrmr_mutual_information_discrete_plugin_v1`, `lasso_l1_logistic_v1`,
  `random_k_local_generator_v1`, `full_candidate_features_v1`,
  `rf_relevance_correlation_redundancy`; aliases `mrmr` →
  `legacy_rf_relevance_corr`, `none_explicit` → `full_features`.

## Post-change preservation evidence

| Check | Result |
|---|---|
| Prompt 6 published package | **41 / 41 files byte-identical**, 0 changed / added / removed |
| Repository validator | `active_results.status = passed`, 388 artifacts verified, 20 registered runs, 12 removed paths absent |
| Frozen voting protocol | still exactly `("rf_corr_mrmr", "boruta")` |
| `boruta` voter resolution | still `BorutaSelector`, not the new canonical descriptor |
| Legacy selector signatures | unchanged (asserted by `test_legacy_selector_signatures_are_unchanged`) |
| Historical registry routes | `boruta`, `boruta_rfe`, `rfe`, `mrmr`, `pca`, `domain_rule_baseline`, `none` all resolve unchanged |
| Legacy budget wiring | `rfe`/`boruta` → `n_features`, `mrmr` → `k`, unchanged |
| Prompt 7 method identities | unchanged (asserted) |
| Prompt 7 artifacts | still load with no migration — the two new contract fields are optional |
| `clip/checkpointing.py` | unchanged from the Phase 0 snapshot |
| `catboost_info/` in the working tree | absent (`allow_writing_files=False` on every heavy CatBoost fit) |
| Writes to `results/` or the legacy bundle | none; asserted per method |

No `git add -A`, `git add .`, broad glob, `reset`, `checkout`, or `clean` was
used. Nothing was pushed or tagged.
