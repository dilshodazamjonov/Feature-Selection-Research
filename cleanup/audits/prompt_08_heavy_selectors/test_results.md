# Prompt 8 validation and regression gates

All commands from `D:\python projects\Research` with `.\.venv\Scripts\python.exe`.

## Gate results, in order

| # | Gate | Command | Result |
|---|---|---|---|
| 1 | Lint / type checks | — | **Not applicable, not claimed as passed.** `pyproject.toml` configures no `[tool.ruff]` / `[tool.mypy]` and neither is installed. Introducing a linter is outside Prompt 8's scope. |
| 2 | Pre-edit Prompt 7 focused tests | `pytest tests/selectors -q` | 155 passed — matches the handoff exactly |
| 3 | Prompt 8 RFE tests | `pytest tests/selectors/test_heavy_rfe.py -q` | **20 passed** |
| 4 | Prompt 8 Boruta tests | `pytest tests/selectors/test_heavy_boruta.py -q` | **22 passed** |
| 5 | Prompt 8 CatBoost-SHAP tests | `pytest tests/selectors/test_heavy_catboost_shap.py -q` | **25 passed** |
| 6 | Leakage / seed / ordering / budget / config / registry / serialization | `pytest tests/selectors/test_heavy_common.py -q` | **46 passed** |
| 7 | Synthetic heavy integration fixture | `python scripts/verify_heavy_selectors.py` | **PASS**, 4 cases, 0 failures, 3.3 s |
| 8 | Historical RFE / Boruta / voting compatibility | included in gate 6 | passed |
| 9 | Repaired Prompt 5 run-identity preservation | `pytest tests/test_manual_research_orchestration.py -q` | **15 passed** (was 14 passed, 1 failed) |
| 10 | Prompt 6 package consistency and hashes | `pytest tests/test_voting_evidence_provenance.py -q` + hash comparison | 10 passed; **41 / 41 files byte-identical** |
| 11 | Full repository suite | `pytest tests -q` | **870 passed, 0 failed, 31 skipped** in 205.84 s |
| 12 | compileall | `python -m compileall -q src/credit_risk_fs/selectors scripts/verify_heavy_selectors.py scripts/verify_lightweight_selectors.py src/credit_risk_fs/experiments/config.py` | exit 0 |
| 13 | Repository validator | `python cleanup/tools/validate_repository_state.py --root .` | `active_results.status = passed`, 388 artifacts verified, 20 registered runs, 12 removed paths absent |
| 14 | Diff inspection | `git status --short`, `git diff --check` | only Prompt 8 paths plus the authenticated assertion repair; whitespace clean; `clip/checkpointing.py` empty diff |

Selector suite total after Prompt 8: **279 passed** in 13.5 s.

## Suite movement

| Metric | Prompt 7 baseline | After Prompt 8 | Delta |
|---|---|---|---|
| Passed | 745 | **870** | **+125** |
| Failed | 1 | **0** | **−1** (resolved by authenticated evidence) |
| Skipped | 31 | 31 | 0 |
| Warnings | 107 | 107 | 0 |

The 107 warnings are all pre-existing `PerformanceWarning`s in the LendingClub
feature builder, untouched by Prompt 8.

No test was weakened, deleted, or converted to a skip. The one added
`pytest.skip` guards `test_published_fixture_evidence_is_consistent_when_present`,
which skips only when the generated fixture artifact is absent and asserts a clean
pass when it is present.

## Required scoping change to two Prompt 7 files

Because heavy methods share the Prompt 7 registry, two places that assumed *every
registered method is cheap* had to be scoped:

- `scripts/verify_lightweight_selectors.py`
- `tests/selectors/test_lightweight_integration.py`

Both now iterate `method_ids_by_cost_class("light")` instead of every ID. This
**narrows scope; it does not weaken any assertion** — the same five light methods
are still asserted in full, and the heavy methods are covered by
`test_heavy_integration.py` plus 113 focused tests. A hard-coded name list was
deliberately avoided so a future method cannot silently escape either fixture.

Both files were left otherwise untouched, and the light fixture still reports
`PASS (5 selectors, 0 failures)`.

## Numeric tolerances and why they suffice

| Test | Tolerance | Justification |
|---|---|---|
| CatBoost-SHAP oracle vs direct native call | `1e-12` absolute | CatBoost fitting is deterministic for a fixed seed, thread count, and `task_type="CPU"`. The only expected difference is float64 accumulation order inside `mean`. Observed SHAP magnitudes are ~1e-1, so this is ~11 orders of magnitude tighter than any meaningful difference. |
| Serialized score round-trip | `abs=0.0` — exact | JSON round-trips float64 losslessly at full repr. |
| Explanation-sample prevalence | `< 0.02` absolute | Stratified allocation rounds to whole rows, so a small discretization gap is expected; 0.02 on a 100-row sample is two rows. |

No stochastic assertion uses a wide tolerance to obtain a green result. Boruta
support-state policy is asserted against a **deterministic stub**, and the tiny
real-engine test asserts only wiring and status (state counts sum to the candidate
count; a stop reason exists) — never which feature was confirmed.

## Scientific-safety assertions

| Assertion | Evidence |
|---|---|
| No real dataset loaded | `synthetic_fixture_results.json` → `real_dataset_loaded = false`; no Home Credit / LendingClub loader imported in any Prompt 8 module |
| No OOT access | `oot_data_loaded = false`; no OOT path, prediction file, or Prompt 6 metric read |
| No real fold, pilot, or matrix executed | `real_fold_executed = false`; the only script builds a 500-row synthetic frame |
| No write to the legacy root | repository write barrier rejects it before the script's own check |
| No write to active results | `results/` file set identical before and after every heavy fit |
| No `catboost_info/` in the working tree | absent; `allow_writing_files=False` forced |
| No run-index or manifest mutation | validator: 20 registered runs, 388 artifacts verified, unchanged |
| No Prompt 6 hash change | 41 / 41 identical |
| No Prompt 7 identity change | asserted by `test_prompt_07_method_identities_are_unchanged` |
| No frozen-voting change | `ELIGIBLE_VOTERS == ("rf_corr_mrmr", "boruta")`; `boruta` still resolves to `BorutaSelector` |
| No uncontrolled fallback | `ControlledSelectorFailure` has no recovery path; asserted for training failure, SHAP failure, engine failure, shape/finiteness violations |
| No hidden padding of natural support | asserted for `confirmed_top_k`, `confirmed_then_tentative`, and RFE's absent natural support |
| No unapproved dependency | nothing installed; `boruta`, `catboost`, `sklearn` were all already pinned |
| No raw runtime log committed | `logs/` remains git-ignored and untracked |
