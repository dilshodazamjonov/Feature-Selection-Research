# Prompt 7 validation and regression gates

All commands run from `D:\python projects\Research` with `.\.venv\Scripts\python.exe`.

## Gate results

| # | Gate | Command | Result |
|---|---|---|---|
| 1 | Lint / type checks | — | **Not applicable.** The repository configures no linter or type checker: `pyproject.toml` has no `[tool.ruff]`, `[tool.mypy]`, or lint dependency, and neither is installed in `.venv`. Nothing was added, since introducing a linter is outside Prompt 7's scope. |
| 2 | Focused selector unit tests | `pytest tests/selectors -q` | **155 passed** |
| 3 | Registry / configuration tests | `pytest tests/selectors/test_lightweight_registry.py -q` | 26 passed |
| 4 | Serialization / path tests | `pytest tests/selectors/test_lightweight_contract.py -q` | 55 passed |
| 5 | Leakage / determinism tests | included in gates 2–4 | passed |
| 6 | Tiny-fixture integration | `pytest tests/selectors/test_lightweight_integration.py -q` | 9 passed |
| 7 | Tiny-fixture script | `python scripts/verify_lightweight_selectors.py` | **PASS**, 5 selectors, 0 failures |
| 8 | Prompt 5/6 preservation | `pytest tests/test_voting_evidence_provenance.py tests/test_manual_research_orchestration.py -q` | Prompt 6 provenance passes; one pre-existing Prompt 5 failure (see below) |
| 9 | Full repository suite | `pytest tests -q` | **1 failed, 745 passed, 31 skipped** in 162.74 s |
| 10 | compileall | `python -m compileall -q src/credit_risk_fs/selectors scripts/verify_lightweight_selectors.py src/credit_risk_fs/experiments/config.py` | exit 0 |
| 11 | Repository validator | `python cleanup/tools/validate_repository_state.py --root .` | `active_results.status = passed`, 388 artifacts verified, 20 registered runs |
| 12 | Prompt 6 package hashes | hash comparison against `prompt_06_package_hashes_baseline.json` | **41 / 41 identical**; 0 changed, 0 added, 0 removed |
| 13 | Diff inspection | `git status --short`, `git diff --check` | only Prompt 7 paths; whitespace clean |

## Suite movement

| Metric | Before Prompt 7 | After Prompt 7 | Delta |
|---|---|---|---|
| Passed | 604 | **745** | **+141** |
| Failed | 1 | 1 | 0 |
| Skipped | 31 | 31 | 0 |

No test was weakened, deleted, or converted to a skip. The one added
`pytest.skip` is in `test_published_fixture_evidence_is_consistent_when_present`,
which skips only when the tiny-fixture evidence file has not been generated —
justified because the file is a generated artifact, and when it *is* present the
test asserts a clean pass.

## The single failure is pre-existing and unrelated

```
tests/test_manual_research_orchestration.py::test_existing_pilots_and_isolated_capacity_evidence_remain_unchanged
assert len(list((ROOT / "results/runs").glob("*/cdv1-0[01][0-9]-*"))) == 14
E   AssertionError: assert 16 == 14
```

Failing identically before and after Prompt 7. The assertion expects 14
`cdv1-0NN` run directories; there are 16 because Prompt 5 completed runs 015 and
016. Prompt 7 did **not** edit it — it is a Prompt 5 preservation guard, and
relaxing it to make a suite green is precisely the change this audit trail exists
to prevent. Recorded as a baseline exception in `baseline_and_preservation.md`.

## Two latent defects found and fixed during the gates

Both were surfaced by warnings rather than failures, which is why they are worth
recording:

1. **Swallowed warnings.** `L1LogisticSelector` wrapped the estimator fit in
   `warnings.catch_warnings(record=True)` to capture convergence state. That
   context captures *every* warning, so non-convergence warnings were being
   silently discarded. Non-convergence warnings are now re-emitted to the caller.
   Regression test: `test_non_convergence_warnings_are_not_swallowed`.

2. **Deprecated estimator API.** With the swallowing fixed, scikit-learn 1.8
   reported that `penalty` is deprecated (removal in 1.10) and that
   `penalty='l1'` alongside the default `l1_ratio=0.0` is *inconsistent*. Verified
   empirically that `penalty='l1'`, `l1_ratio=1.0`, and both together produce
   bit-identical coefficients on this solver, then switched to the non-deprecated
   spelling with a version guard for older scikit-learn. The same re-emission
   then surfaced a second one — `n_jobs` has had no effect since 1.8 and is
   removed in 1.10 — so it is no longer forwarded, while the configured value is
   still recorded because thread counts are part of the declared run contract.
   Both the API spelling and the forwarding decision appear in every result's
   configuration.

## Scientific-safety assertions

| Assertion | Evidence |
|---|---|
| No real dataset loaded | `tiny_fixture_results.json` → `fixture.real_dataset_loaded = false`; no Home Credit / LendingClub loader is imported anywhere in Prompt 7 code |
| No OOT access | `fixture.oot_data_loaded = false`; no OOT path, prediction file, or Prompt 6 metric is read |
| No model trained, no selector pilot, no matrix run | Prompt 7 adds no execution entry point; the only script builds a 600-row synthetic frame |
| No write to the legacy root | `test_integration_writes_nothing_into_protected_roots`; the repository write barrier rejects it before the script's own check |
| No write to active results | same test; `_assert_scratch_is_isolated` raises `SystemExit` |
| No run-index or manifest mutation | validator reports 20 registered runs and 388 artifacts verified, unchanged |
| No Prompt 6 hash change | gate 12, 41/41 identical |
| No hidden fallback | `ControlledSelectorFailure` has no recovery path; asserted by the convergence, zero-support, non-numeric, high-cardinality, and single-class tests |
| Legacy vs canonical mRMR unambiguous | `test_canonical_mi_mrmr_is_unreachable_from_the_legacy_alias`, `test_legacy_mrmr_alias_still_loads_the_legacy_algorithm` |
| No unapproved dependency | nothing installed; `mutual_info_score` comes from the already-pinned scikit-learn |
