# Prompt 7 baseline and preservation record

Authenticated before any selector code was written.

## Git state at Phase 0

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD | `e48a848436893377917be89c4b91b0351d1f7b11` |
| HEAD subject | `style: apply parked import and lint cleanups` |
| `git status --short` | empty (clean tree) |
| `git diff --stat` | empty |
| Sync with `origin/main` | in sync, nothing ahead or behind |

## Prompt 6 separability: already resolved

The Prompt 7 handoff anticipated that Prompt 6 might still be uncommitted and
that a dedicated checkpoint commit might be needed. **That is not the current
state.** Prompt 6 was committed and pushed before Prompt 7 began:

| Commit | Subject | Contents |
|---|---|---|
| `07777f3` | `research: freeze cross-dataset voting inference evidence package` | 24 files, 6,823 insertions: `src/credit_risk_fs/analysis/`, `configs/analysis/`, both Prompt 6 scripts, six Prompt 6 test files, the Prompt 6 audit root, and the stage document |
| `e48a848` | `style: apply parked import and lint cleanups` | `clip/checkpointing.py` import wrap and `experiments/result_paths.py` lint cleanup |

Consequences for Prompt 7:

- No Prompt 6 checkpoint commit was created, because one already exists and is
  cleanly separated at `07777f3`.
- **The handoff note about `src/credit_risk_fs/clip/checkpointing.py` is stale.**
  That file was an uncommitted user modification at the start of the *previous*
  session. It was committed in `e48a848` at the user's explicit instruction to
  push the parked stash along with the Prompt 6 freeze. There is therefore no
  unrelated working-tree change to exclude, and Prompt 7 neither modified,
  restored, stashed, nor staged that file. `git diff -- src/credit_risk_fs/clip/checkpointing.py`
  was empty at Phase 0.
- No `git add -A`, `git add .`, broad glob, `reset`, `checkout`, or `clean` was
  used at any point. Every Prompt 7 path was staged explicitly.

## Prompt 6 evidence authenticated, not recomputed

`tests/test_voting_evidence_provenance.py` was re-run at Phase 0 and passes,
which re-verifies without recomputation:

- the eight pinned frozen inputs still hash-match;
- all 16 run manifests still match their prediction and selection artifacts on
  disk;
- the Kuncheva universe (529 / 675) still agrees across the protocol, the run
  matrix, and all 80 saved fold rankings;
- the published package's own consistency assertions still hold
  (12 comparisons, 4 Holm families, 2,000 bootstrap replications, PASS status).

A byte-level hash snapshot of all 41 published Prompt 6 package files was written
to `prompt_06_package_hashes_baseline.json` so any later change is detectable.
The three version-controlled Prompt 6 audit JSONs are unchanged.

## Known pre-existing suite failure (baseline exception, not Prompt 7 work)

`tests/test_manual_research_orchestration.py::test_existing_pilots_and_isolated_capacity_evidence_remain_unchanged`

```
assert len(list((ROOT / "results/runs").glob("*/cdv1-0[01][0-9]-*"))) == 14
E   AssertionError: assert 16 == 14
```

Status: **failing before Prompt 7 and still failing after.** The assertion
expects 14 `cdv1-0NN` run directories; there are 16 because Prompt 5 completed
runs 015 and 016. This is a Prompt 5 preservation assertion whose expected count
was never updated. Prompt 7 deliberately did **not** edit it: silently relaxing a
guard that exists to detect unexpected run directories is exactly the kind of
change this audit trail is meant to prevent. It is recorded here as a baseline
exception and is reported separately from Prompt 7's own gates.

## Preservation guarantees asserted by Prompt 7 tests

- `tests/selectors/test_lightweight_integration.py::test_integration_writes_nothing_into_protected_roots`
  proves no fixture artifact reaches `results/` or `D:/ResearchFindings/results`,
  and that the repository write barrier rejects the legacy root before the
  script's own redundant check.
- `tests/selectors/test_lightweight_registry.py::test_voting_protocol_voters_are_unaffected`
  proves the frozen voting protocol still names exactly `rf_corr_mrmr` and
  `boruta`, and that `rf_corr_mrmr` still resolves to the legacy algorithm.
- `tests/selectors/test_lightweight_registry.py::test_pre_existing_registry_routes_still_resolve`
  proves `boruta`, `boruta_rfe`, `rfe`, `pca`, `domain_rule_baseline`, and `none`
  are unchanged.
