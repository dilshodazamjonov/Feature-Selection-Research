# Resolution of the stale 14-versus-16 run-count assertion

Outcome: **repaired under the narrow authorization in Prompt 8 Phase 1**, on the
basis of exact authenticated evidence. This is a correction of stale expected
evidence, not a relaxation of preservation — the repaired guard is strictly
stronger than the original.

## The failing test and its purpose

`tests/test_manual_research_orchestration.py::test_existing_pilots_and_isolated_capacity_evidence_remain_unchanged`

Purpose: detect any change to completed research evidence — pilot rows, pilot
directories, the LendingClub capacity gate, and the canonical `cdv1` run set. The
final line asserted a bare count:

```python
assert len(list((ROOT / "results/runs").glob("*/cdv1-0[01][0-9]-*"))) == 14
```

Pre-edit result:

```
AssertionError: assert 16 == 14
```

The expectation was written when the matrix held 14 completed runs and was never
updated after Prompt 5 completed runs 015 and 016.

## Authentication of the exact canonical 16-run set

| Source | Evidence |
|---|---|
| Frozen configuration lock — `results/comparisons/cross_dataset_voting_configuration_lock.json` | `status = all_dev_validated_oot_configuration_locked`, `git_commit = 3bd9d0b93421`, `git_tag = cross-dataset-voting-observability-v2`, **16 `run_ids`, 16 unique** |
| Run directories on disk | 16 matching `results/runs/*/cdv1-0[01][0-9]-*`, **16 unique basenames, 0 duplicates** |
| `results/run_index.csv` | 16 rows for exactly those IDs, **all `completed`**; 20 `cdv1-` rows total, the other 4 being `cdv1-pilot-*` |
| Prompt 6 `input_inventory.csv` | 16 run IDs, **set-equal to the lock** |

Set algebra: `lock == disk`, `lock - disk = ∅`, `disk - lock = ∅`. No extra,
duplicate, partially named, or unrelated directory is counted by the glob — the
four pilots do not match it because their names are `cdv1-pilot-*`.

The authenticated set:

```
cdv1-001-homecredit-reference-rf-corr-mrmr-lr-s42
cdv1-002-homecredit-voting-k100-lr-s42
cdv1-003-homecredit-voting-k200-lr-s42
cdv1-004-homecredit-voting-k300-lr-s42
cdv1-005-homecredit-reference-rf-corr-mrmr-catboost-s42
cdv1-006-homecredit-voting-k100-catboost-s42
cdv1-007-homecredit-voting-k200-catboost-s42
cdv1-008-homecredit-voting-k300-catboost-s42
cdv1-009-lendingclub-v2-reference-rf-corr-mrmr-lr-s42
cdv1-010-lendingclub-v2-voting-k100-lr-s42
cdv1-011-lendingclub-v2-voting-k200-lr-s42
cdv1-012-lendingclub-v2-voting-k300-lr-s42
cdv1-013-lendingclub-v2-reference-rf-corr-mrmr-catboost-s42
cdv1-014-lendingclub-v2-voting-k100-catboost-s42
cdv1-015-lendingclub-v2-voting-k200-catboost-s42
cdv1-016-lendingclub-v2-voting-k300-catboost-s42
```

## The repair

The bare count was replaced with an **exact identity-set assertion**. The
expectation is a literal frozen constant, `CANONICAL_CDV1_RUN_IDS`, and the
authoritative lock file is cross-checked against it:

```python
observed = {path.name for path in (ROOT / "results/runs").glob("*/cdv1-0[01][0-9]-*")}
assert observed == CANONICAL_CDV1_RUN_IDS
locked = json.loads(... "cross_dataset_voting_configuration_lock.json" ...)
assert set(locked["run_ids"]) == CANONICAL_CDV1_RUN_IDS
assert len(locked["run_ids"]) == 16
```

Why the expectation is written out literally rather than derived from the
directory listing or from the lock file alone: a derived expectation would be
self-fulfilling. With the literal set, the guard fails if a run directory is
added, removed, or renamed **and** if the test and the authoritative registry ever
diverge.

Explicitly **not** done: no `>= 16`, no broadened glob, no warning, no skip, no
dynamically self-fulfilling expectation. No run directory was created, deleted,
renamed, or rewritten.

## Strictness verification

| Probe | Result |
|---|---|
| Literal set size | 16 |
| `observed == literal` | True |
| Would fail if one directory were removed | True |
| Would fail if one directory were added | True |

## Before / after

| | Result |
|---|---|
| Before | `tests/test_manual_research_orchestration.py`: **14 passed, 1 failed** |
| After | `tests/test_manual_research_orchestration.py`: **15 passed, 0 failed** |
| Full suite before (Prompt 7 baseline) | 745 passed, **1 failed**, 31 skipped |
| Full suite after | **870 passed, 0 failed, 31 skipped** |
