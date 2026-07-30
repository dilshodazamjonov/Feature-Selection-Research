# Full-baseline Cell 004 safe resume

Status: **prepared; real resume not started by the agent**

## Authenticated stopped state

The original repository state was `main` at
`9230e1599eb4099ff255f514590f973f6e13f467`, with no worktree changes. Control
artifacts authenticate Cells 001, 002, and 003 as immutable completed cells.
Cell `fbv1-004-lendingclub_v2-catboost-full-features-s42` is historically
`timed_out` with `stop_code=wall_clock_limit`.

The supervisor recorded 10,800.025 active seconds against the 10,800-second
limit, a controlled stop lifecycle, exit code 15 after the grace period, complete
child/queue cleanup, and zero survivors. Its last authenticated stage is
`data_validated`; it has no completed folds in the checkpoint and no `_SUCCESS`
marker. Fifteen fold/final-selection files are unfinalized attempt evidence, not
a completed checkpoint.

The stop was recorded at approximately 00:52 local time. Windows booted at
approximately 05:07, more than four hours later, so the restart did not cause the
stop. Read-only process and lock inspection found no worker, orphan, or execution
lock. No dataset was opened for these checks.

## Recovery authorization

`configs/execution/full_baseline_timeout_recovery_cell_004_v1.json` is an
append-only authorization record. It records the historical state, stop reason,
checkpoint identity, frozen scientific hashes, current recovery-code hashes,
Cells 001-003 artifact hashes, the exact 15-file partial inventory, workload
policy, validator version, timestamp, checks, and the decision
`RESUMABLE_FROM_CELL_BOUNDARY`.

The live validator is fail-closed. Authorization is rejected if any control file,
identity, scientific/runtime file, earlier completed artifact, or partial file
differs; if completion evidence appears; if the controlled-stop and cleanup
evidence is incomplete; if a worker/orphan or lock exists; or if the repository
is dirty. `timed_out` is never made generically resumable.

At execution time the checkpoint layer independently requires the complete
passing validation result. The restarted attempt begins at the Cell 004 boundary,
not Fold 5 or the interrupted CatBoost fit. Before new work, it moves unfinalized
outputs and snapshots the prior manifest, checkpoint, and resource evidence under
`incomplete/attempt_history/attempt_01`. The prior 10,800.025 active seconds stay
in history; the new attempt receives a fresh active timeout. All new outputs
continue to use atomic artifact/checkpoint handling.

## Workload and timeout correction

The earlier defect used only selector cost: `full_features` is light, so a cell
whose final model was CatBoost incorrectly received the three-hour light limit.
The versioned runtime policy now resolves each component and takes the maximum
cost class and maximum timeout across dataset, selector, and final model.

For Cell 004:

- selector cost: `light`;
- final-model cost: `heavy`;
- dataset cost: `light`;
- effective cost: `heavy`;
- fresh active-computation wall-clock limit: 21,600 seconds (6 hours).

Six hours reuses the repository's established Boruta heavy ceiling and adds a
three-hour margin beyond the observed attempt. RAM waiting remains excluded from
active-computation time.

## Exact operator commands

From `D:\python projects\Research`, inspect the data-free plan first:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py --plan-resume
```

It must report 001-003 `SKIP`, Cell 004 `RESTART`, boundary `cell_boundary`,
effective cost `heavy`, and timeout `21600s (6h)`. Then, and only when ready to
start the real workload, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py
```

The agent did not execute either command while preparing this repair and did not
load, inspect, hash, or evaluate OOT data.
