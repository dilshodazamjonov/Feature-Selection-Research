# Full-baseline Cell 004 safe resume

Status: **second boundary restart prepared; real resume not started by the agent**

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

The first attempt remains authenticated by
`configs/execution/full_baseline_timeout_recovery_cell_004_v1.json`. The latest
attempt is separately authenticated by
`configs/execution/full_baseline_timeout_recovery_cell_004_attempt_02_v1.json`.
The new append-only record includes the historical state, stop reason,
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
not Fold 5 or the interrupted CatBoost fit. The original stopped attempt is
already preserved under `incomplete/attempt_history/attempt_01`. Before new work,
the latest stopped attempt and its current 15 partials will be preserved under
the deterministic `attempt_02` recovery boundary. All new outputs continue to use
atomic artifact/checkpoint handling.

## Second timeout and Windows sleep correction

The six-hour retry began at 05:16 UTC. Resource samples jump from 12,579.584 to
28,817.480 elapsed seconds while process-tree CPU increases by only 8.25 seconds.
The 16,237.896-second gap authenticates a Windows sleep/hibernate interval. The
old Windows monotonic clock counted that interval, so the supervisor stopped on
wake even though corrected active time was only 12,579.645 seconds.

The supervisor now uses Windows `QueryUnbiasedInterruptTime`, which is monotonic
but excludes sleep and hibernation. Resource evidence records the clock identity,
total supervisor-awake time, detected suspended time, and that suspension was
excluded. RAM-wait time is still excluded separately.

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

Cell 004 spent roughly 2.5 hours reaching final-model fitting, and CatBoost's
observed initial estimate was another 6 hours 40 minutes. The CatBoost-final
component therefore receives a 43,200-second (12-hour) active ceiling, leaving a
practical margin while preserving controlled-stop protection. Selector and
scientific model settings are unchanged.

## Exact operator commands

From `D:\python projects\Research`, inspect the data-free plan first:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py --plan-resume
```

It must report 001-003 `SKIP`, Cell 004 `RESTART`, boundary `cell_boundary`,
effective cost `heavy`, and timeout `43200s (12h)`. Then, and only when ready to
start the real workload, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py
```

The agent did not execute either command while preparing this repair and did not
load, inspect, hash, or evaluate OOT data.
