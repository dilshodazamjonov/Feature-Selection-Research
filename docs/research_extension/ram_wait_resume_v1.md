# RAM wait-and-resume policy v1

Status: **implemented and synthetic-tested; real baseline not resumed by the agent**

## Incident finding

The previous cell-003 stop was a controlled supervisor decision, not a Python
out-of-memory exception. The final resource sample crossed the old 8 GiB
available-system-RAM abort floor: available RAM reached 8,045,408,256 bytes
(7.49 GiB). The process-tree peak recorded for the attempt was 16,452,210,688
bytes (15.32 GiB), below its old 28 GiB process-RSS abort setting. An earlier
heartbeat showed about 7.7 GiB process RAM and about 15 GiB available, but that
was not the terminal sample. There was no operational 15 GiB abort floor in the
authenticated policy.

## Runtime behavior

`configs/execution/ram_wait_resume_v1.yaml` is a runtime-only policy, separate
from the frozen scientific configuration and historical capacity policy.

- The emergency boundary is `max(1 GiB, 2% of total physical RAM)`.
- The default recovery threshold is 4 GiB available RAM.
- RAM is sampled every 5 seconds.
- Recovery must be observed on three consecutive samples; a lower sample resets
  the counter.
- RAM waiting is indefinite until stable recovery, Ctrl+C, or a genuine
  unrecoverable condition.
- The parent command remains alive and automatically continues after recovery.
- Process-tree RSS is recorded but never causes a warning or stop by itself.
- The historical available-RAM and process-RSS fields in
  `local_laptop_safe_v1.yaml` remain only to preserve old artifact/configuration
  identity; they are not operational termination limits.
- Low RAM no longer fails preflight. The inter-run readiness barrier waits before
  spawning the next cell.
- RAM-wait duration is subtracted from cell and stage wall-clock accounting.

The resource artifact records the resolved emergency margin, configured and
resolved recovery threshold, check/log intervals, stable-check count, each RAM
wait transition, total waiting duration, and active-computation duration.

## Cooperative and opaque work

CSV and LendingClub identity-sidecar reads are chunked. The worker checks the RAM
gate before requesting each next chunk, before concatenation/alignment, before
table and normalization boundaries, and at the major data-preparation allocation
boundaries. Source order, projections, dtypes, row contracts, folds, and selector
inputs are unchanged.

Before a new stage or large loader allocation, a direct worker reading prevents
the operation from starting below the recovery threshold. The worker waits at
that safe boundary while the parent applies the same three-check recovery rule.

Opaque library calls cannot be paused internally. If the emergency boundary is
crossed after CatBoost, Boruta, SHAP, pandas concatenation, or another opaque call
has started, the Windows supervisor suspends only the authenticated spawned
worker process tree, retains the parent and artifacts, then resumes the same
process tree after stable recovery. It does not silently restart the algorithm.
If the OS refuses an all-or-nothing process-tree suspension, the supervisor fails
closed with `ram_pause_unavailable`; the checkpoint remains explicitly resumable.

Literal OOM recovery is not guaranteed. Python/Windows may raise `MemoryError`,
and the OS may kill a process if physical RAM plus pagefile capacity is exhausted.
A Python `MemoryError` is recorded honestly with its full traceback in
`logs/debug.log`. An unexplained OS kill is a worker crash, not a normal RAM wait.

## Existing checkpoint compatibility

Cell `fbv1-003-lendingclub_v2-lr-full-features-s42` was created at predecessor
commit `a66f27f1481943c72293a93e6ada09e5f4b39ec1`. The compatibility file
`configs/execution/full_baseline_ram_wait_compatibility_v1.json` permits only
that exact historical RAM-stopped identity to cross the mechanics-only commit
boundary. It authenticates the run ID, original RAM stop, frozen configuration,
data/protocol/row-alignment/config hashes, scientific implementation files, and
the new runtime files. Any mismatch fails closed. Completed cells remain
immutable and are skipped.

## Logging

Wait entry, each five-minute update, and resume are priority records flushed to
both the terminal and append-only `logs/runs.log`, for example:

```text
[2026-07-29 12:10:00 UTC] WAIT   | LendingClub loading paused | Process RAM 7.7 GiB | Available 1.2 GiB
[2026-07-29 12:15:00 UTC] WAIT   | LendingClub loading paused | Waiting 5m | Process RAM 7.6 GiB | Available 1.5 GiB
[2026-07-29 12:19:20 UTC] RESUME | Available RAM stable at 5.1 GiB | Continuing LendingClub loading
```

## Manual commands

Read-only status:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py --status
```

Resume the existing incomplete pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_baseline.py
```

The implementation agent did not run the resume command.
