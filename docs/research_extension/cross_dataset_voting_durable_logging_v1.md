# Cross-Dataset Voting Durable Logging v1

The manual research runner creates and appends UTF-8 JSON Lines to `logs\runs.log` before plan authentication or research execution. Each invocation has a unique `session_id`; the file is never truncated, is excluded narrowly by `/logs/runs.log`, and is outside result manifests and scientific hashes.

Every line is one JSON object with `schema_version`, millisecond UTC `timestamp_utc` ending in `Z`, `level`, `pid`, `session_id`, `event`, and `message`. Applicable records add `run_id`, `dataset`, `model`, `seed`, `phase`, `fold_id`, `stage`, `component`, monotonic elapsed seconds, worker/parent memory, repository-relative artifact paths, warning or stop codes, exception classes, and tracebacks. Values are bounded and sanitized; data frames, arrays, model objects, predictions, row data, targets, environment values, and secrets are not retained.

The lifecycle is visible through `logging_initialized`, `session_started`, authenticated release/hash and plan decisions, stage/component events, resource/supervisor events, checkpoint events, and one session terminal state. Potentially expensive stages have start and terminal records. The parent supervisor writes a `stage_heartbeat` at least every 30 seconds while a worker stage remains active, including available process and memory evidence. Boruta receives explicit configuration/count records and the truthful heartbeat `Boruta fit active; internal iteration unavailable.` because its installed API exposes no computation-preserving iteration callback.

The parent process owns the sole append file and terminal sink. Spawned workers use a separate bounded 1,024-record queue; routine records never block. Priority records wait for a bounded 250 ms, and any routine or priority loss is made durable through `logging_backpressure` counters. This transport is separate from stage and result queues. The listener starts before workers, drains during bounded shutdown, and never owns scientific objects. Canonical parent lifecycle, stop, and final records bypass queue pressure and flush immediately; all records currently flush immediately, exceeding the one-second ordinary-record requirement.

Interpret the latest event as follows:

- Repeating `stage_heartbeat` records mean the supervised stage is active.
- `resource_warning`, `RESOURCE_STOP_LATCHED`, `stage_aborted`, and `session_controlled_stop` identify a controlled resource stop without changing first-cause precedence.
- `stage_interrupted`, `worker_interrupted`, or `session_interrupted` identify an interrupt.
- `stage_failed`, `worker_failed`, or `session_failed` include exception evidence and a full traceback where Python supplied one.
- `worker_finalized` reports exit code, cleanup confirmation, and survivor PIDs; `session_completed` is successful runner finalization.

To watch the canonical log live in PowerShell:

```powershell
Get-Content .\logs\runs.log -Wait -Tail 50
```

To filter run 014 or one fold:

```powershell
Select-String -Path .\logs\runs.log -Pattern "cdv1-014"
Get-Content .\logs\runs.log | ConvertFrom-Json | Where-Object { $_.run_id -eq "cdv1-014-lendingclub-v2-voting-k100-catboost-s42" -and $_.fold_id -eq 5 }
```

Retain `logs\runs.log` for as long as its operational history is useful. Rotate or delete it only while the runner is stopped; a later session recreates it. Third-party libraries may still print raw progress, but canonical start, heartbeat, and completion records remain sufficient to distinguish activity from a stall. An operating-system hard kill can prevent a final application record, so use the already-flushed heartbeat and supervisor records as the last durable evidence.

The only supported continuation procedure and exact command are in [cross_dataset_voting_resume_after_run_014_v1.md](cross_dataset_voting_resume_after_run_014_v1.md).
