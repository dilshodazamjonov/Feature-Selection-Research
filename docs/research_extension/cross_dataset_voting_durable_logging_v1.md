# Cross-Dataset Voting Durable Logging v1

The manual research runner opens three append-only UTF-8 logs before plan authentication or research execution:

- `logs\runs.log` contains concise UTC progress lines for people and is mirrored to the terminal.
- `logs\events.jsonl` contains the detailed machine-readable audit events.
- `logs\debug.log` contains full Python tracebacks for unexpected errors.

All three files are flushed after every record, preserved across resumed runs, narrowly ignored by Git, and excluded from checkpoint eligibility, scientific manifests, and scientific hashes. If an older `runs.log` contains JSON events, the next logging session moves those events into `events.jsonl` and rewrites their visible progress as human text before appending.

Human lines use this form:

```text
[2026-07-24 06:20:10 UTC] START | Run 014 | LendingClub | Voting K=100 | CatBoost
[2026-07-24 06:22:15 UTC] INFO  | Fold 3/5 | Boruta started
[2026-07-24 06:22:45 UTC] ACTIVE | Fold 3/5 | Boruta running | Elapsed 30s | RAM 6.4 GiB | Available 18.2 GiB
[2026-07-24 06:35:40 UTC] DONE  | Fold 3/5 | Boruta completed in 13m 25s
```

The human log never prints raw JSON, PIDs, session IDs, schema versions, or full tracebacks. CatBoost and sklearn progress is suppressed around estimator fitting; the runner instead emits stage starts, 30-second heartbeats, completions, warnings, and stops. A manual `Ctrl+C` produces a short `STOP` message without a traceback. An unexpected error produces a concise `ERROR` message with a pointer to `logs\debug.log`.

The parent process owns the file and terminal sinks. Spawned workers use a bounded queue, while canonical parent lifecycle and terminal events bypass queue pressure. Queue loss is reported as a human warning and retained with counters in `events.jsonl`. Machine audit values remain bounded and sanitized; data frames, arrays, model objects, predictions, row data, targets, environment values, and secrets are not retained.

To watch progress live in PowerShell:

```powershell
Get-Content .\logs\runs.log -Wait -Tail 50
```

To filter the human log for run 014 or fold 5:

```powershell
Select-String -Path .\logs\runs.log -Pattern "Run 014"
Select-String -Path .\logs\runs.log -Pattern "Fold 5/5"
```

For programmatic audit filtering, read the separate JSONL stream:

```powershell
Get-Content .\logs\events.jsonl | ConvertFrom-Json | Where-Object { $_.run_id -eq "cdv1-014-lendingclub-v2-voting-k100-catboost-s42" -and $_.fold_id -eq 5 }
```

Retain the files for as long as their operational history is useful. Rotate or delete them only while the runner is stopped; a later session recreates missing files. An operating-system hard kill can prevent a final application record, so the last already-flushed heartbeat or supervisor record is the final durable evidence.

The only supported continuation procedure and exact command are in [cross_dataset_voting_resume_after_run_014_v1.md](cross_dataset_voting_resume_after_run_014_v1.md).
