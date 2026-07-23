# Cross-Dataset Voting Resume After Run 011 v1

This is the sole supported continuation procedure for the interrupted `cdv1` workflow. Prompt 6.1 authenticated runs 001–010, reconciled run 011, repaired bounded worker-tree shutdown, and did not execute or resume any research run. OOT remained unopened.

Run from `D:\python projects\Research` only after confirming that `main` is clean and the annotated tag `cross-dataset-voting-resume-safety-v1` peels to `HEAD`. Do not edit code, configuration, tags, or `configs/execution/cdv1_resume_compatibility_bridge_v1.json` after resumption begins.

## One-command manual resume

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

The command authenticates the new annotated mechanics release, all frozen hashes, and the exact old/new compatibility bridge before data access. It hashes and validates the immutable DEV artifacts from runs 001–010, then resumes run 011 without repeating folds 1–2. The exact resume boundary is DEV fold 3 at `dev_data_loading`; selection encoding and both voters are recomputed for fold 3 because its in-memory Boruta work was never atomically published. Folds 4–5 and runs 012–016 then proceed sequentially.

Runs 001–010 are reusable only for the exact bridge from original commit `f00f474b6f263ee2619a178524c7c0fdf806024f` and original tag `cross-dataset-voting-pre-execution-v1` to the tagged mechanics release. Any runtime hash, frozen hash, run ID, artifact hash/size, tag, or resume-boundary mismatch fails closed. The bridge is not a general Git-drift bypass.

The unchanged execution limits remain one run, one fold, zero loader workers, at most four estimator threads, CPU only, GPU disabled, and the original RAM/disk/headroom thresholds. Every attempt confirms that the owned worker tree is gone, closes queues without joining a feeder indefinitely, drops large return payloads, records post-cleanup parent RSS/system RAM, and runs a bounded readiness barrier before the next run.

For a true resource abort, `resource_usage.json`, the run manifest, and the checkpoint retain the first `primary_stop_code`; later interrupts or worker errors appear under `secondary_events`. The supervisor requests cooperative stop for at most 20 seconds, waits at most 10 seconds after process-tree termination, then waits at most 10 seconds after forced kill. Including bounded queue collection, the configured supervisor bound is approximately 41 seconds. A failure to remove the exact owned PID/create-time tree is recorded as `worker_tree_termination_failed` and blocks further work.

The historical run-011 record is intentionally not relabelled: its structured evidence records `manual_interrupt`, no resource warning, and no resource abort latch. The `ram_system_headroom` reserve warning was emitted by run 010, which recovered and completed without crossing the unchanged 8 GiB abort floor.

After a controlled stop, wait for PowerShell to return and check that no process command line contains both this repository path and the research launcher/worker module. A read-only check is:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" | Where-Object { $_.CommandLine -match 'python projects\\Research' -and $_.CommandLine -match 'cross_dataset_voting|credit_risk_fs' } | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine
```

If the check returns nothing and the working tree/tag/bridge remain unchanged, rerun the same one-command manual resume. Controlled resource stops and incomplete phases return non-zero. Never delete, rename, move, or manually edit the interrupted run directory; the checkpoint owner alone decides what can be reused or quarantined.

All 16 DEV configurations must validate before the configuration lock is created or any OOT loader can run. Only then may the same workflow continue through the locked full-DEV/OOT phase. Prompt 6.1 did not execute the resume command shown above.
