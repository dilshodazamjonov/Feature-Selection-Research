# Cross-Dataset Voting Resume After Run 014 v1

This is the sole supported continuation procedure for the current `cdv1` workflow. Prompt 6.2 authenticated runs 001–013 and run 014’s finalized folds 1–4, added observability without scientific changes, and did not execute research or open OOT.

Run from `D:\python projects\Research` only after confirming that `main` is clean and annotated tag `cross-dataset-voting-observability-v1` peels to `HEAD`. Do not edit code, configuration, tags, results, or `configs/execution/cdv1_durable_logging_compatibility_bridge_v1.json` after resumption begins.

## One-command manual resume

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

The runner authenticates the original, Prompt 6.1 safety, and observability releases; all frozen hashes; the exact compatibility bridge; and every bridged artifact hash and size before data access. It reuses completed runs 001–013 and run 014 folds 1–4. The exact earliest boundary is DEV fold 5 at `dev_data_loading`; all fold-5 work, including selection encoding and Boruta, is recomputed because no fold-5 artifact was atomically finalized. OOT remains closed until all 16 DEV runs validate.

The same unchanged CPU, parallelism, thread, RAM, disk, resource-precedence, and shutdown policies remain in force. A drift in any release tag, runtime hash, frozen hash, run identity, artifact hash/size, or boundary blocks reuse. `logs\runs.log` begins appending before authentication and is operational evidence only; it is not part of checkpoint eligibility or scientific manifests.

In a second PowerShell window, use the live-tail command documented in [cross_dataset_voting_durable_logging_v1.md](cross_dataset_voting_durable_logging_v1.md). A controlled resource stop returns nonzero after checkpoint and cleanup evidence are finalized. Wait for the launcher to return before another attempt; never delete or edit the interrupted run directory.

Prompt 6.2 did not execute the resume command shown above.
