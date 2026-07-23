# Cross-Dataset Voting Research Runbook v1

This is the sole manual launch procedure for the frozen 16-configuration research workflow. Prompt 6 validated the entry point with planning, mocks, fixtures, and temporary roots; Prompt 6 did **not** execute the command, any research ID, any real DEV fold, or any OOT access.

The command must be run from `D:\python projects\Research` with a clean working tree at the annotated tag `cross-dataset-voting-pre-execution-v1`. Do not edit code or configuration after DEV begins. The command authenticates the Git commit/tag, frozen hashes, Python environment, and dependency lock before any data access.

## One-command manual execution

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

This one invocation performs a live CPU/resource preflight and then runs sequentially with one research run, one fold, zero loader workers, at most four estimator threads, GPU disabled, the unchanged RAM/disk/headroom policy, and the validated LendingClub memory-safe refinement.

Expected scale and phases are:

1. Authenticate the release and expand exactly 16 IDs: 12 voting runs and four rerun-required references.
2. Complete and validate all 80 DEV fold executions (five per configuration). OOT is inaccessible in this phase.
3. Freeze the fully validated 16-configuration set. Any missing/invalid DEV artifact closes the OOT barrier.
4. Perform 16 sequential full-DEV fits and one locked OOT evaluation per configuration. Early OOT results cannot change later configurations.
5. After all 16 OOT runs validate, publish the consolidated 12 paired comparisons, fixed bootstrap/DeLong inference, within-family Holm corrections, and final completeness evidence.
6. Run the repository/result validator. Terminal success is the line `CROSS_DATASET_VOTING_RESEARCH_COMPLETE` with exit code 0.

No wall-clock duration is promised. The workload includes 80 fold fits, 16 full-DEV fits, and large projected datasets; the monitor may stop safely when a frozen resource guardrail is reached.

To interrupt safely, press `Ctrl+C` once and allow the supervisor to preserve the current checkpoint and controlled stop evidence. Do not delete a partial run or `_SUCCESS` marker. To resume, return to the same clean tagged commit and invoke the exact same command above. Resume validates Git/configuration provenance and every finalized artifact before reusing completed folds or phases; completed runs remain immutable.

A controlled failure prints `CONTROLLED_STOP <stable-reason>`, returns a non-zero exit code, and leaves resumable evidence. Do not raise limits, add `--force`, change seeds/features/budgets/models, enable GPU, or work around the DEV-to-OOT barrier. If code or configuration must change, stop and create a newly reviewed protocol/release rather than resuming this one.

Frozen SHA-256 references:

- Scientific protocol: `f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0`
- Row-alignment contract: `fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0`
- Voting protocol: `51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49`
- Execution policy: `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012`
- LendingClub memory refinement: `4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7`
