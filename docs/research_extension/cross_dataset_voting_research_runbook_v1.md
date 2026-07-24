# Cross-Dataset Voting Research Runbook v1

This records the original full-launch procedure for the frozen 16-configuration research workflow. Prompt 6 validated the entry point with planning, mocks, fixtures, and temporary roots; Prompt 6 did **not** execute the command, any research ID, any real DEV fold, or any OOT access. The current supported continuation is [cross_dataset_voting_resume_after_run_014_v1.md](cross_dataset_voting_resume_after_run_014_v1.md), at annotated tag `cross-dataset-voting-observability-v2`; logging operations are documented in [cross_dataset_voting_durable_logging_v1.md](cross_dataset_voting_durable_logging_v1.md).

The historical tags `cross-dataset-voting-pre-execution-v1` and `cross-dataset-voting-resume-safety-v1` remain immutable and must not be moved. Do not use a historical tag to resume.

## One-command manual execution

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

The invocation authenticates the release and expands exactly 16 IDs: 12 voting configurations and four rerun-required references. It executes one run and one fold at a time, with zero loader workers, at most four estimator threads, CPU-only execution, the unchanged resource policy, and the validated LendingClub memory refinement.

All 80 DEV folds must complete and validate before the global DEV barrier can unlock one full-DEV fit and one locked OOT evaluation per configuration. After all 16 OOT runs validate, the workflow publishes the fixed comparison and completeness evidence. OOT results cannot alter later configurations. Terminal success is `CROSS_DATASET_VOTING_RESEARCH_COMPLETE` with exit code 0.

No wall-clock duration is promised. To interrupt safely, press `Ctrl+C` once and let the supervisor finalize checkpoint and cleanup evidence. A controlled stop prints one concise `STOP` line and returns nonzero; a manual interrupt has no traceback. Do not change datasets, folds, seeds, features, budgets, models, resource limits, or tags, and do not delete partial run state.

Frozen SHA-256 references:

- Scientific protocol: `f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0`
- Row-alignment contract: `fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0`
- Voting protocol: `51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49`
- Execution policy: `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012`
- LendingClub memory refinement: `4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7`
