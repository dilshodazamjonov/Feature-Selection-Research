# Resource-safe execution

## Gate and ownership

Prompt 1.1 is `PASS`; its frozen protocol and OOT lock remain unchanged. `AGENTS.md` was requested as a starting file but is absent and no repository-conventional equivalent exists.

The hardened path extends the existing experiment system:

| Concern | Canonical owner |
|---|---|
| Active/legacy root safety, run paths, atomic registry replacement | `credit_risk_fs.experiments.result_paths` |
| Run manifest, status, standard artifacts, resource summary | `credit_risk_fs.experiments.tracking` |
| Registered run lifecycle used by single and matrix entry points | `credit_risk_fs.experiments.execution` |
| Versioned policy, hardware detection, estimates, preflight | `credit_risk_fs.experiments.resource_policy` |
| Spawned worker supervision and live process-tree sampling | `credit_risk_fs.experiments.resource_monitor` |
| Validated atomic artifact publication | `credit_risk_fs.experiments.atomic_io` |
| Stage/fold checkpoint and explicit resume validation | `credit_risk_fs.experiments.checkpointing` |
| Required-column calculation and canonical loading | `credit_risk_fs.pipelines.common` and `credit_risk_fs.data.loaders.DataLoader` |
| CLI exposure | `scripts/preflight_execution.py`, `scripts/run_single.py`, and `scripts/run_matrix.py` |

`single_run.py` and the matrix `runner.py` both call `execute_registered_run`; no parallel results, tracking, runtime, or runner framework was added.

## Policy and preflight

The default profile is `configs/execution/local_laptop_safe_v1.yaml`, schema `resource_safe_execution_policy_v1`. It resolves from an explicit repository root or absolute config path. Negative, zero, contradictory, nested, oversubscribed, or capacity-impossible settings fail. Process and thread limits may only resolve downward. On smaller systems, RAM resolution retains at least 25% headroom and a 6 GiB absolute reserve where capacity permits.

Default limits are one experiment, one fold, zero DataLoader subprocesses, four estimator threads, and no nested parallelism. The worker fixes native BLAS/OpenMP pools at one and caps joblib/loky plus estimator adapters before scientific libraries load. CatBoost receives `thread_count`; supported random-forest estimators and selectors receive a positive bounded `n_jobs`. A wider dataset- or method-specific override fails.

Preflight detects logical/physical CPUs, total/available RAM, results/temp free space, GPU name, total/free VRAM, driver visibility, and process-level NVML telemetry. It verifies active/legacy separation, an active-root write probe, the legacy write barrier, stale locks, partial files, minimum disk, current RAM headroom, and GPU requirements. GPU execution fails by default when process telemetry is required but unavailable. `--allow-gpu-without-telemetry` is the only override and is recorded in the report.

An optional declared-shape estimate reports projected input bytes, dense working-copy bytes, an explicit method multiplier, prediction bytes, atomic temporary space, safety factors, and headroom. Unknown method behavior returns `estimate_unavailable`; it never invents a low multiplier or samples rows/features.

```powershell
.\.venv\Scripts\python.exe scripts/preflight_execution.py `
  --root . `
  --config configs/execution/local_laptop_safe_v1.yaml
```

The live audit report is `cleanup/audits/resource_safe_execution/hardware_preflight.json`; preflight never registers an experiment.

## Supervision and terminal states

Every expensive run executes in one Windows `spawn` worker. The parent samples the worker and recursive children once per configured interval. Samples contain elapsed time, PIDs, process-tree RSS and CPU evidence, system-available RAM, process GPU bytes when NVML supports them, results/temp free space, stage, and fold. The parent records one warning per resource. At an abort threshold it stops scheduling, sets a cooperative event, waits the grace period, then terminates/kills only the known worker tree and confirms cleanup.

Stable stop codes are `ram_process_limit`, `ram_system_headroom`, `gpu_process_limit`, `disk_results_limit`, `disk_temp_limit`, `manual_interrupt`, `preflight_rejected`, and `worker_crash`. Resource aborts end as `aborted_resource_limit`; user interruption ends as `interrupted`; neither is rewritten as completed or ordinary failure. The parent always finalizes the manifest, checkpoint, resource evidence, and run-index row after worker termination.

Monitoring reduces risk; it cannot mathematically guarantee that a native library, device driver, or the operating system will never fail between samples.

## Atomic artifacts and checkpoints

Canonical JSON/YAML, CSV, Parquet, config, manifest, checkpoint, metric, prediction, selection, and resource writes use a unique same-directory `.partial` file. The writer closes and flushes it, `fsync`s it, validates format/schema/row count as applicable, calculates size and SHA-256, then publishes with `os.replace`. Prediction validation also records the ordered row-identity hash. Streaming logs remain streaming; their finalized size/hash enter the manifest.

A failed publication never changes the prior final file. Partials remain visibly incomplete. Explicit resume validates finalized files, quarantines partial and untracked stage output under that run's `incomplete/` directory, and never treats it as final.

Checkpoint schema `experiment_stage_checkpoint_v1` uses:

`initialized`, `data_validated`, `selection_completed`, `model_fit_completed`, `dev_prediction_completed`, `oot_prediction_completed`, `evaluation_completed`, `completed`, `failed`, `aborted_resource_limit`, and `interrupted`.

It records run identity, dataset/selector/model/split/seed/budgets, resolved config hash, protocol and row-contract hashes, data fingerprint hash, Git commit/dirty state, completed stages/folds, artifact integrity/provenance, resource peaks, last successful stage, terminal code, and timestamps. Resume is explicit and checks every identity field plus artifact size, checksum, schema, row count, and provenance. An interrupted stage starts again; generic Python-object serialization and CatBoost snapshots are not used. Completed runs are immutable. Prior terminal resource evidence is archived before a validated retry.

## Projected loading

Before loading, `calculate_required_columns` produces an explicit projection per table. CSV uses `usecols` and optional chunks; Parquet passes `columns` and reads footer/schema metadata without loading the full file. A selected-feature stage declares identity, target, split/time, and required feature columns. A stage scientifically requiring the full candidate universe still receives an explicit schema-derived list; `columns=None` is rejected on the experiment path. Load reports record requested/loaded columns, rows, dtypes, and in-memory dtype bytes. No silent downcast or sparse densification was added.

## Commands

Future new run (after the next prompt authorizes scientific execution):

```powershell
.\.venv\Scripts\python.exe scripts/run_single.py `
  --dataset homecredit `
  --model lr `
  --selector mrmr `
  --execution-policy configs/execution/local_laptop_safe_v1.yaml
```

Explicit validated resume:

```powershell
.\.venv\Scripts\python.exe scripts/run_single.py `
  --dataset homecredit `
  --model lr `
  --selector mrmr `
  --execution-policy configs/execution/local_laptop_safe_v1.yaml `
  --resume <run-id-or-absolute-run-directory>
```

Matrix execution is sequential and deterministic by default. `--resume` on the matrix entry point must identify exactly one run matching the current configuration. No command in this Prompt 2 work ran a Home Credit, LendingClub, or third-dataset experiment.
