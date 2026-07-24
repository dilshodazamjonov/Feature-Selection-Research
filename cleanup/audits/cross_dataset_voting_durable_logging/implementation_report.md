# Durable Timestamped Research Logging — Implementation Report

## Authenticated starting state

Work began on clean `main` at `69c4bb565e6dcb4caad90b01237f0be134f72e88`, equal to `origin/main`. Annotated tag `cross-dataset-voting-resume-safety-v1` had tag object `1ef8f94bf261a54da5e0c52c1402d5f7567b2b53` and still peeled to that commit. The pre-execution tag also remained annotated and unchanged. No repository-associated research Python process or active per-run execution lock existed. The persistent coordination file `results/run_index.csv.lock` was not an execution lock.

The canonical pre-change result inventory was 831 files and 1,439,586,868 bytes. Its exact path/size/SHA-256 table is `immutable_prechange_results_manifest.csv`.

## Run 014 reconciliation and resume boundary

Canonical checkpoint, manifest, resource, and finalized-artifact evidence identifies `cdv1-014-lendingclub-v2-voting-k100-catboost-s42` as `aborted_resource_limit`, with immutable first cause `ram_system_headroom`. DEV folds 1–4 are complete and their 36 finalized checkpoint artifacts pass exact size and SHA-256 checks. No fold-5 directory or publication exists; the last resource samples name fold 5 `voter_boruta` as in-memory activity only.

The stop latched after system-available RAM crossed the unchanged 8 GiB floor. Cooperative grace was 20 seconds, process-tree termination followed, exit was confirmed, worker exit code was 15, cleanup was confirmed, and no survivor remained. The authenticated earliest resume boundary is **DEV fold 5 at `dev_data_loading`**. Fold 5 must be recomputed in full; folds 1–4 are reusable.

## Observed gaps and files changed

Before this patch, project loggers emitted local-time text without session/run/fold/stage identity, the runner printed raw control messages, per-run `run.log` files were fragmented, CatBoost emitted raw verbosity, Boruta was silent during its blocking fit, and worker stage/resource messages were not mirrored into one cross-session durable file.

The implementation adds `research_logging.py` as the single JSONL owner; instruments the manual runner, execution/checkpoint owners, resource supervisor, canonical DEV/OOT worker, voters, RFE, preprocessing, model fit, prediction, evaluation, and finalization; adds synthetic workers and focused tests; evolves the fail-closed compatibility bridge; adds the narrow ignore rule; and publishes the operator runbooks and this read-only-derived audit. Frozen protocols, policies, data, results, and historical bridges/tags are not modified.

## Final logging architecture

The exact default path is `logs\runs.log`. The runner creates its directory and opens UTF-8 append mode before plan or release authentication. Each JSON line follows `research_run_log_v1` and always includes millisecond UTC `timestamp_utc`, `level`, `pid`, unique `session_id`, `event`, and `message`. Applicable scalar fields include run, dataset, model, seed, phase, fold, stage, component, monotonic elapsed time, process/memory evidence, repository-relative paths, stop codes, exception classes, and tracebacks. Every record currently flushes immediately to both the file and terminal.

The parent owns the only file handle. Spawned workers use a separate bounded 1,024-record queue. Routine writes never block; priority writes have a bounded 250 ms attempt. Any loss is disclosed durably through cumulative and delta `logging_backpressure` counters. Parent lifecycle/stop/final records bypass the queue. The listener starts before worker spawn and performs a bounded drain/join after terminal records. Values are size/depth bounded and unsupported scientific objects become type-only omission markers.

Synchronous parent stages and supervised worker stages heartbeat at no more than 30-second intervals. Boruta start records include row/feature counts and its non-sensitive frozen configuration. While its blocking API is active, the parent writes `Boruta fit active; internal iteration unavailable.` with elapsed and memory evidence. Completion reports confirmed/selected, tentative, rejected, and ranked counts. No callback, iteration, percentage, ETA, estimator, sampling, seed, parallelism, or stopping behavior was invented or changed.

Coverage is enumerated in `stage_coverage.json`. It includes provenance, plan/resume, readiness, data loading, target/feature contracts, row boundaries, preprocessing/encoding, each actually configured voter, rank aggregation/tie-breaking, RFE, model fit, prediction, metrics, checkpoint/artifact hashing, fold/run completion, global DEV/OOT barrier work, warnings, first-cause stop latch, grace, terminate, force-kill, exit confirmation, survivor verification, exceptions, interrupts, and session finalization.

## Validation and preservation

Focused validation passed 67 tests. The full suite passed 527 tests with 31 intentional skips and 107 pre-existing warning instances. Compileall, `git diff --check`, and the repository/result validator passed; the validator authenticated 18 registered rows and 194 validator-owned artifacts. The exact scientific-equivalence suite passed for ranking, selected features, probabilities, metrics, planning, checkpoints, and resume interpretation. Only synthetic, mocked, fixture, generated, and plan-only paths were used.

The post-change canonical result inventory remains exactly 831 files and 1,439,586,868 bytes. Comparing every result path, size, and SHA-256 found zero additions, removals, or changes. `logs\runs.log` is outside results, outside checkpoint artifacts and manifests, and ignored only by `/logs/runs.log`.

The strict bridge is `configs/execution/cdv1_durable_logging_compatibility_bridge_v1.json`. It authenticates runs 001–013, run 014’s four finalized folds and exact boundary, both unchanged historical annotated releases, frozen hashes, historical checkpoint commits, current runtime hashes, and the new annotated tag. Any mismatch fails closed.

## Release and limitations

The intended commit message is `feat: add durable timestamped research run logging`. The release identity is the annotated tag `cross-dataset-voting-observability-v1`, bound to the final clean `main` commit; its exact commit SHA and local/remote verification are reported in the final operator handoff. The Prompt 6.1 tag is preserved byte-for-byte.

Known limitations are operational only: third-party raw progress may still appear beside canonical JSON; an operating-system hard kill can prevent the final record, leaving prior flushed heartbeats as evidence; retention/rotation is manual while the runner is stopped; and extreme worker-log pressure can drop bounded queue records, always reported by backpressure counters. None changes scientific or stop semantics.

## Exact operator commands

Resume from the authenticated fold-5 boundary:

```powershell
Set-Location "D:\python projects\Research"
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

Watch the durable log in a second PowerShell window:

```powershell
Set-Location "D:\python projects\Research"
Get-Content .\logs\runs.log -Wait -Tail 50
```

Prompt 6.2 did not execute either command, did not run research, and did not open OOT.
