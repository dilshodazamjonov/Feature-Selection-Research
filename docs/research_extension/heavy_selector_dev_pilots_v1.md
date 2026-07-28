# Heavy-selector DEV pilot pipeline v1

Status: implemented; provisional configuration pending six-cell review.

This pipeline is a configuration and capacity gate for the three Prompt 8 heavy
selectors. It is not an experiment comparison and produces no performance metric.
It does not modify or participate in the frozen rank-voting protocol.

## Fixed matrix and order

One invocation executes exactly six cells, sequentially:

1. Home Credit, DEV fold 1, catboost_shap
2. LendingClub v2, DEV fold 1, catboost_shap
3. Home Credit, DEV fold 1, boruta_random_forest
4. LendingClub v2, DEV fold 1, boruta_random_forest
5. Home Credit, DEV fold 1, rfe_catboost
6. LendingClub v2, DEV fold 1, rfe_catboost

The method-major order guarantees the requested cheapest-first boundary across the
whole run. No cell concurrency and no fold concurrency are permitted.

## Reused contracts

The worker calls the canonical prepare_voting_pilot_dev_data DEV materializer,
canonical_fold_projection with fold_id 1, and OriginalFeatureNumericEncoder. It
fits only the fold-1 training partition, gives the encoded frame authenticated
stable-row-ID indices, and resolves the selector through the shared Prompt 7–8
registry.

The full DEV frame and raw fold-training copy are released before selector fit.
The existing resource policy supplies system-available-RAM and process-tree-RSS
limits. Each cell runs in one spawned process with a method-specific wall-clock
limit and the policy-bounded thread count. The pipeline has no OOT loader, model
fit, prediction, metric, or final-refit path.

## Provisional configuration

The single inventory is
configs/experiments/heavy_selector_dev_pilots_v1.yaml. It contains:

- selector estimator parameters, seeds, budgets, support policy, and sample size;
- method-specific wall-clock limits and policy-controlled estimator threads;
- dataset configuration sources, authenticated universe counts, and exclusions;
- an explicit provisional_pending_six_cell_review lifecycle;
- the Prompt 7 MI-mRMR n_bins 10 decision, marked provisional and not executed.

No setting in this file is represented as accepted or finally frozen. Review of
all six authenticated artifacts is required before a separate acceptance change.

## Durable state and resume

Each cell owns:

results/heavy_selector_dev_pilots_v1/cells/(cell-id)/artifact.json

The file is published atomically and authenticated by a canonical-payload SHA-256.
States are running, completed, failed, manually_interrupted, timed_out, and
resource_aborted. A completion is skippable only when its schema, cell identity,
configuration hash, method implementation, DEV-fold identity, candidate count,
selected-feature order, resource fields, method evidence, and authentication hash
validate.

A valid completion is never overwritten. Missing, non-terminal, stale-config, or
corrupt artifacts are invalid and are resumed from the earliest such cell.
Completed cells later in the sequence remain eligible for authenticated skipping.
A controlled stop ends the current invocation after atomic terminal-state
publication and worker-tree cleanup.

## Evidence

Every completion includes:

- authenticated DEV and fold-1 row hashes, source hashes, feature-universe hash,
  row/feature counts, preprocessing identity, and exact fit scope;
- method and implementation IDs, full effective selector configuration, seed,
  thread count, budget, natural support, feasibility, ordered selections, runtime,
  peak process-tree RSS, and minimum system-available RAM;
- CatBoost-SHAP fit/calculation counts, sample size and identity, SHAP type and
  aggregation;
- Boruta forest/engine settings, realized iteration/fit count, support policy, and
  confirmed/tentative/rejected counts;
- RFE initial/final counts, requested/realized removals, full history, iteration
  count, estimator-fit count, and final selection count.

Terminal and heartbeat events are appended to logs/runs.log and the adjacent audit
stream. Unexpected tracebacks are written to logs/debug.log.

## Status

The status mode parses only the version-controlled configuration and cell JSON
artifacts. It never invokes a research dataset loader. It reports completion,
earliest current/next cell, runtimes, memory, controlled-stop reasons, Boruta
support counts, RFE fit counts, and CatBoost-SHAP sample evidence.

## Manual use

From the repository root, first run the focused preflight tests, then invoke the
single resumable entry point:

    .\.venv\Scripts\python.exe -m pytest tests\test_heavy_selector_dev_pilots.py -q
    .\.venv\Scripts\python.exe scripts\run_heavy_selector_dev_pilots.py

Read-only status:

    .\.venv\Scripts\python.exe scripts\run_heavy_selector_dev_pilots.py --status
