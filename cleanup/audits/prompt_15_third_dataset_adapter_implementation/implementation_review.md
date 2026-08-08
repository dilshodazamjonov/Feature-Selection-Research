# Prompt 15 implementation review

## Verdict

Pass. The authenticated Home Credit Model Stability 2024 protocol has been translated into an inert, explicit, deterministic PyArrow adapter and CLI. The implementation was validated entirely with exact-schema synthetic Parquet fixtures. It performed no real-data, model, selector, prediction, metric, pilot, DEV, or OOT workload.

## Entry and authority

The entry gate passed on `main` at the exact required starting commit `8bb283c0d71cad2f09ecbefd08148f3211b62a7d`, with ahead/behind `0/0` and a clean starting worktree. Both required historical commits were ancestors.

The sole scientific authority was authenticated before implementation:

- lock: `configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json`
- file SHA-256: `e4b9f9f13286f15db0887c9dead09eb7e13f7912af786f2f2bc9c53d126b1860`
- internal authentication SHA-256: `638e1fa2aa54bf98b771206b56ac13f6a6b77e2093deb291b794081d1a475df6`
- approved review digest: `3f537d1b5e79faad3a2f047ec13dbe4b1797e11d4d64c4d92a06e09762a53f1e`
- freeze commit: `e2a9a80f2894ce7f9b8ef3769e073b6bbe2228a9`
- authenticated Stage-1 artifact bindings: 13 of 13

The executable contract contains 13 logical tables, 19 included inputs, 14 explicitly excluded depth-2 Parquet paths, 434 included raw predictor rules, 27 excluded rules, and no unresolved rule.

## Implementation

The implementation is in:

- `src/credit_risk_fs/data/homecredit_model_stability_2024/contract.py`
- `src/credit_risk_fs/data/homecredit_model_stability_2024/adapter.py`
- `scripts/build_homecredit_model_stability_2024.py`

Importing either the package or CLI is inert. Execution requires explicit input root, output root, authenticated protocol lock, stage, and fixture/research mode. Fixture mode is labeled `synthetic_fixture_not_research`; research mode requires every included input size and digest from the lock.

The base table is the sole population authority. The adapter enforces non-null unique `case_id`, a non-null binary target, strict non-null ISO `date_decision`, and final order by `date_decision` then numeric `case_id`. Target, key, date, `MONTH`, and `WEEK_NUM` remain audit/control columns and are never predictors.

Depth-0 tables are concatenated in the lock's partition and family order, checked for exact Arrow schema and one-to-one keys, then base-left joined without row loss, multiplication, or reordering. Included features use the locked family prefix. Date features become signed whole days relative to the same base case's `date_decision`.

Depth-1 tables are processed one family and one bounded base shard at a time. Rows are ordered by `case_id`, `num_group1`, locked numeric source part, and physical offset within that part. Each family is reduced to one compact row per base case before joining. Numeric, date, boolean, and categorical operations exactly follow the lock, including sample variance with denominator `n-1`, explicit missing counts, Unicode NFC/casefold/original-UTF-8 lexical mode ties, and a family row count.

The lock specifies ordered first/last operations but does not separately state whether nulls are skipped. The implementation uses the literal ordered row value: a null in the first or last physical row remains null. This preserves the protocol's required distinction between an observed-null related row and no related row and makes no performance- or target-informed choice. The convention is explicit in `adapter_contract.json` and directly tested.

Known depth-2 paths may be present in an official input layout, but inventory inspection records only their path presence and never opens them. Any request to execute a depth-2 path is rejected. Unknown tables, unknown partitions, unknown Parquet files, changed protocol instances, changed lock bytes, and changed schemas also fail closed.

## Leakage and downstream boundary

The matrix builder performs no imputation, encoding, scaling, selection, ranking, or model preprocessing. It retains constants and all-missing approved features. Its predictor list is generated only from the authenticated include rules and excludes target, case identifiers, group identifiers, split controls, and all lock-excluded rules.

The output metadata exposes the locked DEV/OOT boundary and five-fold grouped-time-series interface. Fit-scope tokens accept DEV membership only. A downstream OOT transform is rejected unless the state was fitted on full DEV; a fold-training state can transform DEV only.

## Resource, atomicity, and provenance

PyArrow reads only required columns through bounded Parquet batches. The five-column base population authority is held in memory, while relational work is bounded by a configurable contiguous base-order shard. Each family/shard compact result is atomically published and checkpointed before the next family. No raw-table pandas join or parallel worker exists.

A resource hook receives batch, family/shard, reuse, and matrix/shard events and may fail closed. Prompt 15 deliberately freezes no RAM or runtime threshold.

Checkpoints bind adapter version, both protocol digests, input inventory identity, logical family, schema, aggregation rules, shard identity, base-case identity, and compact output digest. Stale or corrupted checkpoints are rejected. A valid incompatible completed build is authenticated and rejected rather than overwritten.

Final matrix shards, metadata, lineage, split membership, runtime, and status are hashed in the build manifest. Status is published before the manifest, the manifest is validated, and `_SUCCESS` is written last with the manifest digest. Mutation of the bound status is detected.

## Validation

The focused adapter suite passed 27 tests. Compatible existing protocol, atomic I/O, grouped temporal split, checkpoint, and deterministic/sequential resource-policy selections passed 32 tests. The final validation total is 59 passed, zero failed, and zero warnings in the recorded final commands.

One preserved historical Stage-1-only test was intentionally deselected because it asserts that the later canonical-lock directory must not exist. That assertion directly contradicts both the repository's later canonical-lock test and Prompt 15's binding authority. It was not changed or counted as an implementation failure.

An earlier overbroad exploratory command emitted 107 existing pandas fragmentation warnings from LendingClub feature engineering. They are unrelated to the new adapter, contain no scientific conflict, and were absent from the final targeted validation commands.

The traceability register contains 42 protocol rules, all marked implemented and tested; none is unresolved or waived.

The locked future-stage accounting is asserted without execution: 27 resource-pilot selector fits, 30 resource-pilot evaluation cells, 135 DEV selector fits, 150 DEV fold evaluations, 27 OOT full-DEV selector refits, and 30 OOT evaluations. Prompt 15 executed zero of each.

## Prompt 16 limitations and next boundary

Prompt 16 must be a bounded real-data adapter/resource pilot only. It must supply hard resource thresholds and measure base memory, Parquet rescan I/O, throughput, compression, checkpoint reuse, and a safe shard size. Case-filtered bounded shards can rescan each registered partition once per shard; this is deliberately conservative and must be measured before any broader execution. DEV and OOT scientific execution remain closed.

At this gate, raw dataset paths resolved is false; raw dataset files opened is zero; two-dataset numeric outcomes opened is false; workers, fits, predictions, metrics, and evaluations started are all zero.
