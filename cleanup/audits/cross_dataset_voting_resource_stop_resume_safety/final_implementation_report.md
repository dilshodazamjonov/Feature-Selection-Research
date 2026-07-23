# Cross-Dataset Voting Resource-Stop and Resume-Safety Report

## Outcome

The canonical supervisor now treats a resource stop as a bounded state machine: it latches the first stop cause, requests cooperative exit, waits a finite grace period, terminates the exact owned process tree, force-kills verified remainders, confirms exit, closes bounded queues, and finalizes durable state. Inter-run readiness checks block the next research run if owned children remain, system RAM or disk is below the unchanged abort floor, or cleanup cannot be confirmed.

No research, resume, full-DEV, or OOT command was executed during this repair. All validation used repository metadata or small synthetic inputs.

## Incident reconciliation

The alleged run-011 `ram_system_headroom` hang is contradicted by the structured record. The only reserve warning occurred during run 010 fold 5: available RAM crossed the 10 GiB warning reserve for 18 samples, reached 9.71 GiB, never crossed the unchanged 8 GiB abort floor, recovered to 24.84 GiB, and the run completed.

Run 011 recorded no resource warning, no abort-threshold sample, and a minimum 16.39 GiB available RAM. It was manually interrupted during fold-3 Boruta work. Its folds 1 and 2 are complete and hash-valid; fold 3 never atomically published. The honest continuation boundary is therefore DEV fold 3 at `dev_data_loading`, followed by recomputation of that fold's encoding and voters. The historical stop remains `manual_interrupt`.

The pre-repair supervisor nevertheless contained real latent weaknesses: an interrupt could overwrite an already-latched resource cause, queue collection was not fully bounded, queue feeder cleanup could wait indefinitely, exact child ownership was not durably represented, and the run loop lacked a cleanup/readiness barrier. Those defects are repaired without changing scientific configuration, row identity, voting configuration, or execution thresholds.

## Compatibility and preservation

The compatibility bridge is deliberately narrow. It authenticates the original annotated tag and commit, the new annotated mechanics tag, frozen hashes, runtime hashes, exact run IDs, immutable artifact hashes and sizes, and the run-011 resume boundary. It permits reuse only for runs 001-011 from this release family and is not a general Git-drift bypass.

The immutable pre-edit artifact manifest contains 1,106 files. The final comparison found zero hash, size, or path mismatches. Runs 001-010 passed their canonical validators and all checkpoint artifact hashes. OOT remains unopened.

## Verification

The old-tag and repaired-tree scientific paths produced identical rankings, selected features, class-1 probabilities, and metrics on the same small deterministic synthetic fixture. Synthetic process tests exercised cooperative exit, uncooperative termination, nested-tree cleanup, force-kill escalation, a saturated stage queue, and compact-payload enforcement. No owned survivor or orphan remained.

Focused tests passed 52/52, including a final 15/15 resource-lifecycle subset. The complete suite passed 519 tests with 31 expected skips and 107 pre-existing warning-class instances. The repository validator, byte compilation, whitespace checks, artifact reconciliation, and plan-only 16-configuration/80-fold CLI check passed.

## Manual continuation

The sole supported resume command is documented in `docs/research_extension/cross_dataset_voting_resume_after_run_011_v1.md`. It was prepared but not executed. The release commit and annotated tag are verified after their Git objects are created because a commit cannot contain its own hash or annotated-tag object without circularity.
