# Prompt 12 combination pilot retention review

## Technical summary

The pilot is fully authenticated: 24 of 24 evaluation cells and 18 of 18 unique selector fits are complete, with exactly 168 expected active files, no unexpected files, and no active invalid state. The final cell completed at final_prediction with worker exit 0 and authentication SHA-256 8701e61330498c79e7cb9befd39d17568c5173d2433160457594b559bb230f02.

The outcome-blind recommendation is to retain all four preregistered combination methods in frozen execution order. No method failed authentication, declared support behavior, or runtime-resource review. This recommendation is not approval. The DEV gate remains closed, pilot_approval_lock.json remains absent, and neither DEV nor OOT is authorized.

## The pilot checkpoints and operational boundary are trustworthy

The safe status path independently returned 24 authenticated evaluation cells, 18 authenticated selections, no first incomplete cell, raw_dataset_paths_resolved=false, and workers_started=0. The active pilot tree contains 168 expected files and its deterministic tree digest is ec34e5ad4cd7542ccf74cc851c76367589261a8a88ed0588f1e88aa8cb31903f.

Cell 001's live state is completed and authenticated. Its archived manual_interrupt remains historical. The latest historical stop was recorded at 2026-08-01T11:22:45.490Z; the subsequent session completed successfully at 2026-08-01T13:35:58.021Z with no errors. No experiment worker, orphan, execution lock, stale lock, or partial file remains.

Conventional manifest.json, checkpoint.json, and _SUCCESS files are absent by design. Authenticated per-selection and per-cell JSON states are the completion checkpoints and their absence must not be treated as incomplete.

## Stage support is valid, with two declared natural-support bounds

Sixteen of the 18 selection identities reported feasibility_state=completed. The two remaining identities are not execution failures: both Home Credit CatBoost Boruta-first refiners requested 40 features but Boruta confirmed 26. They correctly emitted infeasible_natural_support, did not pad, did not run an impossible refiner, and completed valid downstream evaluation cells.

The statistical voter met all exact 20/40 budgets. IV-to-Boruta completed all 100/200/300 intermediate pools and returned natural support of 20/25/28 for Home Credit and 86/118/134 for LendingClub v2. The two Boruta-first refiners met LR-20 in both datasets and CatBoost-40 in LendingClub v2.

The support mismatch must remain visible in later comparisons. It does not justify method removal because the frozen protocol expressly defines natural support below the requested budget and forbids filler.

## Runtime and resource margins clear the pilot gate

All 42 latest-session supervised work units completed with worker exit 0, no stop code, and no supervisor resource warning. The maximum selector fit took 547.407 seconds, or 1.2671% of its wall limit. The maximum wall-limit fraction across selector fits was 1.6324%; across evaluations it was 0.9628%.

The highest observed process-tree RSS was 6.984 GiB, below the 24 GiB warning and 28 GiB abort thresholds. The lowest observed system-available RAM was 22.201 GiB, which remained 14.201 GiB above the 8 GiB abort floor. The pilot therefore provides no operational reason to drop a method.

These margins are pilot evidence, not a promise for later expanding folds. DEV must keep the frozen sequential execution, thread caps, monitoring, wall limits, and abort behavior.

## Retain all four methods without predictive tuning

The proposed retained method IDs, in the exact order accepted by the gate, are:

1. statistical_normalized_average_rank
2. iv_then_boruta
3. boruta_then_mrmr_mutual_information
4. boruta_then_rfe_catboost

This scope expands to 120 DEV evaluation cells backed by 90 selector fits across five folds. Predictive outcomes were not extracted, ranked, compared, or used for this recommendation. Retaining the complete frozen set avoids outcome-driven attrition and preserves all registered comparison families.

The two Boruta-first methods are retained with a mandatory natural-support label for Home Credit CatBoost when realized support remains below 40. IV-to-Boruta also remains explicitly natural-support rather than fixed-K.

## Method and robustness notes

Authentication follows the runner's canonical per-artifact SHA-256, identity, terminal-state, referenced-file size, and referenced-file hash checks. Active inventory counts and a deterministic whole-tree digest provide a second package-level reconciliation. Runtime aggregation uses authenticated supervisor fields and latest-success-session finalization events only.

The pilot is one DEV fold. Natural support and resource use may change across the remaining expanding folds. That uncertainty is handled by preserving labels and runtime controls, not by altering frozen membership after viewing outcomes.

## Required next step

The user must explicitly approve both the committed review digest and the exact four-method retention scope in a later message. Only after that approval may an authenticated pilot_approval_lock.json be created, the DEV gate validated, and the non-executed DEV command returned.

Until then:

- Do not create the approval lock.
- Do not run or resume the pilot.
- Do not start DEV.
- Do not access OOT.

## Further question

Does the user approve the identified review digest and the proposed retained method IDs exactly as ordered above?
