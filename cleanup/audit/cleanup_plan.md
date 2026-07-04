# Repository cleanup plan

Generated from a complete SHA-256 inventory. Deletion remains gated by registry, manifest, report, test, and pending-analysis checks.

## Classification summary

| Category | Files | Bytes |
|---|---:|---:|
| AMBIGUOUS_REVIEW_REQUIRED | 1132 | 508080714 |
| CANONICAL_ACTIVE | 1738 | 1128568131 |
| COMPACT_AUDIT_EVIDENCE | 7 | 127486 |
| DRY_RUN_OR_SMOKE_OUTPUT | 23 | 850302 |
| FAILED_OR_INCOMPLETE_OUTPUT | 69 | 94985918 |
| INVALID_SCIENTIFIC_OUTPUT | 459 | 190180710 |
| MIGRATION_BACKUP | 61 | 2617723 |
| REPRODUCIBILITY_SUPPORT | 32514 | 1488601840 |
| REQUIRED_PENDING_ANALYSIS_INPUT | 757 | 10878624711 |
| SUPERSEDED_VALID_OUTPUT | 11 | 524239 |
| TEMPORARY_OR_CACHE | 5717 | 124966016 |

## Execution gates

1. Validate the current registry bundle and final v2 package hashes.
2. Preserve all pending significance, feature-drift, and LLM-cost inputs.
3. Migrate invalid old-policy registry rows to compact tombstones.
4. Back up ignored scientific deletion candidates outside the repository.
5. Delete only HIGH-confidence entries listed in deletion_manifest.csv.
6. Revalidate after each atomic cleanup group.
