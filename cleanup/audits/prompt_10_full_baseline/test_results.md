# Prompt 10 verification results

The local agent did not run a real full-baseline command or dataset worker. The
user-started first cell failed before final DEV fitting or any OOT evaluation;
its artifacts were preserved under the superseded archive documented in
`runtime_incident_001.md`.

| Gate | Result |
|---|---|
| Python compilation for runner/module/execution integration | pass |
| Data-free `--status` | pass; 0/36 completed and corrected first cell identified as missing |
| DEV-only `--audit-pilots` | pass; 6/6 authenticated, OOT flags false |
| Corrected full-baseline pipeline tests | 9 passed in 1.11 s |
| Affected runner/resource/selector scope | 72 passed in 14.33 s |
| Full repository suite | **892 passed, 31 skipped, 0 failed** in 179.47 s |
| Repository-state validator | pass; 20 runs and 388 artifacts authenticated |
| Patch whitespace check | pass |
| Canonical full-baseline progress after correction | 0/36; failed old-configuration attempt archived recoverably |

The 107 full-suite warnings are the existing pandas fragmentation warnings from
the LendingClub feature builder. Prompt 10 added no failure, error, or skip.

The tests cover the exact 36-cell Cartesian product and deterministic order,
selector/final-model constructor compatibility, the original-feature selection
boundary with fold-varying categorical encodings, fixed budgets, the
1,500-iteration CatBoost final model, confirmed-only Boruta semantics,
fail-closed protocol mutation, data-free status, completed-cell skipping,
checkpoint resume ordering, controlled resource stop, corrupt completed-artifact
rejection, and propagation of a per-cell wall-clock limit into the existing
process supervisor.
