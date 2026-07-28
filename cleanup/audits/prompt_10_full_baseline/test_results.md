# Prompt 10 verification results

No real full-baseline command or dataset worker was run.

| Gate | Result |
|---|---|
| Python compilation for runner/module/execution integration | pass |
| Data-free `--status` | pass; 0/36 completed, first cell identified, no result root created |
| DEV-only `--audit-pilots` | pass; 6/6 authenticated, OOT flags false |
| Focused new runner + registered execution tests | 13 passed in 5.50 s (final rerun) |
| Affected runner/resource/selector scope | 118 passed in 18.82 s |
| Full repository suite | **891 passed, 31 skipped, 0 failed** in 225.16 s |
| Repository-state validator | pass; 20 runs and 388 artifacts authenticated |
| Patch whitespace check | pass |
| `results/full_baseline_v1` after implementation/testing | absent |

The 107 full-suite warnings are the existing pandas fragmentation warnings from
the LendingClub feature builder. Prompt 10 added no failure, error, or skip.

The tests cover the exact 36-cell Cartesian product and deterministic order,
selector/final-model constructor compatibility, fixed budgets, the 1,500-iteration
CatBoost final model, confirmed-only Boruta semantics, fail-closed protocol
mutation, data-free status, completed-cell skipping, checkpoint resume ordering,
controlled resource stop, corrupt completed-artifact rejection, and propagation
of a per-cell wall-clock limit into the existing process supervisor.
