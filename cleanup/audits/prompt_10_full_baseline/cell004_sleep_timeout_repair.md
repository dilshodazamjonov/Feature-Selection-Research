# Cell 004 Windows-sleep timeout repair audit

Status: **implementation prepared; real pipeline not run by the agent**

## Authenticated incident

- Cell 004 historical state: `timed_out/wall_clock_limit`
- resumed attempt: `attempt_01`
- configured limit: 21,600 seconds
- sleep-inclusive reported active time: 28,817.541 seconds
- largest sample gap: 16,237.896 seconds
- process CPU growth during the gap: 8.25 seconds
- corrected awake active time: 12,579.645 seconds
- cleanup: controlled stop, exit 15 after grace, child/queue cleanup confirmed,
  zero survivors
- completion evidence: no completed fold, completed checkpoint, or `_SUCCESS`
- live state: no worker/orphan and no execution lock
- current partial evidence: 15 exact-hashed unfinalized files
- earlier history: the first stopped attempt and all its partials authenticate
  under `incomplete/attempt_history/attempt_01`

The evidence shows system sleep/hibernate, not CatBoost or RAM failure. The prior
clock included Windows suspend time and triggered immediately after wake.

## Repair

The Windows supervisor now uses `QueryUnbiasedInterruptTime`, which excludes
sleep/hibernate. It records the clock source, supervisor-awake elapsed seconds,
excluded suspend seconds, and RAM-wait seconds separately. Other platforms retain
their monotonic clock.

The final-model CatBoost component now has a 43,200-second (12-active-hour) limit.
This covers the observed roughly 2.5-hour pre-fit work plus CatBoost's 6-hour-40-
minute initial estimate with margin. Frozen selector behavior, CatBoost settings,
folds, inputs, seeds, and OOT gates remain unchanged.

The v2 recovery record is
`configs/execution/full_baseline_timeout_recovery_cell_004_attempt_02_v1.json`.
It authenticates the suspension evidence, both current and archived attempt
artifacts, current runtime/scientific hashes, Cells 001-003, and a Cell 004
boundary-only restart. Any mismatch remains fail-closed.

## Validation

- focused clock/timeout/resume/resource/checkpoint scope: 69 passed
- full repository suite: 926 passed, 31 skipped, 107 existing pandas
  fragmentation warnings, 0 failed in 629.32 seconds
- real research workload: not executed
- OOT data: not accessed
