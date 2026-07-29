# RAM wait-and-resume patch audit

Status: **validated; real pipeline not executed or resumed by the agent**

## Root cause authenticated before editing

The historical cell-003 attempt was stopped by the old
`ram_system_headroom` control when the terminal sample reached 7.49 GiB
available RAM, below the old 8 GiB abort floor. The final recorded process-tree
peak was 15.32 GiB, below the old 28 GiB RSS abort threshold. This was a
controlled supervisor stop, not `MemoryError` and not an OS kill. The checkpoint
contains only the `initialized` completed stage and remains resumable.

## Validated behavior

- no available-RAM or process-RSS supervisor termination;
- emergency wait boundary `max(1 GiB, 2% total physical RAM)`;
- configurable 4 GiB recovery threshold, three consecutive checks;
- 5-second sampling, immediate/five-minute/immediate WAIT/RESUME logging;
- parent process stays alive and RAM waiting is indefinite;
- RAM wait excluded from active cell/stage timeout accounting;
- preflight records low RAM without blocking, then parent readiness waits;
- cooperative chunk/large-allocation boundaries for dataset and identity loads;
- recovery barrier before opaque stages;
- exact owned-process-tree suspend/resume for an already-running opaque call;
- genuine `MemoryError` traceback routed to `logs/debug.log`;
- Ctrl+C retains existing clean shutdown and resumable checkpoint behavior; and
- resource/manifest artifacts record the RAM policy, events, active time, and
  waiting time.

The opaque fallback is fail-closed if Windows refuses an all-or-nothing
process-tree suspension. Literal exhaustion and an OS process kill cannot be
made crash-proof and are not reported as normal RAM waits.

## Tests

| Gate | Result |
|---|---|
| New RAM wait/recovery tests | 12 passed |
| Focused + supervisor/resume/checkpoint scope | 97 passed in 38.44 s |
| Full repository suite | **904 passed, 31 skipped, 0 failed** in 205.11 s |
| Full-suite warnings | 107 existing pandas fragmentation warnings |
| Python compilation | pass |
| Patch whitespace validation | pass |

Memory/time readings were mocked. Tests created no real memory pressure and did
not sleep for five real minutes.

## Scientific and artifact preservation

The following hashes were captured before editing and matched again after the
full suite:

| Item | SHA-256 |
|---|---|
| Frozen full-baseline configuration | `f03647c376fe834f9bb1c3d6834ed42732ef3e7e1047eeff352af49b31ed607f` |
| Historical execution policy | `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012` |
| Cell 001 config / manifest / success | `1ace5199...` / `db5245e6...` / `30a307d9...` |
| Cell 002 config / manifest / success | `46aec762...` / `1ecca88c...` / `dcb49cd2...` |
| Cell 003 config / checkpoint / manifest / resource | `a9813c3e...` / `0e167ada...` / `4424de74...` / `375cda07...` |
| Matrix run index / status | `3ab648c2...` / `bbc26423...` |

No file under `results/full_baseline_v1` was changed. The compatibility bridge
authenticated the exact cell-003 run/config/protocol/data/row-alignment identity
and the exact frozen scientific/runtime file hashes. Its pre-commit SHA-256 is
`387001fca3c72f9c14d65e1e3a68ddce7d20729d2a584c86888562198ad8c52b`.

The agent inspected only logs and control metadata for the historical stop. It
did not open real prediction contents, load a real dataset, access real OOT data,
evaluate OOT, or invoke `scripts/run_full_baseline.py` without `--status`.
