# Cell 004 timeout-resume repair audit

Status: **implementation validated; real pipeline not run or resumed**

## Before-edit authentication

- branch: `main`
- commit: `9230e1599eb4099ff255f514590f973f6e13f467`
- worktree: clean
- frozen full-baseline SHA-256:
  `f03647c376fe834f9bb1c3d6834ed42732ef3e7e1047eeff352af49b31ed607f`
- Cells 001-003: completed checkpoints, manifests, resource evidence, required
  artifacts, declared hashes, and `_SUCCESS` markers all authenticated
- Cell 004: `timed_out/wall_clock_limit`, 10,800.025 active seconds, controlled
  supervisor cleanup, no completed checkpoint/fold, no `_SUCCESS`
- partial evidence: 15 exact-hashed unfinalized files
- live state: no worker/orphan, no execution lock, zero survivors
- Windows boot: later than the controlled stop, therefore not causal
- OOT access: none

## Defects and resolution

The runner treated every `timed_out` state as non-resumable. It now keeps that
historical state immutable but can separately authorize this exact controlled
attempt after all static and live checks pass. Any failed check produces
`NOT_RESUMABLE`; other failures are not generalized into resumable states.

The runner also derived cost only from the selector. Runtime policy now composes
dataset, selector, and final-model components by maximum cost and maximum timeout.
This resolves Cell 004 as light/heavy/light -> heavy with a fresh 21,600-second
active limit.

Status, plan, execution request, supervisor timeout, logs, and manifests consume
the same resolved classification object. Prior attempt evidence is archived under
a deterministic attempt identity before the fresh attempt can run.

## Validation gates

- static Cell 004 authorization: 31/31 checks passed with 15 partials recorded
- focused timeout/resume regressions: 19 passed
- runner/checkpoint/resource/RAM/logging scope: 88 passed, 2 skipped
- full suite: 923 passed, 31 skipped, 107 existing pandas fragmentation
  warnings, 0 failed in 195.48 seconds
- whitespace and compilation: pass
- scientific configuration and Cells 001-003 artifacts: unchanged by repair
- real experiment or full-baseline command: not executed
- OOT data: not accessed
