# Cross-Dataset Voting Pre-Execution Freeze Audit

Primary gate target: `READY_FOR_USER_MANUAL_RESEARCH_EXECUTION`

Audit date: 2026-07-23

Repository instructions: no `AGENTS.md` or repository-conventional equivalent was present.

## Git and release identity

- Starting branch: `main`.
- Starting HEAD: `888e971a6f1a80261b62658e79c126be0e503556`.
- Starting `origin/main`: `888e971a6f1a80261b62658e79c126be0e503556` after `git fetch --prune origin`; divergence was `0 0`.
- Required commit message: `research: freeze cross-dataset voting pipeline before execution`.
- Ending HEAD / local `main` / remote `origin/main`: the peeled commit of annotated tag `cross-dataset-voting-pre-execution-v1` (the exact content-addressed SHA is recorded in the final release handoff; embedding a commit's own SHA in this same committed file is not possible).
- Tag annotation: `Validated cross-dataset voting pipeline before manual research execution`.
- Tag-object SHA and peeled commit SHA: resolved and recorded in the final release handoff after the annotated tag is created and pushed.
- No matching pre-existing `cross-dataset-voting-pre-execution-*` tag existed at the start.

## Frozen inputs authenticated

- Scientific protocol: `f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0`.
- Row-alignment contract: `fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0`.
- Voting protocol: `51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49`.
- Execution policy: `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012`.
- LendingClub memory-safe mechanics: `4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7`.
- Frozen matrix: `1a0739fd4c78e450d15b63bb9958de314f071ea567ed016ad287eb7943c6ace8`.
- Pure planning configuration-set hash: `172f6f3bc24daa525aa14b997d5f7e355a8ba41c1bd7c1755140172ec248eb5e`.
- Matrix expansion: 16 IDs, 12 voting, four rerun-required references, 80 DEV folds, and 16 full-DEV/OOT fits.

## Validation gates

- Focused command: `.\.venv\Scripts\python.exe -m pytest tests/test_manual_research_orchestration.py tests/test_cross_dataset_voting_integration.py tests/test_lendingclub_memory_refinement.py tests/test_checkpointing.py tests/test_resource_policy.py tests/test_result_paths.py tests/test_execution_dry_run.py tests/test_paired_inference.py -q`.
- Focused result: `97 passed in 9.99s`.
- Full command: `.\.venv\Scripts\python.exe -m pytest tests -q`.
- Full result: `503 passed, 31 skipped, 107 warnings in 142.49s` (final rerun after the resume-status review).
- The 31 skips remain the documented missing external/legacy evidence integrations. The 107 warnings remain the existing LendingClub pandas fragmentation `PerformanceWarning` class. No new skip or warning class was introduced.
- Repository validator: passed with 84 active artifacts verified, four registered runs, four required result directories, 359 read-only legacy files, and 12 removed paths verified absent.
- `compileall src scripts`: passed.
- `git diff --check`: passed; only Git's existing CRLF-to-LF normalization notices were emitted.
- Planning mode was executed; it performed no data load, run registration, or result-directory creation. The real manual command was not executed.

## Before/after integrity

| Evidence | Before | After validation |
|---|---:|---:|
| Canonical pilot index rows | 4 | 4 |
| Canonical pilot run roots | 4 | 4 |
| Research index rows | 0 | 0 |
| Research run roots | 0 | 0 |
| Active `.execution.lock` files | 0 | 0 |
| Canonical `*.partial` files | 0 | 0 |
| Isolated capacity run roots | 3 | 3 |
| Isolated capacity locks | 0 | 0 |
| Isolated capacity partials | 0 | 0 |
| Frozen legacy files | 359 | 359 |
| Frozen legacy bytes | 110,084,164 | 110,084,164 |

The existing pilot immutability test and capacity artifact-manifest tests passed. No research configuration, research fold, full-DEV research model, or research prediction was executed. No OOT path was opened. No external API, LLM, GPU training, CLIP, embedding, SHAP, or inference workload was invoked.

## One manual command

Run from `D:\python projects\Research` only after this release tag is present:

```powershell
.\.venv\Scripts\python.exe scripts\run_cross_dataset_voting_research.py
```

Prompt 6 did not execute this command. The same command validates and resumes eligible checkpoints; it has a hard all-DEV barrier before the first OOT loader call.

## Exact committed path inventory

1. `.gitignore`
2. `cleanup/audits/resource_safe_execution/hardware_preflight.json`
3. `scripts/run_matrix.py`
4. `src/credit_risk_fs/data/loaders.py`
5. `src/credit_risk_fs/evaluation/paired_inference.py`
6. `src/credit_risk_fs/experiments/compare.py`
7. `src/credit_risk_fs/experiments/config.py`
8. `src/credit_risk_fs/experiments/execution.py`
9. `src/credit_risk_fs/experiments/matrix.py`
10. `src/credit_risk_fs/experiments/rank_voting.py`
11. `src/credit_risk_fs/experiments/runner.py`
12. `src/credit_risk_fs/experiments/tracking.py`
13. `src/credit_risk_fs/pipelines/common.py`
14. `src/credit_risk_fs/preprocessing/encoding.py`
15. `src/credit_risk_fs/selectors/mrmr.py`
16. `src/credit_risk_fs/selectors/registry.py`
17. `src/credit_risk_fs/selectors/rfe.py`
18. `tests/selectors/test_boruta.py`
19. `cleanup/audits/cross_dataset_voting_integration_pilot/implementation_validation.json`
20. `cleanup/audits/cross_dataset_voting_integration_pilot/matrix_expansion_validation.json`
21. `cleanup/audits/cross_dataset_voting_integration_pilot/pilot_manifest.json`
22. `cleanup/audits/cross_dataset_voting_integration_pilot/pilot_resource_summary.csv`
23. `cleanup/audits/cross_dataset_voting_integration_pilot/pilot_stage_resources.csv`
24. `cleanup/audits/cross_dataset_voting_integration_pilot/validation_summary.json`
25. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/artifact_manifest.json`
26. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/baseline_state.json`
27. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/before_memory_ownership_map.csv`
28. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/capacity_scenario_manifest.json`
29. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/equivalence_validation.json`
30. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/full_dev_resource_summary.json`
31. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/implementation_validation.json`
32. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/largest_fold_resource_summary.json`
33. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/post_refinement_capacity_projection.csv`
34. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/stage_resource_trace.csv`
35. `cleanup/audits/lendingclub_memory_refinement_capacity_gate/validation_summary.json`
36. `cleanup/tools/build_cross_dataset_voting_pilot_audit.py`
37. `configs/execution/lendingclub_memory_safe_refinement_v1.yaml`
38. `configs/experiments/cross_dataset_rank_voting_pilot_v1.yaml`
39. `docs/research_extension/cross_dataset_voting_integration_pilot_v1.md`
40. `docs/research_extension/cross_dataset_voting_research_runbook_v1.md`
41. `docs/research_extension/lendingclub_memory_refinement_capacity_gate_v1.md`
42. `scripts/run_cross_dataset_voting_research.py`
43. `src/credit_risk_fs/experiments/cross_dataset_research.py`
44. `src/credit_risk_fs/experiments/manual_research.py`
45. `src/credit_risk_fs/experiments/prediction_contract.py`
46. `tests/test_cross_dataset_voting_integration.py`
47. `tests/test_lendingclub_memory_refinement.py`
48. `tests/test_manual_research_orchestration.py`
49. `cleanup/audits/cross_dataset_voting_pre_execution_freeze/freeze_audit.md`

## Intentionally excluded

- `cleanup/audits/lendingclub_memory_refinement_capacity_gate/capacity_execution/`: isolated real capacity-run outputs, including its registry, logs, checkpoints, model/config evidence, and success markers; retained locally and ignored, never force-added.
- `results/`: canonical pilot artifacts and active-results registry remain ignored and byte-unchanged.
- `data/`, `artifacts/`, `logs/`, `tests_runtime/`: datasets, caches, logs, and transient test/runtime outputs.
- `.env`, `.venv/`, Python caches, pytest caches, model/pickle/parquet outputs, credentials, secrets, and machine-local temporary files.
- The immutable external legacy bundle at `D:\ResearchFindings\results` remained read-only and outside the repository.

## Limitations retained

- The 31 documented external-evidence skips and 107 pre-existing performance warnings remain.
- Planning and tests prove orchestration, provenance, barrier, limits, resume, and controlled-stop behavior with fixtures/mocks/temporary roots; the real command remains deliberately unexecuted until the user launches it.
- Literal final commit/tag object SHAs are provided in the post-push handoff rather than self-referenced inside the same content-addressed commit.
