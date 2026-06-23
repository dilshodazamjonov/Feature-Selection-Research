# RUN

# Complete CLIP-v2 Run

## Preview The Full Pipeline

```powershell
uv run python scripts/run_clip_v2_pipeline.py --plan
```

Label: `READ-ONLY`. Purpose: preview every CLIP-v2 stage and command without doing work. Expected output: stage order, commands, and fresh-start archive preview. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: start a clean full run.

## Start A Clean Full Run

LONG-RUNNING: This command trains CLIP-v2, fits eight downstream models, generates predictions, builds reports, runs tests, and performs the final audit. It may run for several hours.

```powershell
uv run python scripts/run_clip_v2_pipeline.py --fresh-start --execute
```

Label: `LONG-RUNNING`. Purpose: archive partial CLIP-v2 outputs, rebuild CLIP-v2 from scratch, run all eight downstream runs one by one, build aggregates/reports, run tests, and run final audit. Expected output: `PASS - CLIP-v2 is scientifically defensible and ready to archive` from the final audit. Modifies files: yes, under `results/clip_v2/`, `results/clip_v2_archives/`, and `reports/clip_v2_*`. Trains a model: yes. Safe to interrupt: yes, press `Ctrl+C`; then preview resume before executing it. Next: commit and tag only after final audit PASS.

## Resume After Interruption

```powershell
uv run python scripts/run_clip_v2_pipeline.py --resume
```

Label: `READ-ONLY`. Purpose: preview which incomplete or stale stages would rerun. Expected output: resume plan. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute resume after reviewing the plan.

```powershell
uv run python scripts/run_clip_v2_pipeline.py --resume --execute
```

Label: `LONG-RUNNING`. Purpose: resume from the first incomplete, interrupted, failed, or stale stage. Expected output: stage progress and final audit result. Modifies files: yes. Trains a model: possibly, depending on the resume point. Safe to interrupt: yes. Next: check status.

## Check Status

```powershell
uv run python scripts/run_clip_v2_pipeline.py --status
```

Label: `READ-ONLY`. Purpose: inspect `results/clip_v2/pipeline_state.json`, lock status, and stage states. Expected output: JSON state summary. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: resume or inspect logs.

Pipeline log:

```text
results/clip_v2/pipeline_execution.log
```

Detailed stage-by-stage commands below are retained for debugging individual stages.

## Final CLIP Comparison

Read-only planning:

```powershell
uv run python scripts/run_clip_final_comparison.py --plan
```

The plan command must not write scientific outputs or mark stages complete. Do not begin execution unless the plan reports `implementation_mode: executable_research_pipeline` and synthetic end-to-end execution tests pass.

Real execution:

```powershell
uv run python scripts/run_clip_final_comparison.py --fresh-start --execute
```

Expected real run counts are 184 core candidate-pool runs, including 120 random-screening repetitions, plus 20 representation-seed downstream runs and 28 ablation downstream runs. A stage is valid only after real predictions, metrics, selected features, hashes, and completion markers validate.

Resume after interruption:

```powershell
uv run python scripts/run_clip_final_comparison.py --resume
uv run python scripts/run_clip_final_comparison.py --resume --execute
```

Status:

```powershell
uv run python scripts/run_clip_final_comparison.py --status
```

Output stays under `results/clip_final_comparison/`; incomplete prior outputs are archived under `results/clip_final_comparison_archives/<timestamp>/`.

## 1. What This Repository Does

This repository compares credit-risk feature-selection methods on Home Credit and LendingClub v2. It includes statistical selectors, LLM-based screening, CLIP-v1, and CLIP-v2. DEV data is used for fitting and feature selection. OOT data is held out as the main evidence because it tests whether a selector survives a later time window.

## 2. Experiment Versions

CLIP-v1 = frozen text metadata + DEV missing rate. It is immutable and referenced by `results/clip_versions/v1/freeze_manifest.json`.

CLIP-v2 = frozen text metadata + a 13-dimensional compact target-free statistical vector.

## 3. Prerequisites

Use Windows PowerShell from the repository root. The project expects Python 3.13 through `uv`, Git, enough disk space for large local CSV/result files, and enough RAM to load the modeling data. GPU is optional; current CLIP configs use CPU.

Install `uv` if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Label: `LIGHTWEIGHT`. Purpose: install the Python environment manager. Expected output: `uv` installed or already available. Modifies files: user tool cache outside the repo. Trains a model: no. Safe to interrupt: yes. Next: open the project.

## 4. Open The Project

```powershell
cd 'D:\python projects\Research'
```

Label: `READ-ONLY`. Purpose: move into the repository. Expected output: no output. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: configure local cache directories.

Change the path if the repository is stored somewhere else.

## 5. Configure Local Cache And Temporary Directories

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
$env:TMP="$PWD\.tmp"
$env:TEMP="$PWD\.tmp"

New-Item -ItemType Directory -Force $env:UV_CACHE_DIR | Out-Null
New-Item -ItemType Directory -Force $env:TMP | Out-Null
```

Label: `LIGHTWEIGHT`. Purpose: keep uv and pytest temp files inside the repo on Windows. Expected output: no output. Modifies files: creates `.uv-cache` and `.tmp`. Trains a model: no. Safe to interrupt: yes. Next: install dependencies.

## 6. Install Dependencies

```powershell
uv sync
```

Label: `LIGHTWEIGHT`. Purpose: create or update the project virtual environment. Expected output: package sync summary. Modifies files: `.venv` and uv cache. Trains a model: no. Safe to interrupt: yes. Next: verify CLIP-v1 freeze.

## 7. Verify CLIP-v1 Freeze

```powershell
uv run python scripts/freeze_clip_v1.py --verify
```

Label: `READ-ONLY`. Purpose: verify the frozen CLIP-v1 hash manifest. Expected output: verification passed. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: verify the Git tag.

```powershell
git tag --list clip-v1-frozen
```

Label: `GIT`. Purpose: confirm the local freeze tag exists. Expected output: `clip-v1-frozen`. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: run unit tests.

## 8. Run Unit Tests Before CLIP-v2

```powershell
uv run pytest tests/clip -q --basetemp "$PWD\.tmp\pytest-clip"
```

Label: `READ-ONLY`. Purpose: run CLIP-focused tests without full model experiments. Expected output: pytest pass/fail summary. Modifies files: temp files only. Trains a model: no full training. Safe to interrupt: yes. Next: run the broader tests.

```powershell
uv run pytest tests -q --basetemp "$PWD\.tmp\pytest-all"
```

Label: `READ-ONLY`. Purpose: run the broader test suite. Expected output: pytest pass/fail summary. Modifies files: temp files only. Trains a model: no full training. Safe to interrupt: yes. Next: build the v2 statistical view.

## 9. Build The CLIP-v2 Statistical View

```powershell
uv run python scripts/build_clip_v2_statistical_view.py --dry-run
```

Label: `READ-ONLY`. Purpose: validate the v2 statistical-view plan. Expected output: JSON plan with 13 descriptors. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute the build.

```powershell
uv run python scripts/build_clip_v2_statistical_view.py --execute
```

Label: `LONG-RUNNING`. Purpose: compute DEV-only v2 descriptors and fit the scaler on Home Credit train features only. Expected output: summary JSON and files in `results/clip_v2/statistical_view/`. Modifies files: yes. Trains a model: no. Safe to interrupt: use `Ctrl+C`, then run `--status`. Next: build contrastive pairs.

Status check:

```powershell
uv run python scripts/build_clip_v2_statistical_view.py --status
```

Label: `READ-ONLY`. Purpose: check whether expected v2 statistical files exist. Expected output: complete/incomplete JSON. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: continue or rerun the build.

## 10. Build CLIP-v2 Contrastive Pairs

```powershell
uv run python scripts/build_clip_v2_contrastive_data.py --dry-run
```

Label: `READ-ONLY`. Purpose: check pair-building prerequisites. Expected output: JSON plan and missing prerequisites if any. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute pair build.

```powershell
uv run python scripts/build_clip_v2_contrastive_data.py --execute
```

Label: `LIGHTWEIGHT`. Purpose: create positive-pair, negative-policy, split, and tensor-schema artifacts. Expected output: files in `results/clip_v2/contrastive_data/`. Modifies files: yes. Trains a model: no. Safe to interrupt: yes, rerun after checking status. Next: train CLIP-v2.

## 11. Train CLIP-v2

```powershell
uv run python scripts/train_clip_v2_encoder.py --dry-run
```

Label: `READ-ONLY`. Purpose: inspect v2 model shape, seeds, and missing prerequisites. Expected output: JSON with `statistical_input_dimension: 13`. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: optional smoke test.

```powershell
uv run python scripts/train_clip_v2_encoder.py --seed 11 --smoke-test --execute
```

Label: `LIGHTWEIGHT`. Purpose: run a tiny training smoke test after inputs exist. Expected output: smoke-test checkpoint path. Modifies files: yes, under `results/clip_v2/training/smoke_test/`. Trains a model: yes, tiny smoke only. Safe to interrupt: yes. Next: full multi-seed training.

LONG-RUNNING COMMAND: this may take substantial time. Run it manually and do not close PowerShell.

```powershell
uv run python scripts/train_clip_v2_encoder.py --all-seeds --execute
```

Label: `LONG-RUNNING`. Purpose: train all configured CLIP-v2 seeds. Expected output: seed summaries and checkpoints in `results/clip_v2/training/`. Modifies files: yes. Trains a model: yes. Safe to interrupt: `Ctrl+C`, then inspect status before resuming manually. Next: audit training.

Inspect Python CPU use:

```powershell
Get-Process python | Select-Object Id, CPU, WorkingSet, StartTime
```

Label: `READ-ONLY`. Purpose: see whether Python is still active. Expected output: process rows. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: wait or stop with `Ctrl+C`.

## 12. Audit CLIP-v2 Training

```powershell
uv run python scripts/audit_clip_v2.py --status
```

Label: `READ-ONLY`. Purpose: inspect v2 artifact directories. Expected output: complete/incomplete JSON. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: validate selector integration.

Proceed only when the audit verdict is acceptable and the selected checkpoint/anchor are present.

## 13. Validate Selector Integration

```powershell
uv run python scripts/validate_clip_v2_selector_integration.py --dry-run
```

Label: `READ-ONLY`. Purpose: validate selector paths and policies without fitting LR or CatBoost. Expected output: JSON with missing prerequisites or pass status. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: plan downstream evaluation.

```powershell
uv run python scripts/validate_clip_v2_selector_integration.py --execute
```

Label: `LIGHTWEIGHT`. Purpose: after CLIP-v2 training exists, verify checkpoint/anchor hashes and materialize versioned v2 score caches. Expected output: `status: passed` plus cache paths in `results/clip_v2/selector_integration/`. Modifies files: yes, score caches only. Trains a model: no. Safe to interrupt: yes, rerun after checking the error. Next: check evaluation status.

## 14. Plan Downstream Evaluation

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --status
```

Label: `READ-ONLY`. Purpose: inspect expected CLIP-v2 evaluation runs. Expected output: eight run statuses. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: review the plan.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --plan
```

Label: `READ-ONLY`. Purpose: print the eight-run plan. Expected output: Home Credit and LendingClub v2 crossed with LR/CatBoost and two selectors. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: run one experiment at a time.

## 15. Run Downstream Evaluation One Experiment At A Time

Do not use a default `--all` command for CLIP-v2. Run reviewed commands one at a time.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset homecredit --model lr --selector clip_v2 --execute
```

Label: `LONG-RUNNING`. Purpose: Home Credit LR with CLIP-v2. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes, then use status/resume. Next: run the next command.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset homecredit --model lr --selector clip_v2_then_mrmr --execute
```

Label: `LONG-RUNNING`. Purpose: Home Credit LR with CLIP-v2 then DEV-only mRMR. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: CatBoost commands.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset homecredit --model catboost --selector clip_v2 --execute
```

Label: `LONG-RUNNING`. Purpose: Home Credit CatBoost with CLIP-v2. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: next CatBoost command.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset homecredit --model catboost --selector clip_v2_then_mrmr --execute
```

Label: `LONG-RUNNING`. Purpose: Home Credit CatBoost with CLIP-v2 then DEV-only mRMR. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: LendingClub v2 commands.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset lendingclub_v2 --model lr --selector clip_v2 --execute
```

Label: `LONG-RUNNING`. Purpose: LendingClub v2 LR with frozen CLIP-v2 selector. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: next LR command.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset lendingclub_v2 --model lr --selector clip_v2_then_mrmr --execute
```

Label: `LONG-RUNNING`. Purpose: LendingClub v2 LR with CLIP-v2 then DEV-only mRMR. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: CatBoost commands.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset lendingclub_v2 --model catboost --selector clip_v2 --execute
```

Label: `LONG-RUNNING`. Purpose: LendingClub v2 CatBoost with frozen CLIP-v2 selector. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: final run.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --dataset lendingclub_v2 --model catboost --selector clip_v2_then_mrmr --execute
```

Label: `LONG-RUNNING`. Purpose: LendingClub v2 CatBoost with CLIP-v2 then DEV-only mRMR. Expected output: one completed run directory. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: rebuild aggregates.

Valid completed runs should be skipped by status-aware execution once that execution path is fully enabled.

## 16. Resume After Interruption

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --status
```

Label: `READ-ONLY`. Purpose: inspect completed, incomplete, and missing runs. Expected output: run status JSON. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: print resume plan.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --resume
```

Label: `READ-ONLY`. Purpose: print a resume plan only. Expected output: planned remaining runs. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: review before execution.

```powershell
uv run python scripts/run_clip_v2_final_evaluation.py --resume --execute
```

Label: `LONG-RUNNING`. Purpose: execute reviewed resume plan. Expected output: completed remaining runs. Modifies files: yes. Trains a model: yes. Safe to interrupt: yes. Next: rebuild aggregates.

## 17. Rebuild Aggregate Tables

```powershell
uv run python scripts/rebuild_clip_v2_evaluation_aggregates.py --dry-run
```

Label: `READ-ONLY`. Purpose: inspect completed runs to aggregate. Expected output: completed run count. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute aggregate rebuild.

```powershell
uv run python scripts/rebuild_clip_v2_evaluation_aggregates.py --execute
```

Label: `LIGHTWEIGHT`. Purpose: rebuild summary tables from saved run artifacts. Expected output: aggregate CSV files. Modifies files: yes. Trains a model: no. Safe to interrupt: yes. Next: build final analysis.

## 18. Build Final CLIP-v2 Analysis

```powershell
uv run python scripts/build_clip_v2_final_analysis.py --dry-run
```

Label: `READ-ONLY`. Purpose: check final-analysis inputs. Expected output: analysis plan. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute final analysis.

```powershell
uv run python scripts/build_clip_v2_final_analysis.py --execute
```

Label: `LIGHTWEIGHT`. Purpose: read saved predictions and produce final tables/plots. Expected output: files in `results/clip_v2/final_analysis/`. Modifies files: yes. Trains a model: no. Safe to interrupt: yes. Next: build reports.

## 19. Build Reports

```powershell
uv run python scripts/build_clip_v2_final_analysis.py --execute
```

Label: `LIGHTWEIGHT`. Purpose: generate Markdown-ready CLIP-v2 report inputs. Expected output: `reports/clip_v2_credit_risk_report.md`, DOCX/PDF when rendering support is available. Modifies files: yes. Trains a model: no. Safe to interrupt: yes. Next: final audit.

Expected report paths:

```text
reports/clip_v2_credit_risk_report.md
reports/clip_v2_credit_risk_report.docx
reports/clip_v2_credit_risk_report.pdf
reports/clip_v2_scientific_verdict.md
reports/clip_v2_limitations.md
reports/clip_v2_reproducibility_manifest.json
```

## 20. Run Final CLIP-v2 Audit

```powershell
uv run python scripts/audit_clip_v2.py --dry-run
```

Label: `READ-ONLY`. Purpose: preview final audit checks. Expected output: pass/fail checks; incomplete runs should fail until all artifacts exist. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: execute final audit after all runs, aggregates, analysis, and reports exist.

```powershell
uv run python scripts/audit_clip_v2.py --execute
```

Label: `READ-ONLY`. Purpose: enforce the final archive gate after the full CLIP-v2 study is complete. Expected output: `PASS - CLIP-v2 is scientifically defensible and ready to archive`. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: commit and tag only after PASS.

Required final verdict before scientific claims: all eight runs complete, metrics recompute, no leakage, and LendingClub v2 remains external.

## 21. Optional Ablation Study

Leave-one-statistic-out ablation asks whether one v2 descriptor is driving the result. It is optional and should happen only after the main CLIP-v2 study is complete.

```powershell
uv run python scripts/plan_clip_v2_ablation.py --plan
```

Label: `OPTIONAL`. Purpose: print the ablation plan. Expected output: full v2 plus seven leave-one-out variants. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: decide whether to run later.

Do not use OOT results to decide which descriptor to retain.

## 22. Git Commit And Tags

```powershell
git status --short
```

Label: `GIT`. Purpose: inspect changed files. Expected output: modified/untracked file list. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: review staged files.

```powershell
git diff --cached --stat
```

Label: `GIT`. Purpose: review staged changes. Expected output: staged file summary. Modifies files: no. Trains a model: no. Safe to interrupt: yes. Next: commit.

Confirm `.tmp`, `.uv-cache`, and cache directories are not staged before committing.

```powershell
git commit -m "feat: add CLIP v2 compact statistical view pipeline"
```

Label: `GIT`. Purpose: commit verified CLIP-v2 code/artifacts. Expected output: commit hash. Modifies files: Git history. Trains a model: no. Safe to interrupt: avoid interrupting. Next: push if desired.

```powershell
git tag -a clip-v2-complete -m "Verified CLIP-v2 compact target-free statistical view study"
```

Label: `GIT`. Purpose: tag the verified CLIP-v2 study after final audit. Expected output: no output on success. Modifies files: Git tag. Trains a model: no. Safe to interrupt: avoid interrupting. Next: push.

## 23. Troubleshooting

Windows temporary-directory permission error: set `$env:TMP` and `$env:TEMP` to `$PWD\.tmp`, then rerun the command.

Interrupted Python process: run the matching `--status` command before any resume command.

Apparently frozen pytest: check CPU with `Get-Process python`. Some tests import heavy packages.

Apparently frozen CatBoost: check CPU and memory. CatBoost can be quiet for long periods.

Stale `.in_progress` directory: do not delete blindly. Run status/resume first and inspect the recovery plan.

Missing model cache: rerun the dry-run command for that stage. It will print missing prerequisites.

Hash mismatch: do not overwrite artifacts to hide it. Re-verify the upstream freeze or rebuild only the affected stage after understanding the mismatch.

Low disk space: stop before long-running commands and move old optional outputs outside the repo if needed.

Out-of-memory error: close other processes and rerun one dataset/model at a time.

Rerun only one failed step by using explicit `--dataset`, `--model`, and `--selector` where supported.

Do not delete result directories blindly because they contain hashes, recovery state, and evidence needed for audit.

## 24. Output Directory Guide

`results/clip/`: frozen CLIP-v1 working artifacts.

`results/clip_versions/v1/`: hash-based CLIP-v1 freeze package.

`results/clip_v2/`: all CLIP-v2 statistical, contrastive, training, selector, evaluation, analysis, and audit outputs.

`reports/`: final Markdown/DOCX/PDF reports and scientific summaries.

`configs/clip_v2/`: CLIP-v2 configs.

## 25. Scientific Interpretation

CLIP-v1 is a missingness-only semantic-statistical experiment. CLIP-v2 tests whether replacing that one-dimensional statistical branch with a compact richer target-free vector improves contrastive feature screening.

Better contrastive loss does not prove better credit scoring. The final evidence remains OOT AUC, OOT KS, Lift@10, score PSI, external LendingClub v2 behavior, seed stability, semantic coverage, and redundancy.

LendingClub v2 must remain external. The LLM workflow remains a frozen comparison.
