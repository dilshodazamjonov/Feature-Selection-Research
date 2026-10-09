# Repository freeze report (submission 2026-10)

Branch `freeze/submission-2026-10`, started 2026-10-09 from `main` at `ddbb067` ("added: final rerun code", 2026-10-08).
Manuscript: `~/projects/article/submission/manuscript.tex` (title "Row-Independent Feature Pre-Screening with Language Models for High-Dimensional Credit Scoring: A Frozen Evaluation on Temporal and Recency-Shift Holdouts", 1,899 lines, last modified 2026-10-08 17:30) and `submission/supplementary.tex` (Tables S1 to S6). The root `~/projects/article/manuscript.tex` carries the older title and is not the submission file.

This checkout is the WSL2 Linux clone (bash), not the Windows laptop on which the experiments ran. Paths below are relative to the repository root unless prefixed `ARTICLE:` (`~/projects/article`).

## 1. Inventory of the working tree (before any change)

514 tracked files at `ddbb067`; `git status` clean; local `main` equal to `origin/main`.

| dir | files | GB | newest mtime | tracked | purpose (as far as visible) |
|---|---:|---:|---|---:|---|
| `data/` | 70 | 11.414 | 2026-07-22 | 0 | raw and processed Kaggle data (ignored; the user decides) |
| `.venv/` | 34,280 | 5.524 | 2026-10-05 | 0 | uv environment |
| `outputs/` | 1,568 | 4.319 | 2026-09-27 | 0 | `outputs/todo_fill/`: Gate-4 fill checkpoints (`hcms2024_matrix/` 3.4 GB cached Stability frames, `frames/` 786 MB, `gate4/` 167 MB fits/rankings, `reruns/` 62 MB, `b6/` 6 MB obfuscation rankings and fits, `selections/`) |
| `cleanup/` | 318 | 0.063 | 2026-09-20 | 11 | `cleanup/audits/` audit layer of earlier protocol stages (manifests, selection stability, PSI, runtime), `cleanup/audit/` repository-cleanup inventories |
| `tests_runtime/` | 364 | 0.009 | 2026-09-20 | 0 | pytest runtime scratch |
| `todo_done/` | 59 | 0.006 | 2026-09-27 | 0 | Gate-4 fills (37 to 47) with `fill_manifest.json` / `fill_log.md`, `final_run/`, `reruns_run/`, `final/` |
| `src/` | 374 | 0.006 | 2026-09-20 | 187 | package source (untracked files are `__pycache__`) |
| `scripts/` | 165 | 0.003 | 2026-09-26 | 76 | entry points incl. `fill_todo*.py` |
| `tests/` | 140 | 0.002 | 2026-09-20 | 115 | test suite |
| `configs/` | 67 | 0.001 | 2026-09-20 | 67 | dataset/model/selector/protocol configs |
| `artifacts/` | 51 | 0.001 | 2026-09-20 | 0 | `artifacts/llm_cache/`: 51 cached LLM ranking responses (populated locally; the old README says empty in a clone) |
| `todo/` | 5 | 0.000 | 2026-09-27 | 0 | rerun skeletons (`README_reruns.md`, pool-control and response-repeat CSVs) |
| `results/` | 41 | 0.000 | 2026-10-08 | 18 | tracked `stability_metrics_output/` (18 CSVs) plus ignored `final_research_package_v2/`, `final_three_dataset_synthesis_v1/`, `finalized_research/`, `research_summary/`, `final_experiments/` (manifests only) |
| `reports/` | 17 | 0.000 | 2026-09-20 | 0 | `reports/archive/*.md` generated reports |
| `logs/` | 8 | 0.000 | 2026-09-20 | 0 | 2026-06-29 transfer-experiment logs |
| `final_artifacts/` | 4 | 0.000 | 2026-09-24 | 0 | 39a/44/45/47 CSVs from the 2026-09-22 fill run |
| `docs/` | 31 | 0.000 | 2026-09-20 | 31 | documentation |
| `credit_risk_fs/` | 2 | 0.000 | 2026-09-20 | 1 | stray top-level shim package (`__init__.py`) |

Top-level files: `.env` (ignored, never committed), `.env.example`, `.gitattributes`, `.gitignore`, `.python-version`, `METHODOLOGY_AUDIT_9_POINTS.md` (ignored), `README.md`, `pyproject.toml`, `pytest.ini`, `uv.lock`. Whole checkout 22 GB; disk has 901 GB free.

**Expected trees that are absent on this machine.** `experiments/` (the ~100 GB fold checkpoints), `results/runs/`, `results/prompt_16_homecredit_model_stability_2024/` (the Stability 2024 accepted request/response, prompt and ranking), `results/run_index.csv`, `results/full_baseline_v1/`, `results/selector_combinations_v1/`, `results/final_experiments/id_preserved_predictions/`, `cleanup/audits/**/report.html` are all absent; there is no `/mnt/d`. The frozen HO prediction vectors, freeze manifests and fold outputs cited by `ARTICLE:priority_1/pairwise.csv` live under `D:/python projects/Research...` on the Windows machine. The compact-evidence archive of Section 8 can therefore only hold what is here: the audit layer, the LLM cache, the Gate-4 fills, the two-dataset package and the article's result CSVs.

## 2. Orientation notes

- The manuscript's only Stability 2024 cohort is the 2020-02-26 cutoff (DEV 1,221,743 / HO 304,916, 8,349 HO events); the tracked `results/stability_metrics_output/` (commit `b778999`, 2026-10-08 18:02, author Dilshod A'zamjonov) carries that cohort. The article folder's `priority_1/full_matrix.csv`, `priority_1/pairwise.csv`, `priority_2/*_20200201.csv`, `priority_2/44_mechanical.csv` (ST rows), `catboost_native_cat_ho_auc.csv`, `diversity_capped_ho_auc.csv`, `l1_selector_ho_auc.csv`, `selected_feature_psi.csv` and `repeated_calls_ho_auc.csv` hold the superseded 2020-02-01 cohort for Stability (Pure LLM 0.8344 / 0.8784, n_ho 369,147) and are not sources for any Stability number.
- `README.md` at `ddbb067` is stale (old title, cohort 2020-02-01 / 1,157,512 / 369,147, "artifacts/llm_cache is empty"); its "verified local support" and "snapshot" sections list the known gaps (KSG/Ross absent, IV interval, L1 class weights, missing caches). Prompt 2 rewrites it.
- `.gitignore` ignores every `*.csv` with four `cleanup/audits/**` exceptions and `*/*.json`; `results/*` is ignored with exceptions for `README.md`, `run_index.csv` (neither tracked) and `stability_metrics_output/`. `.gitattributes` marks four untracked folders as `binary`.
- Tracked Stability CSV defects seen on first read: CRLF in `03`, `04`, `11`, `16`, `17`; trailing `,,,,,` row in `03`; `favourable_range` in `04` undefined and inconsistent (`catboost, mRMR` mean 0.8274 outside 0.8283-0.8322); `ranges` in `10` (`40-100 or 35`); `Full feature reference` naming; `00_run_ids.csv` lacks `17_budget_sweep.csv` and any source path; the seven run ids match no file on this machine.
- Gate-4 provenance: `todo_done/fill_log.md` and `fill_manifest.json` (2026-09-22) record that the local refits do not reproduce the paper's cached-ranking HO AUCs (e.g. HC CB Pure LLM 0.7583 vs 0.7935) because the local `artifacts/llm_cache` full-DEV rankings are truncated / the Stability named ranking is a regenerated control, and that 15 to 27 LendingClub obfuscation and repeat calls failed the frozen contract. The article's `priority_2/` CSVs (moved there by the user on 2026-09-22) are the values the manuscript prints.
- `scripts/fill_todo_reruns.py` (pool controls, 50-call response repeats) writes `todo_done/poolcontrols_*.csv`, `responserepeats_*.csv`; none of these is reported in the manuscript and none enters `results/`.
- `src/credit_risk_fs/selectors/llm_screening.py`: cache files are named `<scope>_<fold_key>_<sha256 of cache_key payload>.json` under `cache_dir` (default `results/_llm_rankings_cache`; the local files sit in `artifacts/llm_cache/`); each payload records `request_sha256`, `prompt_sha256`/`prompt_hash`, `raw_content_sha256`, `metadata_signature`, `description_sha256` per record, `response_model`, `response_id`, token counts. Gate-4 calls (`scripts/todo_fill/gate4/llm.py`) are stored as `outputs/todo_fill/gate4/rankings/<step>/<name>.json` with `ranking`, `ranking_ids`, `n_candidates`, `prompt_sha256`, `response_id`, `attempts`.

(Sections 3 to 9 follow as the work proceeds.)
