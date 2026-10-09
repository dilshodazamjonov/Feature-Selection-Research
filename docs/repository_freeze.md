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


## 3. Evidence index (prompt Section 2)

`docs/freeze_evidence_index.csv`: 198 rows, one per manuscript element examined (126 `found`, 47 `derived`, 25 `missing`; 92 match the manuscript, 60 partially, 31 do not, 15 not checked). Columns: `manuscript_element, dataset, target_csv, source_path, source_sha256, source_bytes, source_mtime, status, matches_manuscript, local_value, manuscript_value, note, agent`. The rows were produced by seven read-only evidence sweeps (cached rankings; classical stability; subsets, dimension and PSI; headline tables; voting, runtime and cohort facts; protocol estimators; the tracked Stability folder), each with a full report kept in the local evidence archive (`paper_evidence_2026-10/reports/`). Where two sources disagreed, the one matching the manuscript was used and the disagreement is listed in Section 10.

Where the numbers live (scrubbed paths; `ARTICLE:` = `~/projects/article`):

- `ARTICLE:priority_1/full_matrix.csv` (HO AUC, KS, Capture@10, Brier, five fold AUCs per selector; Home Credit and LendingClub rows) and `priority_1/pairwise.csv` (Tables 3 and 4 deltas and intervals; its `score` column records the prediction vectors on `D:/python projects/Research_pre_cleanup_backup_20260704/results/final_experiments/id_preserved_predictions/...` and `results/full_baseline_v1/...`, none of which is on this machine). Their Stability rows are the superseded 2020-02-01 cohort and were not used.
- `ARTICLE:priority_1/dev_feature_budget_auc.csv` (Table C.1, Figure C.1b, exact to six decimals for Home Credit and LendingClub).
- `ARTICLE:priority_2/*.csv` (robustness outputs: sub-populations, diversity caps, native categoricals, L1, per-fold responses, lineage-only, obfuscation lists and AUCs, PSI, one-hot dimension, Boruta sizes, missingness bands, IV cap, mRMR timing, ten Stability repeats) and `ARTICLE:appendix_e_subsets.csv` (the twelve subsets; identical to supplementary Tables S1 to S6 and to `priority_2/headline_subsets.csv`).
- `results/final_research_package_v2/final_results_tables.csv` (two-dataset package: full-feature references, the full_baseline_v1 selectors, voting P=200, runtime and memory of those runs).
- `cleanup/audits/prompt_11_selector_combinations/baseline_selection_stability.csv` (Table 9 classical rows), `cleanup/audits/prompt_13_combination_dev_review/stage_support_audit.csv` (IV-then-Boruta confirmed counts), `configs/protocols/row_alignment_contract_v1.json` (cohort counts and windows), `cleanup/audits/resource_safe_execution/hardware_preflight.json` (machine), `cleanup/audits/third_dataset_protocol_freeze/*` (Stability 461/434, split and fold boundaries).
- `reports/archive/homecredit_report.md` and `cross_dataset_v2_analysis.md` (Home Credit hybrid stability at d = 529, Home Credit LR and LendingClub Pure LLM wall-clock seconds).
- `artifacts/llm_cache/*.json` (51 files: the 6 Home Credit and 24 LendingClub v2 calls of the manuscript plus 21 older files), `outputs/todo_fill/b6/rankings` (obfuscation arms), `outputs/todo_fill/gate4/rankings/44` (lineage-only calls).
- `results/stability_metrics_output/` at `ddbb067` (the Stability 2024 values; see Section 5 and finding 5).

## 4. Layout as built (prompt Section 3)

Three uniform folders with 25 files each (`00` to `22`, plus `19b_obfuscation_ranks.csv` and `19c_obfuscation_fold_jaccard.csv`), `results/headline_subsets.csv`, `results/llm_rankings/` and `results/README.md`. Decisions recorded:

- Selector name `Full features` everywhere (the Stability files said `Full feature reference`).
- `11_monthly_ho_auc.csv` and `16_ten_responses.csv` of the Stability folder were folded into the long-schema `11_subpopulation_ho_auc.csv` and `16_response_spread.csv` and removed from the tree (their content is preserved row for row; the two files remain in history).
- `11_subpopulation_ho_auc.csv` carries a `backbone` column (the prompt's schema omitted it; LendingClub reports both backbones).
- `17_budget_sweep.csv` uses the long schema everywhere (`backbone, K, selector, family, dev_mean_auc, is_llm_leader, is_classical_leader`); Home Credit and LendingClub hold all 13 selectors at K = 10..50; Stability holds only the leader and pooled values the tracked three-column file had (leaders named through `04` at the documented budget, blank elsewhere).
- Table 7 is split into `19` (arm AUCs and overlaps), `19b` (EXT_SOURCE ranks per call and arm) and `19c` (pairwise fold Jaccard at K = 20).
- `08_secondary_metrics.csv` holds the whole matrix for Home Credit and LendingClub (KS, Brier, Capture@10 for all 13 selectors; log loss and Lift@10 only where the same run exists in the two-dataset package) and only the best row for Stability (all that was recorded).
- Row counts (HC / LC / ST): 01: 4/7/3; 02: 3/14/2; 03: 28/28/28; 04: 28/28/28; 05: 5/4/4; 06: 4/4/4; 07: 26/26/6; 08: 28/28/1; 09: 120/120/120; 10: 36/36/36; 11: 1/30/20; 12: 2/2/2; 13: 3/3/2; 14: 2/2/2; 15: 2/2/4; 16: 12/12/22; 17: 130/130/30; 18: 2/2/2; 19: 4/1/1; 19b: 54/1/1; 19c: 30/1/1; 20: 6/6/2; 21: 6/24/1; 22: 6/12/5; 00: 111/85/29 provenance rows. `headline_subsets.csv`: 360 rows (6 x 20 + 6 x 40; the prompt's "240" was a miscount). `llm_rankings/`: 81 files, 90 index rows (one row, the Stability named ranking, has no file).

## 5. Stability folder repairs (prompt Section 4)

1. `03`: trailing `,,,,,` row removed; `Full feature reference` renamed; `status` column replaced by `note`.
2. `04`: `favourable_range` dropped (undefined; `catboost, mRMR` 0.8274 lay outside its own 0.8283-0.8322; the widths are a constant 0.0039 in 21 of 26 rows, an order of magnitude narrower than fold spreads, so it is not a fold min-max). No per-fold validation AUC of the 2020-02-26 cohort exists locally, so `dev_mean_auc` is kept alone and the note says so.
3. `10`: `ranges` dropped; `n_selected` from `ARTICLE:priority_2/boruta_subset_sizes.csv` and `below_budget` added (the one below-budget Stability cell is LLM then Boruta, CatBoost, fold 1, 35 of 40). The tracked `n_confirmed` values are identical for the LR and CatBoost rows of every selector and partition, and for that fold-1 cell read 45 (above budget) while 35 features were selected; kept as tracked, flagged in the note.
4. `12`: Pure LLM row added (0.8716 native = one-hot, unchanged by construction) and the one-hot column filled from `03`.
5. `13`: `selector`, `ho_auc_cached` (from `03`) and `overlap_with_cached` (15/20, 30/40 from `ARTICLE:priority_2/44_mechanical.csv`) added.
6. `00_run_ids.csv`: 29 rows, every file listed, each with the source path (`REPO:results/stability_metrics_output/<file>@ddbb067`), the SHA-256 of that tracked blob and the run id the original file recorded. The seven run ids match no file on this machine (grep over the repository, the article folder and `git log -S`).
7. Line endings: every file LF (the five CRLF files were renormalised by `.gitattributes` and rewritten); `07` now ends with a newline.
8. `01`/`02`: uniform column names; the local provenance paths (`cleanup/audits/...`, `outputs/todo_fill/gate4/third_candidates.json`, `results/final_three_dataset_synthesis_v1/...`) moved into `note`; `02` gained the 461-reviewed / 434-approved row from the protocol-freeze audit.
9. Added `18` (Nogueira 0.8052 / 0.7766, Jaccard 0.6908 / 0.6506, 11 / 22 names in all draws, 30 / 69 distinct, recomputed from the ten repeated lists, d = 1,068, B = 10), `19`/`19b`/`19c` (note rows), `20` (note rows), `21` (one row: token usage not recorded), `22` (522.15 s vs 20.52 s on fold 1, matched; peak RSS rows blank, no local source; machine line from the hardware preflight).
10. Cross-check after the edits: every Stability number of the manuscript that has a cell matches (Tables 2, 3, 4, 5, 8, 9, 10, 11, C.1, F.2, F.3, F.4, the monthly counts and delta ranges, the PSI sentences, the native-categorical and L1 sentences).

## 6. Home Credit and LendingClub folders, headline subsets, cached rankings (prompt Section 5)

Built from the sources of Section 3 by two assembly scripts kept in the evidence archive (`paper_evidence_2026-10/build/`); every cell is copied from a named file or marked derived in its `note`. Highlights:

- `03`/`04`/`05`/`08`: `full_matrix.csv` and `pairwise.csv` reproduce Tables 3, 4, 5 and 11 digit for digit (52 of 52 HO AUCs, 8 of 8 deltas and intervals, all DEV means and leaders, all KS/Brier/Capture@10 best rows); the full-feature references and the Figure 3 coordinates come from the two-dataset package.
- `07`: classical rows from the prompt-11 audit (Jaccard exact; Home Credit Nogueira filed at d = 529 and rescaled to d = 373 by the exact identity on the fold-overlap sum, which reproduces the manuscript except the HC CatBoost IV/WOE cell, 0.8404 vs 0.8403); Home Credit hybrid rows from the archived Home Credit report (d = 529, rescaled; all six reproduce); Pure LLM rows recomputed from the cached fold lists; the four LendingClub hybrid Nogueira / mean-Jaccard cells have no local source (min and max do).
- `18`: Table F.4 reproduces from the cached fold lists (6 / 42, 8 / 98, 4 / 39, 14 / 71 names in all draws / distinct).
- `19`: Table 7 AUCs and fold ranges from `obfuscated_ho_auc.csv`; the full-DEV overlaps (11/20, 17/40, 14/20, 24/40) reproduce against the cached named rankings using the article's arm lists; the fold overlaps (12-18 mean 14.2, 15-28 mean 23.2, 14-19 mean 16.2, 26-33 mean 28.8) reproduce only against the named re-calls of the obfuscation run (`namedrecall` files), not against the cached fold rankings (which give 6-12, 9-19, 3-15, 15-20).
- `21`: the 30 cached calls (6 Home Credit + 24 LendingClub) sum to exactly 806,757 input and 56,495 output tokens.
- `headline_subsets.csv`: 360 rows; all four local sources agree name for name and in order; overlaps 4/20, 16/40, 4/20, 21/40, 1/20, 4/40 and the Section 7.5 protected-attribute ranks reproduce.
- `llm_rankings/`: 6 Home Credit named, 24 LendingClub v2 named, 6 arm A, 6 arm B, 6 named re-calls, 31 lineage-only (one failed call kept as a record), the ten Stability repeats as lists, and one index row without a file for the Stability named ranking (not on this machine). No secret string in any copied file.

## 7. Verification (prompt Section 6)

```
SCHEMA: ok
CHECK: 965 manuscript values compared, 861 matched, 97 blank (no CSV source - see docs/repository_freeze.md), 7 flagged (known discrepancies listed in scripts/manuscript_oracle.py), 0 mismatched
tests/test_manuscript_tables.py: 2 passed
```

The 97 values without a CSV source: Table B.1 voting P = 100 and P = 300 rows (voting AUC, delta, 95 % CI, Holm p, delta KS; 56 cells) and the P = 200 intervals and Holm p (12 cells) because the `cross_dataset_voting_inference_v1` payload is absent (its manifest lists the files); Table 9 LendingClub Stable core + LLM fill and LLM then Boruta Nogueira and mean Jaccard (8); Appendix F LendingClub CatBoost wall-clock minutes for seven pipelines and the 176.6-minute selection stage (8); Section 6.7 peak RSS ranges (6); Section 5.1 LendingClub DEV 36-month count and DEV term event rates (3), the 93 source columns and the eight composition counts (9); Table 2 Home Credit candidate count 373 (1); Table 11 Lift@10 for Home Credit and LendingClub (2; Capture@10 is present and Lift is its decile ratio).

The 7 flagged values: Table 7 note "named 1, 2, 3" (the cached ranking and Appendix D give EXT_SOURCE_3, _2, _1 at ranks 1, 2, 3); Table 9 HC CatBoost IV/WOE 0.8403 (recomputed 0.8404); Table 9 / F.4 HC Pure LLM 0.5403 and 0.4092 (recomputed 0.5404 and 0.4091 from the cached lists at d = 373).

## 8. `.gitignore` and `.gitattributes` (prompt Section 7)

Rewritten as specified, with one correction: the local-tree rules are anchored to the repository root (`/experiments/`, `/artifacts/`, `/outputs/`, ...). The unanchored `experiments/` of the prompt would have ignored the tracked `src/credit_risk_fs/experiments/` (35 files) and `configs/experiments/` (10 files). After the rewrite `git ls-files -ci --exclude-standard` is empty and `git status --ignored results/` lists only the five ignored legacy folders. `git add --renormalize .` changed line endings only (five Stability CSVs).

## 9. Compact evidence archive and the deletion question (prompt Section 8)

**There is no `experiments/` tree and no `results/runs/` on this machine**; the ~100 GB of fold checkpoints, the frozen HO prediction vectors and the Stability run directories live on the Windows laptop. The checkout is 22 GB: `data/` 11.4 GB (raw data; not on the list), `.venv/` 5.5 GB (not on the list), `outputs/` 4.3 GB (of which `outputs/todo_fill/hcms2024_matrix/` 3.4 GB and `frames/` 0.8 GB are cached training frames), everything else under 0.1 GB.

Archive written to `../Feature-Selection-Research_archive/paper_evidence_2026-10/` (1,087 files, 30.7 MB, `MANIFEST.csv` with path, bytes, sha256; `README.txt` lists the contents): the 51 cached LLM responses, the obfuscation rankings and fits, the Gate-4 rankings, base rankings, selection overrides, fit records (`*.json`; the `*.npy` score vectors of the local refits were left out), `todo_done/`, `final_artifacts/`, `todo/`, `cleanup/audits/`, the two-dataset and synthesis packages, `reports/`, `logs/`, and the article folder's `priority_1/`, `priority_2/`, `appendix_e_subsets.csv`, `manuscript.tex`, `supplementary.tex`. The verification (`--schema`, `--check`, pytest) was re-run after the copy: unchanged.

Deletion list prepared (nothing executed): `outputs/` (4.3 GB), `todo_done/` (6 MB), `todo/`, `final_artifacts/`, `artifacts/` (llm_cache, copied), `reports/`, `logs/`, `cleanup/audits/`, `cleanup/audit/` (63 MB together), `tests_runtime/` (9 MB), and the git-ignored `results/` folders (`final_experiments/`, `final_research_package_v2/`, `final_three_dataset_synthesis_v1/`, `finalized_research/`, `research_summary/`, 0.4 MB). Not listed: `data/`, `.venv/`. The question is in the final message.

## 10. Pre-submission findings (prompt Section 9)

1. **mRMR estimator.** No Kraskov (KSG) or Ross estimator exists anywhere in the repository; `mi_mrmr.py` rejects the k-NN estimator in its docstring and uses `sklearn.metrics.mutual_info_score` over quantile-binned variables (n_bins 10, objective `mid`) for relevance and redundancy alike. Three of the four stand-alone mRMR headline cells (HC CatBoost 0.7599, LC LR 0.6859, LC CatBoost 0.7020) equal, to the last digit, the `mrmr_mutual_information` runs fbv1-014/015/016 of the two-dataset package, so the discretised estimator produced them. The Home Credit LR cell is different: its five fold AUCs, KS and Brier in `full_matrix.csv`, and the 20-name subset of Appendix D, are exactly those of the discretised run fbv1-013, whose HO AUC is **0.7351**, while the printed HO AUC **0.7699** was scored from `.../id_preserved_predictions/predictions/homecredit/lr/statistical_mrmr/oot_predictions.csv` (Windows backup), the output bucket of the legacy RF/correlation `mrmr` selector (`matrix.py`, `recompute_pairwise_csv.py`). The row therefore mixes two runs. Were 0.7351 the HC LR mRMR value, the HO-rule classical leader of that case would be IV then Boruta (0.7534) and the gap to Stable core + LLM fill (0.7489) would be 0.0045, below the materiality bar, so the one case "the classical leader wins by 0.021" rests on this cell. The LLM then mRMR cells have only one implementation in the repository (`LLMThenStatSelector` with `RandomForestRelevanceMRMRSelector`, the RF/correlation stage; the pairwise paths point to `hybrid_llm_then_mrmr`), so the sentence "applies mutual-information mRMR on the fold training partition" (Section 4.3) is contradicted by every local artefact. Sentences to change before submission: Section 3.2 (KSG/Ross), Section 4.3, Table E.1 ("mutual-information mRMR distinguished from the RF/correlation filter"), and the Home Credit LR mRMR cell needs the author's decision.
2. **Candidate records offered to the LLM.** The cached Home Credit calls carried 374 / 369 / 365 / 374 / 372 / 391 feature records (fold1..fold5, full DEV; union 422, intersection 335; none equals 373) and the LendingClub v2 calls 155 / 154 / 155 / 157 / 160 / 161, not 675. The lists vary by partition because the matrix runner applied `MissingRateFilter(0.95)` and then an **IV/WOE prefilter** (`iv_filter_kwargs = {min_iv: 0.01, max_iv_for_leakage: 0.5}`) before building the records (`runner.py`, `registry.py`); the Stability path used no IV prefilter and 1,068 records. So on Home Credit and LendingClub the "target-free" semantic ranking was preceded by a target-aware screen, and Table 2 ("the same candidate set" for every selector), Section 4.4 ("over the same candidate list in every call") and the Table 9 note are contradicted. The Home Credit count 373 has no local source at all (`scripts/todo_fill/steps.py` records it as not reproducible; the local LLM-eligibility screen gives 391). Two further contradictions in the same files: 22 of the 24 LendingClub responses do not hold exactly the requested number of names (e.g. the full-DEV budget-20 call returned 93, the budget-100 call 52) and seven responses contain names outside the candidate records (all at ranks above 40), against the validation statement of Section 4.4; and EXT_SOURCE_1 is absent from the Home Credit fold-4 and fold-5 responses, against "in every named Home Credit call the three external scores occupy ranks 1, 2 and 3" (true for full DEV and folds 1 to 3). The token sum of exactly these 30 files equals the manuscript's 806,757 / 56,495, which confirms they are the manuscript's calls.
3. **Cached rankings.** Present and copied: the 6 Home Credit and 24 LendingClub v2 named responses, the 12 obfuscation arms, the lineage-only calls of 2026-09-22 (they reproduce the Table 8 overlaps 12/20 for the two LR cases but not 24/40, 24/40, 26/40, 15/20, 30/40). Absent: the accepted Stability 2024 ranking (the `dev_llm_supplement_v3` tree is not here; only its top-40 survives in the subset files) and the raw JSON of the ten Stability repeats (their 100-name lists and response ids are in the article folder and reproduce Table F.4 exactly). The obfuscation run's own "named" calls are fresh re-calls, not the cache; Table 7's fold-overlap column reproduces only against them, its full-DEV column only against the cache.
4. **`headline_subsets.csv`** created (360 rows); consistent with Tables S1 to S6, `appendix_e_subsets.csv`, `priority_2/headline_subsets.csv` and the tracked Stability subsets.
5. **Stability 2024 cohort and provenance.** No value from the 2020-02-01 cohort entered `results/`; the tracked folder is the 2020-02-26 cohort and every manuscript Stability number with a cell matches it. But the folder's own support on this machine is thin: beyond the counts (01, 02, the monthly n, 461/434, all backed by the protocol lock and the split and population-profile audits) no local run output supports any AUC, interval, PSI or stability value, and the seven run ids match nothing. Several tracked values coincide with other artefacts in ways the author should explain before submission: the Table 8 Stability LR interval [+0.0034, +0.0086] and mechanical AUC 0.8396 are identical to the superseded-cohort row of `priority_2/44_mechanical.csv` (0.839634, [0.00339331, 0.00859909], computed for cached 0.8344 and a delta of +0.0052), and share their trailing digits with the local Gate-4 fit of the same cohort (0.639634, [-0.05959909, -0.04939331]); the tracked `06` PSI means and medians equal the superseded-cohort file to four decimals for three leaders while all four maxima and three max features differ; the tracked depth-0 controls (0.7804, 0.8231) differ from the local same-cohort fits (`gate4/fits/23b3994e...`, `bac24ade...`: 0.774842, 0.820919) and the local refits of the regenerated Stability ranking give 0.694 / 0.769 against the tracked 0.8322 / 0.8716; `10`'s `n_confirmed` is duplicated across backbones; `04`'s dropped range column had constant widths. `scripts/fill_todo_reruns.py` still defaults `--third-cohort` to the 2020-02-01 cohort and calls it "the paper's" (Prompt 3).
6. **IV/WOE interval and L1 class weights.** As frozen and run, neither the stand-alone IV/WOE ranker (`iv.py`: rank by IV, keep top K, no bounds) nor the IV-then-Boruta screen (`combinations.py`: top-P pool by IV rank, P = 100 on Home Credit, 300 on LendingClub, 200 on Stability) applies the 0.01-0.50 interval; the interval exists only in the Pure LLM candidate prefilter (`registry.py`) and the rerun IV-pool control. The statement that the upper bound removed nothing agrees with `iv_above_cap_status.csv` (no feature above 0.50 on the fold-1 rows), but the sentence that the interval "serves the IV/WOE ranker and the screening stage of IV then Boruta" does not describe the runs. The frozen L1 runs (fbv1-017..020) have `class_weight` unset (unweighted), and their HO AUCs (0.7453 / 0.7742 HC, 0.6811 / 0.7096 LC) are not the manuscript's; the manuscript's L1 values (0.674-0.842) come from `priority_2/l1_selector_ho_auc.csv` and the tracked Stability file, which match no local run (the "run afterwards" with balanced weights is not on this machine).
7. **Stability token usage.** `21_llm_usage.csv` for Stability holds one row whose note says "token usage not recorded" and nothing else.
8. **Other disagreements and gaps.**
   - Several article result files behind the robustness tables are the local Gate-4 refit outputs plus round constants: `priority_2/perfold_rankings_ho_auc.csv` (Table 10 HC/LC) equals `todo_done/40_perfold_ho.csv` + 0.01 / 0.04 / 0.02 / 0.02 per case; `priority_2/44_mechanical.csv` (Table 8) equals `todo_done/44_mechanical.csv` + 0, 0.04, 0.07, 0.032, 0.07 (HC LR, HC CB, LC LR, LC CB, LC CB LLM then mRMR) with interval bounds that reuse the local bounds' digit strings; `priority_2/catboost_native_cat_ho_auc.csv` LC LLM then mRMR 0.77379 vs local 0.750379; `priority_2/obfuscated_ho_auc.csv` arm-A LR fold AUCs are the local arm-B LR fits and vice versa, and arm-A CatBoost folds are the local arm-A fits + 0.05/0.04/0.04/0.01/0.01. The manuscript prints the article files' values; the local files are the only computed artefacts. The author must confirm which run produced the printed values.
   - The HC LR Stable core row of `priority_2/44_mechanical.csv` has its mechanical and cached columns swapped (copied in the manuscript's reading).
   - Section 6.1 says the mixed-rule interval was not computed; `priority_2/conservative_pair_interval.csv` holds one ([-0.0285, -0.0250]) of unrecorded provenance (already noted in the article's CHANGES.md).
   - The archived two-dataset run behind Table 9's LendingClub Pure LLM stability (0.6280 / 0.4828, 0.5517 / 0.4011) reports HO AUCs 0.7137 / 0.6927 for the same pipelines, not the manuscript's 0.7335 / 0.7402; the archived Home Credit run behind the hybrid stability rows likewise has other HO AUCs (e.g. LLM then Boruta LR 0.6488 vs 0.7337).
   - Per-feature PSI: the prompt-11 audit lists EXT_SOURCE_3 at 0.0207, PREV_INTEREST_ESTIMATE_VAR 0.0774, AMT_REQ_CREDIT_BUREAU_QRT 0.2864 and PREV_DAYS_LAST_DUE_1ST_VERSION_MAX 0.3254 for Home Credit subset members, above every maximum in `selected_feature_psi.csv` (the manuscript's source), while the LendingClub values agree exactly.
   - Home Credit HO window: contract `[-240, 0]` vs manuscript "day -240 to -1".
   - Table B.1: only the four P = 200 AUC pairs, delta KS and fold Jaccard are local; the inference doc states voted-subset Jaccard 0.396-0.462 while the package gives 0.4897 for HC LR P = 200 (manuscript 0.40-0.49).
   - Runtime: the 12.1-minute RF/correlation filter (HC LR) is the archived legacy report's 728 s, whereas the two-dataset package records 1,535 s for the same selector; the peak-RSS ranges match no local table (package: 7.80-11.22 GB, 19.98-30.44 GB); the CPU model is in no audit file; a GPU (RTX 4060 Laptop) was present but unused.
   - No local source for: the 2019-05-01 snapshot date, the 93 source columns and the eight composition counts, the LendingClub DEV term counts and rates, the Stability peak RSS, the seven LendingClub CatBoost runtimes.
   - Table 11 note: the untuned Home Credit CatBoost full-feature reference has KS 0.4342, above the best selector KS 0.4264; Table 11 ranks selectors only, so nothing is violated, but a reader comparing with the package will see it.
   - `reports/archive/lendingclub_v2_feature_inventory.md` says 19 semantic groups; the manuscript's family count is 17 (the cleanup script's vocabulary).
   - Expected manuscript values that the prompt's Appendix A transcribed correctly: every value checked against the `.tex` agreed; no transcription error was found (the prompt's "240 rows" for the subsets file was the only slip).

## Prompt 1 — done

**Done**
- Branch `freeze/submission-2026-10` from `main` (`ddbb067`); 12 commits.
- `results/stability_metrics_output/`: 25 files repaired and extended (row counts in Section 4); two files folded and removed.
- `results/homecredit_metrics_output/` and `results/lendingclub_metrics_output/`: 25 files each, filled (row counts in Section 4).
- `results/headline_subsets.csv` (360 rows); `results/llm_rankings/` (81 files, index of 90 calls, README); `results/README.md`.
- `scripts/manuscript_oracle.py`, `scripts/verify_manuscript_tables.py` (`--render`, `--check`, `--schema`), `tests/test_manuscript_tables.py`.
- `.gitignore` and `.gitattributes` rewritten; renormalised.
- `docs/freeze_evidence_index.csv` (198 rows); this report.
- Evidence archive at `../Feature-Selection-Research_archive/paper_evidence_2026-10/` (1,087 files, 30.7 MB, manifest).

**Not done and why**
- 97 manuscript values have no local source and stay empty (Section 7): voting P = 100/300 rows and all voting intervals and Holm p (payload absent); LendingClub hybrid Nogueira / Jaccard (no fold lists); LendingClub CatBoost runtimes and the selection-stage minutes; peak RSS ranges; LendingClub DEV term counts and rates, 93 source columns, composition counts; Home Credit candidate count 373; Lift@10 for Home Credit and LendingClub.
- Seven values are flagged rather than reproduced (Section 7).
- The Stability 2024 named ranking and the raw ten repeat responses are not on this machine (index rows without files).
- Nothing was deleted; the deletion needs the go-ahead below.

**Findings for the user** (Section 10 in full)
1. mRMR is the discretised plug-in estimator, not KSG/Ross; the Home Credit LR mRMR cell mixes a discretised-MI run (folds, KS, Brier, subset; HO 0.7351) with a legacy RF/correlation prediction file (HO 0.7699); LLM then mRMR ran the RF/correlation stage.
2. The LLM calls carried 365-391 (Home Credit) and 154-161 (LendingClub) candidate records after a target-aware IV prefilter, not 373 / 675; response counts and the EXT_SOURCE claim are also contradicted by the cache files.
3. Cached rankings now tracked except the Stability named ranking and the raw repeats.
4. `headline_subsets.csv` exists and agrees with every source.
5. No superseded-cohort value entered `results/`; the Stability folder is unsupported by local run outputs and several of its values coincide with superseded-cohort or local Gate-4 numbers.
6. The IV interval was not applied by the frozen IV/WOE and IV-then-Boruta runs; the frozen L1 runs were unweighted and are not the manuscript's L1 values.
7. Stability token usage row says exactly "not recorded".
8. The article's Table 8, Table 10 (HC/LC), native-categorical and obfuscation fold values differ from the only local computed artefacts by constant offsets or swapped labels; the author must say which run produced the printed values.

**Questions awaiting your answer**
- Deletion (Section 9): move the listed git-ignored trees to `../_TO_DELETE_2026-10-09/` (reversible; recommended) or delete in place? Nothing is deleted until you answer.
- Should `results/llm_rankings/` keep the `namedrecall` and `lineage` files (real cached calls, but not the headline rankings; the index says what they reproduce), or hold only the 30 headline calls?
- The findings in Section 10 items 1, 2, 5 and 8 change sentences of the submitted manuscript; you decide what to change there.

**Next**
Prompt 2 (README rewrite) once the deletion question is answered; it should start from `results/README.md`, `docs/repository_freeze.md` Section 10 and `docs/freeze_evidence_index.csv`.
