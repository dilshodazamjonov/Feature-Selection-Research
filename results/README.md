# results/

The result files of the study, one folder per dataset with the same 25 files and headers, the twelve headline
subsets, and the accepted LLM rankings. Every file is written by `scripts/run_study.py --stage results` from the run
outputs under `outputs/study/<dataset>/`; nothing in these folders is entered by hand.

```
results/
├── README.md
├── headline_subsets.csv               # full-DEV subsets of the two HO-rule family leaders of each case
├── homecredit_metrics_output/         # Home Credit
├── lendingclub_metrics_output/        # LendingClub v2
├── stability_metrics_output/          # Stability 2024
└── llm_rankings/                      # accepted LLM responses + index.csv + README.md
```

Regenerate a dataset's files after its stages have run, then check structure and values:

```bash
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage results
uv run python scripts/verify_manuscript_tables.py --schema    # file set, headers, LF endings, 00_run_ids coverage
uv run python scripts/verify_manuscript_tables.py --check     # compares the files with scripts/manuscript_oracle.py
uv run python scripts/verify_manuscript_tables.py --render    # prints the tables from the files
```

`--check` rounds each value to the precision of its reference value and reports `match`, `mismatch` or `blank`, and it
recomputes the derived ranges and differences from the same files; it exits non-zero on any mismatch.

## Files (in each dataset folder)

| File | Contents | Columns |
|---|---|---|
| `00_run_ids.csv` | for every file: the run id, the stage manifest it was built from and that manifest's SHA-256 | `file, run_id, source_path, source_sha256` |
| `01_cohort.csv` | DEV and HO windows, rows, events, event rates; DEV and HO together; the release size; the five fold partitions; LendingClub term rows | `split, window_*, unique_index_units, n_rows, n_events, event_rate, share_of_rows` |
| `02_candidate_set.csv` | the missingness screen on the fold-1 training rows; features above the IV upper bound; LendingClub missingness bands, source columns and feature composition; the Stability 2024 availability review | `screen, screen_rows, universe, dropped, retained` |
| `03_ho_auc.csv` | HO ROC-AUC of the thirteen strategies and the full-feature reference | `backbone, selector, family, K, ho_auc` |
| `04_dev_fold_auc.csv` | validation AUC of the five folds and their mean | `fold1_auc..fold5_auc, dev_mean_auc` |
| `05_leader_delta_auc.csv` | family leaders under the HO, DEV and mixed rules; paired HO AUC difference with its 95 % target-stratified paired bootstrap interval (2,000 resamples, seed 20260721) | `rule, llm_*, classical_*, delta_auc, ci_low, ci_high, n_ho` |
| `06_leader_feature_psi.csv` | mean, median and maximum type-aware feature PSI of each HO-rule leader's full-DEV subset | `mean_psi, median_psi, max_psi, max_psi_feature` |
| `07_selection_stability.csv` | Nogueira stability over the candidate set and pairwise Jaccard over the ten fold pairs | `K, d, nogueira, jaccard_mean, jaccard_min, jaccard_max` |
| `08_secondary_metrics.csv` | HO KS, Brier score, log loss, Capture@10 and Lift@10 | `ks, brier, log_loss, capture_at_10, lift_at_10` |
| `09_leader_subsets.csv` | the HO-rule leaders' full-DEV subsets in selection order, with source family, categorical flag, model input width, protected-attribute flag and feature PSI | `rank, feature, source_family, is_categorical, n_model_columns, is_protected, feature_psi` |
| `10_boruta_attainment.csv` | confirmed features and attained budget of the three Boruta-based selectors in every fold and the full-DEV refit | `partition, n_confirmed, K, n_selected, below_budget` |
| `11_subpopulation_ho_auc.csv` | HO-rule leader comparison on row filters of the HO scores: LendingClub matured cohort, terms and issue months; Stability 2024 decision months | `subpopulation, n, *_ho_auc, delta_auc, ci_*` |
| `12_native_categorical.csv` | CatBoost leaders refitted with native categorical handling | `ho_auc_onehot, ho_auc_native` |
| `13_lineage_only.csv` | Pure LLM and the LLM-assisted leader with descriptions rebuilt from lineage | `ho_auc_mechanical, ho_auc_cached, delta_auc, ci_*, overlap_with_cached` |
| `14_l1_selector.csv` | L1-penalised logistic-regression selector | `ho_auc, classical_leader_ho_auc, shortfall` |
| `15_diversity_controls.csv` | family-capped classical leader (all datasets) and depth-0 classical leader (Stability 2024) | `control_ho_auc, uncapped_ho_auc, llm_assisted_ho_auc, difference, cap_per_family, n_families` |
| `16_response_spread.csv` | HO AUC of Pure LLM with each other cached response (per-fold rankings; Stability 2024 repetitions) and with the full-DEV ranking | `draw_source, draw_id, ho_auc, is_headline_response` |
| `17_budget_sweep.csv` | mean DEV AUC of every strategy at K = 10, 20, 30, 40, 50, family leaders, pooled mean | `K, selector, family, dev_mean_auc, is_*_leader` |
| `18_response_agreement.csv` | agreement among those responses' top-K lists | `draws, d, nogueira, jaccard_mean, names_in_all_draws, distinct_names` |
| `19_obfuscation.csv`, `19b_obfuscation_ranks.csv`, `19c_obfuscation_fold_jaccard.csv` | name-obfuscation arms (Home Credit): HO and fold AUC, overlaps with the named subsets, every ranking, fold-pair Jaccard | arm AUCs, overlaps; `arm, call, feature, rank`; `arm, K, fold_i, fold_j, jaccard` |
| `20_rank_voting.csv` | classical rank-voting family (Home Credit, LendingClub): pools P = 100, 200, 300 against the RF/correlation reference; paired bootstrap interval; two-sided paired DeLong p-value with Holm adjustment within the three pools | `pool_P, reference_auc, voting_auc, delta_auc, ci_*, holm_p, delta_ks, *_fold_jaccard_mean` |
| `21_llm_usage.csv` | every named ranking call: budget, uses, tokens, snapshot | `call_id, partition, budget, logical_requests_served, input_tokens, output_tokens, model_snapshot` |
| `22_runtime_and_memory.csv` | wall-clock time and peak resident memory of every pipeline; the matched mRMR timing (Stability 2024 fold 1); peak-memory range | `pipeline, stage, wall_clock_minutes, seconds, peak_rss_gb, matched_conditions` |

A file that does not apply to a dataset holds its header only.

## Schema rules

- CSV, UTF-8, LF line endings, one header row; the same 25 files with the same headers in every dataset folder.
- Selector names: `Domain rules`, `Random K`, `PCA`, `IV/WOE`, `mRMR`, `Boruta RF`, `IV then Boruta`, `RFE CatBoost`,
  `CatBoost SHAP` (family `classical`); `Pure LLM`, `LLM then mRMR`, `LLM then Boruta`, `Stable core + LLM fill`
  (family `LLM-assisted`); `Full features` (family `reference`). `backbone` is `lr` or `catboost`.
- Values are written at full precision; booleans are `TRUE` / `FALSE`; an empty cell means the quantity was not
  produced (the `note` column says why).
- `00_run_ids.csv` points to `outputs/study/<dataset>/manifests/<stage>.json`, which lists every run output of the stage
  with its SHA-256. The manifests and run outputs are local to the machine that ran the study.
