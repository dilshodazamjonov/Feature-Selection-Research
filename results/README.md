# results/

Only what the manuscript reports is tracked here. The three `*_metrics_output/` folders hold identical file sets
and column headers (one folder per dataset), `headline_subsets.csv` is the file Appendix D promises, and
`llm_rankings/` holds the cached LLM responses the Data availability statement promises. Everything else that
once lived under `results/` (run trees, packages, synthesis folders) is git-ignored local working state.

```
results/
├── README.md
├── headline_subsets.csv               # Appendix D / supplementary Tables S1-S6: the twelve headline subsets (360 rows)
├── homecredit_metrics_output/         # Home Credit
├── lendingclub_metrics_output/        # LendingClub v2
├── stability_metrics_output/          # Stability 2024 (cohort 2020-02-26)
└── llm_rankings/                      # cached LLM responses + index.csv + README.md
```

Render the manuscript tables from these CSVs and check every value against the manuscript:

```bash
uv run python scripts/verify_manuscript_tables.py --render
uv run python scripts/verify_manuscript_tables.py --check     # exit 1 on any mismatch
uv run python scripts/verify_manuscript_tables.py --schema    # structural checks
uv run pytest tests/test_manuscript_tables.py -q
```

`scripts/manuscript_oracle.py` holds the manuscript's printed values (transcribed from the submission `.tex`);
the checker rounds CSV values to the manuscript's precision before comparing. Values with no local source are
left empty and reported as "no CSV source" (107 at the freeze; listed in `docs/repository_freeze.md`). Sixteen values
are flagged, not failed (`KNOWN_DISCREPANCIES` in the oracle): five Nogueira values that the local artefacts give one
unit away in the fourth decimal, the two EXT_SOURCE ranks of the Table 7 note, which the cached full-DEV ranking
orders the other way round, and nine Appendix C values (the pooled CatBoost change at K = 50 and the eight pooled
coordinates of Figure C.1a) that reproduce only with a Stability 2024 budget sweep that predates what the
manuscript's source calls the Stability 2024 correction. Besides the tables, `--check` recomputes the prose ranges
and differences of Sections 6.1 to 6.6 and Appendices C and F from the same CSVs.

## Manuscript element -> file -> column

| Manuscript element | File (in each dataset folder) | Columns |
|---|---|---|
| Table 2 (datasets), Section 5.1-5.2 counts | `01_cohort.csv`, `02_candidate_set.csv` | `n_rows, n_events, event_rate, window_*`; `universe, dropped, retained` |
| Table 5 (HO AUC of every selector), "Full" column of Table 3 | `03_ho_auc.csv` | `ho_auc` |
| Table 4 DEV means, DEV-rule leaders, Table C.1 at the documented budget | `04_dev_fold_auc.csv` | `fold1_auc..fold5_auc, dev_mean_auc` |
| Tables 3 and 4 (leaders, delta AUC, 95% CI), mixed rule | `05_leader_delta_auc.csv` | `rule, llm_*, classical_*, delta_auc, ci_low, ci_high, n_ho` |
| Table 3 PSI columns, Section 6.6.2 | `06_leader_feature_psi.csv` | `mean_psi, median_psi, max_psi, max_psi_feature` |
| Table 9, Figure 4 | `07_selection_stability.csv` | `d, nogueira, jaccard_mean, jaccard_min, jaccard_max` |
| Table 11 | `08_secondary_metrics.csv` | `ks, brier, log_loss, capture_at_10, lift_at_10` |
| Appendix D, Table F.3, Section 7.6 | `09_leader_subsets.csv` (and `headline_subsets.csv`) | `rank, feature, source_family, is_categorical, n_model_columns, is_protected` |
| Appendix F "Boruta budget attainment" | `10_boruta_attainment.csv` | `n_confirmed, n_selected, below_budget` |
| Table F.1 and the twelve issue months (LendingClub); the nine months (Stability) | `11_subpopulation_ho_auc.csv` | `subpopulation, backbone, n, *_ho_auc, delta_auc, ci_*` |
| Appendix F "Native categorical handling" | `12_native_categorical.csv` | `ho_auc_onehot, ho_auc_native` |
| Table 8 | `13_lineage_only.csv` | `ho_auc_mechanical, ho_auc_cached, delta_auc, ci_*, overlap_with_cached` |
| Appendix F "L1-penalised selector" | `14_l1_selector.csv` | `ho_auc, classical_leader_ho_auc, shortfall` |
| Table F.2, Table 1 family caps | `15_diversity_controls.csv` | `control_ho_auc, uncapped_ho_auc, llm_assisted_ho_auc, difference, cap_per_family, n_families` |
| Table 10 | `16_response_spread.csv` | `draw_source, draw_id, ho_auc, is_headline_response` |
| Table C.1, Figure C.1 | `17_budget_sweep.csv` | `K, selector, family, dev_mean_auc, is_*_leader` |
| Table F.4 | `18_response_agreement.csv` | `draws, d, nogueira, jaccard_mean, names_in_all_draws, distinct_names` |
| Table 7 and its note (Home Credit only) | `19_obfuscation.csv`, `19b_obfuscation_ranks.csv`, `19c_obfuscation_fold_jaccard.csv` | arm AUCs, fold AUCs, overlaps; EXT_SOURCE ranks; fold-pair Jaccard |
| Table B.1, Section 7.2 (Home Credit, LendingClub) | `20_rank_voting.csv` | `pool_P, reference_auc, voting_auc, delta_auc, ci_*, holm_p, delta_ks, *_fold_jaccard_mean` |
| Section 6.7 token usage | `21_llm_usage.csv` | `call_id, input_tokens, output_tokens, model_snapshot` |
| Section 6.7 and Appendix F runtime, memory, machine | `22_runtime_and_memory.csv` | `pipeline, stage, wall_clock_minutes, seconds, peak_rss_gb, matched_conditions` |

## Uniform-schema rules

- CSV only: UTF-8, LF line endings, header row, no trailing empty rows (enforced by `.gitattributes` and `--schema`).
- The same 25 files with the same headers in every dataset folder. A file that does not apply to a dataset holds
  the header and a `note` row saying why.
- Selector names are the manuscript's Table 1 names (`Full features` is the reference); `family` is `classical`,
  `LLM-assisted` or `reference`; `backbone` is `lr` or `catboost`; dataset folders are `homecredit`,
  `lendingclub_v2`, `stability_2024`.
- Values are stored at the precision of the source artefact; the manuscript rounds. Deltas and intervals carry the
  source precision.
- Every file has a `note` column naming the source of each row and stating when a value is derived (means,
  differences, rescaled Nogueira) or absent ("no local artefact").

## Provenance convention

`00_run_ids.csv` in each folder lists, for every file, the source artefact(s) it was copied from with SHA-256.
Paths are written `REPO:<path>` (this repository, usually git-ignored working state) or `ARTICLE:<path>` (the
manuscript repository's result files). **Those paths are local and are not in a fresh clone**; the tracked CSVs
are the published record. For Stability 2024 the sources are the CSVs committed by the author on 2026-10-08
(commit b778999) from a rerun on the 2020-02-26 cohort whose run outputs are not on the freeze machine; the run
ids recorded there match no local file. `docs/repository_freeze.md` records every gap and discrepancy.
