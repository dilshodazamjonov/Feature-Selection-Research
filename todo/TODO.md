# What is still owed — `todo/`

Rewritten 2026-09-21 for Gate 4 (the revision after the second-opinion read). **43 of 61 steps done, 18 open.** Nine of the open steps need a file from you; they are the nine CSVs in this folder. The other nine open steps need no CSV: 42, 46, 48, 51, 52 are omitted by your decision; 49, 50, 53 are mine and wait on 43 and 44; 24 is the final build and is strictly last. Step numbers are the frozen numbers of `progress.md`; a step closes there, not here.

Every CSV is a skeleton. The dimension columns (dataset, backbone, selector, K, partition, arm, subset) are filled in from Tables 4 and 5 and `priority_1/full_matrix.csv`, and where a reference value already exists in the paper it is filled in too (`ho_auc_cached`, `ho_auc_onehot`, `ho_auc_named`) so you can check the join before running anything. The empty columns are what is owed. Comma-delimited. Dataset and selector strings match `full_matrix.csv` exactly. Lists of feature names go in one cell, names separated by `|`, so that no second file is needed.

Fill the file in place. Do not rename it and do not add files; when a file is complete, tell me and I move it to `priority_2/` and close the step in `progress.md`.

Every bootstrap below is the routine that produced `priority_1/pairwise.csv`: paired on the observation identifier, 2,000 target-stratified resamples, seed 20260721. Every refit is the frozen full-DEV pipeline of the paper: same preprocessing maps, same backbone configurations, seed 42, scored once on the HO population.

| File | Step | Rows | Cost to you | What is owed | Closes the step alone? |
|---|---|---|---|---|---|
| `38_conservative_interval.csv` | 38 | 1 | 5 minutes | One bootstrap row | Yes |
| `39a_onehot_dimension.csv` | 39a | 12 | 10 minutes | Two counts per headline subset | Yes, for 39a |
| `40_perfold_ho.csv` | 40 | 20 | 20 refits, no API | One HO AUC per fold ranking | Yes |
| `37_lc_maturity.csv` | 37 | 60 | Row filters on frozen predictions plus bootstrap | AUCs and intervals on LendingClub sub-populations | Yes |
| `39b_native_catboost.csv` | 39b | 6 | 6 CatBoost refits plus folds | HO AUC with native categoricals | Yes, for 39b |
| `44_mechanical.csv` | 44 | 7 | 31 calls plus 7 refits | Mechanical-description arm | Yes; the description template itself is described in the note below |
| `47_obfuscation.csv` | 47 | 8 | 26 calls plus 8 refits | Obfuscation arms on LendingClub and Stability 2024 | Yes |
| `43_diversity_capped.csv` | 43 | 8 | One new selector, 8 cells with folds | Source-family-capped classical selector | Yes |
| `45_repeated_calls.csv` | 45 | 40 | 30 calls plus 40 refits | Ten repeats on Home Credit and LendingClub | Yes; run only if 40 shows a sign change, otherwise optional |

Suggested order: 38, 39a, 40 first (no API, an afternoon). Then 44 on LendingClub and 37 in parallel. Then 47 and 43. Then 39b, and 45 only if 40 wobbles.

---

## 38. `38_conservative_interval.csv` — one bootstrap row

The third leader rule in Section 6.1 pairs LLM then mRMR with mRMR on Home Credit logistic regression and prints the difference as a point estimate with "interval not computed". This row fills it.

- Inputs: the frozen HO prediction files of `hybrid_llm_mrmr` and `statistical_mrmr`, Home Credit, logistic regression, K = 20, all 120,053 HO rows paired on `SK_ID_CURR`.
- Check before sending: `auc_A` must reproduce 0.74322934 and `auc_B` 0.76989 from `full_matrix.csv`. If either differs, the wrong prediction file was picked.
- Expected: delta about −0.027, interval about ±0.001 wide, excluding zero. Changes nothing else in the paper.

## 39a. `39a_onehot_dimension.csv` — effective dimension of the twelve headline subsets

For each headline subset, after the full-DEV preprocessing map of its backbone: how many of its K original features are categorical, and how many columns the model actually receives after one-hot expansion (rare levels dropped at the paper's thresholds, missing token included).

- Goes into a new small table beside K in Section 6.1, and answers the referee question "40 features or 400 columns?".
- No branch: whatever the counts are, they are reported.

## 40. `40_perfold_ho.csv` — the five per-fold rankings scored to HO

You already hold five cached pure LLM rankings per dataset on Home Credit and LendingClub (fold1 to fold5), over the identical candidate list. Take each fold's ranking, keep the top 20 (LR) or top 40 (CatBoost), refit on full DEV with the frozen pipeline, score on HO. Twenty refits, no API call.

- This gives a five-draw spread of HO AUC per case at zero API cost, the same quantity the ten repeats measured on Stability 2024.
- Read as: sign of the margin against the classical leader of Table 4 in every draw, and spread against the 0.010 bar. If the sign holds in all five draws per case, 45 becomes optional.

## 37. `37_lc_maturity.csv` — LendingClub margins on the loans that could have matured

The LendingClub holdout keeps only loans resolved by 2019-05-01, and 57.7% of them could not have reached maturity by then (`priority_2/lc_maturity_by_cohort.csv`). This file shows whether the +0.030 and +0.051 margins hold where that filter cannot act.

- Row filters on the frozen HO prediction vectors, no refit: (a) `matured_36m_2016-01_to_2016-05`, the 123,898 36-month loans issued 2016-01 to 2016-05, all of which could have matured; (b) each issue month 2016-01 to 2016-12; (c) each term, 36 and 60.
- Four pairs per filter: the two leader pairs of Table 4 (`ho` rule) and the two of Table 5 (`dev` rule). Bootstrap each on the filtered rows; `n_ho` is the row count after the filter.
- Branches: margins hold on (a), one Results paragraph and a new table by month and term, and the Section 6.1 resolution argument becomes a measured statement; margins shrink below 0.010 on (a), the LendingClub margins are rescoped to the resolved population in the abstract and Section 7.1, and the table is still reported.

## 39b. `39b_native_catboost.csv` — CatBoost with native categorical handling

The paper one-hot encodes categoricals for both backbones (Section 4.2). Re-run the three CatBoost headline cases with CatBoost's own `cat_features` instead, everything else frozen: same 40 features, same configuration, same seed. Both leaders of each case, so six refits, plus the five fold values.

- `ho_auc_onehot` is prefilled from `full_matrix.csv` for the comparison.
- Branches: the LLM-assisted leader stays above the classical leader in all three cases, one sentence in Section 6.1 and the table; the ordering flips in any case, reported as a limitation of the one-hot design with the case named. Either way Section 4.2 gains the justification for one-hot under CatBoost (the budget counts original variables; one preprocessing map serves both backbones).

## 44. `44_mechanical.csv` — mechanical-description arm

The question: was the LLM's advantage carried by the descriptions you wrote, or by the lineage alone? Rebuild the description field of every candidate record from lineage only, by code, and re-run the ranking.

- Template, one per dataset, applied to every feature the same way, nothing written by hand. Home Credit: `table <source table>; raw variable <name>; operation <aggregation or none>; type <numeric|categorical|flag>`. LendingClub: `<feature type: raw|ratio|flag|bin|interaction|missing indicator>; source columns <a>, <b>; type <numeric|categorical>`. Stability 2024: `table <name>; depth <0|1>; raw variable <name>; operation <count|mean|min|max|std|last|mode|nunique>; window <from the raw name, or none>; type <rule>`.
- Keep the feature name, source family, lineage formula and semantic group fields exactly as they are. Only the description changes.
- Calls exactly as the cached run: snapshot `gpt-4.1-mini-2025-04-14`, temperature 0, the Appendix A template, `definitions_only`, same candidate sets (373, 675, 1,068), same strict validation with up to three retries. Six calls of 100 on Home Credit, 24 on LendingClub, one on Stability 2024.
- Refit the seven pipelines listed in the file (pure LLM in every case; LLM then mRMR on LendingClub CatBoost, where it is the leader), score HO, bootstrap mechanical against cached on the same HO rows. `ho_auc_cached` is prefilled.
- Put the full-DEV top-100 names in the last column so the overlap with the cached ranking can be computed here.
- Run LendingClub first: it is the only dataset whose base definitions are your own text, and it carries the largest margins. Home Credit's base text is Kaggle's dictionary and Stability's aggregation clause is already mechanical, so the drop there, if any, should be small.
- Branches, decided now: mechanical within 0.010 of cached in all seven rows, the descriptions did not carry the result and the provenance paragraph of Section 4.6 becomes background; mechanical falls by more than 0.010 in any row, the hand-written meaning did real work there, the paper says so and reframes that case as expert metadata routed through a language model.
- Also keep for the release archive: the mechanical description files and the cached responses of the new calls. They are not CSVs owed here.

## 47. `47_obfuscation.csv` — name and definition obfuscation on LendingClub and Stability 2024

Section 6.4 ran this on Home Credit only. Repeat it where the margins are largest.

- Arm A: every feature name replaced by a seeded identifier (`F001` …), definitions kept; every literal occurrence of the original name scrubbed from inside the definition text too. Arm B: names kept, definitions removed. Shuffle seed recorded, permutation drawn per partition, as in the Home Credit run (seed 20260914 there).
- LendingClub at the two reported budgets only (20 and 40), full DEV plus five folds, both arms: 24 calls. Stability 2024: one full-DEV call per arm, 2 calls; its fold columns stay empty.
- Refit pure LLM at K = 20 (LR) and K = 40 (CatBoost), score HO. `ho_auc_named` is prefilled. `overlap_with_named_fulldev` is the number of names the arm's full-DEV top-K shares with the named subset of Appendix E; put the arm's top-K names in the last column so I can verify it.
- Branches as for Home Credit: AUC within 0.010 of named, the Section 7.1 paragraph generalises to three datasets; AUC drops by more than 0.010 on either dataset, the name-recognition concession returns for that dataset in Limitations and the abstract range is rescoped.

## 43. `43_diversity_capped.csv` — a classical selector with a cap per source family

Appendix E shows that on Stability 2024 the pure LLM subsets come entirely from the `static_0` table while the classical leaders spread across six or seven tables, and on Home Credit CatBoost 26 of the LLM's 40 come from the bureau tables. A referee will ask whether any rule that concentrates on the same tables would do as well. This selector answers it.

- `family_cap`: take the HO-rule classical leader of the case (Table 4) and re-run it with a cap of at most ⌈K / number of families⌉ features per source family, filling by the selector's own ranking; write the selector you used in `base_selector`. Six cases.
- `depth0_only`: on Stability 2024 only, the same classical leader restricted to the depth-0 candidates (`static_0` and `static_cb_0`). Two cases. This is the direct test of the LLM's choice.
- Frozen backbones, fold-local plus full DEV, HO scored once.
- Branches, decided now: the capped or depth-0 selector matches or exceeds the LLM leader on Stability 2024, and the Stability margin is attributed to source concentration in Sections 6.1 and 7.1 with the semantic reading withdrawn there; it falls short by more than 0.010, and the semantic reading stands with the control reported as the test it passed.

## 45. `45_repeated_calls.csv` — ten repeats on Home Credit and LendingClub

Only if `40_perfold_ho.csv` shows a sign change in any case, or if you prefer to have it regardless. Exactly the Stability 2024 procedure of Section 6.3: the full-DEV ranking call repeated ten times, temperature 0, same snapshot, template and candidate records; Home Credit one call of 100 per repeat (budgets are prefixes), LendingClub one call per budget (20 and 40) per repeat; about 30 calls. Each response through the frozen pipeline to HO, 40 refits.

- Record the provider response id per call and the top-100 names per repeat so Nogueira and Jaccard can be computed here, as for Stability 2024.
- Read as median, s.d., range, and whether the sign and the 0.010 bar hold in every repeat, per case. Extends the repeat table to three datasets.
- Branches: sign holds in every repeat, Section 6.3 generalises the Stability paragraph; sign reverses in any repeat on a positive case, that case is reported as sign-unstable in the abstract range and Section 7.1.

---

## Open steps that need no file

- **42, 46, 48, 51, 52** — omitted by your decision of 2026-09-21 (matched timing, sham prompt, prompt paraphrases, second language model, backbone sensitivity). They stay listed in `progress.md` as open and can be revived for a response letter.
- **49, 50** `[Claude]` — framing rewrite and the final Table 2 row; wait on 43 and 44.
- **53** `[Both]` — archive update with every new cached ranking and manifest; the Data availability statement promises the repository holds them.
- **24** `[Claude]` — final build, strictly last.
