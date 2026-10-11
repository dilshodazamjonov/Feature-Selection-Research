# Row-Independent Feature Pre-Screening with Language Models for High-Dimensional Credit Scoring

### A Frozen Evaluation on Temporal and Recency-Shift Holdouts

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)

**Dilshod A'zamjonov**<sup>1,3,*</sup> and **Sejuti Rahman**<sup>2</sup>

<sup>1</sup> Department of Engineering, New Uzbekistan University, Tashkent, Uzbekistan ·
<sup>2</sup> Department of Computing, New Uzbekistan University, Tashkent, Uzbekistan ·
<sup>3</sup> Central Bank of the Republic of Uzbekistan, Tashkent, Uzbekistan ·
<sup>*</sup> corresponding author, d.azamjonov@newuu.uz

This repository runs a frozen comparison of classical feature selection and LLM-assisted pre-screening on three
public credit portfolios. A language model reads only the variable book (feature names, definitions and lineage, never
a data row) and ranks the screened candidate set; the ranking is used directly or as a pool for a statistical selector.
Thirteen strategies and a full-feature reference are fitted with logistic regression (K = 20 original features) and
CatBoost (K = 40) on five expanding-window development folds, refitted on full development data (DEV) and scored once
on a held-out population (HO) later in time or shifted in applicant recency. One command per dataset runs the whole
protocol and writes every result file.

---

## Contents

1. [Protocol](#1-protocol)
2. [Datasets and the DEV/HO design](#2-datasets-and-the-devho-design)
3. [Feature-selection strategies](#3-feature-selection-strategies)
4. [LLM pre-screening](#4-llm-pre-screening)
5. [Backbones, budgets and preprocessing](#5-backbones-budgets-and-preprocessing)
6. [Metrics and inference](#6-metrics-and-inference)
7. [Robustness checks](#7-robustness-checks)
8. [Leakage and reproducibility controls](#8-leakage-and-reproducibility-controls)
9. [Running the study](#9-running-the-study)
10. [Result files](#10-result-files)
11. [Repository layout](#11-repository-layout)
12. [Superseded entry points](#12-superseded-entry-points)
13. [Citation, licence and declarations](#citation)

---

## 1. Protocol

The same steps run for every dataset (`scripts/run_study.py`, package `src/credit_risk_fs/study/`):

1. **Assembly and outcome.** The engineered universe is assembled: 529 predictors for Home Credit, 675 for LendingClub
   v2 and 1,959 for Stability 2024 (generated from the 434 raw predictors approved by the prediction-time availability
   review of the Stability 2024 protocol lock).
2. **Split and folds.** Each dataset is split along its ordering index into DEV and HO, and DEV into five
   expanding-window folds (`GroupedTimeSeriesSplit`, five splits, one index unit assigned to neither partition). DEV
   rows are ordered by the index, ties broken by the observation identifier.
3. **Candidate set.** Every selector chooses from the same candidate set: the features with at most 90 % missing values
   (missing or infinite) on the training partition of the first fold. The screen uses no outcome, is computed once and is
   held fixed across the folds. On Home Credit, functions of the ordering index (previous-application decision days and
   their aggregates) are outside the candidate set.
4. **Selection.** Each strategy of [Section 3](#3-feature-selection-strategies) produces a subset of K original
   features from the training partition of each fold and of full DEV; target-aware selectors see only those rows.
5. **Fitting and validation.** The fixed backbones of [Section 5](#5-backbones-budgets-and-preprocessing) are fitted on
   each subset; the fold validation partitions give the development AUC and the five fold subsets.
6. **Freeze and HO scoring.** Everything is refitted on full DEV; rankings, subsets, preprocessing maps and settings are
   written before the full-DEV model scores HO, once.

The LLM ranking, domain rules, Random K, principal components and the missingness screen are target-free. Information
value, mRMR, Boruta, recursive elimination, SHAP ranking, the stable-core construction and model fitting are
target-aware and confined to DEV training partitions.

## 2. Datasets and the DEV/HO design

| | Home Credit | LendingClub v2 | Stability 2024 |
|---|---|---|---|
| Source | Home Credit Default Risk (Kaggle) | All Lending Club loan data (Kaggle) | Home Credit – Credit Risk Model Stability (Kaggle) |
| Unit and event | loan application; payment difficulties (provider target) | 36- or 60-month loan resolved by the 2019-05-01 snapshot; charged off or defaulted | credit decision; default (provider target) |
| Ordering index | `DAYS_DECISION` of the most recent previous application (recency) | days from the issue month to 2018-12-01 (calendar time) | `date_decision` (calendar time) |
| DEV | day −600 to −241 | day −1795 to −1096 (issued 2014-01 to 2015-12) | 2019-01-01 to 2020-02-25 |
| HO | day −240 to −1 | day −1065 to −730 (issued 2016-01 to 2016-12) | 2020-02-26 to 2020-10-05 |
| Universe | 529 original features (application record and auxiliary tables) | 675 engineered features from the loan book | 1,959 engineered features (depth-0 tables `static_0`, `static_cb_0`; ten depth-1 families) |
| Paired-observation identifier | `SK_ID_CURR` | original loan id (`record_identity_v1` sidecar) | `case_id` |
| Study configuration | `configs/study/homecredit.yaml` | `configs/study/lendingclub_v2.yaml` | `configs/study/stability_2024.yaml` |

The Home Credit and LendingClub loaders (`src/credit_risk_fs/pipelines/common.py`) check the DEV and HO row counts
(Home Credit 99,092 and 120,053; LendingClub v2 598,649 and 293,105) and the ordered row-identity hashes in
`configs/protocols/row_alignment_contract_v1.json`, and refuse to load data that differ. LendingClub v2 excludes every
post-origination field and the platform's pricing and grading fields (`int_rate`, `grade`, `sub_grade`, `installment`,
`funded_amnt`). Stability 2024 is read from the matrix built by `scripts/build_homecredit_model_stability_2024.py`
under the protocol lock `configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json`; every
generated variable keeps its lineage (raw table, raw variable, depth, operation, type rule).

Construction code: `src/credit_risk_fs/feature_engineering/homecredit/` (Home Credit),
`src/credit_risk_fs/feature_engineering/lendingclub/` with `scripts/prepare_lendingclub_v2.py` (LendingClub v2; feature
inventory `data/lendingclub_v2/metadata/feature_inventory.csv`), `src/credit_risk_fs/data/homecredit_model_stability_2024/`
(Stability 2024).

## 3. Feature-selection strategies

Each selector ranks the candidates by its own criterion and keeps the top K original features
(`src/credit_risk_fs/study/selection.py`).

| Method | Family | Selection role | Labels | Implementation |
|---|---|---|---|---|
| Full features | Reference | no selection; the candidate set (Home Credit, LendingClub v2) or the unscreened universe (Stability 2024) | – | `study/modeling.py` (sparse preprocessing map) |
| Domain rules | Classical | deterministic credit-domain scorer written for the Home Credit variable book, applied unchanged elsewhere | no | `selectors/domain_rules.py` |
| Random K | Classical | uniformly random subset of size K (seed 42) | no | `selectors/lightweight/controls.py` |
| PCA | Classical | K leading components of the preprocessed candidates (components, not original features) | no | `study/modeling.py` |
| IV/WOE | Classical | ranking by information value over ten training-partition quantile bins; eligible when 0.01 ≤ IV ≤ 0.50 | yes | `selectors/lightweight/iv.py` |
| mRMR | Classical | mutual-information relevance minus mean mutual-information redundancy, greedy | yes | `selectors/lightweight/mi_mrmr.py` |
| Boruta RF | Classical | features whose random-forest importance (500 trees of depth 6) beats the best shadow at α = 0.05 with two-step correction; top K confirmed, never padded | yes | `selectors/heavy/boruta_rf.py` |
| IV then Boruta | Classical | IV eligibility interval, then Boruta on the eligible features | yes | `study/selection.py` |
| RFE CatBoost | Classical | removes 20 % of the surviving features per step by the importance of a CatBoost ranker (500 iterations, depth 6, learning rate 0.05) until K remain | yes | `selectors/heavy/rfe_catboost.py` |
| CatBoost SHAP | Classical | largest mean absolute native SHAP value over a 10,000-row training sample | yes | `selectors/heavy/catboost_shap.py` |
| Pure LLM | LLM-assisted | top K names of the cached semantic ranking of the partition | no | `study/llm.py` |
| LLM then mRMR | LLM-assisted | semantic pool of 60 (LR) or 100 (CatBoost) names, then mRMR | yes | `study/selection.py` |
| LLM then Boruta | LLM-assisted | the same pool, then Boruta, truncated to K | yes | `study/selection.py` |
| Stable core + LLM fill | LLM-assisted | RF/correlation filter on five bootstrap resamples (80 % with replacement, seeds 42 to 46); features chosen in at least four form the core, ordered by frequency and mean rank; the LLM pool ranking fills the rest, without padding | yes | `selectors/stable_core_llm_fill.py` |

**Details.** *mRMR* estimates mutual information on a target-stratified sample of 20,000 training rows (seed 42) with
nearest-neighbour estimators: Ross for a continuous feature against a discrete variable, Kraskov et al. for two
continuous features (three neighbours); two discrete variables use the plug-in estimator. Its greedy order is computed to
50 features, so every budget is a prefix. *Boruta* runs `BorutaPy` with 500 trees and a maximum of 10 iterations
(`configs/study/common.yaml`). The *RF/correlation filter* (`selectors/mrmr.py`, random-forest relevance over mean
absolute correlation) serves the stable-core construction and the voting family only. *Domain rules*: 42 named-variable
rules, family and string-pattern rules for bureau, instalment, credit-card and previous-application aggregates, 15
property-attribute rules and nine semantic-group fallbacks give each feature a priority, adjusted for the aggregation
suffix; features its leakage rules flag are ranked last; ties fall to the name. Selection operates on original
variables: the raw frame serves IV, mRMR, the LLM stages, domain rules and Random K; a one-column-per-feature encoding
(median-imputed numerics, ordinal-coded categoricals) serves Boruta, RFE, SHAP, the RF/correlation filter and the L1
selector. On LendingClub v2 the statistical stage of LLM then mRMR selects one-hot columns of the semantic pool, so its
budget counts columns.

**Diversity controls** (not strategies; they enter no family or leader choice): the HO-rule classical leader refitted
with at most ⌈K/F⌉ features per source family, F being the number of source families of the universe (6, 17 and 12),
filled in the base selector's own ranking order (mRMR applies the cap inside its greedy search; Boruta-based selectors
fill from confirmed features only); and, on Stability 2024, the same leader restricted to the depth-0 candidates.

**Classical rank-voting family** (Home Credit and LendingClub v2): on each training partition the normalised ranks
(r − 1)/(d − 1) of three classical rankings over the candidate set (the RF/correlation filter, Boruta, and recursive
elimination to one feature) are averaged; the top P ∈ {100, 200, 300} form the pool (P = 200 primary), which CatBoost
RFE reduces to K. The reference is the RF/correlation filter at K on the candidate set. Two-sided paired DeLong tests are
Holm-adjusted within each case's three pools.

## 4. LLM pre-screening

GPT-4.1 mini (snapshot `gpt-4.1-mini-2025-04-14`), Chat Completions, temperature 0, strict JSON-schema response. One
prompt template serves all three datasets under a definition-only information contract: the records carry each
candidate's name, description, source family, lineage formula and semantic group, and no row, outcome or statistic. A
response is accepted only with exactly the requested number of distinct names copied from the records; a failing
response is re-requested up to three times; there is no fallback and no padding.

| Dataset | Ranking calls |
|---|---|
| Home Credit | one call of 100 names for each of the five fold training partitions and full DEV; prefixes give every budget |
| LendingClub v2 | one call per partition and budget (20, 40, 60, 100): 24 calls |
| Stability 2024 | one full-DEV call of 100 names serving every fold, plus ten repetitions of it |

Ablation calls: name obfuscation on Home Credit (arm A: names replaced by identifiers `F001` to `F529` from a permutation
seeded with 20260914, drawn per partition; arm B: descriptions blanked), and descriptions rebuilt from lineage in all
three datasets. The template, the record format per dataset and the call plan are in
[`docs/llm_prompt_protocol.md`](docs/llm_prompt_protocol.md).

## 5. Backbones, budgets and preprocessing

| Backbone | Budget | Configuration (identical across selectors; `configs/study/common.yaml`) |
|---|---|---|
| Logistic regression | K = 20 | scikit-learn `liblinear`, L2, C = 1, at most 1,000 iterations, balanced class weights, seed 42 |
| CatBoost | K = 40 | depth 10, learning rate 0.01, L2 leaf regularisation 95, at least 290 observations per leaf, `rsm` 0.90, random strength 0.125, depthwise growth, Bernoulli subsample 0.55, Logloss with AUC evaluation, balanced class weights, seed 42, four threads, a fixed 1,500 iterations without early stopping |

The same configurations serve every feature count, including the full-feature references. The budget counts original
variables; the one-hot expansion follows selection.

Preprocessing (`src/credit_risk_fs/preprocessing/encoding.py`, `study/modeling.py`), fitted on each training partition
and on full DEV: numeric infinities set to missing, mean imputation, standardisation; categoricals receive an explicit
missing token and are one-hot encoded; levels unseen in training are ignored; levels rarer than 10 observations (Home
Credit, Stability 2024) or 50 (LendingClub v2) are dropped. CatBoost receives the same one-hot expansion; native
categorical handling is a robustness check.

## 6. Metrics and inference

- **Discrimination:** HO ROC-AUC (headline). **Secondary:** KS (maximum separation), Brier score, log loss, Capture@10
  and Lift@10 over the top ⌈10 %⌉ scores, ties broken by identifier.
- **Drift:** feature PSI over type-aware states learned on DEV and applied to HO (DEV quantile bins plus a missing state
  for continuous variables; 0/1/missing/unexpected for binary variables; exact DEV levels for integer variables with at
  most 20 levels; DEV categories plus missing and unseen states for categoricals); mean, median and maximum over a
  subset.
- **Reproducibility:** Nogueira stability over the candidate set and mean, minimum and maximum pairwise Jaccard over the
  ten fold pairs.
- **Leader rules:** *HO rule*, the selector of each family with the highest HO AUC; *DEV rule*, the highest mean
  validation AUC over the five folds; *mixed rule*, the DEV-rule LLM-assisted leader against the HO-rule classical leader.
- **Paired comparisons:** HO AUC difference, LLM-assisted minus classical, on identical observations paired by
  identifier, with a 95 % interval from 2,000 target-stratified paired bootstrap resamples (seed 20260721). A difference
  of at least 0.010 AUC is treated as material. The two-sided paired DeLong test with Holm's adjustment is used only for
  the voting family.

Code: `src/credit_risk_fs/study/metrics.py` (bootstrap, PSI, KS, Capture/Lift, stability) and
`src/credit_risk_fs/evaluation/paired_inference.py` (DeLong, Holm).

## 7. Robustness checks

Run by the `robustness` stage: native categorical handling for the CatBoost leaders; the L1-penalised selector
(`liblinear`, C = 0.05, balanced class weights, the K largest non-zero coefficients, no padding); the diversity controls;
response sampling (each per-fold ranking, or each Stability 2024 repetition, carried through the full-DEV pipeline);
name obfuscation; descriptions rebuilt from lineage; and the matched mRMR timing on the first Stability 2024 fold (the
unscreened 1,959-feature universe against the 100-feature semantic pool). The `results` stage adds the HO-rule leader
comparison on sub-populations of the HO scores (LendingClub matured cohort, terms and issue months; Stability 2024
decision months). The `sweep` stage refits every strategy at K = 10, 20, 30, 40 and 50 on the development folds only.

## 8. Leakage and reproducibility controls

- HO rows enter no screening, selection, ranking or tuning step; every target-aware component is fitted on a training
  partition and refitted once on full DEV.
- The LLM receives definitions only; the obfuscated requests are checked to contain no original name.
- Rankings, subsets and preprocessing maps are written before the full-DEV model scores HO, and HO scores are stored
  for the paired comparisons.
- Seeds: 42 (classifiers, selectors, the mRMR sample, Random K), 42 to 46 (stable-core resamples), 20260721 (paired
  bootstrap), 20260914 (obfuscation permutation).
- `uv.lock` pins the environment (Python 3.13, scikit-learn 1.8.0, CatBoost 1.2.10, Boruta 0.4.3). Temperature 0 does not
  make the language model deterministic; the cached responses make the LLM-assisted results reproducible.

## 9. Running the study

### 9.1 Environment

```bash
uv sync                      # Python 3.13 environment from uv.lock
cp .env.example .env         # set OPENAI_API_KEY (needed only by the llm stage); never commit .env
```

### 9.2 Data (not tracked; obtain from the providers and accept their terms)

```bash
# Home Credit Default Risk: put the competition CSV files in data/homecredit/raw/ (or data/inputs/) and
# HomeCredit_columns_description.csv in data/ (or data/homecredit/metadata/columns_description.csv), then:
uv run python scripts/prepare_homecredit.py

# All Lending Club loan data: put accepted_2007_to_2018Q4.csv in data/lendingclub/raw/, then:
uv run python scripts/prepare_lendingclub.py
uv run python scripts/prepare_lendingclub_v2.py
uv run python scripts/build_lendingclub_identity_sidecar.py

# Home Credit - Credit Risk Model Stability: put the competition files in data/homecredit_model_stability_2024/
# (the relative paths recorded in the protocol lock), then build the matrix:
uv run python scripts/build_homecredit_model_stability_2024.py \
  --input-root data/homecredit_model_stability_2024 \
  --output-root data/homecredit_model_stability_2024_matrix \
  --protocol-lock configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json \
  --stage build --mode research
```

The LendingClub loader checks the SHA-256 of the tracked feature inventory and of the identity sidecar against its
manifest; the Home Credit and LendingClub loaders check row counts and ordered row-identity hashes; the Stability 2024
build checks every input file against the protocol lock.

### 9.3 The study, dataset by dataset

```bash
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage prepare     # data, folds, screen, records
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage llm         # all ranking calls (API)
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage main        # 13 strategies + reference, 2 backbones, 5 folds + full DEV
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage robustness
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage voting
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage sweep
uv run python scripts/run_study.py --config configs/study/homecredit.yaml --stage results     # the 25 result files
```

Repeat with `configs/study/lendingclub_v2.yaml` and `configs/study/stability_2024.yaml`, or run
`--stage all` per dataset:

```bash
for dataset in homecredit lendingclub_v2 stability_2024; do
  uv run python scripts/run_study.py --config configs/study/$dataset.yaml --stage all
done
```

Every stage resumes: finished units of work (rankings, fitted pipelines, robustness fits) are read back from
`outputs/study/<dataset>/` instead of being recomputed, so an interrupted stage is restarted with the same command.
Cached units are reused as they are: after changing a setting in `configs/study/`, delete the outputs it affects
(for example `outputs/study/<dataset>/rankings/`, `main/`, `robustness/`, `voting/`, `sweep/`) before running again;
an LLM ranking whose prompt no longer matches the current records is refused rather than reused. The
`llm` stage makes 19 calls on Home Credit (6 named, 12 obfuscated, 1 lineage-only), 28 on LendingClub v2 (24 named, 4
lineage-only) and 12 on Stability 2024 (1 named, 10 repetitions, 1 lineage-only); `--offline` forbids API calls. The
`voting` stage does nothing on Stability 2024.

**Resources.** CPU only, four estimator threads. The full protocol takes days of computation, most of it in Boruta,
recursive elimination, the stable-core resamples and the CatBoost fits on LendingClub v2 and Stability 2024. Plan for
about 16 GB of RAM on Home Credit, 32 GB on LendingClub v2 and more than 32 GB on Stability 2024 (its full-feature
reference and PCA are fitted on 1.2 million DEV rows).

### 9.4 Checks

```bash
uv run python scripts/verify_manuscript_tables.py --schema    # structure of the three result folders
uv run python scripts/verify_manuscript_tables.py --check     # value-by-value comparison with scripts/manuscript_oracle.py
uv run python scripts/verify_manuscript_tables.py --render    # the tables, rendered from the result files
uv run pytest -q
```

A new run makes new LLM calls, so the rankings, and every value that depends on them, differ from any earlier run;
`--check` lists each difference.

### 9.5 Run outputs

```
outputs/study/<dataset>/
  data/           cached DEV and HO frames (Home Credit, LendingClub v2) or the Stability 2024 row index
  prepare/        candidates.json, screen.csv, folds.npz, families.json, records.json, cohort.json, ho_extras.parquet
  llm/            rankings/<call_id>.json (accepted responses), calls.csv
  rankings/       per-partition selector rankings (JSON, with fit time)
  main/           <backbone>/<strategy>/<partition>.json (subset, AUC, metrics, time, memory), scores/*.npy (HO scores)
  robustness/     native, l1, family_cap, depth0, response, obfuscation, lineage, matched_mrmr_timing.json, scores/
  voting/         per-partition voted and reference subsets, HO fits and scores
  sweep/          <backbone>/K<k>/<strategy>/<fold>.json
  manifests/      SHA-256 of every output of each stage (cited by results/*/00_run_ids.csv)
```

## 10. Result files

`results/homecredit_metrics_output/`, `results/lendingclub_metrics_output/` and `results/stability_metrics_output/` hold
the same 25 files with the same headers; `results/headline_subsets.csv` holds the HO-rule leaders' full-DEV subsets and
`results/llm_rankings/` the accepted LLM responses with `index.csv`. [`results/README.md`](results/README.md) describes
every file and column.

## 11. Repository layout

```
src/credit_risk_fs/
  study/                   the study runner: config, data, records, llm, selection, modeling, metrics, stages, results
  data/                    dataset loaders; Stability 2024 adapter and protocol contract
  feature_engineering/     Home Credit and LendingClub v2 universes
  feature_metadata/        definition-only feature records (definitions.py)
  selectors/               llm_prompt.py and llm_screening.py (LLM stage), domain_rules.py, lightweight/ (IV, mRMR,
                           L1, controls), heavy/ (Boruta RF, RFE CatBoost, CatBoost SHAP), mrmr.py (RF/correlation
                           filter), stable_core_llm_fill.py, llm_then_stat.py
  preprocessing/           encoding, LendingClub cleaning, labelling
  evaluation/              paired DeLong and Holm, stability measures
scripts/                   run_study.py, data preparation, verify_manuscript_tables.py
configs/study/             common.yaml and one configuration per dataset
configs/protocols/         row-alignment contract, Stability 2024 protocol lock
results/                   result files (Section 10)
docs/                      llm_prompt_protocol.md and protocol notes
tests/                     unit and contract tests
```

## 12. Superseded entry points

`scripts/run_study.py` replaces the earlier execution paths. These remain in the tree for history and are not used by
the study:

- `scripts/run_matrix.py` and `src/credit_risk_fs/experiments/runner.py` (matrix runner); `scripts/run_prompt_16_third_dataset.py`,
  `scripts/run_prompt_16_final_oot.py` and `src/credit_risk_fs/experiments/prompt_16_*` (earlier Stability 2024 runners);
  `scripts/fill_todo.py`, `scripts/fill_todo_gate4.py` and `scripts/todo_fill/` (earlier robustness fills);
  `scripts/run_b6_obfuscation_ablation.py` and `scripts/b6_obfuscation.py` (earlier obfuscation runner);
  `src/credit_risk_fs/selectors/combinations.py` (earlier IV-then-Boruta route with a top-P screen);
  `docs/repository_freeze.md` (record of the earlier freeze).
- `scripts/recenter_pairwise_inference.py` shifts bootstrap intervals onto externally supplied differences, which is
  incompatible with the paired bootstrap of [Section 6](#6-metrics-and-inference); remove it:

```bash
git rm scripts/recenter_pairwise_inference.py
```

Tests written for the earlier paths (for example tests of the IV prefilter or of the earlier prompts) fail against the
current selectors and can be removed together with those paths.

## Citation

```bibtex
@unpublished{azamjonov2026rowindependent,
  author = {A'zamjonov, Dilshod and Rahman, Sejuti},
  title  = {Row-Independent Feature Pre-Screening with Language Models for High-Dimensional Credit Scoring: A Frozen Evaluation on Temporal and Recency-Shift Holdouts},
  note   = {Submitted to \emph{Big Data Research}},
  year   = {2026},
  url    = {https://github.com/dilshodazamjonov/Feature-Selection-Research}
}
```

## Licence, data terms and declarations

- **Code and result files:** no licence file is included yet; until one is added, all rights are reserved.
- **Data:** the three datasets are publicly available on Kaggle and their redistribution is governed by the terms of the
  respective competitions and dataset licences:
  [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data),
  [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
  [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).
  They are not tracked here.
- **Funding and interests:** no specific grant from funding agencies in the public, commercial or not-for-profit
  sectors; the authors declare no known competing financial interests or personal relationships.
- **Disclaimer:** the views expressed are those of the authors and do not necessarily reflect those of the Central Bank
  of the Republic of Uzbekistan. Only public datasets are used.
- **Generative AI:** Claude Code (Anthropic) assisted in developing parts of the code and documentation of this
  repository. GPT-4.1 mini is a component of the method under study.
