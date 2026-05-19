# Run Catalog

This file lists the main commands and the full experiment matrix supported by the current repo.

## Main Orchestrator

Top-level entrypoint:

```powershell
python run_experiment.py --dataset all
```

Useful variants:

```powershell
python run_experiment.py --dataset homecredit
python run_experiment.py --dataset lendingclub
python run_experiment.py --dataset all --dry-run
python run_experiment.py --dataset homecredit --models lr
python run_experiment.py --dataset lendingclub --models catboost
python run_experiment.py --dataset all --force
```

What it does:
1. prepare dataset
2. check setup
3. run matrix
4. aggregate results
5. make plots

## Lower-Level Scripts

Preparation:

```powershell
python scripts/prepare_homecredit.py
python scripts/prepare_lendingclub.py
```

Setup checks:

```powershell
python scripts/check_setup.py --dataset homecredit
python scripts/check_setup.py --dataset lendingclub
```

Matrix runs:

```powershell
python scripts/run_matrix.py --dataset homecredit
python scripts/run_matrix.py --dataset lendingclub
python scripts/run_matrix.py --dataset homecredit --dry-run
python scripts/run_matrix.py --dataset lendingclub --dry-run
```

Single runs:

```powershell
python scripts/run_single.py --dataset homecredit --model lr --selector mrmr
python scripts/run_single.py --dataset lendingclub --model lr --selector mrmr
```

Aggregation and plots:

```powershell
python scripts/aggregate_results.py --dataset homecredit
python scripts/aggregate_results.py --dataset lendingclub
python scripts/make_plots.py --dataset homecredit
python scripts/make_plots.py --dataset lendingclub
```

## Matrix Size

Per dataset:

- `2` models
- `8` selector variants
- `16` total experiment runs
- about `6` actual fresh LLM ranking calls on a cold cache

For both datasets together:

- `32` total experiment runs
- about `12` actual fresh LLM ranking calls on a cold cache

## Selector Variants

Supported selector names:

- `mrmr`
- `boruta`
- `pca`
- `domain_rule_baseline`
- `llm`
- `llm_then_mrmr`
- `llm_then_boruta`
- `stable_core_llm_fill`

Models:

- `lr`
- `catboost`

Datasets:

- `homecredit`
- `lendingclub`

## Full Matrix Per Dataset

The same 16-run matrix applies to both datasets.

### Logistic Regression

- `lr + mrmr`
- `lr + boruta`
- `lr + pca`
- `lr + domain_rule_baseline`
- `lr + llm`
- `lr + llm_then_mrmr`
- `lr + llm_then_boruta`
- `lr + stable_core_llm_fill`

### CatBoost

- `catboost + mrmr`
- `catboost + boruta`
- `catboost + pca`
- `catboost + domain_rule_baseline`
- `catboost + llm`
- `catboost + llm_then_mrmr`
- `catboost + llm_then_boruta`
- `catboost + stable_core_llm_fill`

## Concrete Single-Run Commands

### Home Credit

```powershell
python scripts/run_single.py --dataset homecredit --model lr --selector mrmr
python scripts/run_single.py --dataset homecredit --model lr --selector boruta
python scripts/run_single.py --dataset homecredit --model lr --selector pca
python scripts/run_single.py --dataset homecredit --model lr --selector domain_rule_baseline
python scripts/run_single.py --dataset homecredit --model lr --selector llm
python scripts/run_single.py --dataset homecredit --model lr --selector llm_then_mrmr
python scripts/run_single.py --dataset homecredit --model lr --selector llm_then_boruta
python scripts/run_single.py --dataset homecredit --model lr --selector stable_core_llm_fill
python scripts/run_single.py --dataset homecredit --model catboost --selector mrmr
python scripts/run_single.py --dataset homecredit --model catboost --selector boruta
python scripts/run_single.py --dataset homecredit --model catboost --selector pca
python scripts/run_single.py --dataset homecredit --model catboost --selector domain_rule_baseline
python scripts/run_single.py --dataset homecredit --model catboost --selector llm
python scripts/run_single.py --dataset homecredit --model catboost --selector llm_then_mrmr
python scripts/run_single.py --dataset homecredit --model catboost --selector llm_then_boruta
python scripts/run_single.py --dataset homecredit --model catboost --selector stable_core_llm_fill
```

### LendingClub

```powershell
python scripts/run_single.py --dataset lendingclub --model lr --selector mrmr
python scripts/run_single.py --dataset lendingclub --model lr --selector boruta
python scripts/run_single.py --dataset lendingclub --model lr --selector pca
python scripts/run_single.py --dataset lendingclub --model lr --selector domain_rule_baseline
python scripts/run_single.py --dataset lendingclub --model lr --selector llm
python scripts/run_single.py --dataset lendingclub --model lr --selector llm_then_mrmr
python scripts/run_single.py --dataset lendingclub --model lr --selector llm_then_boruta
python scripts/run_single.py --dataset lendingclub --model lr --selector stable_core_llm_fill
python scripts/run_single.py --dataset lendingclub --model catboost --selector mrmr
python scripts/run_single.py --dataset lendingclub --model catboost --selector boruta
python scripts/run_single.py --dataset lendingclub --model catboost --selector pca
python scripts/run_single.py --dataset lendingclub --model catboost --selector domain_rule_baseline
python scripts/run_single.py --dataset lendingclub --model catboost --selector llm
python scripts/run_single.py --dataset lendingclub --model catboost --selector llm_then_mrmr
python scripts/run_single.py --dataset lendingclub --model catboost --selector llm_then_boruta
python scripts/run_single.py --dataset lendingclub --model catboost --selector stable_core_llm_fill
```

## Recommended Final Run

If you want the full paper package in one shot:

```powershell
python run_experiment.py --dataset all
```

If you want to check scheduling only:

```powershell
python run_experiment.py --dataset all --dry-run
```

## Practical Notes

- One clean final run for each dataset is normally enough for this paper.
- The LLM cache keeps actual OpenAI calls much lower than the number of LLM-related experiment folders.
- The slowest selectors are usually `boruta` and `stable_core_llm_fill`.
- `catboost` is currently the heaviest model family in runtime.
