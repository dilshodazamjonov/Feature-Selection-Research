# Research Implementation Report

## Topic

**Evaluating Feature Selection in Credit Scoring: A Comparison of Statistical, LLM-Based, and Hybrid Approaches.**

Working hypothesis:

**LLMs can approximate domain-driven feature selection and complement statistical methods.**

## Executive Status

**Implementation status: mostly complete**

**Research execution status: not fully complete yet**

The codebase now supports the intended experiment families, the shared temporal evaluation framework, separate reporting plots, and an overnight runner. The remaining work is not major architecture work anymore. The remaining work is to actually run the experiment matrix you want and interpret the resulting outputs.

## What Is Implemented

### 1. Shared temporal experiment framework

Implemented in:

- [pipelines/common.py](</d:/python projects/Research/pipelines/common.py>)
- [training/kfold_trainer.py](</d:/python projects/Research/training/kfold_trainer.py>)
- [training/fold.py](</d:/python projects/Research/training/fold.py>)

Current behavior:

- data loading and feature engineering are centralized
- temporal split is based on the engineered descendant of `previous_application.DAYS_DECISION`
- default split windows are:
  - `DEV`: `-600 <= time < -240`
  - `OOT`: `-240 <= time <= 0`
- older rows are dropped
- grouped time-series CV is used inside `DEV`
- final model is refit on full `DEV` and evaluated on `OOT`

This matches the core research need for realistic temporal credit-scoring evaluation.

### 2. Part 1: Statistical baselines

Implemented in:

- [experiments/statistical_baselines.py](</d:/python projects/Research/experiments/statistical_baselines.py>)
- [scripts/run_statistical_comparison.py](</d:/python projects/Research/scripts/run_statistical_comparison.py>)

Supported selectors:

- `mrmr`
- `boruta`
- `pca`

This covers the statistical-baseline part of the topic.

### 3. Part 2: LLM vs statistical

Implemented in:

- [experiments/llm_vs_statistical.py](</d:/python projects/Research/experiments/llm_vs_statistical.py>)
- [feature_selection/llm_selector.py](</d:/python projects/Research/feature_selection/llm_selector.py>)
- [scripts/run_llm_vs_statistical.py](</d:/python projects/Research/scripts/run_llm_vs_statistical.py>)

Current LLM behavior:

- selector runs fold-locally on training data only
- `95%` missing-rate filter is applied first
- IV filtering is applied after that
- only summarized metadata is sent to the LLM
- no raw row-level data is sent
- selection is cached by metadata/training signature

This aligns well with the claim that the LLM is acting as a domain-driven selector rather than a direct predictive model.

### 4. Part 3: Hybrid LLM -> statistical

Implemented in:

- [experiments/hybrid_comparison.py](</d:/python projects/Research/experiments/hybrid_comparison.py>)
- [feature_selection/hybrid.py](</d:/python projects/Research/feature_selection/hybrid.py>)
- [scripts/run_hybrid_comparison.py](</d:/python projects/Research/scripts/run_hybrid_comparison.py>)

Current behavior:

- LLM selects raw engineered features first
- preprocessing happens after that
- downstream statistical selector is then applied

This is the correct implementation pattern for testing whether LLM preselection helps statistical selectors.

### 5. Separate plotting/reporting layer

Implemented in:

- [plots.py](</d:/python projects/Research/plots.py>)
- [evaluation/plotting.py](</d:/python projects/Research/evaluation/plotting.py>)

Important design choice:

- plotting is separate from training
- real experiment runs do not generate comparison plots automatically
- finished experiment folders can be compared later by passing their paths to `plots.py`

This is a good research workflow because training and reporting are now separated cleanly.

### 6. YAML config and easier model switching

Implemented in:

- [config.yaml](</d:/python projects/Research/config.yaml>)
- [experiments/config.py](</d:/python projects/Research/experiments/config.py>)

Current behavior:

- model can be changed with `model_selector`
- model-specific parameters are configurable in one place
- scripts accept `--config`
- scripts also accept `--model-selector` override

This makes repeated experiment runs much easier.

### 7. Overnight orchestrator

Implemented in:

- [main.py](</d:/python projects/Research/main.py>)
- [experiments/run_all.py](</d:/python projects/Research/experiments/run_all.py>)
- [scripts/run_all_experiments.py](</d:/python projects/Research/scripts/run_all_experiments.py>)

Current behavior:

- runs Part 1
- then Part 2
- then Part 3

This is useful for unattended overnight execution.

### 8. Stability summary CSV

Implemented in:

- [training/kfold_trainer.py](</d:/python projects/Research/training/kfold_trainer.py>)

Current output:

- `results/stability_confidence_summary.csv`

Metrics currently included:

- `gini`
- `ks`
- `psi_feature_mean`
- `psi_feature_max`
- `psi_model`
- `jaccard_similarity`

Columns:

- `metric`
- `value`
- `ci95_lower`
- `ci95_upper`

This is aligned with the stability part of the research question.

## Final Check Results

Code-level checks completed successfully:

- `py_compile` passed for the main orchestration and experiment files
- `python main.py --help` passed
- `python plots.py --help` passed
- `python scripts/run_statistical_comparison.py --help` passed
- `python scripts/run_llm_vs_statistical.py --help` passed
- `python scripts/run_hybrid_comparison.py --help` passed
- `python scripts/run_single_experiment.py --help` passed
- `python scripts/run_all_experiments.py --help` passed

What was not done in this final audit:

- I did not rerun all full experiments on the real data in this final pass
- I did not rerun expensive LLM jobs just for validation
- I did not regenerate full empirical plots because that depends on completed experiment outputs

So the code is checked, but the full empirical study still depends on actually running the jobs.

## Does It Meet The Topic Requirements?

## Short answer

**Yes for implementation design. Not yet fully yes for completed research evidence.**

## Why the answer is yes on implementation

Your topic needs:

1. a statistical baseline comparison
2. an LLM-based selector
3. a hybrid LLM + statistical selector
4. temporal credit-scoring evaluation
5. overlap, stability, and performance comparisons

The codebase now supports all of those.

## Why the answer is not fully yes yet for completed research

The study is only complete after the required experiment matrix is actually run and summarized.

In particular, the following still depend on execution rather than implementation:

1. empirical evidence that LLM approximates domain-driven selection
2. empirical evidence that hybrid helps or does not help
3. final cross-model comparisons such as `lr` vs `catboost`
4. final interpretation of stability over time

## Important caveats

### 1. The overnight runner only executes one hybrid baseline per run

`main.py` runs one hybrid configuration based on the configured `hybrid_comparison.stat_selector`.

That means:

- if config uses `mrmr`, overnight run covers `LLM -> mRMR`
- if config uses `boruta`, overnight run covers `LLM -> Boruta`

If your final study wants both hybrids in the final reported results, you still need either:

- two overnight runs with different config values, or
- a future enhancement to run both hybrid baselines automatically

### 2. Part 2 can include PCA, but it is not the default

The LLM-vs-statistical pipeline supports passing different statistical selectors, but the default config is:

- `mrmr`
- `boruta`

If your final study explicitly wants `LLM vs PCA` in Part 2, add `pca` to the config or CLI for that run.

### 3. The report framework exists, but the thesis/paper narrative is not autogenerated

The code now produces the right artifacts, but it does not automatically write the final academic interpretation for the topic. That still requires your analysis after runs finish.

## Recommended Experiment Matrix For The Topic

To fully support the topic claim, the minimum practical run set is:

### Model family 1

- `lr` statistical baselines
- `lr` LLM vs statistical
- `lr` hybrid with `mrmr`
- `lr` hybrid with `boruta`

### Model family 2

- `catboost` statistical baselines
- `catboost` LLM vs statistical
- `catboost` hybrid with `mrmr`
- `catboost` hybrid with `boruta`

That gives you both:

- selector-family comparison
- model-family comparison

## What Is Done vs Not Done

## Done

- experiment architecture
- temporal split logic
- statistical baselines pipeline
- LLM selector pipeline
- hybrid selector pipeline
- metadata-only LLM selection
- per-run artifact folders
- YAML config
- separate plotting tool
- overnight runner
- stability CI summary output
- docs and structure cleanup

## Not done automatically

- running the full experiment matrix
- automatically comparing both hybrid baselines in a single overnight run
- automatically generating the final thesis conclusions
- statistical significance testing beyond the current CI summary

## Overall Verdict

**The implementation now supports the research topic well and is in a runnable state.**

**The code side is substantially done.**

**The research side is not fully done until you run the required experiment matrix and interpret the outputs.**

If your question is:

**“Do I have the code needed to perform this research properly?”**

The answer is:

**Yes, with one practical note: run both hybrid variants and any extra Part 2 baselines you want in the final study.**
