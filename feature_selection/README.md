# IVFilter Class — Information Value Feature Selection

## Overview

`IVFilter` is a lightweight, research-oriented Python class used to compute **Information Value (IV)** and **Weight of Evidence (WOE)** for features in a binary classification dataset.

It is designed to be used inside a **machine learning pipeline** with the standard **`fit` / `transform` interface**, ensuring proper separation between training and validation data during cross-validation.

The class:

* Computes IV using **quantile binning for numeric features** and **category grouping for categorical features**
* Uses **vectorized NumPy operations** for efficient computation
* Generates **CSV reports** for reproducibility and research documentation
* Flags potential **data leakage**
* Selects features based on a configurable **minimum IV threshold**

Outputs are saved under:

```
data/output/
```

---

# Pipeline Position

`IVFilter` is intended to run **after preprocessing and before feature scaling / dimensionality reduction**.

Example pipeline structure:

```
Raw Data
   ↓
Preprocessor
   ↓
IVFilter
   ↓
Scaling
   ↓
PCA
   ↓
Random Forest
```

---

# Features

* Vectorized IV / WOE computation
* Quantile-based binning (`qcut`)
* Automatic handling of missing values
* Leakage detection heuristics
* CSV reporting for research reproducibility
* Compatible with cross-validation workflows

---

# Installation

No installation required beyond common data science libraries.

Required packages:

```
numpy
pandas
```

---

# Class Initialization

```python
from iv_filter import IVFilter

iv = IVFilter(
    n_bins=10,
    min_iv=0.02,
    max_iv_for_leakage=0.5,
    min_bin_pct=None,
    verbose=True,
    save_bin_level_stats=True
)
```

### Parameters

| Parameter              | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `n_bins`               | Number of quantile bins for numeric features     |
| `min_iv`               | Minimum IV required for a feature to be selected |
| `max_iv_for_leakage`   | Threshold to flag suspiciously high IV           |
| `min_bin_pct`          | Optional minimum bin percentage warning          |
| `verbose`              | Prints processing logs                           |
| `save_bin_level_stats` | Saves detailed per-bin statistics                |

---

# Basic Usage

### Fit IV on training data

```python
iv = IVFilter(n_bins=10, min_iv=0.02)

iv.fit(X_train, y_train)

print(iv.iv_table_.head())
```

### Transform dataset

```python
X_selected = iv.transform(X_train)
```

### Fit and transform together

```python
X_selected = iv.fit_transform(X_train, y_train)
```

---

# Cross Validation Usage

When performing **cross-validation**, the IV calculation must be fit **inside each fold** using only the training portion.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):

    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    iv = IVFilter(n_bins=10, min_iv=0.02)

    iv.fit(X_train_fold, y_train_fold)

    X_train_sel = iv.transform(X_train_fold)
    X_val_sel = iv.transform(X_val_fold)

    # Continue pipeline
```

---

# Example Full Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# preprocessing
X_prep = preprocessor.fit_transform(X_train)

# IV feature selection
iv = IVFilter(n_bins=10, min_iv=0.02)
iv.fit(X_prep, y_train)

X_selected = iv.transform(X_prep)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_pca, y_train)
```

---

# Output Files

The class generates several reports under:

```
data/output/
```

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `iv_table.csv`             | Feature IV ranking             |
| `iv_selected_features.csv` | Features kept after filtering  |
| `iv_dropped_features.csv`  | Features removed due to low IV |
| `iv_feature_summaries.csv` | Per-feature statistics         |
| `iv_bin_level_stats.csv`   | Per-bin WOE and IV statistics  |
| `iv_leakage_flags.csv`     | Leakage detection flags        |

---

# Leakage Detection

The class flags potential leakage when:

* `IV >= max_iv_for_leakage`
* A bin has **perfect separation** (all good or all bad)

These flags are saved in:

```
iv_leakage_flags.csv
```

Such features should be inspected manually before modeling.

---

# Research Reproducibility

For research workflows, the class automatically stores:

* IV ranking tables
* Feature selection results
* Bin-level WOE statistics

These outputs allow experiments to be reproduced and included in **appendix sections of research papers**.

---

# Notes

* IV is calculated for **binary classification problems only**
* Numeric variables use **quantile binning**
* Categorical variables are grouped by category
* Rare categories are not automatically merged

---

# License

Internal research utility.

---

# LLM Selector Note

This repository also contains metadata-driven selectors under:

- `feature_selection/llm_selector.py`
- `feature_selection/hybrid.py`

The LLM selector is designed to mimic expert-style feature review in retail
credit risk. It does not receive raw training rows. Instead, it reviews
feature-level metadata and ranks variables the way a domain expert might review
a variable pack before model development.

Current metadata exposed to the LLM includes:

- `description`
- `table`
- `dtype`
- `missing_rate`
- `non_null_count`
- `mean`
- `min`
- `max`
- `std`
- `var`
- `p05`, `p25`, `p50`, `p75`, `p95`
- `unique_count` for non-numeric features

The lightweight structured LLM response can also include:

- `reasoning_summary`
- `selection_principles`

The prompt is intentionally domain-aware and stability-aware. It biases the LLM
toward:

- interpretable business signals
- broad coverage and low missingness
- stable operational collection processes
- simpler representatives among redundant aggregates
- features that are more likely to remain useful out of time

This keeps the LLM method aligned with the research framing of using the model
as a substitute for expert-driven feature selection rather than as a row-level
predictive model.
