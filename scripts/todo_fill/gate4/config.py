"""Constants shared by the Gate-4 steps: leaders, families, file/step registry, unit costs."""

from __future__ import annotations

import math

from scripts.todo_fill.common import HOMECREDIT, LENDINGCLUB, THIRD

#: CSV file -> step id.
FILES: dict[str, str] = {
    "37_lc_maturity.csv": "37",
    "38_conservative_interval.csv": "38",
    "39a_onehot_dimension.csv": "39a",
    "39b_native_catboost.csv": "39b",
    "40_perfold_ho.csv": "40",
    "43_diversity_capped.csv": "43",
    "44_mechanical.csv": "44",
    "45_repeated_calls.csv": "45",
    "47_obfuscation.csv": "47",
}
STEP_FILE: dict[str, str] = {step: name for name, step in FILES.items()}
ALL_STEPS: tuple[str, ...] = ("38", "39a", "40", "37", "39b", "44", "47", "43", "45")  # TODO.md suggested order

#: Value columns that count as "owed" per file (dimension columns are pre-filled).
VALUE_COLUMNS: dict[str, list[str]] = {
    "37_lc_maturity.csv": ["auc_A", "auc_B", "delta_auc_A_minus_B", "ci95_low", "ci95_high", "n_ho"],
    "38_conservative_interval.csv": ["auc_A", "auc_B", "delta_auc_A_minus_B", "ci95_low", "ci95_high", "n_ho"],
    "39a_onehot_dimension.csv": ["n_categorical_in_subset", "columns_after_onehot"],
    "39b_native_catboost.csv": ["ho_auc_native", "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc"],
    "40_perfold_ho.csv": ["ho_auc"],
    "43_diversity_capped.csv": ["base_selector", "ho_auc", "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc"],
    "44_mechanical.csv": [
        "ho_auc_mechanical", "delta_mechanical_minus_cached", "ci95_low", "ci95_high", "n_ho",
        "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc", "fulldev_top100_names_pipe_separated",
    ],
    "45_repeated_calls.csv": ["ho_auc", "response_id", "top100_names_pipe_separated"],
    "47_obfuscation.csv": [
        "ho_auc_arm", "fold1_auc", "fold2_auc", "fold3_auc", "fold4_auc", "fold5_auc",
        "overlap_with_named_fulldev", "fulldev_topK_names_pipe_separated",
    ],
}

#: Table 4 (HO rule) classical leader per (dataset, backbone) -- the base selector of step 43.
HO_RULE_CLASSICAL_LEADER: dict[tuple[str, str], str] = {
    (HOMECREDIT, "lr"): "mRMR",
    (HOMECREDIT, "catboost"): "RFE CatBoost",
    (LENDINGCLUB, "lr"): "IV then Boruta",
    (LENDINGCLUB, "catboost"): "IV then Boruta",
    (THIRD, "lr"): "RFE CatBoost",
    (THIRD, "catboost"): "CatBoost SHAP",
}

#: Reference values quoted in TODO.md for the step-38 join check.
STEP38_EXPECTED = {"auc_A": 0.74322934, "auc_B": 0.76989}

#: Third-dataset depth-0 source families (``static_0`` and ``static_cb_0`` tables).
THIRD_DEPTH0_FAMILIES: tuple[str, ...] = ("static", "static_cb")

#: Fixed row count of the third dataset's availability-filtered LLM candidate set (TODO.md).
THIRD_EXPECTED_CANDIDATES = 1068

#: LendingClub cached-run budgets: the legacy matrix issued one call per feature budget.
LENDINGCLUB_CACHE_BUDGETS: tuple[int, ...] = (20, 40, 60, 100)

#: Unit costs (seconds) measured on this machine, used only for the dry-run estimates.
CATBOOST_SECONDS_PER_1K_ROWS = 0.9          # single-thread CatBoost, 1,500 iterations, depth 10
LR_FIT_OVERHEAD_SECONDS = 25.0              # load + preprocess + liblinear
CATBOOST_FIT_OVERHEAD_SECONDS = 30.0
LLM_CALL_SECONDS = 10.0
BOOTSTRAP_SECONDS_PER_100K_ROWS = 15.0
DEV_ROWS = {HOMECREDIT: 99_092, LENDINGCLUB: 598_649, THIRD: 1_221_743}
FOLD_TRAIN_ROWS = {
    HOMECREDIT: {f"fold{i}": int(99_092 * i / 6) for i in range(1, 6)},
    LENDINGCLUB: {f"fold{i}": int(598_649 * i / 6) for i in range(1, 6)},
    THIRD: {"fold1": 200_661, "fold2": 402_103, "fold3": 604_598, "fold4": 810_904, "fold5": 1_012_061},
}
HO_ROWS = {HOMECREDIT: 120_053, LENDINGCLUB: 293_105, THIRD: 304_916}
#: Selector fit costs (seconds, full DEV) measured or extrapolated from the previous fill.
SELECTOR_SECONDS_FULL_DEV = {
    (HOMECREDIT, "mRMR"): 210.0,
    (HOMECREDIT, "RFE CatBoost"): 170.0,
    (HOMECREDIT, "LLM then mRMR"): 90.0,
    (LENDINGCLUB, "IV then Boruta"): 3300.0,
    (LENDINGCLUB, "RFE CatBoost"): 1200.0,
    (LENDINGCLUB, "LLM then mRMR"): 80.0,
    (THIRD, "RFE CatBoost"): 7200.0,
    (THIRD, "CatBoost SHAP"): 1500.0,
}


def fit_seconds(dataset: str, backbone: str, partition: str) -> float:
    rows = DEV_ROWS[dataset] if partition == "full_dev" else FOLD_TRAIN_ROWS[dataset][partition]
    if backbone == "lr":
        return LR_FIT_OVERHEAD_SECONDS
    return CATBOOST_FIT_OVERHEAD_SECONDS + CATBOOST_SECONDS_PER_1K_ROWS * rows / 1000.0


def selector_seconds(dataset: str, label: str, partition: str) -> float:
    full = SELECTOR_SECONDS_FULL_DEV.get((dataset, label), 600.0)
    if partition == "full_dev":
        return full
    return full * FOLD_TRAIN_ROWS[dataset][partition] / DEV_ROWS[dataset]


def bootstrap_seconds(rows: int) -> float:
    return BOOTSTRAP_SECONDS_PER_100K_ROWS * rows / 100_000.0


def family_cap(k: int, n_families: int) -> int:
    return int(math.ceil(k / n_families))
