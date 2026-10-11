"""Study constants and configuration loading.

A dataset configuration (``configs/study/<dataset>.yaml``) names a common file in
``extends`` (``configs/study/common.yaml``) whose protocol settings it inherits; keys
given in the dataset file override the common ones.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLASSICAL = (
    "Domain rules",
    "Random K",
    "PCA",
    "IV/WOE",
    "mRMR",
    "Boruta RF",
    "IV then Boruta",
    "RFE CatBoost",
    "CatBoost SHAP",
)
LLM_ASSISTED = ("Pure LLM", "LLM then mRMR", "LLM then Boruta", "Stable core + LLM fill")
SELECTORS = CLASSICAL + LLM_ASSISTED
FULL = "Full features"
FAMILY = {
    **{name: "classical" for name in CLASSICAL},
    **{name: "LLM-assisted" for name in LLM_ASSISTED},
    FULL: "reference",
}
SLUGS = {
    "Domain rules": "domain_rules",
    "Random K": "random_k",
    "PCA": "pca",
    "IV/WOE": "iv_woe",
    "mRMR": "mrmr",
    "Boruta RF": "boruta_rf",
    "IV then Boruta": "iv_then_boruta",
    "RFE CatBoost": "rfe_catboost",
    "CatBoost SHAP": "catboost_shap",
    "Pure LLM": "pure_llm",
    "LLM then mRMR": "llm_then_mrmr",
    "LLM then Boruta": "llm_then_boruta",
    "Stable core + LLM fill": "stable_core_llm_fill",
    FULL: "full_features",
}
BORUTA_SELECTORS = ("Boruta RF", "IV then Boruta", "LLM then Boruta")
BACKBONES = ("lr", "catboost")
FOLDS = ("fold1", "fold2", "fold3", "fold4", "fold5")
PARTITIONS = FOLDS + ("full_dev",)
FULL_DEV = "full_dev"


class StudyError(RuntimeError):
    """Raised when an input, cache or stage precondition does not hold."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a dataset configuration merged over the common file it extends."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        common = yaml.safe_load((config_path.parent / parent).read_text(encoding="utf-8")) or {}
        data = _merge(common, data)
    for key in ("dataset", "budgets", "llm_pool", "models", "selectors", "llm"):
        if key not in data:
            raise StudyError(f"{config_path}: missing required key {key!r}")
    data["config_path"] = str(config_path)
    data.setdefault("root", str(PROJECT_ROOT))
    return data


def root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["root"])


def resolve(cfg: dict[str, Any], relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root(cfg) / path


def output_dir(cfg: dict[str, Any]) -> Path:
    return resolve(cfg, cfg.get("output_root", "outputs/study")) / str(cfg["dataset"])


def results_dir(cfg: dict[str, Any]) -> Path:
    return resolve(cfg, cfg.get("results_root", "results"))


def budget(cfg: dict[str, Any], backbone: str) -> int:
    return int(cfg["budgets"][backbone])


def pool_size(cfg: dict[str, Any], backbone: str) -> int:
    return int(cfg["llm_pool"][backbone])


def family_cap(budget_k: int, n_families: int) -> int:
    return int(math.ceil(int(budget_k) / int(n_families)))


__all__ = [
    "BACKBONES",
    "BORUTA_SELECTORS",
    "CLASSICAL",
    "FAMILY",
    "FOLDS",
    "FULL",
    "FULL_DEV",
    "LLM_ASSISTED",
    "PARTITIONS",
    "PROJECT_ROOT",
    "SELECTORS",
    "SLUGS",
    "StudyError",
    "budget",
    "family_cap",
    "load_config",
    "output_dir",
    "pool_size",
    "resolve",
    "results_dir",
    "root",
]
