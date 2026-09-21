"""Harvest frozen evidence that survives in the repository's audit layer.

Nothing here fits a model.  Every accessor reads a tracked CSV/JSON/Markdown
file under ``cleanup/audits``, ``results`` or ``reports/archive`` and returns
plain Python structures keyed by the paper's (dataset, model, method) grid.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from scripts.todo_fill.common import (
    HOMECREDIT,
    LENDINGCLUB,
    REPO_ROOT,
    THIRD,
    nogueira_from_frequencies,
)

AUDITS = REPO_ROOT / "cleanup/audits"
BASELINE_STABILITY = AUDITS / "prompt_11_selector_combinations/baseline_selection_stability.csv"
BASELINE_FEATURE_PSI = AUDITS / "prompt_11_selector_combinations/baseline_feature_psi_audit.csv"
BASELINE_RESULTS_LONG = AUDITS / "prompt_11_selector_combinations/baseline_results_long.csv"
COMBINATION_SUMMARY = AUDITS / "prompt_13_combination_dev_review/dev_configuration_summary.csv"
COMBINATION_SELECTION_STABILITY = AUDITS / "prompt_13_combination_dev_review/selection_stability.csv"
COMBINATION_STAGE_SUPPORT = AUDITS / "prompt_13_combination_dev_review/stage_support_audit.csv"
FINAL_TABLE = REPO_ROOT / "results/final_research_package_v2/final_results_tables.csv"
THIRD_DEV_ACCOUNTING = AUDITS / "prompt_16_final_amended_oot/complete_amended_dev_accounting.csv"
THIRD_OOT_REGISTRY = AUDITS / "prompt_16_final_amended_oot/final_34_cell_oot_registry.json"
THIRD_PILOT_ACCOUNTING = AUDITS / "prompt_16_final_third_dataset_execution/pilot_accounting.csv"
HOMECREDIT_REPORT = REPO_ROOT / "reports/archive/homecredit_report.md"
CROSS_DATASET_V2_REPORT = REPO_ROOT / "reports/archive/cross_dataset_v2_analysis.md"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    number = _float(value)
    return None if number is None else int(round(number))


def _pipe_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(round(float(item))) for item in str(value).split("|") if item.strip() != ""]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------ full baseline v1


@dataclass(frozen=True)
class FrozenStability:
    dataset: str
    model: str
    method_id: str
    fold_counts: tuple[int, ...]
    nogueira: float | None
    jaccard_mean: float | None
    jaccard_min: float | None
    jaccard_max: float | None
    union_size: int | None
    source: str


@lru_cache(maxsize=None)
def baseline_stability() -> dict[tuple[str, str, str], FrozenStability]:
    """(dataset, model, method_id) -> frozen full_baseline_v1 fold statistics."""

    out: dict[tuple[str, str, str], FrozenStability] = {}
    for row in _rows(BASELINE_STABILITY):
        key = (row["dataset"], row["model"], row["method_id"])
        out[key] = FrozenStability(
            dataset=row["dataset"],
            model=row["model"],
            method_id=row["method_id"],
            fold_counts=tuple(_pipe_ints(row.get("fold_selected_counts"))),
            nogueira=_float(row.get("nogueira_stability")),
            jaccard_mean=_float(row.get("mean_pairwise_jaccard")),
            jaccard_min=_float(row.get("minimum_pairwise_jaccard")),
            jaccard_max=_float(row.get("maximum_pairwise_jaccard")),
            union_size=_int(row.get("union_size")),
            source=rel(BASELINE_STABILITY),
        )
    return out


@lru_cache(maxsize=None)
def baseline_feature_psi() -> dict[tuple[str, str, str], list[tuple[str, float | None]]]:
    """(dataset, model, method_id) -> [(model-matrix column, psi)] for the frozen full-DEV refit."""

    out: dict[tuple[str, str, str], list[tuple[str, float | None]]] = {}
    for row in _rows(BASELINE_FEATURE_PSI):
        key = (row["dataset"], row["model"], row["method_id"])
        out.setdefault(key, []).append((row["feature_name"], _float(row.get("psi"))))
    return out


# ------------------------------------------------------- selector combinations


@dataclass(frozen=True)
class FrozenCombination:
    dataset: str
    method: str
    iv_pool: int | None
    model: str
    fold_counts: tuple[int, ...]
    jaccard_mean: float | None
    jaccard_min: float | None
    jaccard_max: float | None
    nogueira: float | None
    selection_counts: dict[str, int]
    source: str


def _pool_from_variant(variant: str, iv_pool: str) -> int | None:
    if iv_pool and iv_pool.strip():
        return _int(iv_pool)
    match = re.search(r"iv_pool_(\d+)", variant or "")
    return int(match.group(1)) if match else None


@lru_cache(maxsize=None)
def combination_stability(universe_size: dict[str, int] | None = None) -> dict[tuple[str, str, int | None, str], FrozenCombination]:
    """(dataset, method, iv_pool, model) -> frozen selector_combinations_v1 fold statistics.

    Nogueira is not in the audit; it is reconstructed exactly from the frozen
    per-feature selection counts and fold sizes with the dataset universe as ``d``.
    """

    from scripts.todo_fill.common import UNIVERSE_SIZE

    universe = dict(UNIVERSE_SIZE)
    if universe_size:
        universe.update(universe_size)
    counts: dict[tuple[str, str, int | None, str], dict[str, int]] = {}
    meta: dict[tuple[str, str, int | None, str], dict[str, str]] = {}
    for row in _rows(COMBINATION_SELECTION_STABILITY):
        key = (row["dataset"], row["method"], _pool_from_variant(row.get("variant", ""), row.get("iv_pool", "")), row["final_model"])
        counts.setdefault(key, {})[row["feature"]] = int(round(float(row["selection_count"])))
        meta.setdefault(key, row)
    for row in _rows(COMBINATION_SUMMARY):
        key = (row["dataset"], row["method"], _pool_from_variant(row.get("variant", ""), row.get("iv_pool", "")), row["final_model"])
        meta.setdefault(key, row)
    out: dict[tuple[str, str, int | None, str], FrozenCombination] = {}
    for key, row in meta.items():
        fold_counts = tuple(_pipe_ints(row.get("fold_selected_feature_counts")))
        selection_counts = counts.get(key, {})
        nog = None
        if selection_counts and fold_counts:
            nog = nogueira_from_frequencies(list(selection_counts.values()), list(fold_counts), universe[key[0]])
        out[key] = FrozenCombination(
            dataset=key[0],
            method=key[1],
            iv_pool=key[2],
            model=key[3],
            fold_counts=fold_counts,
            jaccard_mean=_float(row.get("mean_pairwise_jaccard")),
            jaccard_min=_float(row.get("min_pairwise_jaccard")),
            jaccard_max=_float(row.get("max_pairwise_jaccard")),
            nogueira=nog,
            selection_counts=selection_counts,
            source=f"{rel(COMBINATION_SUMMARY)}; {rel(COMBINATION_SELECTION_STABILITY)}",
        )
    return out


@lru_cache(maxsize=None)
def combination_fold_support() -> dict[tuple[str, str, int | None, int], dict[str, Any]]:
    """(dataset, method, iv_pool, fold_id) -> realized count, stage counts, sha256 (fold fits are model independent)."""

    out: dict[tuple[str, str, int | None, int], dict[str, Any]] = {}
    for row in _rows(COMBINATION_STAGE_SUPPORT):
        pool = _pool_from_variant(row.get("variant", ""), "")
        key = (row["dataset"], row["method"], pool, int(float(row["fold_id"])))
        out[key] = {
            "realized_selected_count": _int(row.get("realized_selected_count")),
            "stage_1_count": _int(row.get("stage_1_count")),
            "stage_2_count": _int(row.get("stage_2_count")),
            "selected_features_sha256": row.get("selected_features_sha256"),
            "requested_k": _int(row.get("requested_k")),
            "fit_seconds": _float(row.get("fit_seconds")),
            "source": rel(COMBINATION_STAGE_SUPPORT),
        }
    return out


# ------------------------------------------------------ canonical OOT table


@lru_cache(maxsize=None)
def final_table() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """(dataset, method, configuration, model) -> canonical two-dataset OOT row."""

    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _rows(FINAL_TABLE):
        key = (row["dataset"], row["method"], row["configuration"], row["model"])
        out[key] = {
            "requested_k": _int(row.get("requested_k")),
            "realized_k": _int(row.get("realized_k")),
            "reference_natural_support_k": _int(row.get("reference_natural_support_k")),
            "support_status": row.get("support_status"),
            "oot_auc": _float(row.get("oot_auc")),
            "oot_brier": _float(row.get("oot_brier")),
            "nogueira_stability": _float(row.get("nogueira_stability")),
            "mean_pairwise_jaccard": _float(row.get("mean_pairwise_jaccard")),
            "feature_psi_mean": _float(row.get("feature_psi_mean")),
            "fit_seconds": _float(row.get("fit_seconds")),
            "source": rel(FINAL_TABLE),
        }
    return out


def final_table_row(dataset: str, method: str, model: str, configuration: str | None = None) -> dict[str, Any] | None:
    table = final_table()
    if configuration is not None:
        return table.get((dataset, method, configuration, model))
    for key, row in table.items():
        if key[0] == dataset and key[1] == method and key[3] == model:
            return row
    return None


# -------------------------------------------------------------- third dataset


@lru_cache(maxsize=None)
def third_oot_registry() -> dict[int, dict[str, Any]]:
    """configuration_order -> registry cell (method_id, model, iv_pool_budget, budget)."""

    if not THIRD_OOT_REGISTRY.exists():
        return {}
    payload = json.loads(THIRD_OOT_REGISTRY.read_text(encoding="utf-8"))
    cells: dict[int, dict[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "configuration_order" in value and "method_id" in value:
                cells[int(value["configuration_order"])] = dict(value)
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return cells


@lru_cache(maxsize=None)
def third_dev_accounting() -> dict[tuple[str, str, int | None], dict[int, dict[str, Any]]]:
    """(method_id, model, iv_pool_budget) -> {fold_id: {count, status, sha256}} for the frozen third-dataset DEV phase."""

    registry = third_oot_registry()
    out: dict[tuple[str, str, int | None], dict[int, dict[str, Any]]] = {}
    for row in _rows(THIRD_DEV_ACCOUNTING):
        order = _int(row.get("configuration_order"))
        cell = registry.get(order or -1, {})
        pool = cell.get("iv_pool_budget")
        pool = int(pool) if pool not in (None, "", "null") else None
        key = (row["method_id"], row["model"], pool)
        out.setdefault(key, {})[int(float(row["fold_id"]))] = {
            "selected_feature_count": _int(row.get("selected_feature_count")),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "selected_features_sha256": row.get("selected_features_sha256"),
            "configuration_order": order,
            "source": rel(THIRD_DEV_ACCOUNTING),
        }
    return out


@lru_cache(maxsize=None)
def third_pilot_fit_seconds() -> dict[tuple[str, int | None], float]:
    """(method_id, requested_feature_budget) -> fold-1 pilot fit seconds on the frozen machine."""

    out: dict[tuple[str, int | None], float] = {}
    for row in _rows(THIRD_PILOT_ACCOUNTING):
        if row.get("record_kind") != "selection_fit":
            continue
        seconds = _float(row.get("fit_seconds"))
        if seconds is None:
            continue
        out[(row["method_id"], _int(row.get("requested_feature_budget")))] = seconds
    return out


# ------------------------------------------------------- legacy 8-arm reports


def _markdown_tables(path: Path) -> list[tuple[str, list[dict[str, str]]]]:
    """Return (section heading, rows) for every pipe table in a markdown file."""

    if not path.exists():
        return []
    tables: list[tuple[str, list[dict[str, str]]]] = []
    heading = ""
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            if header and rows:
                tables.append((heading, rows))
            heading = line.lstrip("#").strip()
            header, rows = None, []
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
                continue
            rows.append({header[i]: cells[i] if i < len(cells) else "" for i in range(len(header))})
        else:
            if header and rows:
                tables.append((heading, rows))
            header, rows = None, []
    if header and rows:
        tables.append((heading, rows))
    return tables


@lru_cache(maxsize=None)
def legacy_report_values() -> dict[tuple[str, str, str], dict[str, Any]]:
    """(dataset, model, legacy selector key) -> stability/drift numbers from the archive reports.

    Home Credit comes from ``reports/archive/homecredit_report.md`` (full 8-arm
    tables).  LendingClub v2 legacy numbers only survive in the best-run tables of
    ``reports/archive/cross_dataset_v2_analysis.md``; the v1 LendingClub report is
    deliberately ignored because the paper uses the v2 line.
    """

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for heading, rows in _markdown_tables(HOMECREDIT_REPORT):
        if heading.startswith("Stability Review"):
            for row in rows:
                key = (HOMECREDIT, row.get("model", ""), row.get("selector", ""))
                out.setdefault(key, {}).update(
                    {
                        "nogueira_stability": _float(row.get("nogueira_stability")),
                        "mean_pairwise_jaccard": _float(row.get("mean_pairwise_jaccard")),
                        "stability_source": rel(HOMECREDIT_REPORT),
                    }
                )
        elif heading.startswith("Drift and Robustness Review"):
            for row in rows:
                key = (HOMECREDIT, row.get("model", ""), row.get("selector", ""))
                out.setdefault(key, {}).update(
                    {
                        "selected_feature_count": _int(row.get("selected_feature_count")),
                        "psi_mean": _float(row.get("psi_mean")),
                        "psi_median": _float(row.get("psi_median")),
                        "psi_max": _float(row.get("psi_max")),
                        "drift_source": rel(HOMECREDIT_REPORT),
                    }
                )
    for heading, rows in _markdown_tables(CROSS_DATASET_V2_REPORT):
        for row in rows:
            if "dataset_name" not in row or "selector" not in row or "model" not in row:
                continue
            if "nogueira_stability" not in row:
                continue
            key = (row["dataset_name"], row["model"], row["selector"])
            entry = out.setdefault(key, {})
            entry.setdefault("nogueira_stability", _float(row.get("nogueira_stability")))
            entry.setdefault("mean_pairwise_jaccard", _float(row.get("mean_pairwise_jaccard")))
            if _float(row.get("selected_feature_psi_mean")) is not None:
                entry.setdefault("psi_mean", _float(row.get("selected_feature_psi_mean")))
            entry.setdefault("stability_source", rel(CROSS_DATASET_V2_REPORT))
    return out


LEGACY_KEY_FOR_METHOD = {
    "domain_rule_baseline": "domain_rule_baseline",
    "pca": "pca",
    "mrmr_legacy": "mrmr",
    "llm": "llm",
    "llm_then_mrmr": "llm_then_mrmr",
    "llm_then_boruta": "llm_then_boruta",
    "stable_core_llm_fill": "stable_core_llm_fill",
}


def legacy_values(dataset: str, model: str, method: str) -> dict[str, Any] | None:
    key = LEGACY_KEY_FOR_METHOD.get(method)
    if key is None:
        return None
    return legacy_report_values().get((dataset, model, key))


def collapse_model_matrix_columns(columns: Iterable[str], universe: Iterable[str]) -> list[str]:
    """Map one-hot model-matrix names (``CODE_GENDER_F``) back to raw candidate names.

    Order of first appearance is preserved so a frozen PSI table can be turned
    back into the raw full-DEV selected set.
    """

    universe_set = set(universe)
    longest_first = sorted(universe_set, key=len, reverse=True)
    out: list[str] = []
    for column in columns:
        if column in universe_set:
            base = column
        else:
            base = next((name for name in longest_first if column.startswith(name + "_")), None)
            if base is None:
                base = column
        if base not in out:
            out.append(base)
    return out
