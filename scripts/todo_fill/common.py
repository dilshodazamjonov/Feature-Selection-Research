"""Shared constants, label mapping, manifest, CSV and resource helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _candidate in (REPO_ROOT, SRC_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

HOMECREDIT = "homecredit"
LENDINGCLUB = "lendingclub_v2"
THIRD = "homecredit_model_stability_2024"
DATASETS: tuple[str, ...] = (HOMECREDIT, LENDINGCLUB, THIRD)
BACKBONES: tuple[str, ...] = ("lr", "catboost")
BUDGETS: dict[str, int] = {"lr": 20, "catboost": 40}
FOLDS: tuple[str, ...] = ("fold1", "fold2", "fold3", "fold4", "fold5")
FULL_DEV = "full_dev"
PARTITIONS: tuple[str, ...] = (*FOLDS, FULL_DEV)
SEED = 42

#: Candidate-universe size used as the Nogueira denominator ``d`` for refitted
#: selectors (post-feature-engineering, before any selector).  The four pure LLM
#: rows in ``10_stability.csv`` carry their own pre-filled ``d`` (373/675).
UNIVERSE_SIZE: dict[str, int] = {HOMECREDIT: 529, LENDINGCLUB: 675, THIRD: 1959}

#: LLM candidate-pool budget handed to the statistical stage of the legacy
#: ``llm_then_*`` hybrids (``configs/base.yaml`` -> ``llm.ranking_budget``).
LLM_POOL_BUDGET: dict[str, int] = {"lr": 60, "catboost": 100}
LLM_SHARED_POOL_SIZE = 100

#: ``IV then Boruta`` is configuration specific.  ``scripts/recompute_pairwise_csv.py``
#: maps the paper's Home Credit rows to the registered pool-100 runs and the
#: LendingClub rows to the registered pool-300 runs.  The third dataset's frozen
#: primary pool is 200 (``iv_pool_primary`` in the protocol lock).
IV_THEN_BORUTA_POOL: dict[str, int] = {HOMECREDIT: 100, LENDINGCLUB: 300, THIRD: 200}

#: Paper label -> internal method key.  ``mRMR`` is dataset specific: the two
#: original datasets used the legacy RF-relevance/correlation selector
#: (registry alias ``mrmr``); the third dataset's frozen matrix used canonical
#: mutual-information mRMR (``mrmr_mutual_information``).
LABEL_TO_METHOD: dict[str, str | dict[str, str]] = {
    "Domain rules": "domain_rule_baseline",
    "Random K": "random_k",
    "PCA": "pca",
    "IV/WOE": "iv_woe",
    "mRMR": {HOMECREDIT: "mrmr_legacy", LENDINGCLUB: "mrmr_legacy", THIRD: "mrmr_mutual_information"},
    "Boruta RF": "boruta_random_forest",
    "IV then Boruta": "iv_then_boruta",
    "RFE CatBoost": "rfe_catboost",
    "CatBoost SHAP": "catboost_shap",
    "LLM then mRMR": "llm_then_mrmr",
    "LLM then Boruta": "llm_then_boruta",
    "Stable core + LLM fill": "stable_core_llm_fill",
    "CLIP-ranked": "clip_ranked",
    "Pure LLM": "llm",
    "L1 logistic regression": "lasso_l1_logistic",
    "Full feature reference": "full_features",
}

#: Which experimental line each method belongs to.  ``legacy`` = the original
#: 8-arm matrix (selector fitted after the dense ``Preprocessor`` unless the
#: selector opts into raw selection), ``baseline`` = full_baseline_v1 (contract
#: selectors on the ``OriginalFeatureNumericEncoder`` frame), ``combination`` =
#: selector_combinations_v1, ``cache`` = deterministic truncation of a cached
#: LLM ranking, ``analytic`` = value known by construction.
METHOD_LINE: dict[str, str] = {
    "domain_rule_baseline": "legacy",
    "pca": "legacy",
    "mrmr_legacy": "legacy",
    "llm_then_mrmr": "legacy",
    "llm_then_boruta": "legacy",
    "stable_core_llm_fill": "legacy",
    "llm": "cache",
    "random_k": "baseline",
    "iv_woe": "baseline",
    "mrmr_mutual_information": "baseline",
    "lasso_l1_logistic": "baseline",
    "boruta_random_forest": "baseline",
    "rfe_catboost": "baseline",
    "catboost_shap": "baseline",
    "full_features": "baseline",
    "iv_then_boruta": "combination",
    "clip_ranked": "unsupported",
}

BORUTA_STAGE_LABELS: tuple[str, ...] = ("Boruta RF", "IV then Boruta", "LLM then Boruta")

COPY_VERBATIM: tuple[str, ...] = ("15_cohort.csv", "16_l1.csv", "21_brier.csv")

DESCRIPTION_PATH: dict[str, Path] = {
    HOMECREDIT: REPO_ROOT / "data/homecredit/metadata/columns_description.csv",
    LENDINGCLUB: REPO_ROOT / "data/lendingclub_v2/metadata/columns_description.csv",
}
PREPROCESSOR_KWARGS: dict[str, dict[str, Any]] = {
    HOMECREDIT: {},
    LENDINGCLUB: {"cat_min_frequency": 50},
}


class FillError(RuntimeError):
    """A step could not be completed; recorded in the manifest, never fatal."""


class ResourceSkip(FillError):
    """A fit was skipped because this machine cannot afford it."""


def method_for(label: str, dataset: str) -> str:
    value = LABEL_TO_METHOD[label]
    if isinstance(value, dict):
        return value[dataset]
    return value


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Same contract as ``credit_risk_fs.data.homecredit_model_stability_2024.contract``."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def fmt(value: Any, digits: int = 6) -> str:
    """Format a cell for the CSVs: blanks stay blank, ints stay ints."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    if number.is_integer() and abs(number) < 1e12 and digits >= 0 and isinstance(value, (int,)):
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{number}"


# --------------------------------------------------------------------------- CSV


@dataclass
class Skeleton:
    name: str
    columns: list[str]
    rows: list[dict[str, str]]

    @classmethod
    def read(cls, path: Path) -> "Skeleton":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [
                {column: (row.get(column) or "").strip() for column in columns}
                for row in reader
                if any((value or "").strip() for value in row.values())
            ]
        return cls(name=path.name, columns=columns, rows=rows)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, lineterminator="\n")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({column: row.get(column, "") for column in self.columns})
        os.replace(tmp, path)

    def fill_rate(self, value_columns: Sequence[str]) -> tuple[int, int]:
        filled = 0
        total = 0
        for row in self.rows:
            for column in value_columns:
                total += 1
                if row.get(column, "") != "":
                    filled += 1
        return filled, total


# ---------------------------------------------------------------------- manifest


@dataclass
class Manifest:
    """Provenance for every filled cell plus every skip, in one JSON + one markdown."""

    output_dir: Path
    started_at: float = field(default_factory=time.time)
    cells: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    skips: list[dict[str, Any]] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    csv_summary: dict[str, Any] = field(default_factory=dict)

    def cell(self, step: str, key: Mapping[str, Any], field_name: str, value: Any, source: str, **extra: Any) -> None:
        self.cells.append({"step": step, "key": dict(key), "field": field_name, "value": value, "source": source, **extra})

    def note(self, step: str, message: str, **extra: Any) -> None:
        self.notes.append({"step": step, "message": message, **extra})
        log(f"[{step}] {message}")

    def skip(self, step: str, key: Mapping[str, Any] | None, reason: str, **extra: Any) -> None:
        self.skips.append({"step": step, "key": dict(key or {}), "reason": reason, **extra})
        log(f"[{step}] SKIP {dict(key or {})}: {reason}")

    def timing(self, label: str, seconds: float, **extra: Any) -> None:
        self.timings.append({"label": label, "seconds": float(seconds), **extra})

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - self.started_at,
            "settings": self.settings,
            "csv_summary": self.csv_summary,
            "cells": self.cells,
            "skips": self.skips,
            "notes": self.notes,
            "timings": self.timings,
        }
        write_json(self.output_dir / "fill_manifest.json", payload)
        lines = ["# todo_done fill log", ""]
        lines.append(f"Generated {payload['generated_at_utc']} after {payload['elapsed_seconds']/60:.1f} minutes.")
        lines.append("")
        lines.append("## CSV fill summary")
        lines.append("")
        lines.append("| file | filled cells | total value cells | status |")
        lines.append("|---|---|---|---|")
        for name, summary in sorted(self.csv_summary.items()):
            lines.append(f"| {name} | {summary.get('filled','')} | {summary.get('total','')} | {summary.get('status','')} |")
        lines.append("")
        if self.skips:
            lines.append("## Skipped or blank cells")
            lines.append("")
            for item in self.skips:
                lines.append(f"- **{item['step']}** {item['key']}: {item['reason']}")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for item in self.notes:
                lines.append(f"- **{item['step']}**: {item['message']}")
            lines.append("")
        source_counts: dict[str, int] = {}
        for cell in self.cells:
            source_counts[cell["source"].split(":")[0]] = source_counts.get(cell["source"].split(":")[0], 0) + 1
        lines.append("## Cell sources")
        lines.append("")
        for source, count in sorted(source_counts.items()):
            lines.append(f"- {source}: {count} cells")
        (self.output_dir / "fill_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- logging

_LOG_PATH: Path | None = None


def configure_log(path: Path) -> None:
    global _LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = path


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@contextmanager
def heartbeat(label: str, every_seconds: float = 300.0):
    """Print ``label`` with the elapsed minutes every ``every_seconds`` while the block runs.

    Long selector fits and data loads are single blocking calls; this keeps the terminal
    showing that the process is alive and what it is doing.
    """

    stop = threading.Event()
    started = time.perf_counter()

    def _beat() -> None:
        while not stop.wait(every_seconds):
            log(f"[still running] {label}: {(time.perf_counter() - started) / 60:.0f} min elapsed")

    thread = threading.Thread(target=_beat, name="fill-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self.start


# --------------------------------------------------------------------- resources


def available_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:  # pragma: no cover - psutil is a project dependency
        return float("inf")


def frame_gib(rows: int, columns: int, bytes_per_cell: int = 4) -> float:
    return rows * columns * bytes_per_cell / 2**30


def require_ram(needed_gb: float, label: str) -> None:
    available = available_ram_gb()
    if available < needed_gb:
        raise ResourceSkip(
            f"{label}: needs about {needed_gb:.1f} GiB but only {available:.1f} GiB is available"
        )


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # pragma: no cover
        pass


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def nogueira(sets: Sequence[Sequence[str]], d: int) -> float | None:
    """Nogueira stability with universe size ``d`` (same algebra as the repo)."""

    m = len(sets)
    if m < 2 or d <= 1:
        return None
    sizes = [len(set(s)) for s in sets]
    k_bar = sum(sizes) / m
    if k_bar <= 0 or k_bar >= d:
        return 1.0
    features = set().union(*(set(s) for s in sets))
    variance = 0.0
    for feature in features:
        p = sum(feature in set(s) for s in sets) / m
        variance += (m / (m - 1)) * p * (1 - p)
    observed = variance / d
    expected = (k_bar / d) * (1 - k_bar / d)
    return 1 - observed / expected


def nogueira_from_frequencies(counts: Sequence[int], fold_sizes: Sequence[int], d: int) -> float | None:
    """Nogueira from per-feature selection counts (frozen frequency tables)."""

    m = len(fold_sizes)
    if m < 2 or d <= 1:
        return None
    k_bar = sum(fold_sizes) / m
    if k_bar <= 0 or k_bar >= d:
        return 1.0
    variance = 0.0
    for count in counts:
        p = count / m
        variance += (m / (m - 1)) * p * (1 - p)
    return 1 - (variance / d) / ((k_bar / d) * (1 - k_bar / d))


def pairwise_jaccard_summary(sets: Sequence[Sequence[str]]) -> dict[str, float | None]:
    import itertools

    values = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
