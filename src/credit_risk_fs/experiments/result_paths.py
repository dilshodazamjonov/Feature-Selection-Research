"""Canonical paths and registry operations for active experiment results."""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Mapping


RESULT_SUBDIRECTORIES = (
    "runs",
    "comparisons",
    "figures",
    "final_package",
)

RUN_INDEX_COLUMNS = (
    "run_id",
    "dataset",
    "selector",
    "model",
    "split_protocol",
    "seed",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "runtime_seconds",
    "peak_ram_mb",
    "peak_gpu_mb",
    "run_directory",
    "config_path",
    "manifest_path",
    "notes",
)

DEFAULT_RESULTS_README = """# Active experiment results

This directory contains only new active experiments. Historical finalized
outputs are stored separately and remain immutable.

- `runs/`: isolated runs at `runs/<dataset>/<YYYY-MM-DD_selector_model>/`.
- `comparisons/`: cross-method and cross-dataset comparison tables.
- `figures/`: validated figures built from saved run outputs.
- `final_package/`: curated manuscript and report deliverables.
- `run_index.csv`: active experiments only.

Runs never overwrite one another. A completed run normally records its config,
manifest, selected features, fold selections, metrics, DEV/OOT predictions,
stability, resource usage, and log; the manifest identifies applicable and
present artifacts. Raw data, source code, notebooks, model caches, scratch
files, temporary files, and legacy artifacts do not belong here.
"""

_SAFE_COMPONENT_PATTERN = re.compile(r"[^a-z0-9-]+")


class RunDirectoryCollisionError(FileExistsError):
    """Raised when a requested active run directory already exists."""


def sanitize_component(value: object, *, field_name: str = "path component") -> str:
    """Return a portable lowercase slug that cannot introduce path traversal."""

    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} must not be empty")
    ascii_value = (
        unicodedata.normalize("NFKD", raw)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    sanitized = _SAFE_COMPONENT_PATTERN.sub("_", ascii_value).strip("_-")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"{field_name} has no usable characters: {raw!r}")
    return sanitized


def build_run_id(
    *,
    selector: object,
    model: object,
    run_date: date | str | None = None,
) -> str:
    """Build a readable, deterministic base run ID for one calendar day."""

    if run_date is None:
        date_text = date.today().isoformat()
    elif isinstance(run_date, date):
        date_text = run_date.isoformat()
    else:
        try:
            date_text = date.fromisoformat(str(run_date)).isoformat()
        except ValueError as exc:
            raise ValueError("run_date must use YYYY-MM-DD format") from exc
    return "_".join(
        (
            date_text,
            sanitize_component(selector, field_name="selector"),
            sanitize_component(model, field_name="model"),
        )
    )


def resolve_results_root(
    repository_root: str | Path,
    configured_results_root: str | Path = "results",
) -> Path:
    """Resolve a configured results root against an explicit repository root."""

    repository = Path(repository_root).resolve()
    configured = Path(configured_results_root)
    candidate = configured if configured.is_absolute() else repository / configured
    return candidate.resolve()


def initialize_results_layout(
    repository_root: str | Path,
    *,
    results_root: str | Path = "results",
) -> Path:
    """Create the active experiment-results layout without overwriting files."""

    resolved_results_root = resolve_results_root(repository_root, results_root)
    resolved_results_root.mkdir(parents=True, exist_ok=True)

    for directory_name in RESULT_SUBDIRECTORIES:
        (resolved_results_root / directory_name).mkdir(parents=True, exist_ok=True)

    readme_path = resolved_results_root / "README.md"
    if not readme_path.exists():
        try:
            with readme_path.open("x", encoding="utf-8", newline="") as file:
                file.write(DEFAULT_RESULTS_README)
        except FileExistsError:
            pass

    run_index_path = resolved_results_root / "run_index.csv"
    if not run_index_path.exists():
        try:
            with run_index_path.open("x", encoding="utf-8", newline="") as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerow(RUN_INDEX_COLUMNS)
        except FileExistsError:
            pass

    return resolved_results_root


def ensure_within_directory(path: str | Path, directory: str | Path) -> Path:
    """Resolve *path* and reject it unless it stays within *directory*."""

    resolved_directory = Path(directory).resolve()
    candidate = Path(path)
    resolved_path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_directory / candidate).resolve()
    )
    try:
        resolved_path.relative_to(resolved_directory)
    except ValueError as exc:
        raise ValueError(
            f"path escapes configured directory {resolved_directory}: {resolved_path}"
        ) from exc
    return resolved_path


def planned_run_directory(
    results_root: str | Path,
    *,
    dataset: object,
    run_id: object,
) -> Path:
    """Return a validated active-run path without creating it."""

    root = Path(results_root).resolve()
    runs_root = root / "runs"
    dataset_name = sanitize_component(dataset, field_name="dataset")
    safe_run_id = sanitize_component(run_id, field_name="run_id")
    return ensure_within_directory(
        runs_root / dataset_name / safe_run_id,
        runs_root,
    )


def create_run_directory(
    results_root: str | Path,
    *,
    dataset: object,
    run_id: object,
    collision_policy: str = "error",
) -> Path:
    """Atomically create a run directory, rejecting or suffixing collisions."""

    if collision_policy not in {"error", "suffix"}:
        raise ValueError("collision_policy must be 'error' or 'suffix'")

    requested = planned_run_directory(
        results_root,
        dataset=dataset,
        run_id=run_id,
    )
    requested.parent.mkdir(parents=True, exist_ok=True)
    candidate = requested
    collision_number = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError as exc:
            if collision_policy == "error":
                raise RunDirectoryCollisionError(
                    f"active run directory already exists: {candidate}"
                ) from exc
            collision_number += 1
            candidate = requested.with_name(f"{requested.name}_{collision_number:02d}")


def repository_relative_path(path: str | Path, repository_root: str | Path) -> str:
    """Return a portable repository-relative path and reject external paths."""

    repository = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {candidate}") from exc
    return relative.as_posix()


def _read_run_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or tuple(reader.fieldnames) != RUN_INDEX_COLUMNS:
            raise ValueError(
                f"run index header mismatch at {path}; expected {RUN_INDEX_COLUMNS}"
            )
        return list(reader)


def append_run_index_row(
    results_root: str | Path,
    row: Mapping[str, Any],
) -> None:
    """Append one unique active run without changing existing registry rows."""

    index_path = Path(results_root).resolve() / "run_index.csv"
    rows = _read_run_index(index_path)
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run index row requires run_id")
    if any(existing["run_id"] == run_id for existing in rows):
        raise ValueError(f"run index already contains run_id: {run_id}")
    unknown = set(row) - set(RUN_INDEX_COLUMNS)
    if unknown:
        raise ValueError(f"unknown run index columns: {sorted(unknown)}")

    with index_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RUN_INDEX_COLUMNS,
            lineterminator="\n",
        )
        writer.writerow(
            {
                column: "" if row.get(column) is None else row.get(column, "")
                for column in RUN_INDEX_COLUMNS
            }
        )


def update_run_index_row(
    results_root: str | Path,
    run_id: str,
    updates: Mapping[str, Any],
) -> None:
    """Update exactly one existing active run row using an atomic replacement."""

    index_path = Path(results_root).resolve() / "run_index.csv"
    rows = _read_run_index(index_path)
    unknown = set(updates) - set(RUN_INDEX_COLUMNS)
    if unknown:
        raise ValueError(f"unknown run index columns: {sorted(unknown)}")
    if "run_id" in updates and str(updates["run_id"]) != run_id:
        raise ValueError("run_id cannot be changed")

    matches = [row for row in rows if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one run index row for {run_id!r}, found {len(matches)}"
        )
    for row in rows:
        if row["run_id"] == run_id:
            row.update(
                {
                    key: "" if value is None else str(value)
                    for key, value in updates.items()
                }
            )

    temporary_path = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RUN_INDEX_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, index_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
