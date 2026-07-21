"""Canonical paths and registry operations for active experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
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
LEGACY_RESULTS_ENV = "CREDIT_RISK_LEGACY_RESULTS_ROOT"
LEGACY_EVIDENCE_PROFILE_ENV = "CREDIT_RISK_LEGACY_EVIDENCE_PROFILE"
LEGACY_EVIDENCE_PROFILE_FILENAME = "legacy_evidence_profile.json"
CLIP_COMPLETE_PROFILE = "clip_complete_v1"
CLIP_REQUIRED_ARTIFACT_GROUPS = (
    "clip_readiness",
    "clip_text_baseline",
    "clip_v2_statistical_view",
    "corrected_homecredit_clip_contrastive",
    "corrected_homecredit_clip_training",
)


class RunDirectoryCollisionError(FileExistsError):
    """Raised when a requested active run directory already exists."""


class LegacyResultsConfigurationError(ValueError):
    """Raised when the active and historical result roots are unsafe."""


class HistoricalResultsWriteError(PermissionError):
    """Raised before a repository writer can enter the historical bundle."""


class LegacyEvidenceProfileError(ValueError):
    """Raised when a bundle declares evidence that does not validate."""


def configured_legacy_results_root(
    configured_root: str | Path | None = None,
    *,
    required: bool = False,
) -> Path | None:
    """Resolve the explicit read-only historical root without using the CWD."""

    value = configured_root
    if value is None:
        value = os.environ.get(LEGACY_RESULTS_ENV)
    if value is None or not str(value).strip():
        if required:
            raise LegacyResultsConfigurationError(
                f"historical evidence requires {LEGACY_RESULTS_ENV}"
            )
        return None
    supplied = Path(str(value).strip())
    if not supplied.is_absolute():
        raise LegacyResultsConfigurationError(
            f"{LEGACY_RESULTS_ENV} must be an absolute path: {supplied}"
        )
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise LegacyResultsConfigurationError(
            f"configured historical results root is not a directory: {resolved}"
        )
    return resolved


def validate_results_root_separation(
    active_root: str | Path,
    legacy_root: str | Path,
    *,
    forbidden_legacy_roots: tuple[str | Path, ...] = (),
) -> tuple[Path, Path]:
    """Require independent, non-overlapping active and historical roots."""

    active = Path(active_root).resolve()
    legacy = configured_legacy_results_root(legacy_root, required=True)
    assert legacy is not None
    if active == legacy or active.is_relative_to(legacy) or legacy.is_relative_to(active):
        raise LegacyResultsConfigurationError(
            f"active and historical results roots overlap: active={active}, legacy={legacy}"
        )
    for forbidden_root in forbidden_legacy_roots:
        forbidden = Path(forbidden_root).resolve()
        if legacy == forbidden or legacy.is_relative_to(forbidden):
            raise LegacyResultsConfigurationError(
                f"historical results root resolves inside a forbidden data/test root: {legacy}"
            )
    return active, legacy


def reject_historical_write(
    path: str | Path,
    *,
    legacy_root: str | Path | None = None,
) -> Path:
    """Resolve a prospective write and reject the historical bundle boundary."""

    resolved = Path(path).resolve()
    legacy = configured_legacy_results_root(legacy_root)
    if legacy is not None and (resolved == legacy or resolved.is_relative_to(legacy)):
        raise HistoricalResultsWriteError(
            f"write rejected inside immutable historical results root {legacy}: {resolved}"
        )
    return resolved


def resolve_legacy_artifact(
    former_repository_path: str | Path,
    *,
    legacy_root: str | Path | None = None,
    required: bool = True,
) -> Path:
    """Map a former ``results/...`` read to the explicitly configured bundle."""

    root = configured_legacy_results_root(legacy_root, required=True)
    assert root is not None
    normalized = str(former_repository_path).replace("\\", "/")
    former = Path(normalized)
    if former.is_absolute() or ".." in former.parts:
        raise LegacyResultsConfigurationError(
            f"legacy artifact path must be a traversal-free repository path: {former}"
        )
    parts = former.parts[1:] if former.parts and former.parts[0].lower() == "results" else former.parts
    candidate = (root.joinpath(*parts)).resolve()
    if not candidate.is_relative_to(root):
        raise LegacyResultsConfigurationError(
            f"legacy artifact escapes configured root: {candidate}"
        )
    if required and not candidate.is_file():
        raise FileNotFoundError(
            f"configured historical evidence is missing required artifact: {candidate}"
        )
    return candidate


def validate_legacy_evidence_profile(
    legacy_root: str | Path,
    *,
    required_profile: str = CLIP_COMPLETE_PROFILE,
) -> dict[str, Any] | None:
    """Validate a declared complete legacy evidence bundle, or report absence."""

    root = configured_legacy_results_root(legacy_root, required=True)
    assert root is not None
    declared_by_environment = os.environ.get(LEGACY_EVIDENCE_PROFILE_ENV)
    profile_path = root / LEGACY_EVIDENCE_PROFILE_FILENAME
    if not profile_path.is_file():
        if declared_by_environment:
            raise LegacyEvidenceProfileError(
                f"{LEGACY_EVIDENCE_PROFILE_ENV} declares {declared_by_environment!r} "
                f"but {profile_path} is missing"
            )
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyEvidenceProfileError(
            f"legacy evidence profile cannot be parsed: {profile_path}"
        ) from exc
    profile_name = payload.get("profile")
    if declared_by_environment and declared_by_environment != profile_name:
        raise LegacyEvidenceProfileError(
            "legacy evidence environment/profile manifest mismatch"
        )
    if profile_name != required_profile:
        return None
    if payload.get("status") in {"incomplete", "unavailable"}:
        return None
    if payload.get("status") != "complete":
        raise LegacyEvidenceProfileError(
            f"legacy evidence profile {required_profile} has an invalid status"
        )
    groups = payload.get("artifact_groups")
    if not isinstance(groups, dict):
        raise LegacyEvidenceProfileError("legacy evidence profile artifact_groups is invalid")
    missing_groups = set(CLIP_REQUIRED_ARTIFACT_GROUPS) - set(groups)
    if missing_groups:
        raise LegacyEvidenceProfileError(
            f"legacy evidence profile omits required groups: {sorted(missing_groups)}"
        )
    verified_artifacts = 0
    for group in CLIP_REQUIRED_ARTIFACT_GROUPS:
        artifacts = groups[group]
        if not isinstance(artifacts, list) or not artifacts:
            raise LegacyEvidenceProfileError(
                f"legacy evidence group {group} must declare at least one artifact"
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
                raise LegacyEvidenceProfileError(
                    f"legacy evidence group {group} has an invalid artifact declaration"
                )
            relative = Path(str(artifact["path"]).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise LegacyEvidenceProfileError(
                    f"legacy evidence artifact path is unsafe: {relative}"
                )
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise LegacyEvidenceProfileError(
                    f"declared legacy evidence artifact is missing: {candidate}"
                )
            expected_hash = str(artifact["sha256"]).lower()
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise LegacyEvidenceProfileError(
                    f"declared legacy evidence artifact hash mismatch: {candidate}"
                )
            verified_artifacts += 1
    result = dict(payload)
    result["profile_path"] = str(profile_path)
    result["verified_artifact_count"] = verified_artifacts
    return result


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
    resolved = candidate.resolve()
    legacy = configured_legacy_results_root()
    if legacy is not None:
        validate_results_root_separation(resolved, legacy)
    return reject_historical_write(resolved, legacy_root=legacy)


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

    resolved_directory = reject_historical_write(directory)
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
    return reject_historical_write(resolved_path)


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

    index_path = reject_historical_write(
        Path(results_root).resolve() / "run_index.csv"
    )
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

    index_path = reject_historical_write(
        Path(results_root).resolve() / "run_index.csv"
    )
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

    temporary_path = reject_historical_write(
        index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    )
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
