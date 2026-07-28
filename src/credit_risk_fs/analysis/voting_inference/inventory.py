"""Run discovery, input inventory, and Prompt 5 completion authentication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from credit_risk_fs.analysis.voting_inference.config import (
    AnalysisConfig,
    AuthenticationError,
    read_json,
)
from credit_risk_fs.utils.hashing import sha256_file

RUN_ID_PATTERN = re.compile(
    r"^cdv1-(?P<order>\d{3})-(?P<dataset>homecredit|lendingclub-v2)-"
    r"(?P<configuration>reference-rf-corr-mrmr|voting-k\d+)-"
    r"(?P<model>lr|catboost)-s(?P<seed>\d+)$"
)
DATASET_FROM_RUN_ID = {"homecredit": "homecredit", "lendingclub-v2": "lendingclub_v2"}

DEV_PREDICTION_RELATIVE = "results/dev_predictions.csv"
OOT_PREDICTION_RELATIVE = "results/oot_predictions.csv"
DEV_METADATA_RELATIVE = "results/dev_prediction_metadata.json"
OOT_METADATA_RELATIVE = "results/oot_prediction_metadata.json"
FOLD_SELECTION_RELATIVE = "folds/fold_{fold}/selected_features.csv"
FOLD_CANDIDATE_RELATIVE = "folds/fold_{fold}/candidate_features.csv"
FOLD_VOTER_RELATIVE = "folds/fold_{fold}/voter_rankings.csv"
FOLD_DEV_PREDICTION_RELATIVE = "folds/fold_{fold}/predictions_dev.csv"
FINAL_SELECTION_RELATIVE = "features/final_selected_features.csv"


@dataclass(frozen=True)
class RunRecord:
    """One authenticated completed run, treated as an immutable input."""

    run_id: str
    dataset: str
    model: str
    configuration: str
    candidate_pool_budget: int | None
    arm: str
    designation: str
    comparison_family: str
    directory: Path
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]

    @property
    def dev_predictions(self) -> Path:
        return self.directory / DEV_PREDICTION_RELATIVE

    @property
    def oot_predictions(self) -> Path:
        return self.directory / OOT_PREDICTION_RELATIVE

    @property
    def dev_metadata(self) -> Path:
        return self.directory / DEV_METADATA_RELATIVE

    @property
    def oot_metadata(self) -> Path:
        return self.directory / OOT_METADATA_RELATIVE

    @property
    def final_selection(self) -> Path:
        return self.directory / FINAL_SELECTION_RELATIVE

    def fold_selection(self, fold: int) -> Path:
        return self.directory / FOLD_SELECTION_RELATIVE.format(fold=fold)

    def fold_candidates(self, fold: int) -> Path:
        return self.directory / FOLD_CANDIDATE_RELATIVE.format(fold=fold)

    def fold_voter_rankings(self, fold: int) -> Path:
        return self.directory / FOLD_VOTER_RELATIVE.format(fold=fold)

    def fold_dev_predictions(self, fold: int) -> Path:
        return self.directory / FOLD_DEV_PREDICTION_RELATIVE.format(fold=fold)

    def is_reference(self) -> bool:
        return self.arm == "reference"


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Decompose a frozen run id into its declared configuration facets."""

    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise AuthenticationError(f"run id does not match the frozen pattern: {run_id!r}")
    raw_configuration = match.group("configuration")
    if raw_configuration.startswith("voting-k"):
        configuration = f"voting_k{raw_configuration.removeprefix('voting-k')}"
        budget: int | None = int(raw_configuration.removeprefix("voting-k"))
        arm = "voting"
    else:
        configuration = "reference"
        budget = None
        arm = "reference"
    return {
        "execution_order": int(match.group("order")),
        "dataset": DATASET_FROM_RUN_ID[match.group("dataset")],
        "model": match.group("model"),
        "configuration": configuration,
        "candidate_pool_budget": budget,
        "arm": arm,
        "seed": int(match.group("seed")),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def discover_runs(config: AnalysisConfig) -> list[RunRecord]:
    """Load every registered cross-dataset voting run from the frozen lock."""

    lock = read_json(
        config.repository_root
        / str(config.payload["frozen_inputs"]["configuration_lock"]["path"])
    )
    run_ids = [str(value) for value in lock["run_ids"]]
    if len(run_ids) != len(set(run_ids)):
        raise AuthenticationError("configuration lock contains duplicate run ids")
    expected_total = int(config.expected["total_runs"])
    if len(run_ids) != expected_total:
        raise AuthenticationError(
            f"configuration lock registers {len(run_ids)} runs, expected {expected_total}"
        )
    primary_pool = int(config.expected["primary_candidate_pool"])

    records: list[RunRecord] = []
    for run_id in run_ids:
        facets = parse_run_id(run_id)
        directory = config.run_root / facets["dataset"] / run_id
        if not directory.is_dir():
            raise AuthenticationError(f"registered run directory is missing: {directory}")
        manifest = read_json(directory / "manifest.json")
        run_config = read_json(directory / "config.json")
        for field_name, expected_value in (
            ("dataset", facets["dataset"]),
            ("model", facets["model"]),
        ):
            if str(run_config[field_name]) != expected_value:
                raise AuthenticationError(
                    f"{run_id}: config {field_name}={run_config[field_name]!r} "
                    f"contradicts the run id ({expected_value!r})"
                )
        declared_budget = run_config.get("candidate_pool_budget")
        if declared_budget != facets["candidate_pool_budget"]:
            raise AuthenticationError(
                f"{run_id}: config candidate_pool_budget={declared_budget!r} "
                f"contradicts the run id ({facets['candidate_pool_budget']!r})"
            )
        designation = str(run_config["designation"])
        expected_designation = _expected_designation(
            facets["arm"], facets["candidate_pool_budget"], primary_pool
        )
        if designation != expected_designation:
            raise AuthenticationError(
                f"{run_id}: config designation {designation!r} contradicts the frozen "
                f"structure ({expected_designation!r})"
            )
        records.append(
            RunRecord(
                run_id=run_id,
                dataset=facets["dataset"],
                model=facets["model"],
                configuration=facets["configuration"],
                candidate_pool_budget=facets["candidate_pool_budget"],
                arm=facets["arm"],
                designation=designation,
                comparison_family=str(run_config["comparison_family"]),
                directory=directory,
                manifest=manifest,
                config=run_config,
            )
        )
    return records


def _expected_designation(arm: str, budget: int | None, primary_pool: int) -> str:
    """Map the frozen run structure onto the saved designation vocabulary."""

    if arm == "reference":
        return "reference"
    return "primary" if budget == primary_pool else "sensitivity"


def _prediction_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def build_input_inventory(
    config: AnalysisConfig, runs: list[RunRecord]
) -> pd.DataFrame:
    """Build the per-run input inventory required before metric computation."""

    fold_count = int(config.expected["dev_folds_per_run"])
    rows: list[dict[str, Any]] = []
    for run in runs:
        peaks = run.manifest.get("resource_peaks", {}) or {}
        resource_usage_path = run.directory / "resource_usage.json"
        resource_usage = (
            read_json(resource_usage_path) if resource_usage_path.is_file() else {}
        )
        fold_selection_paths = [run.fold_selection(fold) for fold in range(1, fold_count + 1)]
        present_folds = [path for path in fold_selection_paths if path.is_file()]
        selected = _read_feature_list(run.final_selection)
        rows.append(
            {
                "run_id": run.run_id,
                "dataset": run.dataset,
                "model": run.model,
                "configuration": run.configuration,
                "arm": run.arm,
                "designation": run.designation,
                "comparison_family": run.comparison_family,
                "candidate_pool_budget": run.candidate_pool_budget,
                "expected_final_feature_budget": config.final_budget(run.model),
                "final_selected_feature_count": len(selected),
                "dev_fold_count_expected": fold_count,
                "dev_fold_selection_count_present": len(present_folds),
                "dev_prediction_row_count": _prediction_row_count(run.dev_predictions),
                "oot_prediction_row_count": _prediction_row_count(run.oot_predictions),
                "dev_prediction_sha256": _hash_if_present(run.dev_predictions),
                "oot_prediction_sha256": _hash_if_present(run.oot_predictions),
                "final_selection_sha256": _hash_if_present(run.final_selection),
                "fold_selection_sha256_concatenated": ";".join(
                    _hash_if_present(path) or "" for path in fold_selection_paths
                ),
                "manifest_sha256": _hash_if_present(run.directory / "manifest.json"),
                "config_sha256": _hash_if_present(run.directory / "config.json"),
                "checkpoint_sha256": _hash_if_present(run.directory / "checkpoint.json"),
                "completion_state": str(run.manifest.get("status")),
                "success_marker_present": (run.directory / "_SUCCESS").is_file(),
                "runtime_seconds": _float_or_none(
                    (resource_usage.get("timings_seconds") or {}).get("total")
                ),
                "peak_process_tree_rss_bytes": peaks.get("peak_process_tree_rss_bytes"),
                "minimum_system_available_ram_bytes": peaks.get(
                    "minimum_system_available_ram_bytes"
                ),
                "peak_process_gpu_bytes": peaks.get("peak_process_gpu_bytes"),
                "dev_source_ordered_id_sha256": _metadata_field(
                    run.dev_metadata, "source_order_identity_sha256"
                ),
                "dev_identity_target_sha256": _metadata_field(
                    run.dev_metadata, "identity_target_sha256"
                ),
                "oot_source_ordered_id_sha256": _metadata_field(
                    run.oot_metadata, "source_order_identity_sha256"
                ),
                "oot_identity_target_sha256": _metadata_field(
                    run.oot_metadata, "identity_target_sha256"
                ),
                "row_identifier": str(config.expected["row_identifier"][run.dataset]),
                "exclusion_state": "included",
                "exclusion_evidence": "",
                "run_directory": _relative(run.directory, config.repository_root),
            }
        )
    return pd.DataFrame(rows)


def _hash_if_present(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_field(path: Path, field: str) -> str | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    value = payload.get(field)
    return None if value is None else str(value)


def _read_feature_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    column = "feature" if "feature" in frame.columns else frame.columns[0]
    return [str(value) for value in frame[column].tolist()]


def read_fold_selection(run: RunRecord, fold: int) -> list[str]:
    """Read one fold's selected-feature list in saved selection order."""

    return _read_feature_list(run.fold_selection(fold))


def read_final_selection(run: RunRecord) -> list[str]:
    """Read the full-DEV final selected-feature list in selection order."""

    return _read_feature_list(run.final_selection)


def read_fold_candidate_pool(run: RunRecord, fold: int) -> list[str]:
    """Read one fold's frozen candidate pool in aggregate-rank order."""

    return _read_feature_list(run.fold_candidates(fold))


def fold_candidate_universe_counts(run: RunRecord, fold: int) -> set[int]:
    """Return the distinct candidate-universe sizes a fold ranking declares."""

    path = run.fold_voter_rankings(fold)
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, usecols=lambda name: name in {"candidate_universe_count"})
    if "candidate_universe_count" not in frame.columns:
        # The reference schema omits the field; the ranking covers the universe.
        rows = pd.read_csv(path, usecols=["voter_id"])
        voters = rows["voter_id"].nunique()
        return {int(len(rows) // max(voters, 1))}
    values = pd.to_numeric(frame["candidate_universe_count"], errors="coerce").dropna()
    return {int(value) for value in values.unique()}


def authenticate_prompt_05_completion(
    config: AnalysisConfig, runs: list[RunRecord], inventory: pd.DataFrame
) -> dict[str, Any]:
    """Authenticate Prompt 5 completion from canonical artifacts only."""

    failures: list[str] = []
    warnings: list[str] = []
    fold_count = int(config.expected["dev_folds_per_run"])

    completeness = read_json(
        config.repository_root
        / str(config.payload["frozen_inputs"]["completeness_summary"]["path"])
    )
    if str(completeness.get("status")) != "complete":
        failures.append(
            f"completeness summary status is {completeness.get('status')!r}"
        )
    for key, expected in (
        ("registered_runs", int(config.expected["total_runs"])),
        ("dev_fold_executions", int(config.expected["total_dev_fold_executions"])),
        ("full_dev_oot_fits", int(config.expected["oot_prediction_artifacts"])),
    ):
        if int(completeness.get(key, -1)) != expected:
            failures.append(
                f"completeness summary {key}={completeness.get(key)!r}, expected {expected}"
            )

    run_index = pd.read_csv(
        config.repository_root
        / str(config.payload["frozen_inputs"]["run_index"]["path"])
    )
    indexed = run_index.set_index("run_id", drop=False)
    if run_index["run_id"].duplicated().any():
        failures.append("run index contains duplicate run ids")

    # Every registered run must be complete in every canonical source.
    for run in runs:
        record = inventory.loc[inventory["run_id"] == run.run_id].iloc[0]
        checkpoint = read_json(run.directory / "checkpoint.json")
        if str(checkpoint.get("status")) != "completed":
            failures.append(f"{run.run_id}: checkpoint status {checkpoint.get('status')!r}")
        if str(record["completion_state"]) != "completed":
            failures.append(f"{run.run_id}: manifest status {record['completion_state']!r}")
        if not bool(record["success_marker_present"]):
            failures.append(f"{run.run_id}: missing _SUCCESS marker")
        if run.run_id not in indexed.index:
            failures.append(f"{run.run_id}: absent from results/run_index.csv")
        elif str(indexed.at[run.run_id, "status"]) != "completed":
            failures.append(
                f"{run.run_id}: run index status {indexed.at[run.run_id, 'status']!r}"
            )
        completed_folds = {str(value) for value in checkpoint.get("completed_fold_ids", [])}
        if completed_folds != {str(index) for index in range(1, fold_count + 1)}:
            failures.append(
                f"{run.run_id}: checkpoint completed folds {sorted(completed_folds)}"
            )
        if int(record["dev_fold_selection_count_present"]) != fold_count:
            failures.append(
                f"{run.run_id}: {record['dev_fold_selection_count_present']} of "
                f"{fold_count} fold selections present"
            )
        for fold in range(1, fold_count + 1):
            if not run.fold_dev_predictions(fold).is_file():
                failures.append(f"{run.run_id}: fold {fold} DEV predictions missing")
        if int(record["oot_prediction_row_count"]) != int(
            config.expected["oot_rows"][run.dataset]
        ):
            failures.append(
                f"{run.run_id}: OOT rows {record['oot_prediction_row_count']} != "
                f"{config.expected['oot_rows'][run.dataset]}"
            )
        if int(record["final_selected_feature_count"]) != config.final_budget(run.model):
            failures.append(
                f"{run.run_id}: final selection has "
                f"{record['final_selected_feature_count']} features, expected "
                f"{config.final_budget(run.model)}"
            )
        # Manifest-declared artifact hashes must match the files on disk.
        for name, entry in (run.manifest.get("artifacts") or {}).items():
            declared = entry.get("sha256")
            if not declared:
                continue
            artifact = run.directory / str(entry["path"])
            if not artifact.is_file():
                failures.append(f"{run.run_id}: manifest artifact missing: {name}")
                continue
            if sha256_file(artifact) != str(declared):
                failures.append(f"{run.run_id}: manifest hash mismatch for {name}")

    # Stale or partial directories must not be mistaken for registered runs.
    registered = {run.run_id for run in runs}
    stray: list[str] = []
    for dataset_root in sorted(config.run_root.glob("*")):
        if not dataset_root.is_dir():
            continue
        for directory in sorted(dataset_root.glob("cdv1-*")):
            if directory.name in registered:
                continue
            stray.append(_relative(directory, config.repository_root))
    for directory in stray:
        if "cdv1-pilot-" in directory:
            warnings.append(f"pilot run present and excluded from inference: {directory}")
        else:
            failures.append(f"unregistered cdv1 run directory present: {directory}")

    quarantine_root = config.repository_root / "results/quarantine"
    quarantined_files = (
        [
            _relative(path, config.repository_root)
            for path in quarantine_root.rglob("*")
            if path.is_file()
        ]
        if quarantine_root.is_dir()
        else []
    )
    if quarantined_files:
        warnings.append(
            f"{len(quarantined_files)} quarantined artifact(s) retained outside the run set"
        )
    partial = [
        _relative(path, config.repository_root)
        for path in config.run_root.rglob("*.partial")
    ]
    if partial:
        failures.append(f"unresolved .partial artifacts present: {partial}")

    return {
        "schema_version": "prompt_05_completion_authentication_v1",
        "authenticated_at_stage": "prompt_06_phase_a3",
        "registered_run_count": len(runs),
        "expected_run_count": int(config.expected["total_runs"]),
        "expected_dev_fold_executions": int(
            config.expected["total_dev_fold_executions"]
        ),
        "observed_dev_fold_selection_artifacts": int(
            inventory["dev_fold_selection_count_present"].sum()
        ),
        "observed_oot_prediction_artifacts": int(
            (inventory["oot_prediction_row_count"] > 0).sum()
        ),
        "observed_aggregated_dev_prediction_artifacts": int(
            (inventory["dev_prediction_row_count"] > 0).sum()
        ),
        "duplicate_run_ids": [],
        "approved_exclusions": list(config.expected["approved_exclusions"]),
        "pilot_run_directories_excluded": [
            entry for entry in stray if "cdv1-pilot-" in entry
        ],
        "quarantined_artifact_count": len(quarantined_files),
        "completeness_summary": completeness,
        "warnings": warnings,
        "failures": failures,
        "status": "PASS" if not failures else "BLOCKED",
    }


__all__ = [
    "RunRecord",
    "authenticate_prompt_05_completion",
    "build_input_inventory",
    "discover_runs",
    "fold_candidate_universe_counts",
    "parse_run_id",
    "read_final_selection",
    "read_fold_candidate_pool",
    "read_fold_selection",
]
