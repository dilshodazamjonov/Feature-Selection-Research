"""Versioned stage checkpoints and fail-closed explicit resume validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from credit_risk_fs.experiments.atomic_io import (
    ArtifactMetadata,
    copy_atomic,
    inspect_artifact,
    quarantine_partial_artifacts,
    sha256_file,
    write_json_atomic,
)
from credit_risk_fs.experiments.research_logging import emit_research_event


CHECKPOINT_SCHEMA_VERSION = "experiment_stage_checkpoint_v1"
CHECKPOINT_STAGES = (
    "initialized",
    "data_validated",
    "selection_completed",
    "model_fit_completed",
    "dev_prediction_completed",
    "oot_prediction_completed",
    "evaluation_completed",
    "completed",
    "failed",
    "aborted_resource_limit",
    "interrupted",
)
TERMINAL_STAGES = {"completed", "failed", "aborted_resource_limit", "interrupted"}
SUCCESS_STAGES = tuple(stage for stage in CHECKPOINT_STAGES if stage not in TERMINAL_STAGES)
IDENTITY_FIELDS = (
    "run_id",
    "dataset",
    "selector",
    "model",
    "split_protocol",
    "seed",
    "budgets",
    "resolved_config_hash",
    "protocol_version",
    "protocol_hash",
    "data_hash",
    "row_alignment_hash",
    "git_commit",
    "git_dirty",
)


class ResumeValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResumeValidation:
    run_directory: Path
    checkpoint_path: Path
    reusable_stages: tuple[str, ...]
    completed_fold_ids: tuple[str, ...]
    quarantined_partials: tuple[str, ...]
    resumable: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repository_relative_log_path(path: Path) -> str:
    """Render an artifact path without leaking an absolute workstation path."""

    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists():
            return resolved.relative_to(candidate).as_posix()
    return resolved.name


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"checkpoint identity is missing fields: {missing}")
    normalized = {field: identity[field] for field in IDENTITY_FIELDS}
    if not str(normalized["run_id"]).strip():
        raise ValueError("checkpoint run_id must not be empty")
    return normalized


def resolve_resume_target(results_root: str | Path, target: str | Path) -> Path:
    """Resolve one explicitly named run ID or directory; never choose implicitly."""

    root = Path(results_root).resolve()
    if not str(target).strip():
        raise ResumeValidationError(
            "resume_target_missing", "an explicit run ID or directory is required"
        )
    supplied = Path(target)
    if supplied.is_absolute() or supplied.exists():
        candidate = supplied.resolve()
        if not candidate.is_relative_to(root / "runs"):
            raise ResumeValidationError(
                "resume_target_outside_results",
                f"resume directory is outside active results/runs: {candidate}",
            )
        if not candidate.is_dir():
            raise ResumeValidationError("resume_target_missing", f"run directory is missing: {candidate}")
        return candidate
    run_id = str(target).strip()
    matches = [path.resolve() for path in (root / "runs").glob(f"*/{run_id}") if path.is_dir()]
    if len(matches) != 1:
        raise ResumeValidationError(
            "resume_target_ambiguous",
            f"expected exactly one active run named {run_id!r}, found {len(matches)}",
        )
    return matches[0]


class CheckpointManager:
    def __init__(self, run_directory: str | Path) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.path = self.run_directory / "checkpoint.json"

    def initialize(
        self,
        identity: Mapping[str, Any],
        *,
        resolved_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.path.exists():
            raise FileExistsError(f"checkpoint already exists: {self.path}")
        normalized = _validate_identity(identity)
        now = _utc_now()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": normalized["run_id"],
            "run_directory": str(self.run_directory),
            "identity": normalized,
            "resolved_policy": dict(resolved_policy or {}),
            "status": "running",
            "completed_stages": ["initialized"],
            "completed_fold_ids": [],
            "finalized_artifacts": {},
            "latest_resource_peaks": {},
            "last_successful_stage": "initialized",
            "stop_code": None,
            "primary_stop_code": None,
            "secondary_events": [],
            "stop_lifecycle": [],
            "termination_condition": None,
            "cleanup_evidence": {},
            "error": None,
            "created_at_utc": now,
            "updated_at_utc": now,
        }
        write_json_atomic(self.path, payload, overwrite=False)
        return payload

    def load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ResumeValidationError("checkpoint_missing", f"checkpoint is missing: {self.path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeValidationError("checkpoint_unreadable", f"checkpoint is unreadable: {self.path}") from exc
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ResumeValidationError("checkpoint_schema_mismatch", "checkpoint schema version is unsupported")
        if Path(str(payload.get("run_directory", ""))).resolve() != self.run_directory:
            raise ResumeValidationError("checkpoint_run_directory_mismatch", "checkpoint run directory does not match")
        _validate_identity(payload.get("identity", {}))
        return payload

    def _normalize_artifact(
        self,
        metadata: ArtifactMetadata | Mapping[str, Any],
        *,
        stage: str,
        identity: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        values = metadata.to_dict() if isinstance(metadata, ArtifactMetadata) else dict(metadata)
        path = Path(str(values.get("path", ""))).resolve()
        if not path.is_relative_to(self.run_directory):
            raise ValueError(f"checkpoint artifact is outside run directory: {path}")
        if not path.is_file():
            raise ValueError(f"checkpoint artifact is missing: {path}")
        relative = path.relative_to(self.run_directory).as_posix()
        values["path"] = relative
        values["stage"] = stage
        values["provenance"] = {
            key: identity[key]
            for key in (
                "resolved_config_hash",
                "protocol_hash",
                "data_hash",
                "row_alignment_hash",
                "git_commit",
            )
        }
        return relative, values

    def transition(
        self,
        stage: str,
        *,
        artifacts: Iterable[ArtifactMetadata | Mapping[str, Any]] = (),
        completed_fold_id: str | int | None = None,
        resource_peaks: Mapping[str, Any] | None = None,
        stop_code: str | None = None,
        error: str | None = None,
        termination_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in CHECKPOINT_STAGES:
            raise ValueError(f"unknown checkpoint stage: {stage}")
        payload = self.load()
        if payload["status"] == "completed":
            raise ResumeValidationError("completed_run_immutable", "completed checkpoints are immutable")
        completed = list(payload.get("completed_stages", []))
        if stage in SUCCESS_STAGES and stage not in completed:
            previous_success_indices = [CHECKPOINT_STAGES.index(item) for item in completed if item in SUCCESS_STAGES]
            if previous_success_indices and CHECKPOINT_STAGES.index(stage) < max(previous_success_indices):
                raise ValueError("checkpoint stages cannot move backwards")
            completed.append(stage)
            payload["last_successful_stage"] = stage
        finalized = dict(payload.get("finalized_artifacts", {}))
        for artifact in artifacts:
            relative, entry = self._normalize_artifact(
                artifact,
                stage=stage,
                identity=payload["identity"],
            )
            if relative in finalized and finalized[relative] != entry:
                raise ResumeValidationError(
                    "artifact_overwrite_mismatch",
                    f"checkpoint artifact metadata changed: {relative}",
                )
            finalized[relative] = entry
        fold_ids = list(map(str, payload.get("completed_fold_ids", [])))
        if completed_fold_id is not None and str(completed_fold_id) not in fold_ids:
            fold_ids.append(str(completed_fold_id))
        payload.update(
            {
                "completed_stages": completed,
                "completed_fold_ids": fold_ids,
                "finalized_artifacts": finalized,
                "latest_resource_peaks": dict(resource_peaks or payload.get("latest_resource_peaks", {})),
                "status": stage if stage in TERMINAL_STAGES else "running",
                "stop_code": stop_code,
                "error": error,
                "updated_at_utc": _utc_now(),
            }
        )
        if termination_metadata is not None:
            metadata = dict(termination_metadata)
            payload.update(
                {
                    "primary_stop_code": metadata.get("primary_stop_code"),
                    "secondary_events": list(metadata.get("secondary_events", [])),
                    "stop_lifecycle": list(metadata.get("stop_lifecycle", [])),
                    "termination_condition": metadata.get("termination_condition"),
                    "cleanup_evidence": dict(metadata.get("cleanup_evidence", {})),
                }
            )
        write_json_atomic(self.path, payload)
        stage_artifacts = [
            {
                "path": _repository_relative_log_path(self.run_directory / relative),
                "size_bytes": metadata.get("size_bytes"),
                "sha256": metadata.get("sha256"),
            }
            for relative, metadata in finalized.items()
            if metadata.get("stage") == stage
        ]
        emit_research_event(
            "checkpoint_transition",
            message=f"Checkpoint transitioned to {stage}",
            priority=True,
            run_id=payload.get("run_id"),
            stage=stage,
            component="checkpoint_manager",
            checkpoint_path=_repository_relative_log_path(self.path),
            completed_fold_id=completed_fold_id,
            checkpoint_status=payload.get("status"),
            stop_code=stop_code,
            artifact_count=len(stage_artifacts),
            artifacts=stage_artifacts,
        )
        return payload

    def validate_resume(
        self,
        expected_identity: Mapping[str, Any],
        *,
        quarantine_partials: bool = True,
        resume_authorization: Mapping[str, Any] | None = None,
    ) -> ResumeValidation:
        payload = self.load()
        emit_research_event(
            "checkpoint_validation_started",
            message="Checkpoint resume validation started",
            priority=True,
            run_id=payload.get("run_id"),
            stage=payload.get("last_successful_stage"),
            component="checkpoint_manager",
            completed_fold_ids=payload.get("completed_fold_ids", []),
            finalized_artifact_count=len(payload.get("finalized_artifacts", {})),
            checkpoint_path=_repository_relative_log_path(self.path),
        )
        if payload.get("status") == "completed" or "completed" in payload.get("completed_stages", []):
            raise ResumeValidationError("completed_run_immutable", "completed runs cannot be resumed")
        if payload.get("stop_code") == "wall_clock_limit":
            authorization = dict(resume_authorization or {})
            if (
                authorization.get("decision") != "RESUMABLE_FROM_CELL_BOUNDARY"
                or authorization.get("run_id") != payload.get("run_id")
                or authorization.get("intended_restart_boundary") != "cell_boundary"
                or authorization.get("validator_version")
                != "full_baseline_timeout_resume_validator_v2"
                or authorization.get("historical_terminal_state") != "timed_out"
                or authorization.get("historical_stop_reason") != "wall_clock_limit"
                or authorization.get("reasons")
                or authorization.get("active_processes")
                or authorization.get("lock_paths")
                or authorization.get("checkpoint_identity") != payload.get("identity")
                or not authorization.get("checks")
                or not all(
                    item.get("passed") is True
                    for item in authorization.get("checks", [])
                )
            ):
                raise ResumeValidationError(
                    "timeout_resume_authorization_required",
                    "a controlled timeout requires explicit cell-boundary authorization",
                )
        expected = _validate_identity(expected_identity)
        actual = payload["identity"]
        for field in IDENTITY_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ResumeValidationError(
                    f"resume_mismatch_{field}",
                    f"resume identity mismatch for {field}: expected={expected.get(field)!r}, actual={actual.get(field)!r}",
                )

        for relative, metadata in payload.get("finalized_artifacts", {}).items():
            path = (self.run_directory / relative).resolve()
            if not path.is_relative_to(self.run_directory):
                raise ResumeValidationError("artifact_path_escape", f"artifact escapes run: {relative}")
            if not path.is_file():
                raise ResumeValidationError("artifact_missing", f"finalized artifact is missing: {relative}")
            if int(path.stat().st_size) != int(metadata.get("size_bytes", -1)):
                raise ResumeValidationError("artifact_size_mismatch", f"artifact size mismatch: {relative}")
            if sha256_file(path) != metadata.get("sha256"):
                raise ResumeValidationError("artifact_checksum_mismatch", f"artifact checksum mismatch: {relative}")
            inspected = inspect_artifact(
                path,
                artifact_format=metadata.get("artifact_format"),
                expected_row_count=metadata.get("row_count"),
            )
            if metadata.get("schema") is not None and inspected.schema != metadata.get("schema"):
                raise ResumeValidationError("artifact_schema_mismatch", f"artifact schema mismatch: {relative}")
            provenance = metadata.get("provenance", {})
            for field in (
                "resolved_config_hash",
                "protocol_hash",
                "data_hash",
                "row_alignment_hash",
                "git_commit",
            ):
                if provenance.get(field) != actual.get(field):
                    raise ResumeValidationError(
                        f"artifact_provenance_mismatch_{field}",
                        f"artifact provenance mismatch for {relative}: {field}",
                    )

        attempt_number = len(payload.get("attempt_history", [])) + 1
        attempt_directory = (
            self.run_directory
            / "incomplete"
            / "attempt_history"
            / f"attempt_{attempt_number:02d}"
        )
        quarantined = (
            quarantine_partial_artifacts(
                self.run_directory,
                destination_directory=attempt_directory / "partial_artifacts",
            )
            if quarantine_partials
            else []
        )
        if quarantine_partials:
            tracked = {
                (self.run_directory / relative).resolve()
                for relative in payload.get("finalized_artifacts", {})
            }
            control_names = {
                "checkpoint.json",
                "manifest.json",
                "run_manifest.json",
                "run.log",
                ".execution.lock",
            }
            untracked_root = attempt_directory / "partial_artifacts"
            for path in sorted(self.run_directory.rglob("*")):
                if not path.is_file() or path.is_relative_to(self.run_directory / "incomplete"):
                    continue
                if path.resolve() in tracked or path.name in control_names:
                    continue
                destination = untracked_root / path.relative_to(self.run_directory)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ResumeValidationError(
                        "untracked_quarantine_collision",
                        f"cannot quarantine untracked artifact: {path}",
                    )
                os.replace(path, destination)
                quarantined.append(destination)
        validation = ResumeValidation(
            run_directory=self.run_directory,
            checkpoint_path=self.path,
            reusable_stages=tuple(payload.get("completed_stages", [])),
            completed_fold_ids=tuple(map(str, payload.get("completed_fold_ids", []))),
            quarantined_partials=tuple(str(path) for path in quarantined),
            resumable=True,
        )
        emit_research_event(
            "checkpoint_validation_completed",
            message="Checkpoint resume validation completed",
            priority=True,
            run_id=payload.get("run_id"),
            stage=payload.get("last_successful_stage"),
            component="checkpoint_manager",
            completed_fold_ids=list(validation.completed_fold_ids),
            reusable_stages=list(validation.reusable_stages),
            quarantined_partial_count=len(validation.quarantined_partials),
            checkpoint_path=_repository_relative_log_path(self.path),
        )
        return validation

    def begin_resume_attempt(
        self,
        *,
        historical_manifest_path: str | Path | None = None,
        resume_authorization: Mapping[str, Any] | None = None,
        new_active_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Archive prior terminal resource evidence before a validated retry."""

        payload = self.load()
        if payload.get("status") == "completed":
            raise ResumeValidationError("completed_run_immutable", "completed runs cannot be resumed")
        history = list(payload.get("attempt_history", []))
        attempt_number = len(history) + 1
        attempt_dir = (
            self.run_directory
            / "incomplete"
            / "attempt_history"
            / f"attempt_{attempt_number:02d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prior_status = payload.get("status")
        finalized = dict(payload.get("finalized_artifacts", {}))
        resource = finalized.pop("resource_usage.json", None)
        archived_resource = None
        resource_path = self.run_directory / "resource_usage.json"
        archived_manifest = None
        historical_manifest_status = None
        historical_manifest_sha256 = None
        if historical_manifest_path is not None:
            manifest_path = Path(historical_manifest_path).resolve()
            if not manifest_path.is_relative_to(self.run_directory):
                raise ResumeValidationError(
                    "historical_manifest_path_escape",
                    "historical manifest path escapes run directory",
                )
            historical_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            historical_manifest_status = historical_manifest.get("status")
            historical_manifest_sha256 = sha256_file(manifest_path)
            archived_path = attempt_dir / "manifest.json"
            copy_atomic(manifest_path, archived_path, overwrite=False)
            archived_manifest = archived_path.relative_to(self.run_directory).as_posix()
        checkpoint_snapshot = attempt_dir / "checkpoint_before_resume.json"
        copy_atomic(self.path, checkpoint_snapshot, overwrite=False)
        if resource is not None and resource_path.is_file():
            archived = attempt_dir / "resource_usage.json"
            os.replace(resource_path, archived)
            archived_resource = archived.relative_to(self.run_directory).as_posix()
        prior_active_seconds = None
        if archived_resource is not None:
            try:
                resource_payload = json.loads(
                    (self.run_directory / archived_resource).read_text(encoding="utf-8")
                )
                prior_active_seconds = resource_payload.get(
                    "active_computation_seconds",
                    resource_payload.get("timings_seconds", {}).get("total"),
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                prior_active_seconds = None
        history.append(
            {
                "attempt_id": f"attempt_{attempt_number:02d}",
                "status": prior_status,
                "historical_manifest_status": historical_manifest_status,
                "stop_code": payload.get("stop_code"),
                "primary_stop_code": payload.get(
                    "primary_stop_code", payload.get("stop_code")
                ),
                "secondary_events": list(payload.get("secondary_events", [])),
                "stop_lifecycle": list(payload.get("stop_lifecycle", [])),
                "termination_condition": payload.get("termination_condition"),
                "cleanup_evidence": dict(payload.get("cleanup_evidence", {})),
                "error": payload.get("error"),
                "ended_at_utc": payload.get("updated_at_utc"),
                "resource_metadata": resource,
                "archived_resource_path": archived_resource,
                "archived_manifest_path": archived_manifest,
                "archived_manifest_sha256": historical_manifest_sha256,
                "archived_checkpoint_path": checkpoint_snapshot.relative_to(
                    self.run_directory
                ).as_posix(),
                "prior_active_computation_seconds": prior_active_seconds,
                "resume_authorization": dict(resume_authorization or {}),
                "restart_boundary": (
                    dict(resume_authorization or {}).get(
                        "intended_restart_boundary", "checkpoint"
                    )
                ),
                "new_active_timeout_seconds": new_active_timeout_seconds,
            }
        )
        payload.update(
            {
                "attempt_history": history,
                "finalized_artifacts": finalized,
                "status": "running",
                "stop_code": None,
                "primary_stop_code": None,
                "secondary_events": [],
                "stop_lifecycle": [],
                "termination_condition": None,
                "cleanup_evidence": {},
                "error": None,
                "updated_at_utc": _utc_now(),
            }
        )
        write_json_atomic(self.path, payload)
        return payload
