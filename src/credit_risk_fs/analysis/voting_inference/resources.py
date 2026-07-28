"""Runtime and resource evidence assembled from saved logs and manifests."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from credit_risk_fs.analysis.voting_inference.config import read_json
from credit_risk_fs.analysis.voting_inference.inventory import RunRecord

LOG_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| \w+\s*\| (?P<logger>[^|]+)\| (?P<message>.*)$"
)
STAGE_MARKERS = {
    "selector_rf_corr_mrmr": ("RandomForestRelevanceMRMRSelector", "[FIT] Feature selection completed"),
    "selector_boruta": ("credit_risk_fs.selectors.boruta", "Boruta finished"),
    "selector_rfe": ("credit_risk_fs.selectors.rfe", "RFE finished"),
    "data_load": ("data_loader", "Replaced"),
}


def runtime_resource_row(run: RunRecord) -> dict[str, Any]:
    """Summarise one run's wall-clock, memory, GPU, and recovery evidence."""

    usage_path = run.directory / "resource_usage.json"
    usage = read_json(usage_path) if usage_path.is_file() else {}
    timings = usage.get("timings_seconds") or {}
    peaks = run.manifest.get("resource_peaks") or {}
    history = run.directory / "incomplete/attempt_history"
    attempts = sorted(history.glob("attempt_*_resource_usage.json")) if history.is_dir() else []
    lifecycle = run.manifest.get("stop_lifecycle") or []
    return {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model": run.model,
        "configuration": run.configuration,
        "candidate_pool_budget": run.candidate_pool_budget,
        "total_wall_clock_seconds": _as_float(timings.get("total")),
        "feature_selection_seconds": _as_float(timings.get("feature_selection")),
        "model_training_seconds": _as_float(timings.get("model_training")),
        "prediction_seconds": _as_float(timings.get("prediction")),
        "evaluation_seconds": _as_float(timings.get("evaluation")),
        "dev_phase_completed_at_utc": run.manifest.get("dev_phase_completed_at_utc"),
        "resumed_at_utc": run.manifest.get("resumed_at_utc"),
        "completed_at_utc": run.manifest.get("completed_at_utc"),
        "started_at_utc": run.manifest.get("started_at_utc"),
        "peak_process_tree_rss_bytes": peaks.get("peak_process_tree_rss_bytes"),
        "peak_process_tree_rss_mb": _as_float(usage.get("peak_ram_mb")),
        "minimum_system_available_ram_bytes": peaks.get(
            "minimum_system_available_ram_bytes"
        ),
        "peak_process_gpu_bytes": peaks.get("peak_process_gpu_bytes"),
        "peak_process_gpu_mb": _as_float(usage.get("peak_gpu_mb")),
        "minimum_results_free_disk_bytes": peaks.get("minimum_results_free_disk_bytes"),
        "minimum_temp_free_disk_bytes": peaks.get("minimum_temp_free_disk_bytes"),
        "run_directory_bytes": _directory_bytes(run),
        "resource_sample_count": len(usage.get("samples") or []),
        "resource_warning_count": len(usage.get("warnings") or []),
        "interruption_recovery_attempt_count": len(attempts),
        "stop_lifecycle_state_count": len(lifecycle),
        "primary_stop_code": run.manifest.get("primary_stop_code"),
        "worker_exit_code": run.manifest.get("worker_exit_code"),
        "resumability_status": run.manifest.get("resumability_status"),
        "checkpoint_resume_status": run.manifest.get("resumability_status"),
        "final_completion_state": run.manifest.get("status"),
        "stage_timings_available": bool(
            any(timings.get(key) is not None for key in ("feature_selection", "model_training", "prediction", "evaluation"))
        ),
        "cached_and_uncached_paths_comparable": False,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _directory_bytes(run: RunRecord) -> int:
    return int(sum(path.stat().st_size for path in run.directory.rglob("*") if path.is_file()))


def stage_breakdown_rows(run: RunRecord) -> list[dict[str, Any]]:
    """Derive observed selector/loader stage durations from the saved run log.

    The saved ``timings_seconds`` stage fields are null for these runs, so the
    breakdown is reconstructed from timestamped log markers and labelled as
    log-derived rather than instrumented.
    """

    log_path = run.directory / "run.log"
    if not log_path.is_file():
        return []
    events: list[tuple[pd.Timestamp, str, str]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_LINE.match(line.strip())
        if match is None:
            continue
        events.append(
            (
                pd.Timestamp(match.group("timestamp")),
                match.group("logger").strip(),
                match.group("message").strip(),
            )
        )
    if not events:
        return []
    rows: list[dict[str, Any]] = []
    for stage, (logger_name, marker) in STAGE_MARKERS.items():
        matched = [
            (timestamp, message)
            for timestamp, logger, message in events
            if logger == logger_name and message.startswith(marker)
        ]
        rows.append(
            {
                "run_id": run.run_id,
                "dataset": run.dataset,
                "model": run.model,
                "configuration": run.configuration,
                "stage": stage,
                "log_event_count": len(matched),
                "first_event_utc": str(matched[0][0]) if matched else None,
                "last_event_utc": str(matched[-1][0]) if matched else None,
                "observed_span_seconds": (
                    float((matched[-1][0] - matched[0][0]).total_seconds())
                    if len(matched) > 1
                    else None
                ),
                "source": "run_log_marker",
                "instrumented_stage_timer_available": False,
            }
        )
    span = float((events[-1][0] - events[0][0]).total_seconds())
    rows.append(
        {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "model": run.model,
            "configuration": run.configuration,
            "stage": "logged_execution_span",
            "log_event_count": len(events),
            "first_event_utc": str(events[0][0]),
            "last_event_utc": str(events[-1][0]),
            "observed_span_seconds": span,
            "source": "run_log_marker",
            "instrumented_stage_timer_available": False,
        }
    )
    return rows


__all__ = ["runtime_resource_row", "stage_breakdown_rows"]
