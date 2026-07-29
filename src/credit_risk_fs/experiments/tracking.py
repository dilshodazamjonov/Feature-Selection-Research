from __future__ import annotations

import json
import os
import subprocess
import sys
import ctypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_risk_fs.experiments.config import compute_config_hash
from credit_risk_fs.experiments.atomic_io import (
    copy_atomic,
    inspect_artifact,
    write_json_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.result_paths import (
    reject_historical_write,
    sanitize_component,
)


SUCCESS_MARKER = "_SUCCESS"

STANDARD_ARTIFACTS = {
    "config": "config.json",
    "manifest": "manifest.json",
    "selected_features": "selected_features.csv",
    "fold_selections": "fold_selections.csv",
    "metrics": "metrics.csv",
    "predictions_dev": "predictions_dev.csv",
    "predictions_oot": "predictions_oot.csv",
    "stability": "stability.csv",
    "resource_usage": "resource_usage.json",
    "preflight": "preflight.json",
    "checkpoint": "checkpoint.json",
    "run_log": "run.log",
}

_STANDARD_ARTIFACT_SOURCES = {
    "selected_features.csv": ("features/final_selected_features.csv",),
    "fold_selections.csv": ("features/fold_selected_features.csv",),
    "metrics.csv": ("results/prediction_metrics.csv",),
    "predictions_dev.csv": (
        "results/dev_oof_predictions.csv",
        "results/dev_predictions.csv",
    ),
    "predictions_oot.csv": ("results/oot_predictions.csv",),
    "stability.csv": ("features/feature_stability_metrics.csv",),
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_commit_hash(project_root: str | Path = ".") -> str:
    """Return the current git commit hash, or ``unknown`` outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_git_dirty_state(project_root: str | Path = ".") -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def build_data_version(data_dir: str | Path) -> dict[str, Any]:
    """
    Build a lightweight data fingerprint from file names, sizes, and mtimes.

    This intentionally avoids hashing multi-GB raw files while still recording
    the exact files visible to the run.
    """
    path = Path(data_dir)
    files = []
    if path.exists():
        for file_path in sorted(path.glob("*.csv")):
            stat = file_path.stat()
            files.append(
                {
                    "name": file_path.name,
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    )
                    .replace(microsecond=0)
                    .isoformat(),
                }
            )

    return {
        "path": str(path),
        "file_count": len(files),
        "files": files,
    }


def run_id_for_config(
    *,
    model: str,
    experiment_type: str,
    selector: str,
    config_hash: str,
) -> str:
    """Create a stable run id so completed matrix entries can be resumed."""
    return "_".join(
        (
            sanitize_component(model, field_name="model"),
            sanitize_component(experiment_type, field_name="experiment type"),
            sanitize_component(selector, field_name="selector"),
            sanitize_component(config_hash[:12], field_name="config hash"),
        )
    )


def build_run_manifest(
    *,
    run_id: str,
    model: str,
    selector: str,
    experiment_type: str,
    config: dict[str, Any],
    data_dir: str | Path,
    random_seed: int,
    output_folder: str | Path,
    project_root: str | Path = ".",
    status: str = "running",
    artifact_applicability: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build the audit manifest for one matrix run."""
    config_hash = compute_config_hash(config)
    return {
        "run_id": run_id,
        "timestamp": utc_timestamp(),
        "started_at_utc": utc_timestamp(),
        "status": status,
        "model": model,
        "selector": selector,
        "experiment_type": experiment_type,
        "config_hash": config_hash,
        "config": config,
        "data_path": str(data_dir),
        "data_version": build_data_version(data_dir),
        "random_seed": int(random_seed),
        "git_commit_hash": get_git_commit_hash(project_root),
        "git_dirty": get_git_dirty_state(project_root),
        "output_folder": str(Path(output_folder)),
        "artifacts": {
            name: {
                "applicable": bool((artifact_applicability or {}).get(name, True)),
                "path": relative,
                "present": relative in {"config.json", "manifest.json"},
            }
            for name, relative in STANDARD_ARTIFACTS.items()
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON and return the path."""
    output_path = reject_historical_write(path)
    write_json_atomic(output_path, payload)
    return output_path


def write_run_manifest(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write the canonical manifest plus the historical compatibility name."""

    run_path = Path(run_dir)
    payload["artifacts"] = build_artifact_contract(run_path, payload)
    manifest_path = write_json(run_path / "manifest.json", payload)
    write_json(run_path / "run_manifest.json", payload)
    return manifest_path


def build_artifact_contract(
    run_dir: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Describe applicability and presence for standard per-run artifacts."""

    run_path = Path(run_dir)
    existing = dict((payload or {}).get("artifacts", {}))
    contract: dict[str, dict[str, Any]] = {}
    for name, relative in STANDARD_ARTIFACTS.items():
        previous = existing.get(name, {})
        applicable = bool(previous.get("applicable", True))
        present = (run_path / relative).is_file()
        if name == "manifest":
            present = True
        contract[name] = {
            "applicable": applicable,
            "path": relative,
            "present": present,
        }
        if present and name != "manifest":
            metadata = inspect_artifact(run_path / relative)
            contract[name].update(
                {
                    "size_bytes": metadata.size_bytes,
                    "sha256": metadata.sha256,
                }
            )
    for name, previous in existing.items():
        if name in STANDARD_ARTIFACTS:
            continue
        if not isinstance(previous, dict):
            raise ValueError(f"additional artifact entry must be a mapping: {name}")
        relative = str(previous.get("path", "")).strip()
        applicable = bool(previous.get("applicable", True))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"additional artifact path is invalid: {name}={relative!r}")
        present = (run_path / relative).is_file()
        contract[name] = {
            "applicable": applicable,
            "path": relative,
            "present": present,
        }
        if present:
            metadata = inspect_artifact(run_path / relative)
            contract[name].update(
                {"size_bytes": metadata.size_bytes, "sha256": metadata.sha256}
            )
    return contract


def materialize_standard_artifacts(run_dir: str | Path) -> dict[str, Path]:
    """Create canonical filenames from real run outputs without overwriting."""

    run_path = Path(run_dir)
    materialized: dict[str, Path] = {}
    for target_name, source_names in _STANDARD_ARTIFACT_SOURCES.items():
        target = reject_historical_write(run_path / target_name)
        source = next(
            (run_path / name for name in source_names if (run_path / name).is_file()),
            None,
        )
        if source is None:
            continue
        if target.exists():
            raise FileExistsError(f"standard run artifact already exists: {target}")
        copy_atomic(source, target, overwrite=False)
        materialized[target_name] = target
    return materialized


def _peak_ram_mb() -> tuple[float | None, str]:
    if os.name == "nt":
        try:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if succeeded:
                return counters.PeakWorkingSetSize / (1024 * 1024), "windows_peak_working_set"
        except (AttributeError, OSError):
            pass
    else:
        try:
            import resource

            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
            return peak / divisor, "resource_ru_maxrss"
        except (ImportError, OSError, ValueError):
            pass
    return None, "unavailable"


def _peak_gpu_mb() -> tuple[float | None, str]:
    torch = sys.modules.get("torch")
    if torch is None:
        return None, "torch_not_loaded"
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024), "torch_cuda"
    except (AttributeError, RuntimeError):
        pass
    return None, "cuda_unavailable"


def write_resource_usage(
    run_dir: str | Path,
    runtime_payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist measured timing and process peak-memory information."""

    peak_ram_mb, ram_source = _peak_ram_mb()
    peak_gpu_mb, gpu_source = _peak_gpu_mb()
    payload = {
        "timings_seconds": {
            "feature_selection": runtime_payload.get("feature_selection_time_sec"),
            "model_training": runtime_payload.get("training_time_sec"),
            "prediction": runtime_payload.get("prediction_time_sec"),
            "evaluation": runtime_payload.get("evaluation_time_sec"),
            "total": runtime_payload.get("total_runtime_seconds"),
        },
        "peak_ram_mb": peak_ram_mb,
        "peak_gpu_mb": peak_gpu_mb,
        "peak_ram_measurement": ram_source,
        "peak_gpu_measurement": gpu_source,
    }
    for key in (
        "current_process_tree_rss_bytes",
        "peak_process_tree_rss_bytes",
        "current_process_gpu_bytes",
        "peak_process_gpu_bytes",
        "minimum_system_available_ram_bytes",
        "minimum_results_free_disk_bytes",
        "minimum_temp_free_disk_bytes",
        "worker_exit_code",
        "stop_code",
        "status",
        "samples",
        "warnings",
        "resolved_parallelism",
        "policy_version",
        "preflight_status",
        "checkpoint_path",
        "resumability_status",
        "active_computation_seconds",
        "total_ram_wait_seconds",
        "ram_control",
    ):
        if key in runtime_payload:
            payload[key] = runtime_payload[key]
    if payload.get("peak_process_tree_rss_bytes") is not None:
        payload["peak_ram_mb"] = float(payload["peak_process_tree_rss_bytes"]) / (1024 * 1024)
        payload["peak_ram_measurement"] = "supervised_process_tree_rss"
    if payload.get("peak_process_gpu_bytes") is not None:
        payload["peak_gpu_mb"] = float(payload["peak_process_gpu_bytes"]) / (1024 * 1024)
        payload["peak_gpu_measurement"] = "nvml_process_tree"
    write_json(Path(run_dir) / "resource_usage.json", payload)
    return payload


def mark_completed(run_dir: str | Path) -> Path:
    """Create the success marker used by resume behavior."""
    marker = reject_historical_write(Path(run_dir) / SUCCESS_MARKER)
    write_text_atomic(marker, utc_timestamp() + "\n", overwrite=False)
    return marker


def is_completed_run(run_dir: str | Path) -> bool:
    """Return true when a run has completed outputs and a success marker."""
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        manifest_path = path / "run_manifest.json"
    required = [
        path / SUCCESS_MARKER,
        manifest_path,
        path / "leakage_report.json",
        path / "data_split_manifest.json",
        path / "features" / "final_selected_features.csv",
        path / "features" / "fold_selected_features.csv",
        path / "features" / "selection_frequency.csv",
        path / "features" / "feature_stability_metrics.csv",
        path / "models" / "final_model.model",
        path / "models" / "final_model_bundle.joblib",
        path / "models" / "final_preprocessor.pkl",
        path / "models" / "final_model_metadata.json",
        path / "results" / "experiment_summary.csv",
        path / "results" / "cv_results.csv",
        path / "results" / "oot_test_results.csv",
        path / "results" / "oot_predictions.csv",
        path / "results" / "selected_feature_psi.csv",
        path / "results" / "model_score_psi.csv",
        path / "results" / "credit_risk_utility.csv",
        path / "results" / "runtime_summary.csv",
    ]
    if not all(item.exists() for item in required):
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    experiment_type = manifest.get("experiment_type")
    if experiment_type in {"llm", "hybrid"} and not (
        path / "features" / "llm_rankings_summary.csv"
    ).exists():
        return False
    if experiment_type == "hybrid" and not (
        path / "features" / "llm_hybrid_trace.csv"
    ).exists():
        return False
    return True
