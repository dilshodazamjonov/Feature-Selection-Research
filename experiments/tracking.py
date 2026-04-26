from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.config import compute_config_hash


SUCCESS_MARKER = "_SUCCESS"


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
    safe_selector = selector.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{model}_{experiment_type}_{safe_selector}_{config_hash[:12]}"


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
) -> dict[str, Any]:
    """Build the audit manifest for one matrix run."""
    config_hash = compute_config_hash(config)
    return {
        "run_id": run_id,
        "timestamp": utc_timestamp(),
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
        "output_folder": str(Path(output_folder)),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return output_path


def write_run_manifest(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``run_manifest.json`` inside a run directory."""
    return write_json(Path(run_dir) / "run_manifest.json", payload)


def mark_completed(run_dir: str | Path) -> Path:
    """Create the success marker used by resume behavior."""
    marker = Path(run_dir) / SUCCESS_MARKER
    marker.write_text(utc_timestamp(), encoding="utf-8")
    return marker


def is_completed_run(run_dir: str | Path) -> bool:
    """Return true when a run has completed outputs and a success marker."""
    path = Path(run_dir)
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
