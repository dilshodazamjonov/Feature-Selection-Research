from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from credit_risk_fs.clip.v1_freeze import verify_freeze_package
from credit_risk_fs.clip_final_comparison.io import atomic_write_json
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json


def build_source_experiment_manifest() -> dict[str, Any]:
    v1 = verify_freeze_package()
    if v1.get("status") != "passed":
        raise RuntimeError("CLIP-v1 freeze integrity failed")
    v2_checks = _run_v2_audit()
    if v2_checks.get("status") != "passed":
        raise RuntimeError("CLIP-v2 audit failed")
    required = _required_artifacts()
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"missing source artifacts: {missing}")
    return {
        "created_at_utc": _git(["show", "-s", "--format=%cI", "HEAD"]),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status_short": _git(["status", "--short"]),
        "clip_v1_freeze": v1,
        "clip_v2_audit": v2_checks,
        "source_artifact_hashes": {path: sha256_file(path) for path in required if Path(path).is_file()},
        "checkpoint_hashes": _checkpoint_hashes(),
        "anchor_hashes": _anchor_hashes(),
        "preprocessor_hashes": _preprocessor_hashes(),
        "dataset_split_hashes": _split_hashes(),
        "baseline_result_hashes": _baseline_hashes(),
        "lendingclub_v2_external_to_representation_training": True,
        "oot_not_used_for_preprocessing_feature_selection_checkpoint_selection_or_tuning": True,
        "legacy_lendingclub_absent_from_new_plan": True,
    }


def write_source_experiment_manifest(output_dir: Path) -> Path:
    manifest = build_source_experiment_manifest()
    path = output_dir / "manifests" / "source_experiment_manifest.json"
    atomic_write_json(path, manifest)
    return path


def _run_v2_audit() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/audit_clip_v2.py"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        shell=False,
    )
    if result.returncode != 0:
        return {"status": "failed", "stdout": result.stdout, "stderr": result.stderr}
    import json

    return json.loads(result.stdout)


def _required_artifacts() -> list[str]:
    paths = [
        "results/clip_versions/v1/freeze_manifest.json",
        "results/clip_v2/training/model_selection_manifest.json",
        "results/clip_v2/training/selected_checkpoint.pt",
        "results/clip_v2/training/learned_anchor_manifest.json",
        "results/clip_v2/statistical_view/statistical_preprocessor.json",
        "results/clip_v2/final_evaluation/run_manifest.json",
        "results/clip_v2/final_evaluation/evaluation_summary.csv",
        "results/clip_v2/final_evaluation/aggregate_validation.json",
        "results/clip_v2/final_analysis/analysis_summary.json",
        "configs/clip_v2/selector.yaml",
    ]
    paths.extend(str(path).replace("\\", "/") for path in sorted(Path("results/clip_v2/final_evaluation/predictions").glob("*.parquet")))
    return paths


def _checkpoint_hashes() -> dict[str, str]:
    paths = sorted(Path("results/clip_v2/training/seeds").glob("seed_*/best_checkpoint.pt"))
    selected = Path("results/clip_v2/training/selected_checkpoint.pt")
    if selected.exists():
        paths.append(selected)
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def _anchor_hashes() -> dict[str, str]:
    paths = sorted(Path("results/clip_v2").glob("**/*anchor*manifest*.json"))
    return {path.as_posix(): sha256_file(path) for path in paths if path.is_file()}


def _preprocessor_hashes() -> dict[str, str]:
    paths = sorted(Path("results/clip_v2").glob("**/*preprocessor*.json"))
    return {path.as_posix(): sha256_file(path) for path in paths if path.is_file()}


def _split_hashes() -> dict[str, str]:
    paths = sorted(Path("results/clip_v2").glob("**/*split*manifest*.json"))
    return {path.as_posix(): sha256_file(path) for path in paths if path.is_file()}


def _baseline_hashes() -> dict[str, str]:
    paths = [
        Path("results/clip/final_evaluation/evaluation_summary.csv"),
        Path("results/clip/final_evaluation/run_manifest.json"),
        Path("results/clip_v2/final_evaluation/evaluation_summary.csv"),
        Path("results/clip_v2/final_evaluation/run_manifest.json"),
    ]
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False, timeout=30)
    return result.stdout.strip()
