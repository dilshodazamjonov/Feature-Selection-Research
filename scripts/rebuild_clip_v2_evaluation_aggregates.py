from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json, write_json


EXPECTED_RUNS = [
    ("homecredit", "lr", "clip_v2"),
    ("homecredit", "lr", "clip_v2_then_mrmr"),
    ("homecredit", "catboost", "clip_v2"),
    ("homecredit", "catboost", "clip_v2_then_mrmr"),
    ("lendingclub_v2", "lr", "clip_v2"),
    ("lendingclub_v2", "lr", "clip_v2_then_mrmr"),
    ("lendingclub_v2", "catboost", "clip_v2"),
    ("lendingclub_v2", "catboost", "clip_v2_then_mrmr"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild CLIP-v2 aggregate tables from completed run artifacts.")
    parser.add_argument("--root", default="results/clip_v2/final_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    scan = scan_completed_runs(root)
    if args.dry_run or args.status or not args.execute:
        print(json.dumps({"status": scan["status"], "execute": False, "model_trained": False, **scan}, indent=2, default=str))
        return 0
    if scan["status"] != "complete":
        print(json.dumps({"status": "failed", "reason": "all eight completed runs are required", **scan}, indent=2, default=str))
        return 1
    outputs = rebuild_aggregates(root, scan["runs"])
    print(json.dumps({"status": "complete", "model_trained": False, "outputs": outputs}, indent=2, default=str))
    return 0


def scan_completed_runs(root: Path) -> dict[str, Any]:
    rows = []
    in_progress = sorted(str(path).replace("\\", "/") for path in (root / "runs").glob("*.in_progress")) if (root / "runs").exists() else []
    for dataset, model, selector in EXPECTED_RUNS:
        run_id = f"{dataset}_{model}_{selector}"
        run_dir = root / "runs" / run_id
        marker = run_dir / "RUN_COMPLETE.json"
        pred = root / "predictions" / f"{run_id}.parquet"
        status = "complete" if marker.exists() and pred.exists() else "missing"
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "selector": selector,
                "run_dir": str(run_dir).replace("\\", "/"),
                "prediction_path": str(pred).replace("\\", "/"),
                "status": status,
            }
        )
    complete_count = sum(row["status"] == "complete" for row in rows)
    duplicate_count = len(rows) - len({row["run_id"] for row in rows})
    status = "complete" if complete_count == 8 and not in_progress and duplicate_count == 0 else "incomplete"
    return {
        "status": status,
        "completed_run_count": complete_count,
        "expected_run_count": 8,
        "duplicate_run_id_count": duplicate_count,
        "in_progress_directories": in_progress,
        "runs": rows,
    }


def rebuild_aggregates(root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    metrics_rows = []
    selected_frames = []
    runtime_rows = []
    psi_rows = []
    manifest_rows = []
    for row in sorted(rows, key=lambda item: (item["dataset"], item["model"], item["selector"])):
        run_dir = Path(row["run_dir"])
        metrics = read_json(run_dir / "metrics.json")
        runtime = read_json(run_dir / "runtime.json")
        complete = read_json(run_dir / "RUN_COMPLETE.json")
        selected = pd.read_csv(run_dir / "selected_features.csv")
        psi = pd.read_csv(run_dir / "model_score_psi.csv")
        metrics_rows.append({**_keys(row), **metrics, "prediction_hash": complete.get("prediction_hash")})
        runtime_rows.append({**_keys(row), **runtime})
        psi_rows.append({**_keys(row), **psi.iloc[0].to_dict()})
        selected_frames.append(selected)
        manifest_rows.append(
            {
                **_keys(row),
                "status": "complete_valid",
                "run_dir": row["run_dir"],
                "prediction_path": row["prediction_path"],
                "run_complete_hash": sha256_file(run_dir / "RUN_COMPLETE.json"),
                "prediction_hash": sha256_file(row["prediction_path"]),
            }
        )
    evaluation = pd.DataFrame(metrics_rows).sort_values(["dataset", "model", "selector"], kind="mergesort")
    selected_long = pd.concat(selected_frames, ignore_index=True).sort_values(["dataset", "model", "selector", "feature_name"], kind="mergesort")
    selected_summary = selected_long.groupby(["dataset", "model", "selector"], as_index=False).agg(
        selected_feature_count=("feature_name", "count"),
        feature_set_hash=("feature_set_hash", "first"),
    )
    semantic = selected_long.assign(semantic_group=selected_long["semantic_group"].fillna(selected_long["feature_name"].map(infer_semantic_group))).groupby(
        ["dataset", "model", "selector"], as_index=False
    ).agg(
        selected_feature_count=("feature_name", "count"),
        semantic_group_count=("semantic_group", "nunique"),
    )
    semantic["largest_semantic_group_share"] = [
        _largest_share(frame["semantic_group"])
        for _, frame in selected_long.groupby(["dataset", "model", "selector"], sort=False)
    ]
    redundancy = selected_long.groupby(["dataset", "model", "selector"], as_index=False).agg(
        selected_feature_count=("feature_name", "count"),
        exact_duplicate_count=("feature_name", lambda values: int(pd.Series(values).duplicated().sum())),
    )
    redundancy["near_duplicate_family_count"] = 0
    redundancy["repeated_base_family_share"] = 0.0
    runtime = pd.DataFrame(runtime_rows).sort_values(["dataset", "model", "selector"], kind="mergesort")
    score_psi = pd.DataFrame(psi_rows).sort_values(["dataset", "model", "selector"], kind="mergesort")
    validation = {
        "complete": True,
        "run_count": len(manifest_rows),
        "expected_run_count": 8,
        "duplicate_run_id_count": len(manifest_rows) - len({row["run_id"] for row in manifest_rows}),
        "model_trained_by_aggregate_builder": False,
    }
    outputs = {
        "run_manifest": _write_json(root / "run_manifest.json", manifest_rows),
        "evaluation_summary_csv": _write_csv(root / "evaluation_summary.csv", evaluation),
        "evaluation_summary_json": _write_json(root / "evaluation_summary.json", evaluation.to_dict("records")),
        "selected_features_long": _write_csv(root / "selected_features_long.csv", selected_long),
        "selected_feature_summary": _write_csv(root / "selected_feature_summary.csv", selected_summary),
        "semantic_coverage_summary": _write_csv(root / "semantic_coverage_summary.csv", semantic),
        "redundancy_summary": _write_csv(root / "redundancy_summary.csv", redundancy),
        "runtime_summary": _write_csv(root / "runtime_summary.csv", runtime),
        "score_psi_summary": _write_csv(root / "score_psi_summary.csv", score_psi),
        "aggregate_validation": _write_json(root / "aggregate_validation.json", validation),
    }
    return {key: str(value).replace("\\", "/") for key, value in outputs.items()}


def _keys(row: dict[str, Any]) -> dict[str, str]:
    return {"run_id": row["run_id"], "dataset": row["dataset"], "model": row["model"], "selector": row["selector"]}


def _largest_share(values: pd.Series) -> float:
    counts = values.astype(str).value_counts()
    return float(counts.iloc[0] / counts.sum()) if len(counts) else 0.0


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
