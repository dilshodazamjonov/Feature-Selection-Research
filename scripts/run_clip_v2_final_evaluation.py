from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.selector_validation import load_clip_selector_config, validate_clip_selector_binding  # noqa: E402
from credit_risk_fs.clip.v2_validation import validate_clip_v2_config, validate_no_v1_output_paths  # noqa: E402
from credit_risk_fs.evaluation.metrics import ks_score  # noqa: E402
from credit_risk_fs.experiments.config import (  # noqa: E402
    compute_config_hash,
    load_named_project_config,
    resolve_feature_budget,
    resolve_model_kwargs,
)
from credit_risk_fs.experiments.tracking import build_data_version  # noqa: E402
from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group  # noqa: E402
from credit_risk_fs.pipelines.common import ExperimentConfig, prepare_modeling_data, run_experiment  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file, sha256_text  # noqa: E402
from credit_risk_fs.utils.io import read_json, write_json  # noqa: E402


DATASETS = ("homecredit", "lendingclub_v2")
MODELS = ("lr", "catboost")
SELECTORS = ("clip_v2", "clip_v2_then_mrmr")
RUN_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute CLIP-v2 final downstream evaluation.")
    parser.add_argument("--config", default="configs/clip_v2/evaluation.yaml")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--selector", choices=SELECTORS)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = _load_config(args.config)
    output_root = Path(str(raw.get("final_evaluation_root", "results/clip_v2/final_evaluation")))
    specs = _specs(args)
    binding_error = ""
    try:
        binding = _selector_binding()
    except Exception as exc:
        binding_error = str(exc)
        binding = {"checkpoint_hash": "", "anchor_hash": ""}
        if args.execute:
            raise
    statuses = [_classify_run(output_root, spec, binding) for spec in specs]
    if args.status or args.plan or args.resume or not args.execute:
        planned = _planned_runs(specs, statuses, resume=bool(args.resume))
        print(
            json.dumps(
                {
                    "status": "planned" if not args.status else _overall_status(statuses),
                    "execute": False,
                    "resume_requested": bool(args.resume),
                    "planned_run_count": len(planned),
                    "planned_runs": planned,
                    "runs": statuses,
                    "output_root": str(output_root).replace("\\", "/"),
                    "selector_binding_error": binding_error,
                    "expensive_model_training": False,
                    "requires_execute_for_real_runs": True,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    planned = _planned_runs(specs, statuses, resume=bool(args.resume))
    if not planned:
        print(json.dumps({"status": "nothing_to_execute", "all_requested_runs_complete": True}, indent=2))
        return 0
    return _execute_planned(planned, output_root=output_root, binding=binding)


def _load_config(path: str | Path) -> dict[str, Any]:
    from credit_risk_fs.experiments.config import _parse_simple_yaml

    raw = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    errors = validate_clip_v2_config(raw)
    validate_no_v1_output_paths([str(raw.get("final_evaluation_root", "")), str(raw.get("output_root", ""))])
    if errors:
        raise RuntimeError("; ".join(errors))
    return raw


def _selector_binding() -> dict[str, Any]:
    config = load_clip_selector_config("configs/clip_v2/selector.yaml")
    return validate_clip_selector_binding(config)


def _specs(args: argparse.Namespace) -> list[dict[str, str]]:
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    models = [args.model] if args.model else list(MODELS)
    selectors = [args.selector] if args.selector else list(SELECTORS)
    return [{"dataset": d, "model": m, "selector": s} for d in datasets for m in models for s in selectors]


def _run_id(spec: dict[str, str]) -> str:
    return f"{spec['dataset']}_{spec['model']}_{spec['selector']}"


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if rows and all(row["status"] == "complete_valid" for row in rows):
        return "complete"
    if any(row["status"] not in {"not_started", "complete_valid"} for row in rows):
        return "needs_resume"
    return "incomplete"


def _planned_runs(specs: list[dict[str, str]], statuses: list[dict[str, Any]], *, resume: bool) -> list[dict[str, str]]:
    by_id = {row["run_id"]: row for row in statuses}
    planned = []
    for spec in specs:
        status = by_id[_run_id(spec)]["status"]
        if status == "complete_valid":
            continue
        if status in {"not_started", "interrupted", "failed", "incomplete", "stale"} or resume:
            planned.append(spec)
    return planned


def _classify_run(output_root: Path, spec: dict[str, str], binding: dict[str, Any]) -> dict[str, Any]:
    run_id = _run_id(spec)
    run_dir = output_root / "runs" / run_id
    progress_dir = output_root / "runs" / f"{run_id}.in_progress"
    prediction_path = output_root / "predictions" / f"{run_id}.parquet"
    row = {"run_id": run_id, **spec, "prediction_path": str(prediction_path).replace("\\", "/")}
    if progress_dir.exists():
        return {**row, "status": "interrupted", "safe_to_reuse": False, "notes": "in_progress directory exists"}
    if not run_dir.exists():
        return {**row, "status": "not_started", "safe_to_reuse": False, "notes": ""}
    marker = run_dir / "RUN_COMPLETE.json"
    if not marker.exists():
        return {**row, "status": "incomplete", "safe_to_reuse": False, "notes": "missing RUN_COMPLETE.json"}
    try:
        complete = read_json(marker)
        validation = _validate_run_artifacts(run_dir=run_dir, prediction_path=prediction_path, binding=binding)
    except Exception as exc:
        return {**row, "status": "stale", "safe_to_reuse": False, "notes": str(exc)}
    if complete.get("run_id") != run_id:
        return {**row, "status": "stale", "safe_to_reuse": False, "notes": "completion marker run_id mismatch"}
    return {
        **row,
        "status": "complete_valid",
        "safe_to_reuse": True,
        "prediction_rows": validation["prediction_rows"],
        "feature_set_hash": validation["feature_set_hash"],
        "notes": "",
    }


def _execute_planned(specs: list[dict[str, str]], *, output_root: Path, binding: dict[str, Any]) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(exist_ok=True)
    (output_root / "predictions").mkdir(exist_ok=True)
    prepared_by_dataset = {}
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        run_id = _run_id(spec)
        final_dir = output_root / "runs" / run_id
        progress_dir = output_root / "runs" / f"{run_id}.in_progress"
        if final_dir.exists():
            status = _classify_run(output_root, spec, binding)
            if status["status"] == "complete_valid":
                _progress(index, total, spec, "skip_complete_valid", 0.0)
                continue
            raise RuntimeError(f"{run_id}: refusing to overwrite invalid existing run; inspect --status first")
        if progress_dir.exists():
            raise RuntimeError(f"{run_id}: interrupted in-progress directory exists; inspect --resume first")
        started = time.time()
        progress_dir.mkdir(parents=True, exist_ok=False)
        try:
            _append_log(progress_dir, f"run_start {run_id}")
            config = _experiment_config(spec, progress_dir)
            _write_text(progress_dir / "config_snapshot.yaml", _config_snapshot_text(spec, config))
            if spec["dataset"] not in prepared_by_dataset:
                _progress(index, total, spec, "prepare_data_start", started)
                prepared_by_dataset[spec["dataset"]] = prepare_modeling_data(config)
            prepared = prepared_by_dataset[spec["dataset"]]
            _progress(
                index,
                total,
                spec,
                f"data_ready DEV={len(prepared.X_train)} OOT={len(prepared.X_oot)} candidates={prepared.X_train.shape[1]}",
                started,
            )
            _append_log(progress_dir, "model_fit start")
            run = run_experiment(config, prepared_data=prepared)
            _append_log(progress_dir, "model_fit end")
            _materialize_required_artifacts(progress_dir, spec=spec, binding=binding, prepared_oot_rows=len(prepared.X_oot))
            validation = _validate_run_artifacts(
                run_dir=progress_dir,
                prediction_path=output_root / "predictions" / f"{run_id}.parquet",
                binding=binding,
            )
            _append_log(progress_dir, f"artifact_validation passed {validation}")
            progress_dir.replace(final_dir)
            completion = {
                "run_id": run_id,
                "dataset": spec["dataset"],
                "model": spec["model"],
                "selector": spec["selector"],
                "status": "complete_valid",
                "completed_at_epoch_seconds": time.time(),
                "runtime_seconds": float(time.time() - started),
                "checkpoint_hash": binding["checkpoint_hash"],
                "anchor_hash": binding["anchor_hash"],
                **validation,
            }
            write_json(final_dir / "RUN_COMPLETE.json", completion)
            _progress(index, total, spec, "complete", started)
        except KeyboardInterrupt:
            _append_log(progress_dir, "interrupted KeyboardInterrupt no_completion_marker")
            print("Interrupted. Resume plan: uv run python scripts/run_clip_v2_final_evaluation.py --resume")
            raise
        except Exception as exc:
            _append_log(progress_dir, f"failed {exc!r} no_completion_marker")
            raise
    return 0


def _experiment_config(spec: dict[str, str], run_dir: Path) -> ExperimentConfig:
    project = load_named_project_config(spec["dataset"])
    feature_budget = resolve_feature_budget(project, spec["model"])
    config_hash_payload = {
        "experiment_version": "clip_v2",
        "dataset": spec["dataset"],
        "model": spec["model"],
        "selector": spec["selector"],
        "feature_budget": feature_budget,
        "clip_selector_config": "configs/clip_v2/selector.yaml",
        "random_seed": int(project.get("random_seed", 42)),
    }
    return ExperimentConfig(
        experiment_name=spec["selector"],
        selector_name=spec["selector"],
        dataset_name=spec["dataset"],
        model_name=spec["model"],
        model_kwargs=resolve_model_kwargs(project, spec["model"]),
        data_dir=str(project["data_dir"]),
        description_path=str(project["description_path"]),
        base_output_dir=str(run_dir.parent),
        experiment_output_dir=str(run_dir),
        dev_start_day=int(project["dev_start_day"]),
        oot_start_day=int(project["oot_start_day"]),
        oot_end_day=int(project["oot_end_day"]),
        n_splits=int(project["n_splits"]),
        cv_gap_groups=int(project["cv_gap_groups"]),
        random_state=int(project.get("random_seed", 42)),
        feature_budget=feature_budget,
        excluded_feature_columns=tuple(project.get("excluded_feature_columns", [])),
        preprocessor_kwargs=dict(project.get("preprocessor_kwargs", {})),
        selector_kwargs={
            "config_path": "configs/clip_v2/selector.yaml",
            "dataset": spec["dataset"],
            "model_name": spec["model"],
            "missing_feature_policy": "error",
            "selector_label": spec["selector"],
        },
        experiment_type="clip_v2_final_evaluation",
        config_hash=sha256_text(json.dumps(config_hash_payload, sort_keys=True)),
        data_fingerprint=build_data_version(project["data_dir"]),
    )


def _materialize_required_artifacts(run_dir: Path, *, spec: dict[str, str], binding: dict[str, Any], prepared_oot_rows: int) -> None:
    run_id = _run_id(spec)
    selected_source = run_dir / "features" / "final_selected_features.csv"
    selected = pd.read_csv(selected_source)
    feature_col = "feature_name" if "feature_name" in selected.columns else "feature"
    features = selected[feature_col].astype(str).tolist()
    feature_hash = sha256_text(json.dumps(sorted(features), sort_keys=True))
    selected_out = pd.DataFrame(
        {
            "dataset": spec["dataset"],
            "model": spec["model"],
            "selector": spec["selector"],
            "feature_name": features,
            "feature_set_hash": feature_hash,
            "semantic_group": [infer_semantic_group(feature) for feature in features],
        }
    )
    _atomic_csv(run_dir / "selected_features.csv", selected_out)
    manifest_path = _selection_manifest_path(run_dir, spec["selector"])
    selection_rows = int(len(pd.read_csv(manifest_path))) if manifest_path.exists() else 0
    write_json(
        run_dir / "feature_selection_manifest.json",
        {
            "run_id": run_id,
            "selector": spec["selector"],
            "feature_budget": 20 if spec["model"] == "lr" else 40,
            "selected_count": len(features),
            "selection_manifest_path": str(manifest_path).replace("\\", "/"),
            "selection_manifest_rows": selection_rows,
            "feature_set_hash": feature_hash,
            "clip_checkpoint_hash": binding["checkpoint_hash"],
            "clip_anchor_hash": binding["anchor_hash"],
            "mrmr_scope": "DEV only after CLIP-v2 screening" if spec["selector"] == "clip_v2_then_mrmr" else "not_applicable",
        },
    )
    metrics = pd.read_csv(run_dir / "results" / "oot_test_results.csv").iloc[0].to_dict()
    metrics_json = {key: _json_value(value) for key, value in metrics.items()}
    metrics_json.update({"run_id": run_id, "dataset": spec["dataset"], "model": spec["model"], "selector": spec["selector"]})
    write_json(run_dir / "metrics.json", metrics_json)
    runtime = pd.read_csv(run_dir / "results" / "runtime_summary.csv").iloc[0].to_dict()
    write_json(run_dir / "runtime.json", {key: _json_value(value) for key, value in runtime.items()})
    shutil.copy2(run_dir / "results" / "model_score_psi.csv", run_dir / "model_score_psi.csv")
    shutil.copy2(run_dir / "leakage_report.json", run_dir / "leakage_audit.json")
    source_pred = pd.read_csv(run_dir / "results" / "oot_predictions.csv")
    pred = pd.DataFrame(
        {
            "dataset": spec["dataset"],
            "model": spec["model"],
            "selector": spec["selector"],
            "evaluation_index": np.arange(len(source_pred), dtype=int),
            "y_true": source_pred["y_true"].astype(int),
            "y_pred_proba": pd.to_numeric(source_pred["y_pred_proba"], errors="coerce"),
            "y_pred": source_pred["y_pred"].astype(int),
            "split": "OOT",
            "run_id": run_id,
            "checkpoint_hash": binding["checkpoint_hash"],
            "anchor_hash": binding["anchor_hash"],
            "feature_set_hash": feature_hash,
        }
    )
    if len(pred) != int(prepared_oot_rows):
        raise RuntimeError(f"{run_id}: prediction row count mismatch expected={prepared_oot_rows} observed={len(pred)}")
    prediction_path = run_dir.parents[1] / "predictions" / f"{run_id}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)


def _selection_manifest_path(run_dir: Path, selector: str) -> Path:
    if selector == "clip_v2":
        return run_dir / "llm_responses" / "final_dev" / "clip_v2_selection_manifest.csv"
    return run_dir / "llm_responses" / "final_dev" / "clip_v2_then_mrmr_selection_manifest.csv"


def _validate_run_artifacts(*, run_dir: Path, prediction_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    required = [
        "config_snapshot.yaml",
        "selected_features.csv",
        "feature_selection_manifest.json",
        "metrics.json",
        "runtime.json",
        "model_score_psi.csv",
        "leakage_audit.json",
        "execution.log",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing run artifacts: {missing}")
    if not prediction_path.exists():
        raise RuntimeError(f"missing prediction artifact: {prediction_path}")
    pred = pd.read_parquet(prediction_path)
    expected_cols = {
        "dataset",
        "model",
        "selector",
        "evaluation_index",
        "y_true",
        "y_pred_proba",
        "y_pred",
        "split",
        "run_id",
        "checkpoint_hash",
        "anchor_hash",
        "feature_set_hash",
    }
    missing_cols = sorted(expected_cols - set(pred.columns))
    if missing_cols:
        raise RuntimeError(f"prediction missing columns: {missing_cols}")
    if pred["evaluation_index"].duplicated().any():
        raise RuntimeError("duplicate prediction evaluation_index")
    y_true = pred["y_true"].astype(int)
    y_score = pd.to_numeric(pred["y_pred_proba"], errors="coerce")
    if not set(y_true.unique()).issubset({0, 1}):
        raise RuntimeError("prediction target is not binary")
    if not np.isfinite(y_score).all() or not y_score.between(0.0, 1.0).all():
        raise RuntimeError("prediction probabilities are invalid")
    if not pred["checkpoint_hash"].astype(str).eq(binding["checkpoint_hash"]).all():
        raise RuntimeError("prediction checkpoint hash mismatch")
    if not pred["anchor_hash"].astype(str).eq(binding["anchor_hash"]).all():
        raise RuntimeError("prediction anchor hash mismatch")
    metrics = read_json(run_dir / "metrics.json")
    if len(set(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        if abs(float(metrics.get("auc", metrics.get("oot_auc", auc))) - auc) > 1e-8:
            raise RuntimeError("AUC recomputation mismatch")
    ks_value = ks_score(y_true.to_numpy(), y_score.to_numpy())
    if isinstance(ks_value, tuple):
        ks_value = ks_value[0]
    return {
        "prediction_rows": int(len(pred)),
        "feature_set_hash": str(pred["feature_set_hash"].iloc[0]),
        "prediction_hash": sha256_file(prediction_path),
        "recomputed_ks": float(ks_value),
    }


def _config_snapshot_text(spec: dict[str, str], config: ExperimentConfig) -> str:
    payload = {
        "experiment_version": "clip_v2",
        "dataset": spec["dataset"],
        "model": spec["model"],
        "selector": spec["selector"],
        "feature_budget": config.feature_budget,
        "config_hash": getattr(config, "config_hash", ""),
        "clip_selector_config": "configs/clip_v2/selector.yaml",
    }
    return "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n"


def _progress(index: int, total: int, spec: dict[str, str], stage: str, started: float) -> None:
    elapsed = 0.0 if started == 0.0 else time.time() - started
    print(f"[{index}/{total}] {spec['dataset']} {spec['model']} {spec['selector']} | {stage} | elapsed={elapsed:.1f}s", flush=True)


def _append_log(run_dir: Path, message: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "execution.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
        handle.flush()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
