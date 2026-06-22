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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter, score_coverage  # noqa: E402
from credit_risk_fs.clip.evaluation_aggregation import aggregate_status  # noqa: E402
from credit_risk_fs.clip.selector_validation import (  # noqa: E402
    load_clip_selector_config,
    validate_clip_selector_binding,
)
from credit_risk_fs.evaluation.drift import calculate_psi  # noqa: E402
from credit_risk_fs.evaluation.metrics import evaluate_model, ks_score  # noqa: E402
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


ACTIVE_DATASETS = ("homecredit", "lendingclub_v2")
LEGACY_DATASET = "lendingclub"
MODELS = ("lr", "catboost")
SELECTORS = ("clip", "clip_then_mrmr")
BASELINE_SELECTORS = ("mrmr", "llm", "llm_then_mrmr", "stable_core_llm_fill")
OUTPUT_ROOT = Path("results/clip/final_evaluation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final downstream evaluation for frozen CLIP selectors.")
    parser.add_argument("--dataset", choices=[*ACTIVE_DATASETS, LEGACY_DATASET], default=None)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--selector", choices=SELECTORS, default=None)
    parser.add_argument("--all", action="store_true", help="Run both active datasets.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Audit current final-evaluation run status without training.")
    parser.add_argument("--resume", action="store_true", help="Print a safe resume plan. Requires --execute to run models.")
    parser.add_argument("--execute", action="store_true", help="Actually execute planned expensive model runs.")
    return parser.parse_args()


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset == LEGACY_DATASET:
        raise RuntimeError("legacy LendingClub is forbidden for CLIP final evaluation")
    if getattr(args, "all", False):
        return list(ACTIVE_DATASETS)
    if args.dataset:
        return [args.dataset]
    if getattr(args, "dry_run", False) or getattr(args, "status", False) or getattr(args, "resume", False):
        return list(ACTIVE_DATASETS)
    raise RuntimeError("Specify --dataset homecredit, --dataset lendingclub_v2, --all, or --dry-run.")


def _selected_models(args: argparse.Namespace) -> list[str]:
    model = getattr(args, "model", None)
    return [model] if model else list(MODELS)


def _selected_selectors(args: argparse.Namespace) -> list[str]:
    selector = getattr(args, "selector", None)
    return [selector] if selector else list(SELECTORS)


def _run_specs(
    datasets: list[str],
    models: list[str] | None = None,
    selectors: list[str] | None = None,
) -> list[dict[str, str]]:
    models = models or list(MODELS)
    selectors = selectors or list(SELECTORS)
    return [
        {"dataset": dataset, "model": model, "selector": selector}
        for dataset in datasets
        for model in models
        for selector in selectors
    ]


def _run_id(spec: dict[str, str]) -> str:
    return f"{spec['dataset']}_{spec['model']}_{spec['selector']}"


def _project_config(dataset: str) -> dict[str, Any]:
    return load_named_project_config(dataset)


def _experiment_config(dataset: str, model: str, selector: str, run_dir: Path) -> ExperimentConfig:
    project = _project_config(dataset)
    feature_budget = resolve_feature_budget(project, model)
    model_kwargs = resolve_model_kwargs(project, model)
    config_hash_payload = {
        "dataset": dataset,
        "model": model,
        "selector": selector,
        "feature_budget": feature_budget,
        "clip_selector_config": "configs/clip/selector.yaml",
        "random_seed": int(project.get("random_seed", 42)),
    }
    return ExperimentConfig(
        experiment_name=selector,
        selector_name=selector,
        dataset_name=dataset,
        model_name=model,
        model_kwargs=model_kwargs,
        data_dir=str(project["data_dir"]),
        description_path=str(project["description_path"]),
        base_output_dir=str(OUTPUT_ROOT),
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
            "config_path": "configs/clip/selector.yaml",
            "dataset": dataset,
            "model_name": model,
            "missing_feature_policy": "exclude_with_manifest",
        },
        experiment_type="clip_final_evaluation",
        config_hash=sha256_text(json.dumps(config_hash_payload, sort_keys=True)),
        data_fingerprint=build_data_version(project["data_dir"]),
    )


def _validate_boundaries(datasets: list[str]) -> dict[str, Any]:
    selector_config = load_clip_selector_config("configs/clip/selector.yaml")
    binding = validate_clip_selector_binding(selector_config)
    coverage = score_coverage("configs/clip/selector.yaml")
    if set(datasets) - set(ACTIVE_DATASETS):
        raise RuntimeError(f"unsupported datasets requested: {datasets}")
    if LEGACY_DATASET in datasets:
        raise RuntimeError("legacy LendingClub is forbidden")
    for dataset in datasets:
        if coverage[dataset]["feature_count"] <= 0:
            raise RuntimeError(f"{dataset}: no CLIP score coverage")
    return {"binding": binding, "coverage": coverage, "selector_config": selector_config}


def _dry_run(datasets: list[str]) -> int:
    validation = _validate_boundaries(datasets)
    specs = _run_specs(datasets)
    missing_baselines = []
    for dataset in datasets:
        table = Path("results") / dataset / "final_comparison_table.csv"
        if not table.exists():
            missing_baselines.append(str(table))
    payload = {
        "dry_run": True,
        "active_datasets": datasets,
        "legacy_lendingclub_allowed": False,
        "run_count": len(specs),
        "runs": specs,
        "feature_budgets": {model: resolve_feature_budget(_project_config(datasets[0]), model) for model in MODELS},
        "clip_checkpoint_hash": validation["binding"]["checkpoint_hash"],
        "anchor_hash": validation["binding"]["anchor_hash"],
        "score_coverage": validation["coverage"],
        "missing_baseline_tables": missing_baselines,
        "downstream_model_training": False,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if not missing_baselines else 1


def _feature_set_hash(features: list[str]) -> str:
    return sha256_text(json.dumps(sorted(str(f) for f in features), sort_keys=True))


def _load_metadata(dataset: str) -> pd.DataFrame:
    path = Path(_project_config(dataset)["description_path"])
    if not path.exists():
        return pd.DataFrame(columns=["feature_name", "semantic_group", "source_table_or_formula"])
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    if dataset == "homecredit":
        return pd.DataFrame(
            {
                "feature_name": frame.get("Row", pd.Series(dtype=str)).astype(str),
                "semantic_group": frame.get("semantic_group", pd.Series([pd.NA] * len(frame))).astype(str)
                if "semantic_group" in frame.columns
                else pd.NA,
                "source_table_or_formula": frame.get("Table", pd.Series([pd.NA] * len(frame))).astype(str),
            }
        )
    return pd.DataFrame(
        {
            "feature_name": frame.get("feature", pd.Series(dtype=str)).astype(str),
            "semantic_group": frame.get("semantic_group", pd.Series([pd.NA] * len(frame))).astype(str),
            "source_table_or_formula": frame.get("source_column_or_formula", pd.Series([pd.NA] * len(frame))).astype(str),
        }
    )


def _enrich_selected_features(dataset: str, model: str, selector: str, run_dir: Path, binding: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    selected_path = run_dir / "features" / "final_selected_features.csv"
    selected = pd.read_csv(selected_path)
    feature_col = "feature_name" if "feature_name" in selected.columns else "feature"
    features = selected[feature_col].astype(str).tolist()
    feature_hash = _feature_set_hash(features)
    adapter = ClipScoreAdapter("configs/clip/selector.yaml", dataset=dataset)
    ranking = adapter.rank_candidates(features, missing_feature_policy="exclude_with_manifest")
    metadata = _load_metadata(dataset)
    manifest_path = (
        run_dir / "llm_responses" / "final_dev" / "clip_then_mrmr_selection_manifest.csv"
        if selector == "clip_then_mrmr"
        else run_dir / "llm_responses" / "final_dev" / "clip_selection_manifest.csv"
    )
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_cols = [
        col
        for col in ["feature_name", "clip_score", "clip_rank", "screening_pool_member", "final_selected", "final_rank"]
        if col in manifest.columns
    ]
    merged = pd.DataFrame({"feature_name": features})
    merged["dataset"] = dataset
    merged["model"] = model
    merged["selector"] = selector
    merged["feature_set_hash"] = feature_hash
    merged = merged.merge(ranking[["feature_name", "learned_similarity", "clip_rank"]], on="feature_name", how="left")
    if manifest_cols:
        merged = merged.merge(manifest[manifest_cols].drop_duplicates("feature_name"), on="feature_name", how="left", suffixes=("", "_manifest"))
    merged = merged.merge(metadata, on="feature_name", how="left")
    merged["semantic_group"] = merged["semantic_group"].where(
        merged["semantic_group"].notna() & merged["semantic_group"].astype(str).ne("nan"),
        merged["feature_name"].map(infer_semantic_group),
    )
    merged["clip_score"] = merged.get("clip_score", merged["learned_similarity"])
    merged["checkpoint_hash"] = binding["checkpoint_hash"]
    merged["anchor_hash"] = binding["anchor_hash"]
    merged["missing_score_count"] = int(merged["learned_similarity"].isna().sum())
    merged["duplicate_count"] = int(merged["feature_name"].duplicated().sum())
    merged["blocked_feature_count"] = 0
    merged["mrmr_rank"] = merged["final_rank"] if selector == "clip_then_mrmr" and "final_rank" in merged.columns else pd.NA
    return merged, feature_hash


def _write_prediction_artifact(
    *,
    dataset: str,
    model: str,
    selector: str,
    run_dir: Path,
    output_dir: Path,
    binding: dict[str, Any],
    feature_hash: str,
) -> Path:
    source = pd.read_csv(run_dir / "results" / "oot_predictions.csv")
    pred = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "selector": selector,
            "evaluation_index": np.arange(len(source), dtype=int),
            "y_true": source["y_true"].astype(int),
            "y_pred_proba": pd.to_numeric(source["y_pred_proba"], errors="coerce"),
            "y_pred": source["y_pred"].astype(int),
            "split": "OOT",
            "checkpoint_hash": binding["checkpoint_hash"],
            "anchor_hash": binding["anchor_hash"],
            "feature_set_hash": feature_hash,
            "run_id": run_dir.name,
        }
    )
    out = output_dir / "predictions" / f"{dataset}_{model}_{selector}_oot_predictions.csv"
    _atomic_write_csv(out, pred)
    return out


def _psi_bucket(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.10:
        return "low"
    if value < 0.25:
        return "moderate"
    return "high"


def _semantic_summary(selected_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in selected_long.groupby(["dataset", "model", "selector"], dropna=False):
        counts = frame["semantic_group"].fillna("unknown").astype(str).value_counts()
        source_count = int(frame["source_table_or_formula"].fillna("unknown").astype(str).nunique())
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "selector": keys[2],
                "selected_feature_count": int(len(frame)),
                "semantic_group_count": int(counts.shape[0]),
                "largest_semantic_group_share": float(counts.iloc[0] / len(frame)) if len(frame) else np.nan,
                "source_table_coverage": source_count,
            }
        )
    return pd.DataFrame(rows)


def _base_family(feature: str) -> str:
    parts = str(feature).split("_")
    if len(parts) >= 3 and parts[-1] in {"MEAN", "SUM", "MIN", "MAX", "VAR", "FLAG"}:
        return "_".join(parts[:-1])
    return str(feature)


def _redundancy_summary(selected_long: pd.DataFrame, run_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for keys, frame in selected_long.groupby(["dataset", "model", "selector"], dropna=False):
        families = frame["feature_name"].map(_base_family)
        repeated_share = float(families.duplicated(keep=False).mean()) if len(families) else np.nan
        run_name = f"{keys[0]}_{keys[1]}_{keys[2]}"
        psi_path = run_dirs[run_name] / "results" / "selected_feature_psi.csv"
        psi = pd.read_csv(psi_path) if psi_path.exists() else pd.DataFrame()
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "selector": keys[2],
                "selected_feature_count": int(len(frame)),
                "exact_duplicate_count": int(frame["feature_name"].duplicated().sum()),
                "near_duplicate_family_count": int(families.value_counts().gt(1).sum()),
                "repeated_base_family_share": repeated_share,
                "pairwise_feature_correlation_status": "not_materialized_by_final_pipeline",
                "selected_feature_psi_mean": float(pd.to_numeric(psi.get("psi", pd.Series(dtype=float)), errors="coerce").mean())
                if not psi.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _seed_sensitivity() -> pd.DataFrame:
    selected = read_json("results/clip/training/model_selection_manifest.json")
    rows = []
    for dataset in ACTIVE_DATASETS:
        scores = pd.read_csv(f"results/clip/training/{dataset}_learned_scores.csv")
        selected_top20 = set(scores.sort_values("learned_rank").head(20)["feature_name"].astype(str))
        selected_top40 = set(scores.sort_values("learned_rank").head(40)["feature_name"].astype(str))
        groups = scores.sort_values("learned_rank").head(40)["feature_name"].map(infer_semantic_group)
        rows.append(
            {
                "dataset": dataset,
                "primary_selected_seed": int(selected["selected_seed"]),
                "seed_count": int(len(selected["all_seed_results"])),
                "feature_rank_correlation_across_seeds": np.nan,
                "top20_jaccard_across_seeds": np.nan,
                "top40_jaccard_across_seeds": np.nan,
                "top20_count_primary_checkpoint": int(len(selected_top20)),
                "top40_count_primary_checkpoint": int(len(selected_top40)),
                "semantic_group_count_top40_primary_checkpoint": int(groups.nunique()),
                "downstream_per_seed_evaluation": "skipped_not_computationally_practical",
                "limitation": "retained seed checkpoints exist, but per-seed learned score files were not materialized",
            }
        )
    return pd.DataFrame(rows)


def _load_baselines(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = Path("results") / dataset / "final_comparison_table.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["selector"].astype(str).isin(BASELINE_SELECTORS)].copy()
        frame["source"] = "frozen_baseline"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _paired_auc_bootstrap(left: pd.DataFrame, right: pd.DataFrame, *, n_bootstrap: int = 300, seed: int = 42) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("prediction lengths do not match")
    y = left["y_true"].to_numpy()
    left_score = left["y_pred_proba"].to_numpy()
    right_score = right["y_pred_proba"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], right_score[idx]) - roc_auc_score(y[idx], left_score[idx])))
    if not deltas:
        return {"method": "paired_bootstrap_auc", "status": "failed_no_valid_resamples"}
    arr = np.asarray(deltas)
    return {
        "method": "paired_bootstrap_auc",
        "status": "ok",
        "n": int(n),
        "point_delta": float(roc_auc_score(y, right_score) - roc_auc_score(y, left_score)),
        "ci95_lower": float(np.quantile(arr, 0.025)),
        "ci95_upper": float(np.quantile(arr, 0.975)),
        "p_value_two_sided": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
        "bootstrap_samples": int(len(arr)),
    }


def _statistical_significance(new_summary: pd.DataFrame, baseline_frame: pd.DataFrame, prediction_dir: Path) -> pd.DataFrame:
    rows = []
    for new_row in new_summary.to_dict("records"):
        new_pred = pd.read_csv(prediction_dir / f"{new_row['dataset']}_{new_row['model']}_{new_row['selector']}_oot_predictions.csv")
        baselines = baseline_frame[
            (baseline_frame["dataset_name"].astype(str).eq(str(new_row["dataset"])))
            & (baseline_frame["model"].astype(str).eq(str(new_row["model"])))
            & (baseline_frame["selector"].astype(str).isin(BASELINE_SELECTORS))
        ]
        for base in baselines.to_dict("records"):
            base_pred_path = Path(str(base["output_folder"])) / "results" / "oot_predictions.csv"
            payload = {
                "dataset": new_row["dataset"],
                "model": new_row["model"],
                "new_selector": new_row["selector"],
                "baseline_selector": base["selector"],
                "baseline_output_folder": base["output_folder"],
            }
            if not base_pred_path.exists():
                payload.update({"status": "missing_baseline_predictions", "method": "not_run"})
            else:
                base_pred = pd.read_csv(base_pred_path)
                try:
                    stats = _paired_auc_bootstrap(base_pred, new_pred)
                    payload.update(stats)
                except Exception as exc:
                    payload.update({"status": "failed", "method": "paired_bootstrap_auc", "error": repr(exc)})
            rows.append(payload)
    return pd.DataFrame(rows)


def _collect_run(
    *,
    dataset: str,
    model: str,
    selector: str,
    run_dir: Path,
    output_dir: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    selected_long, feature_hash = _enrich_selected_features(dataset, model, selector, run_dir, binding)
    pred_path = _write_prediction_artifact(
        dataset=dataset,
        model=model,
        selector=selector,
        run_dir=run_dir,
        output_dir=output_dir,
        binding=binding,
        feature_hash=feature_hash,
    )
    oot = pd.read_csv(run_dir / "results" / "oot_test_results.csv").iloc[0].to_dict()
    total_candidates = int(oot.get("total_candidate_feature_count", 0) or 0)
    scored_candidates = int(ClipScoreAdapter("configs/clip/selector.yaml", dataset=dataset).score_frame(use_cache=True)["feature_name"].nunique())
    missing_score_count = max(total_candidates - scored_candidates, 0)
    selected_long["eligible_candidate_count"] = scored_candidates
    selected_long["total_candidate_feature_count"] = total_candidates
    selected_long["missing_score_count"] = missing_score_count
    split = read_json(run_dir / "data_split_manifest.json")
    runtime = pd.read_csv(run_dir / "results" / "runtime_summary.csv").iloc[0].to_dict()
    predictions = pd.read_csv(pred_path)
    recomputed = evaluate_model(predictions["y_true"], predictions["y_pred_proba"], y_pred=predictions["y_pred"])
    if abs(float(recomputed["auc"]) - float(oot["auc"])) > 1e-10:
        raise RuntimeError(f"{run_dir.name}: saved AUC does not match prediction recomputation")
    if abs(float(oot["gini"]) - (2 * float(oot["auc"]) - 1)) > 1e-10:
        raise RuntimeError(f"{run_dir.name}: saved Gini does not equal 2*AUC-1")
    ks_recomputed, _ = ks_score(predictions["y_true"], predictions["y_pred_proba"])
    if abs(float(ks_recomputed) - float(oot["ks"])) > 1e-10:
        raise RuntimeError(f"{run_dir.name}: saved KS does not match prediction recomputation")
    run_metrics = {
        "dataset": dataset,
        "model": model,
        "selector": selector,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "prediction_path": str(pred_path),
        "oot_auc": float(oot["auc"]),
        "oot_gini": float(oot["gini"]),
        "oot_ks": float(oot["ks"]),
        "lift_at_10": float(oot.get("lift_at_10", np.nan)),
        "model_score_psi": float(oot.get("model_score_psi", np.nan)),
        "model_score_psi_bucket": _psi_bucket(float(oot.get("model_score_psi", np.nan))),
        "selected_feature_count": int(oot.get("selected_feature_count", len(selected_long))),
        "eligible_candidate_count": scored_candidates,
        "total_candidate_feature_count": total_candidates,
        "missing_score_count": missing_score_count,
        "blocked_feature_count": 0,
        "duplicate_count": 0,
        "prediction_count": int(len(predictions)),
        "dev_target_rate": float(split["dev"]["target_rate"]),
        "oot_target_rate": float(split["oot"]["target_rate"]),
        "runtime_seconds": float(runtime.get("total_runtime_seconds", np.nan)),
        "feature_set_hash": feature_hash,
        "checkpoint_hash": binding["checkpoint_hash"],
        "anchor_hash": binding["anchor_hash"],
        "log_loss": float(oot.get("log_loss", np.nan)),
        "brier": float(oot.get("brier", np.nan)),
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    _atomic_write_csv(run_dir / "selected_features_enriched.csv", selected_long)
    return {"metrics": run_metrics, "selected_long": selected_long}


def _write_limitations(output_dir: Path) -> None:
    text = """# CLIP Final Evaluation Limitations

- OOT performance is the primary evidence.
- DEV CV is diagnostic because the CLIP representation was prepared at DEV level and not rebuilt fold-locally.
- `clip_then_mrmr` reruns mRMR on current DEV/fold data, but the CLIP screening representation is frozen.
- Statistical view is `missingness_only`; do not claim broad statistical feature-quality learning.
- LendingClub v2 was external application only and was not used for CLIP tuning.
- Seed sensitivity is ranking-only unless per-seed learned score files are materialized.
"""
    (output_dir / "evaluation_limitations.md").write_text(text, encoding="utf-8")


def _write_run_manifest(output_dir: Path, rows: list[dict[str, Any]], binding: dict[str, Any]) -> None:
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "completed",
            "run_count": len(rows),
            "runs": rows,
            "checkpoint_hash": binding["checkpoint_hash"],
            "anchor_hash": binding["anchor_hash"],
            "legacy_lendingclub_allowed": False,
            "frozen_baselines_overwritten": False,
            "dev_cv_policy": "diagnostic_only_clip_not_fold_local",
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        },
    )


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _atomic_write_json(path: Path, payload: Any) -> Path:
    return _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")
        handle.flush()


def _safe_hash(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except Exception:
        return None


def _safe_json(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size == 0:
            return False, "zero-byte"
        json.loads(path.read_text(encoding="utf-8"))
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def _safe_csv(path: Path) -> tuple[bool, str]:
    try:
        if path.stat().st_size == 0:
            return False, "zero-byte"
        pd.read_csv(path, nrows=5)
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def _csv_row_count(path: Path) -> int | None:
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def _validate_artifact(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "zero-byte"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _safe_json(path)
    if suffix == ".csv":
        return _safe_csv(path)
    return True, ""


def _top_prediction_path(output_dir: Path, spec: dict[str, str]) -> Path:
    return output_dir / "predictions" / f"{_run_id(spec)}_oot_predictions.csv"


def _required_run_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "config_snapshot.json",
        run_dir / "features" / "final_selected_features.csv",
        run_dir / "results" / "oot_test_results.csv",
        run_dir / "results" / "oot_predictions.csv",
        run_dir / "results" / "runtime_summary.csv",
        run_dir / "results" / "experiment_summary.csv",
        run_dir / "data_split_manifest.json",
        run_dir / "leakage_audit.json",
        run_dir / "checkpoint_anchor_binding.json",
        run_dir / "source_hashes.json",
        run_dir / "selected_features_enriched.csv",
        run_dir / "models" / "final_model_metadata.json",
    ]


def _expected_source_hashes() -> dict[str, str]:
    paths = {
        "clip_selector_config": Path("configs/clip/selector.yaml"),
        "checkpoint_manifest": Path("results/clip/training/seeds/seed_55/checkpoint_manifest.json"),
        "model_selection_manifest": Path("results/clip/training/model_selection_manifest.json"),
        "anchor_manifest": Path("results/clip/training/learned_anchor_manifest.json"),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _lift_at_fraction(y_true: pd.Series, y_score: pd.Series, fraction: float = 0.10) -> float:
    frame = pd.DataFrame({"y_true": y_true.astype(float), "score": y_score.astype(float)})
    if frame.empty:
        return float("nan")
    top_n = max(1, int(np.ceil(len(frame) * fraction)))
    base_rate = float(frame["y_true"].mean())
    if base_rate == 0:
        return float("nan")
    return float(frame.sort_values("score", ascending=False).head(top_n)["y_true"].mean() / base_rate)


def _validate_predictions_and_metrics(run_dir: Path, top_pred_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "actual_prediction_rows": None,
        "expected_prediction_rows": None,
        "metrics_match": False,
        "notes": [],
    }
    run_pred_path = run_dir / "results" / "oot_predictions.csv"
    oot_path = run_dir / "results" / "oot_test_results.csv"
    split_path = run_dir / "data_split_manifest.json"
    for path in [run_pred_path, oot_path, top_pred_path, split_path]:
        ok, note = _validate_artifact(path)
        if not ok:
            result["notes"].append(f"{path.name}: {note}")
            return result
    run_pred = pd.read_csv(run_pred_path)
    top_pred = pd.read_csv(top_pred_path)
    oot = pd.read_csv(oot_path).iloc[0].to_dict()
    split = read_json(split_path)
    expected_rows = int(split.get("oot", {}).get("row_count", len(run_pred)))
    result["expected_prediction_rows"] = expected_rows
    result["actual_prediction_rows"] = int(len(top_pred))
    required_top_columns = {
        "dataset",
        "model",
        "selector",
        "evaluation_index",
        "y_true",
        "y_pred_proba",
        "y_pred",
        "split",
        "checkpoint_hash",
        "anchor_hash",
        "feature_set_hash",
        "run_id",
    }
    missing_cols = sorted(required_top_columns - set(top_pred.columns))
    if missing_cols:
        result["notes"].append(f"top prediction missing columns: {missing_cols}")
        return result
    if len(run_pred) != expected_rows or len(top_pred) != expected_rows:
        result["notes"].append(f"prediction row mismatch expected={expected_rows} run={len(run_pred)} top={len(top_pred)}")
        return result
    if run_pred[["y_true", "y_pred_proba", "y_pred"]].isna().any().any():
        result["notes"].append("run predictions contain nulls")
        return result
    if top_pred[["y_true", "y_pred_proba", "y_pred"]].isna().any().any():
        result["notes"].append("top predictions contain nulls")
        return result
    recomputed = evaluate_model(run_pred["y_true"], run_pred["y_pred_proba"], y_pred=run_pred["y_pred"])
    tolerances = {"auc": 1e-10, "gini": 1e-10, "ks": 1e-10, "lift_at_10": 1e-8}
    checks = {
        "auc": float(recomputed["auc"]),
        "gini": float(recomputed["gini"]),
        "ks": float(recomputed["ks"]),
    }
    if "lift_at_10" in oot and not pd.isna(oot["lift_at_10"]):
        checks["lift_at_10"] = _lift_at_fraction(run_pred["y_true"], run_pred["y_pred_proba"], 0.10)
    for metric, actual in checks.items():
        saved = float(oot[metric])
        if abs(actual - saved) > tolerances[metric]:
            result["notes"].append(f"{metric} mismatch saved={saved} recomputed={actual}")
            return result
    result["metrics_match"] = True
    result["valid"] = True
    return result


def _validate_hash_binding(run_dir: Path, spec: dict[str, str], binding: dict[str, Any]) -> tuple[bool, list[str], str | None]:
    notes: list[str] = []
    feature_hash: str | None = None
    try:
        snapshot = read_json(run_dir / "config_snapshot.json")
        expected_config_hash = _experiment_config(spec["dataset"], spec["model"], spec["selector"], run_dir).config_hash
        if snapshot.get("config_hash") != expected_config_hash:
            notes.append("config hash differs from current config")
    except Exception as exc:
        notes.append(f"config snapshot invalid: {exc!r}")
    try:
        selected = pd.read_csv(run_dir / "features" / "final_selected_features.csv")
        feature_col = "feature_name" if "feature_name" in selected.columns else "feature"
        feature_hash = _feature_set_hash(selected[feature_col].astype(str).tolist())
        enriched = pd.read_csv(run_dir / "selected_features_enriched.csv")
        if "feature_set_hash" in enriched.columns:
            hashes = set(enriched["feature_set_hash"].dropna().astype(str))
            if hashes and hashes != {feature_hash}:
                notes.append("selected feature hash differs from enriched file")
    except Exception as exc:
        notes.append(f"selected feature hash invalid: {exc!r}")
    try:
        saved_binding = read_json(run_dir / "checkpoint_anchor_binding.json")
        if saved_binding.get("checkpoint_hash") != binding["checkpoint_hash"]:
            notes.append("checkpoint hash differs")
        if saved_binding.get("anchor_hash") != binding["anchor_hash"]:
            notes.append("anchor hash differs")
    except Exception as exc:
        notes.append(f"checkpoint/anchor binding invalid: {exc!r}")
    try:
        source_hashes = read_json(run_dir / "source_hashes.json")
        expected = _expected_source_hashes()
        for key, value in expected.items():
            if source_hashes.get(key) != value:
                notes.append(f"source hash differs: {key}")
    except Exception as exc:
        notes.append(f"source hashes invalid: {exc!r}")
    return not notes, notes, feature_hash


def _completion_payload(
    *,
    spec: dict[str, str],
    run_dir: Path,
    binding: dict[str, Any],
    prediction_rows: int,
    recovered: bool = False,
) -> dict[str, Any]:
    source_hashes = read_json(run_dir / "source_hashes.json")
    feature_hash_ok, notes, feature_hash = _validate_hash_binding(run_dir, spec, binding)
    if not feature_hash_ok:
        raise RuntimeError(f"{_run_id(spec)} cannot be marked complete: {notes}")
    return {
        "run_id": _run_id(spec),
        "dataset": spec["dataset"],
        "model": spec["model"],
        "selector": spec["selector"],
        "completed_at": _utc_now(),
        "config_hash": read_json(run_dir / "config_snapshot.json").get("config_hash"),
        "feature_set_hash": feature_hash,
        "checkpoint_hash": binding["checkpoint_hash"],
        "anchor_hash": binding["anchor_hash"],
        "source_hashes": source_hashes,
        "prediction_file_hash": sha256_file(_top_prediction_path(OUTPUT_ROOT, spec)),
        "metrics_file_hash": sha256_file(run_dir / "results" / "oot_test_results.csv"),
        "prediction_row_count": prediction_rows,
        "completion_status": "complete_valid_recovered" if recovered else "complete_valid",
        "recovered_from_interruption": bool(recovered),
    }


def _validate_completed_run_directory(
    *,
    spec: dict[str, str],
    run_dir: Path,
    binding: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    notes = []
    valid_artifacts = 0
    required = _required_run_artifacts(run_dir)
    for path in required:
        ok, note = _validate_artifact(path)
        if ok:
            valid_artifacts += 1
        else:
            notes.append(f"{path.relative_to(run_dir)}: {note}")
    top_pred = _top_prediction_path(output_dir, spec)
    metric_check = _validate_predictions_and_metrics(run_dir, top_pred)
    if not metric_check["valid"]:
        notes.extend(metric_check["notes"])
    hash_ok, hash_notes, _ = _validate_hash_binding(run_dir, spec, binding)
    if not hash_ok:
        notes.extend(hash_notes)
    return {
        "valid": valid_artifacts == len(required) and metric_check["valid"] and hash_ok,
        "valid_artifact_count": valid_artifacts,
        "required_artifact_count": len(required),
        "actual_prediction_rows": metric_check["actual_prediction_rows"],
        "expected_prediction_rows": metric_check["expected_prediction_rows"],
        "notes": "; ".join(notes),
    }


def _classify_run(spec: dict[str, str], binding: dict[str, Any], output_dir: Path = OUTPUT_ROOT) -> dict[str, Any]:
    run_id = _run_id(spec)
    run_dir = output_dir / "runs" / run_id
    progress_dir = output_dir / "runs" / f"{run_id}.in_progress"
    top_pred_path = _top_prediction_path(output_dir, spec)
    row: dict[str, Any] = {
        "dataset": spec["dataset"],
        "model": spec["model"],
        "selector": spec["selector"],
        "run_id": run_id,
        "status": "unknown",
        "run_directory": str(run_dir),
        "started_at": "",
        "last_modified_at": "",
        "required_artifact_count": 0,
        "valid_artifact_count": 0,
        "prediction_file_exists": top_pred_path.exists(),
        "prediction_file_valid": False,
        "expected_prediction_rows": None,
        "actual_prediction_rows": None,
        "metrics_file_valid": False,
        "completion_marker_exists": False,
        "leakage_audit_valid": False,
        "safe_to_reuse": False,
        "recovery_action": "none",
        "notes": "",
        "_recoverable_without_marker": False,
    }
    notes: list[str] = []
    if progress_dir.exists():
        row["status"] = "in_progress_interrupted"
        row["run_directory"] = str(progress_dir)
        row["last_modified_at"] = pd.Timestamp(progress_dir.stat().st_mtime, unit="s").isoformat()
        row["recovery_action"] = "quarantine_in_progress_directory"
        row["notes"] = ".in_progress directory exists"
        return row
    if not run_dir.exists():
        row["status"] = "not_started"
        row["recovery_action"] = "run_on_resume"
        return row
    files = list(run_dir.rglob("*"))
    file_stats = [path.stat() for path in files if path.is_file()]
    if file_stats:
        row["started_at"] = pd.Timestamp(min(stat.st_mtime for stat in file_stats), unit="s").isoformat()
        row["last_modified_at"] = pd.Timestamp(max(stat.st_mtime for stat in file_stats), unit="s").isoformat()
    required = _required_run_artifacts(run_dir)
    row["required_artifact_count"] = len(required) + 1
    valid_artifacts = 0
    for path in required:
        ok, note = _validate_artifact(path)
        if ok:
            valid_artifacts += 1
        else:
            notes.append(f"{path.relative_to(run_dir)}: {note}")
    row["valid_artifact_count"] = valid_artifacts
    top_ok, top_note = _validate_artifact(top_pred_path)
    row["prediction_file_valid"] = bool(top_ok)
    if not top_ok:
        notes.append(f"top prediction: {top_note}")
    metrics_ok, metrics_note = _validate_artifact(run_dir / "results" / "oot_test_results.csv")
    row["metrics_file_valid"] = bool(metrics_ok)
    if not metrics_ok:
        notes.append(f"metrics: {metrics_note}")
    leakage_ok, leakage_note = _validate_artifact(run_dir / "leakage_audit.json")
    row["leakage_audit_valid"] = bool(leakage_ok)
    if not leakage_ok:
        notes.append(f"leakage audit: {leakage_note}")
    marker_path = run_dir / "RUN_COMPLETE.json"
    marker_ok, marker_note = _validate_artifact(marker_path)
    row["completion_marker_exists"] = marker_path.exists() and marker_ok
    if marker_path.exists() and not marker_ok:
        notes.append(f"completion marker: {marker_note}")
    if valid_artifacts < len(required) or not top_ok:
        row["status"] = "in_progress_interrupted" if valid_artifacts < len(required) else "complete_invalid"
        row["recovery_action"] = "quarantine_partial_outputs"
        row["notes"] = "; ".join(notes)
        return row
    metric_check = _validate_predictions_and_metrics(run_dir, top_pred_path)
    row["expected_prediction_rows"] = metric_check["expected_prediction_rows"]
    row["actual_prediction_rows"] = metric_check["actual_prediction_rows"]
    if not metric_check["valid"]:
        notes.extend(metric_check["notes"])
        row["status"] = "complete_invalid"
        row["recovery_action"] = "quarantine_inconsistent_outputs"
        row["notes"] = "; ".join(notes)
        return row
    hash_ok, hash_notes, _ = _validate_hash_binding(run_dir, spec, binding)
    if not hash_ok:
        notes.extend(hash_notes)
        row["status"] = "complete_invalid"
        row["recovery_action"] = "quarantine_stale_outputs"
        row["notes"] = "; ".join(notes)
        return row
    if not row["completion_marker_exists"]:
        row["status"] = "complete_invalid"
        row["recovery_action"] = "write_recovered_completion_marker_after_validation"
        row["_recoverable_without_marker"] = True
        notes.append("all artifacts validate but RUN_COMPLETE.json is absent")
        row["notes"] = "; ".join(notes)
        return row
    try:
        marker = read_json(marker_path)
        if marker.get("prediction_file_hash") != sha256_file(top_pred_path):
            notes.append("completion marker prediction hash differs")
        if marker.get("metrics_file_hash") != sha256_file(run_dir / "results" / "oot_test_results.csv"):
            notes.append("completion marker metrics hash differs")
        if marker.get("prediction_row_count") != metric_check["actual_prediction_rows"]:
            notes.append("completion marker prediction row count differs")
    except Exception as exc:
        notes.append(f"completion marker invalid: {exc!r}")
    if notes:
        row["status"] = "complete_invalid"
        row["recovery_action"] = "quarantine_inconsistent_outputs"
        row["notes"] = "; ".join(notes)
        return row
    row["status"] = "complete_valid"
    row["valid_artifact_count"] = row["required_artifact_count"]
    row["safe_to_reuse"] = True
    row["recovery_action"] = "reuse"
    row["notes"] = "validated"
    return row


def _file_integrity_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        record: dict[str, Any] = {
            "path": relative,
            "size": path.stat().st_size,
            "last_modified_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
            "sha256": _safe_hash(path),
            "valid": True,
            "issue": "",
        }
        ok, note = _validate_artifact(path)
        record["valid"] = bool(ok)
        record["issue"] = note
        records.append(record)
    return records


def _write_recovery_artifacts(
    *,
    audit_rows: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
    output_dir: Path,
    recovery_actions: list[dict[str, Any]] | None = None,
    pre_file_records: list[dict[str, Any]] | None = None,
) -> None:
    recovery_dir = output_dir / "interruption_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    pre_file_records = pre_file_records or file_records
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in audit_rows]
    audit_frame = pd.DataFrame(public_rows)
    _atomic_write_csv(recovery_dir / "interrupted_run_audit.csv", audit_frame)
    _atomic_write_json(recovery_dir / "interrupted_run_audit.json", public_rows)
    _atomic_write_csv(recovery_dir / "file_integrity_audit.csv", pd.DataFrame(file_records))
    _atomic_write_json(
        recovery_dir / "pre_recovery_file_hashes.json",
        {row["path"]: {"sha256": row["sha256"], "size": row["size"], "last_modified_at": row["last_modified_at"]} for row in pre_file_records},
    )
    quarantine_records = _quarantine_records_from_disk(output_dir)
    if recovery_actions:
        quarantine_records.extend([action for action in recovery_actions if "quarantine_path" in action])
    if quarantine_records:
        _atomic_write_csv(recovery_dir / "quarantine_manifest.csv", pd.DataFrame(quarantine_records))
        _atomic_write_json(recovery_dir / "quarantine_manifest.json", quarantine_records)
    lines = [
        "# CLIP Final Evaluation Interruption Recovery Plan",
        "",
        f"Generated at: {_utc_now()}",
        "",
        "## Run Status",
        "",
    ]
    for row in public_rows:
        lines.append(f"- {row['run_id']}: {row['status']} -> {row['recovery_action']}")
    lines.extend(["", "## Recovery Actions", ""])
    if recovery_actions:
        for action in recovery_actions:
            lines.append(f"- {action['run_id']}: {action['action']} ({action.get('reason', '')})")
    else:
        lines.append("- No mutation performed by this command.")
    lines.extend(
        [
            "",
            "## Execution Policy",
            "",
            "- `--status` audits only and runs no models.",
            "- `--resume` prints a plan and runs no models unless `--execute` is present.",
            "- Real execution writes to `<run_id>.in_progress/` and writes `RUN_COMPLETE.json` last.",
        ]
    )
    _atomic_write_text(recovery_dir / "recovery_plan.md", "\n".join(lines) + "\n")


def _recover_markers(audit_rows: list[dict[str, Any]], binding: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    actions = []
    for row in audit_rows:
        if not row.get("_recoverable_without_marker"):
            continue
        spec = {"dataset": row["dataset"], "model": row["model"], "selector": row["selector"]}
        run_dir = output_dir / "runs" / row["run_id"]
        payload = _completion_payload(
            spec=spec,
            run_dir=run_dir,
            binding=binding,
            prediction_rows=int(row["actual_prediction_rows"]),
            recovered=True,
        )
        _atomic_write_json(run_dir / "RUN_COMPLETE.json", payload)
        actions.append(
            {
                "run_id": row["run_id"],
                "action": "wrote_recovered_completion_marker",
                "reason": "all required artifacts and recomputed metrics validated before marker creation",
            }
        )
    return actions


def _quarantine_invalid_runs(audit_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    actions = []
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    quarantine_root = output_dir / "interruption_recovery" / "quarantine" / timestamp
    for row in audit_rows:
        if row["status"] in {"complete_valid", "not_started"} or row.get("_recoverable_without_marker"):
            continue
        run_id = row["run_id"]
        source_dirs = [output_dir / "runs" / run_id, output_dir / "runs" / f"{run_id}.in_progress"]
        moved = False
        for source in source_dirs:
            if not source.exists():
                continue
            source_files = [path for path in source.rglob("*") if path.is_file()]
            dest = quarantine_root / run_id / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            for file_path in source_files:
                relative = file_path.relative_to(source)
                dest_file = dest / relative
                stat = dest_file.stat()
                actions.append(
                    {
                        "run_id": run_id,
                        "action": "quarantined_file",
                        "reason": row["notes"],
                        "original_path": str(file_path),
                        "quarantine_path": str(dest_file),
                        "original_hash": sha256_file(dest_file),
                        "size": int(stat.st_size),
                        "modification_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
                    }
                )
            moved = True
        top_pred = _top_prediction_path(output_dir, {"dataset": row["dataset"], "model": row["model"], "selector": row["selector"]})
        if moved and top_pred.exists():
            stat = top_pred.stat()
            dest = quarantine_root / run_id / "predictions" / top_pred.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(top_pred), str(dest))
            actions.append(
                {
                    "run_id": run_id,
                    "action": "quarantined_prediction",
                    "reason": row["notes"],
                    "original_path": str(top_pred),
                    "quarantine_path": str(dest),
                    "original_hash": sha256_file(dest),
                    "size": int(stat.st_size),
                    "modification_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
                }
            )
    return actions


def _quarantine_records_from_disk(output_dir: Path) -> list[dict[str, Any]]:
    quarantine_root = output_dir / "interruption_recovery" / "quarantine"
    if not quarantine_root.exists():
        return []
    records = []
    for file_path in sorted(quarantine_root.rglob("*")):
        if not file_path.is_file():
            continue
        parts = file_path.relative_to(quarantine_root).parts
        if len(parts) < 4:
            continue
        _, run_id, kind = parts[:3]
        if kind == "predictions":
            original_path = str(output_dir / "predictions" / file_path.name)
        else:
            original_path = str(output_dir / "runs" / kind / Path(*parts[3:]))
        stat = file_path.stat()
        records.append(
            {
                "run_id": run_id,
                "action": "quarantined_file",
                "reason": "existing quarantine record",
                "original_path": original_path,
                "quarantine_path": str(file_path),
                "original_hash": sha256_file(file_path),
                "size": int(stat.st_size),
                "modification_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
            }
        )
    return records


def _audit_runs(
    specs: list[dict[str, str]],
    *,
    binding: dict[str, Any],
    output_dir: Path = OUTPUT_ROOT,
    recover: bool = False,
    write_artifacts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
    pre_file_records = _file_integrity_records(output_dir)
    audit_rows = [_classify_run(spec, binding, output_dir) for spec in specs]
    actions: list[dict[str, Any]] = []
    if recover:
        actions.extend(_recover_markers(audit_rows, binding, output_dir))
        actions.extend(_quarantine_invalid_runs(audit_rows, output_dir))
        audit_rows = [_classify_run(spec, binding, output_dir) for spec in specs]
    file_records = _file_integrity_records(output_dir)
    if write_artifacts:
        _write_recovery_artifacts(
            audit_rows=audit_rows,
            file_records=file_records,
            output_dir=output_dir,
            recovery_actions=actions,
            pre_file_records=pre_file_records,
        )
    return audit_rows, file_records, actions


def _print_status(audit_rows: list[dict[str, Any]]) -> None:
    public = [{key: value for key, value in row.items() if not key.startswith("_")} for row in audit_rows]
    aggregates = aggregate_status(OUTPUT_ROOT)
    print(
        json.dumps(
            {
                "run_count": len(public),
                "runs": public,
                "aggregate_status": aggregates,
                "prompt7_complete": bool(
                    sum(1 for row in public if row["status"] == "complete_valid") == 8
                    and aggregates.get("aggregate_complete") is True
                ),
            },
            indent=2,
            default=str,
        )
    )


def _print_execution_plan(specs: list[dict[str, str]], audit_rows: list[dict[str, Any]], *, execute: bool) -> list[dict[str, str]]:
    by_id = {row["run_id"]: row for row in audit_rows}
    planned = []
    skipped = []
    for spec in specs:
        row = by_id[_run_id(spec)]
        if row["status"] == "complete_valid" and row["safe_to_reuse"]:
            skipped.append(_run_id(spec))
        else:
            planned.append(spec)
    payload = {
        "execute": bool(execute),
        "expensive_model_training": bool(execute and planned),
        "planned_run_count": len(planned),
        "planned_runs": planned,
        "skipped_valid_completed_runs": skipped,
        "safe_resume_command": "uv run python scripts/run_clip_final_evaluation.py --resume --execute",
    }
    print(json.dumps(payload, indent=2, default=str))
    return planned


def _promote_completed_run(progress_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory: {final_dir}")
    progress_dir.replace(final_dir)


def _execute_specs(specs: list[dict[str, str]], datasets: list[str]) -> int:
    started = time.time()
    validation = _validate_boundaries(datasets)
    binding = validation["binding"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "runs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "predictions").mkdir(exist_ok=True)

    run_rows = []
    selected_frames = []
    run_dirs: dict[str, Path] = {}
    specs_by_dataset: dict[str, list[dict[str, str]]] = {dataset: [] for dataset in datasets}
    for spec in specs:
        specs_by_dataset[spec["dataset"]].append(spec)
    total = len(specs)
    completed_so_far = 0
    for dataset in datasets:
        prepared = prepare_modeling_data(_experiment_config(dataset, "lr", "clip", OUTPUT_ROOT / "_prep"))
        for spec in specs_by_dataset[dataset]:
            completed_so_far += 1
            model = spec["model"]
            selector = spec["selector"]
            run_name = _run_id(spec)
            final_dir = OUTPUT_ROOT / "runs" / run_name
            progress_dir = OUTPUT_ROOT / "runs" / f"{run_name}.in_progress"
            status = _classify_run(spec, binding, OUTPUT_ROOT)
            if status["status"] == "complete_valid" and status["safe_to_reuse"]:
                print(f"[{completed_so_far}/{total}] reusing {run_name}")
                collected = _collect_run(
                    dataset=dataset,
                    model=model,
                    selector=selector,
                    run_dir=final_dir,
                    output_dir=OUTPUT_ROOT,
                    binding=binding,
                )
                run_rows.append(collected["metrics"])
                selected_frames.append(collected["selected_long"])
                run_dirs[run_name] = final_dir
                continue
            if final_dir.exists() or progress_dir.exists():
                raise RuntimeError(f"{run_name}: existing incomplete/invalid artifacts must be quarantined before execution")
            config = _experiment_config(dataset, model, selector, progress_dir)
            config_snapshot = {
                "dataset": dataset,
                "model": model,
                "selector": selector,
                "feature_budget": config.feature_budget,
                "config_hash": config.config_hash,
                "clip_selector_config": "configs/clip/selector.yaml",
            }
            progress_dir.mkdir(parents=True, exist_ok=False)
            log_path = progress_dir / "execution.log"
            stage_started = time.time()
            try:
                print(f"[{completed_so_far}/{total}] {run_name}: artifact-writing start")
                _append_log(log_path, f"run_start dataset={dataset} model={model} selector={selector}")
                _atomic_write_json(progress_dir / "config_snapshot.json", config_snapshot)
                _append_log(log_path, "model_fit start")
                print(f"[{completed_so_far}/{total}] {run_name}: model-fit start")
                run = run_experiment(config, prepared_data=prepared)
                run_dir_effective = run.exp_dir
                _append_log(log_path, f"model_fit end elapsed_seconds={time.time() - stage_started:.1f}")
                print(f"[{completed_so_far}/{total}] {run_name}: prediction/metric artifact validation start")
                collected = _collect_run(
                    dataset=dataset,
                    model=model,
                    selector=selector,
                    run_dir=run_dir_effective,
                    output_dir=OUTPUT_ROOT,
                    binding=binding,
                )
                shutil.copy2(run_dir_effective / "leakage_report.json", run_dir_effective / "leakage_audit.json")
                _atomic_write_json(
                    run_dir_effective / "checkpoint_anchor_binding.json",
                    {
                        "checkpoint_hash": binding["checkpoint_hash"],
                        "anchor_hash": binding["anchor_hash"],
                        "statistical_view_scope": binding["statistical_view_scope"],
                    },
                )
                _atomic_write_json(run_dir_effective / "source_hashes.json", _expected_source_hashes())
                validation_row = _validate_completed_run_directory(
                    spec=spec,
                    run_dir=run_dir_effective,
                    binding=binding,
                    output_dir=OUTPUT_ROOT,
                )
                if not validation_row["valid"]:
                    raise RuntimeError(f"{run_name}: artifacts failed validation before promotion: {validation_row['notes']}")
                _promote_completed_run(progress_dir, final_dir)
                payload = _completion_payload(
                    spec=spec,
                    run_dir=final_dir,
                    binding=binding,
                    prediction_rows=int(validation_row["actual_prediction_rows"]),
                    recovered=False,
                )
                _atomic_write_json(final_dir / "RUN_COMPLETE.json", payload)
                _append_log(final_dir / "execution.log", "run_complete marker_written_last")
                print(f"[{completed_so_far}/{total}] {run_name}: completed elapsed_seconds={time.time() - stage_started:.1f}")
                run_rows.append(collected["metrics"])
                selected_frames.append(collected["selected_long"])
                run_dirs[run_name] = final_dir
            except KeyboardInterrupt:
                _append_log(log_path, "interrupted KeyboardInterrupt no_completion_marker")
                print(f"{run_name}: interrupted. Safe resume: uv run python scripts/run_clip_final_evaluation.py --resume")
                raise
            except Exception as exc:
                _append_log(log_path, f"failed {exc!r} no_completion_marker")
                raise

    if not run_rows:
        print(json.dumps({"status": "no_runs_executed", "runtime_seconds": time.time() - started}, indent=2))
        return 0
    evaluation = pd.DataFrame(run_rows)
    selected_long = pd.concat(selected_frames, ignore_index=True)
    baselines = _load_baselines(datasets)
    clip_comparison = evaluation.rename(
        columns={"dataset": "dataset_name", "model": "model", "selector": "selector"}
    ).copy()
    clip_comparison["source"] = "clip_final_evaluation"
    comparison_cols = [col for col in baselines.columns if col in clip_comparison.columns]
    comparison = pd.concat([baselines, clip_comparison[comparison_cols]], ignore_index=True, sort=False)
    semantic = _semantic_summary(selected_long)
    redundancy = _redundancy_summary(selected_long, run_dirs)
    runtime = evaluation[["dataset", "model", "selector", "run_id", "runtime_seconds"]].copy()
    score_psi = evaluation[["dataset", "model", "selector", "model_score_psi", "model_score_psi_bucket"]].copy()
    seed = _seed_sensitivity()
    significance = _statistical_significance(evaluation, baselines, OUTPUT_ROOT / "predictions")

    _atomic_write_csv(OUTPUT_ROOT / "evaluation_summary.csv", evaluation)
    _atomic_write_json(OUTPUT_ROOT / "evaluation_summary.json", evaluation.to_dict("records"))
    _atomic_write_csv(OUTPUT_ROOT / "comparison_with_frozen_baselines.csv", comparison)
    _atomic_write_csv(OUTPUT_ROOT / "selected_features_long.csv", selected_long)
    selected_long.groupby(["dataset", "model", "selector"], as_index=False).agg(
        selected_feature_count=("feature_name", "count"),
        feature_set_hash=("feature_set_hash", "first"),
        missing_score_count=("missing_score_count", "max"),
        duplicate_count=("duplicate_count", "max"),
        blocked_feature_count=("blocked_feature_count", "max"),
    ).pipe(lambda frame: _atomic_write_csv(OUTPUT_ROOT / "selected_feature_summary.csv", frame))
    _atomic_write_csv(OUTPUT_ROOT / "semantic_coverage_summary.csv", semantic)
    _atomic_write_csv(OUTPUT_ROOT / "redundancy_summary.csv", redundancy)
    _atomic_write_csv(OUTPUT_ROOT / "runtime_summary.csv", runtime)
    _atomic_write_csv(OUTPUT_ROOT / "score_psi_summary.csv", score_psi)
    _atomic_write_csv(OUTPUT_ROOT / "seed_sensitivity_summary.csv", seed)
    _atomic_write_csv(OUTPUT_ROOT / "statistical_significance_summary.csv", significance)
    _write_limitations(OUTPUT_ROOT)
    _write_run_manifest(OUTPUT_ROOT, run_rows, binding)
    print(json.dumps({"status": "completed", "run_count": len(run_rows), "runtime_seconds": time.time() - started}, indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    datasets = _selected_datasets(args)
    models = _selected_models(args)
    selectors = _selected_selectors(args)
    specs = _run_specs(datasets, models=models, selectors=selectors)
    if args.dry_run:
        return _dry_run(datasets)
    validation = _validate_boundaries(datasets)
    binding = validation["binding"]
    if args.status:
        audit_rows, _, _ = _audit_runs(specs, binding=binding, output_dir=OUTPUT_ROOT, recover=False, write_artifacts=False)
        _print_status(audit_rows)
        return 0
    if args.resume:
        audit_rows, _, _ = _audit_runs(specs, binding=binding, output_dir=OUTPUT_ROOT, recover=True)
        planned = _print_execution_plan(specs, audit_rows, execute=args.execute)
        if not args.execute:
            return 0
        return _execute_specs(planned, datasets)
    audit_rows, _, _ = _audit_runs(specs, binding=binding, output_dir=OUTPUT_ROOT, recover=bool(args.execute))
    planned = _print_execution_plan(specs, audit_rows, execute=args.execute)
    if not args.execute:
        return 0
    return _execute_specs(planned, datasets)


if __name__ == "__main__":
    raise SystemExit(main())
