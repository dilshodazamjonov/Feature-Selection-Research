from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


ACTIVE_DATASETS = ("homecredit", "lendingclub_v2")
MODELS = ("lr", "catboost")
SELECTORS = ("clip", "clip_then_mrmr")
BASELINE_SELECTORS = ("mrmr", "llm", "llm_then_mrmr", "stable_core_llm_fill")
EXPECTED_RUNS = tuple((dataset, model, selector) for dataset in ACTIVE_DATASETS for model in MODELS for selector in SELECTORS)
EXPECTED_BUDGETS = {"lr": 20, "catboost": 40}
SOURCE_HASH_INPUTS = {
    "clip_selector_config": Path("configs/clip/selector.yaml"),
    "checkpoint_manifest": Path("results/clip/training/seeds/seed_55/checkpoint_manifest.json"),
    "model_selection_manifest": Path("results/clip/training/model_selection_manifest.json"),
    "anchor_manifest": Path("results/clip/training/learned_anchor_manifest.json"),
}
AGGREGATE_FILES = (
    "evaluation_summary.csv",
    "evaluation_summary.json",
    "run_manifest.json",
    "comparison_with_frozen_baselines.csv",
    "selected_features_long.csv",
    "selected_feature_summary.csv",
    "semantic_coverage_summary.csv",
    "redundancy_summary.csv",
    "runtime_summary.csv",
    "score_psi_summary.csv",
    "statistical_significance_summary.csv",
)


@dataclass(frozen=True)
class CompletedRun:
    dataset: str
    model: str
    selector: str
    run_id: str
    run_dir: Path
    prediction_path: Path
    marker: dict[str, Any]
    metrics: dict[str, Any]
    runtime: dict[str, Any]
    split: dict[str, Any]
    leakage: dict[str, Any]
    selected: pd.DataFrame
    selected_enriched: pd.DataFrame


def expected_run_id(dataset: str, model: str, selector: str) -> str:
    return f"{dataset}_{model}_{selector}"


def expected_source_hashes() -> dict[str, str]:
    return {key: sha256_file(path) for key, path in SOURCE_HASH_INPUTS.items()}


def feature_set_hash(features: list[str]) -> str:
    return sha256_text(json.dumps(sorted(str(feature) for feature in features), sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ks_score(y_true: pd.Series, y_score: pd.Series) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _lift_at_10(y_true: pd.Series, y_score: pd.Series) -> float:
    frame = pd.DataFrame({"y_true": y_true.astype(float), "score": y_score.astype(float)})
    top_n = max(1, int(np.ceil(len(frame) * 0.10)))
    base_rate = float(frame["y_true"].mean())
    if base_rate == 0:
        return float("nan")
    return float(frame.sort_values("score", ascending=False).head(top_n)["y_true"].mean() / base_rate)


def _assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: actual={actual} expected={expected}")


def validate_prediction_metrics(run: CompletedRun) -> dict[str, float]:
    predictions = pd.read_csv(run.prediction_path)
    expected_rows = int(run.split["oot"]["row_count"])
    if len(predictions) != expected_rows:
        raise ValueError(f"{run.run_id}: prediction rows {len(predictions)} != expected {expected_rows}")
    if int(run.marker["prediction_row_count"]) != expected_rows:
        raise ValueError(f"{run.run_id}: completion marker prediction row count mismatch")
    required_columns = {
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
    if set(predictions.columns) != required_columns:
        raise ValueError(f"{run.run_id}: prediction columns mismatch")
    prediction_run_ids = set(predictions["run_id"].astype(str).head(10))
    if not all(value == run.run_id or value == f"{run.run_id}.in_progress" for value in prediction_run_ids):
        raise ValueError(f"{run.run_id}: prediction run_id does not match canonical run")
    auc = float(roc_auc_score(predictions["y_true"], predictions["y_pred_proba"]))
    gini = float(2 * auc - 1)
    ks = _ks_score(predictions["y_true"], predictions["y_pred_proba"])
    lift = _lift_at_10(predictions["y_true"], predictions["y_pred_proba"])
    _assert_close(f"{run.run_id} auc", auc, float(run.metrics["auc"]), 1e-10)
    _assert_close(f"{run.run_id} gini", gini, float(run.metrics["gini"]), 1e-10)
    _assert_close(f"{run.run_id} ks", ks, float(run.metrics["ks"]), 1e-10)
    _assert_close(f"{run.run_id} lift_at_10", lift, float(run.metrics["lift_at_10"]), 1e-8)
    return {"auc": auc, "gini": gini, "ks": ks, "lift_at_10": lift}


def _validate_completed_run(root: Path, run_dir: Path, expected_sources: dict[str, str]) -> CompletedRun:
    if run_dir.name.endswith(".in_progress"):
        raise ValueError(f"{run_dir}: .in_progress directory is not a completed run")
    marker_path = run_dir / "RUN_COMPLETE.json"
    if not marker_path.exists():
        raise ValueError(f"{run_dir.name}: missing RUN_COMPLETE.json")
    marker = _read_json(marker_path)
    dataset = str(marker.get("dataset"))
    model = str(marker.get("model"))
    selector = str(marker.get("selector"))
    run_id = expected_run_id(dataset, model, selector)
    if (dataset, model, selector) not in EXPECTED_RUNS:
        raise ValueError(f"{run_dir.name}: unexpected dataset/model/selector")
    if marker.get("run_id") != run_id or run_dir.name != run_id:
        raise ValueError(f"{run_dir.name}: run identity mismatch")
    if marker.get("completion_status") not in {"complete_valid", "complete_valid_recovered"}:
        raise ValueError(f"{run_id}: invalid completion status")
    if ".in_progress" in json.dumps(marker, sort_keys=True):
        raise ValueError(f"{run_id}: completion marker contains .in_progress reference")
    prediction_path = root / "predictions" / f"{run_id}_oot_predictions.csv"
    metrics_path = run_dir / "results" / "oot_test_results.csv"
    if not prediction_path.exists():
        raise ValueError(f"{run_id}: missing top-level prediction file")
    if sha256_file(prediction_path) != marker.get("prediction_file_hash"):
        raise ValueError(f"{run_id}: prediction hash mismatch")
    if sha256_file(metrics_path) != marker.get("metrics_file_hash"):
        raise ValueError(f"{run_id}: metrics hash mismatch")
    source_hashes = _read_json(run_dir / "source_hashes.json")
    if source_hashes != expected_sources or marker.get("source_hashes") != expected_sources:
        raise ValueError(f"{run_id}: source hash mismatch")
    snapshot = _read_json(run_dir / "config_snapshot.json")
    if snapshot.get("config_hash") != marker.get("config_hash"):
        raise ValueError(f"{run_id}: config hash mismatch")
    selected = pd.read_csv(run_dir / "features" / "final_selected_features.csv")
    feature_col = "feature_name" if "feature_name" in selected.columns else "feature"
    selected_features = selected[feature_col].astype(str).tolist()
    if len(selected_features) != len(set(selected_features)):
        raise ValueError(f"{run_id}: duplicate selected features")
    expected_budget = EXPECTED_BUDGETS[model]
    if len(selected_features) != expected_budget:
        raise ValueError(f"{run_id}: selected feature count {len(selected_features)} != budget {expected_budget}")
    if feature_set_hash(selected_features) != marker.get("feature_set_hash"):
        raise ValueError(f"{run_id}: feature-set hash mismatch")
    enriched = pd.read_csv(run_dir / "selected_features_enriched.csv")
    if int(pd.to_numeric(enriched.get("blocked_feature_count", pd.Series([0])), errors="coerce").fillna(0).max()) != 0:
        raise ValueError(f"{run_id}: blocked selected features present")
    run = CompletedRun(
        dataset=dataset,
        model=model,
        selector=selector,
        run_id=run_id,
        run_dir=run_dir,
        prediction_path=prediction_path,
        marker=marker,
        metrics=pd.read_csv(metrics_path).iloc[0].to_dict(),
        runtime=pd.read_csv(run_dir / "results" / "runtime_summary.csv").iloc[0].to_dict(),
        split=_read_json(run_dir / "data_split_manifest.json"),
        leakage=_read_json(run_dir / "leakage_audit.json"),
        selected=selected,
        selected_enriched=enriched,
    )
    validate_prediction_metrics(run)
    if run.leakage.get("oot_used_in_feature_selection") is not False:
        raise ValueError(f"{run_id}: leakage audit does not confirm OOT exclusion")
    if not run.leakage.get("temporal_split_disjoint"):
        raise ValueError(f"{run_id}: temporal split is not disjoint")
    return run


def discover_completed_runs(root: Path) -> list[CompletedRun]:
    run_root = root / "runs"
    expected_sources = expected_source_hashes()
    runs = []
    seen: set[tuple[str, str, str]] = set()
    for run_dir in sorted(run_root.iterdir() if run_root.exists() else []):
        if not run_dir.is_dir() or run_dir.name.endswith(".in_progress"):
            continue
        marker_path = run_dir / "RUN_COMPLETE.json"
        if not marker_path.exists():
            continue
        run = _validate_completed_run(root, run_dir, expected_sources)
        key = (run.dataset, run.model, run.selector)
        if key in seen:
            raise ValueError(f"duplicate run key: {key}")
        seen.add(key)
        runs.append(run)
    expected = set(EXPECTED_RUNS)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"completed run coverage mismatch: missing={missing} extra={extra}")
    return sorted(runs, key=lambda run: (run.dataset, run.model, run.selector, run.run_id))


def build_evaluation_summary(runs: list[CompletedRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "dataset": run.dataset,
                "model": run.model,
                "selector": run.selector,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "prediction_path": str(run.prediction_path),
                "oot_auc": float(run.metrics["auc"]),
                "oot_gini": float(run.metrics["gini"]),
                "oot_ks": float(run.metrics["ks"]),
                "lift_at_10": float(run.metrics["lift_at_10"]),
                "model_score_psi": float(run.metrics["model_score_psi"]),
                "selected_feature_count": int(run.metrics["selected_feature_count"]),
                "runtime_seconds": float(run.runtime.get("total_runtime_seconds", np.nan)),
                "dev_row_count": int(run.split["dev"]["row_count"]),
                "oot_row_count": int(run.split["oot"]["row_count"]),
                "dev_target_rate": float(run.split["dev"]["target_rate"]),
                "oot_target_rate": float(run.split["oot"]["target_rate"]),
                "checkpoint_hash": run.marker["checkpoint_hash"],
                "anchor_hash": run.marker["anchor_hash"],
                "feature_set_hash": run.marker["feature_set_hash"],
                "config_hash": run.marker["config_hash"],
                "completion_status": run.marker["completion_status"],
                "completed_at": run.marker["completed_at"],
                "statistical_view_scope": "missingness_only",
                "auc_gini_ks_lift_recomputed": True,
                "score_psi_recomputed": False,
                "score_psi_source": "per-run results/model_score_psi.csv cross-checked against oot_test_results.csv",
                "score_psi_recomputation_limitation": "DEV/train score vectors are not persisted; PSI was not independently recomputed.",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "selector", "run_id"]).reset_index(drop=True)


def build_selected_features_long(runs: list[CompletedRun]) -> pd.DataFrame:
    frames = []
    for run in runs:
        feature_col = "feature_name" if "feature_name" in run.selected.columns else "feature"
        frame = pd.DataFrame({"feature_name": run.selected[feature_col].astype(str).tolist()})
        enriched = run.selected_enriched.copy()
        if "feature_name" not in enriched.columns:
            enriched["feature_name"] = frame["feature_name"]
        enriched = enriched.drop_duplicates("feature_name", keep="first")
        frame = frame.merge(enriched, on="feature_name", how="left", suffixes=("", "_enriched"))
        frame["dataset"] = run.dataset
        frame["model"] = run.model
        frame["selector"] = run.selector
        frame["run_id"] = run.run_id
        frame["feature_set_hash"] = run.marker["feature_set_hash"]
        frame["final_selected"] = frame.get("final_selected", True)
        frame["final_rank"] = pd.to_numeric(frame.get("final_rank", pd.Series(range(1, len(frame) + 1))), errors="coerce")
        frame["clip_score"] = pd.to_numeric(frame.get("clip_score", frame.get("learned_similarity")), errors="coerce")
        frame["clip_rank"] = pd.to_numeric(frame.get("clip_rank", pd.Series(dtype=float)), errors="coerce")
        frame["screening_pool_member"] = frame.get("screening_pool_member", True)
        if "mrmr_score" not in frame.columns:
            frame["mrmr_score"] = pd.NA
        if "mrmr_rank" not in frame.columns:
            frame["mrmr_rank"] = frame["final_rank"] if run.selector == "clip_then_mrmr" else pd.NA
        if "semantic_group" not in frame.columns:
            frame["semantic_group"] = frame["feature_name"].map(infer_semantic_group)
        if "source_table_or_formula" not in frame.columns:
            frame["source_table_or_formula"] = pd.NA
        columns = [
            "dataset",
            "model",
            "selector",
            "run_id",
            "feature_name",
            "final_selected",
            "final_rank",
            "clip_score",
            "clip_rank",
            "screening_pool_member",
            "mrmr_score",
            "mrmr_rank",
            "semantic_group",
            "source_table_or_formula",
            "feature_set_hash",
        ]
        frames.append(frame[columns])
    long = pd.concat(frames, ignore_index=True)
    if long.duplicated(["run_id", "feature_name"]).any():
        raise ValueError("duplicate selected feature within a run")
    return long.sort_values(["dataset", "model", "selector", "final_rank", "feature_name"]).reset_index(drop=True)


def build_selected_feature_summary(selected_long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        selected_long.groupby(["dataset", "model", "selector", "run_id"], as_index=False)
        .agg(
            selected_feature_count=("feature_name", "count"),
            feature_set_hash=("feature_set_hash", "first"),
            duplicate_selected_feature_count=("feature_name", lambda values: int(pd.Series(values).duplicated().sum())),
            semantic_group_count=("semantic_group", lambda values: int(pd.Series(values).fillna("unknown").nunique())),
        )
        .sort_values(["dataset", "model", "selector", "run_id"])
        .reset_index(drop=True)
    )
    for row in summary.to_dict("records"):
        expected = EXPECTED_BUDGETS[row["model"]]
        if int(row["selected_feature_count"]) != expected:
            raise ValueError(f"{row['run_id']}: selected feature count does not match budget")
    return summary


def _base_family(feature: str) -> str:
    parts = str(feature).split("_")
    if len(parts) >= 3 and parts[-1] in {"MEAN", "SUM", "MIN", "MAX", "VAR", "FLAG"}:
        return "_".join(parts[:-1])
    return str(feature)


def build_semantic_summary(selected_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in selected_long.groupby(["dataset", "model", "selector", "run_id"], dropna=False):
        groups = frame["semantic_group"].fillna("unknown").astype(str)
        counts = groups.value_counts()
        families = frame["feature_name"].map(_base_family)
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "selector": keys[2],
                "run_id": keys[3],
                "selected_feature_count": int(len(frame)),
                "semantic_group_count": int(groups.nunique()),
                "largest_semantic_group_share": float(counts.iloc[0] / len(frame)) if len(frame) else np.nan,
                "source_table_coverage": int(frame["source_table_or_formula"].fillna("unknown").astype(str).nunique()),
                "repeated_family_count": int(families.value_counts().gt(1).sum()),
                "repeated_family_share": float(families.duplicated(keep=False).mean()) if len(families) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "selector", "run_id"]).reset_index(drop=True)


def build_redundancy_summary(selected_long: pd.DataFrame, runs: list[CompletedRun]) -> pd.DataFrame:
    run_lookup = {run.run_id: run for run in runs}
    rows = []
    for keys, frame in selected_long.groupby(["dataset", "model", "selector", "run_id"], dropna=False):
        run = run_lookup[keys[3]]
        families = frame["feature_name"].map(_base_family)
        psi_path = run.run_dir / "results" / "selected_feature_psi.csv"
        psi = pd.read_csv(psi_path) if psi_path.exists() else pd.DataFrame()
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "selector": keys[2],
                "run_id": keys[3],
                "selected_feature_count": int(len(frame)),
                "exact_duplicate_count": int(frame["feature_name"].duplicated().sum()),
                "near_duplicate_family_count": int(families.value_counts().gt(1).sum()),
                "repeated_base_family_share": float(families.duplicated(keep=False).mean()) if len(families) else np.nan,
                "pairwise_feature_correlation_status": "not_materialized_by_final_pipeline",
                "selected_feature_psi_mean": float(pd.to_numeric(psi.get("psi", pd.Series(dtype=float)), errors="coerce").mean())
                if not psi.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "selector", "run_id"]).reset_index(drop=True)


def build_runtime_summary(runs: list[CompletedRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "dataset": run.dataset,
                "model": run.model,
                "selector": run.selector,
                "run_id": run.run_id,
                "clip_scoring_time_sec": np.nan,
                "mrmr_screening_time_sec": float(run.runtime.get("final_feature_selection_time_sec", np.nan))
                if run.selector == "clip_then_mrmr"
                else np.nan,
                "downstream_model_fitting_time_sec": float(run.runtime.get("final_training_time_sec", np.nan)),
                "prediction_time_sec": float(run.runtime.get("final_evaluation_time_sec", np.nan)),
                "total_runtime_seconds": float(run.runtime.get("total_runtime_seconds", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "selector", "run_id"]).reset_index(drop=True)


def build_score_psi_summary(runs: list[CompletedRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        saved_psi = float(run.metrics["model_score_psi"])
        psi_file = pd.read_csv(run.run_dir / "results" / "model_score_psi.csv").iloc[0]
        _assert_close(f"{run.run_id} model_score_psi", saved_psi, float(psi_file["model_score_psi"]), 1e-12)
        rows.append(
            {
                "dataset": run.dataset,
                "model": run.model,
                "selector": run.selector,
                "run_id": run.run_id,
                "model_score_psi": saved_psi,
                "model_score_psi_bucket": "low" if saved_psi < 0.10 else "moderate" if saved_psi < 0.25 else "high",
                "psi_source": "per-run results/model_score_psi.csv and oot_test_results.csv",
                "psi_recomputed": False,
                "psi_recomputation_limitation": "DEV/train score vectors are not persisted; value was cross-checked only.",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model", "selector", "run_id"]).reset_index(drop=True)


def load_frozen_baselines(datasets: tuple[str, ...] = ACTIVE_DATASETS) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = Path("results") / dataset / "final_comparison_table.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["selector"].astype(str).isin(BASELINE_SELECTORS)].copy()
        frame["result_origin"] = "frozen_baseline"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_baseline_comparison(evaluation: pd.DataFrame) -> pd.DataFrame:
    baselines = load_frozen_baselines()
    clip_rows = evaluation.rename(columns={"dataset": "dataset_name"}).copy()
    clip_rows["result_origin"] = "clip_extension"
    all_columns = sorted(set(baselines.columns).union(clip_rows.columns))
    return pd.concat(
        [baselines.reindex(columns=all_columns), clip_rows.reindex(columns=all_columns)],
        ignore_index=True,
        sort=False,
    )


def _paired_auc_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    n_bootstrap: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        return {"status": "skipped", "skip_reason": "prediction row counts do not align"}
    y = candidate["y_true"].to_numpy()
    if not np.array_equal(baseline["y_true"].to_numpy(), y):
        return {"status": "skipped", "skip_reason": "prediction targets do not align"}
    base_score = baseline["y_pred_proba"].to_numpy()
    new_score = candidate["y_pred_proba"].to_numpy()
    point = float(roc_auc_score(y, new_score) - roc_auc_score(y, base_score))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], new_score[idx]) - roc_auc_score(y[idx], base_score[idx])))
    if not deltas:
        return {"status": "skipped", "skip_reason": "no valid bootstrap samples"}
    arr = np.asarray(deltas)
    return {
        "status": "ok",
        "skip_reason": "",
        "point_estimate_difference": point,
        "ci95_lower": float(np.quantile(arr, 0.025)),
        "ci95_upper": float(np.quantile(arr, 0.975)),
        "p_value": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
        "paired_row_count": int(len(y)),
        "bootstrap_samples": int(len(arr)),
    }


def build_statistical_significance(evaluation: pd.DataFrame) -> pd.DataFrame:
    baselines = load_frozen_baselines()
    rows = []
    baseline_cache: dict[str, pd.DataFrame | None] = {}
    candidate_cache: dict[str, pd.DataFrame] = {}
    for new_row in evaluation.to_dict("records"):
        new_pred = candidate_cache.setdefault(str(new_row["prediction_path"]), pd.read_csv(new_row["prediction_path"]))
        subset = baselines[
            (baselines["dataset_name"].astype(str).eq(str(new_row["dataset"])))
            & (baselines["model"].astype(str).eq(str(new_row["model"])))
            & (baselines["selector"].astype(str).isin(BASELINE_SELECTORS))
        ]
        for baseline in subset.to_dict("records"):
            baseline_path = Path(str(baseline.get("output_folder", ""))) / "results" / "oot_predictions.csv"
            payload = {
                "dataset": new_row["dataset"],
                "model": new_row["model"],
                "new_selector": new_row["selector"],
                "baseline_selector": baseline["selector"],
                "comparison_metric": "auc",
                "method": "paired_bootstrap_auc",
                "multiple_testing_correction": "benjamini_hochberg_by_dataset_model",
                "baseline_prediction_path": str(baseline_path),
            }
            if not baseline_path.exists():
                payload.update(
                    {
                        "status": "skipped",
                        "skip_reason": "missing_baseline_predictions",
                        "point_estimate_difference": np.nan,
                        "ci95_lower": np.nan,
                        "ci95_upper": np.nan,
                        "p_value": np.nan,
                        "paired_row_count": np.nan,
                    }
                )
            else:
                baseline_key = str(baseline_path)
                if baseline_key not in baseline_cache:
                    baseline_cache[baseline_key] = pd.read_csv(baseline_path)
                stats = _paired_auc_bootstrap(baseline_cache[baseline_key], new_pred)
                payload.update(stats)
            rows.append(payload)
    frame = pd.DataFrame(rows)
    if "p_value" in frame.columns:
        frame["p_value_bh"] = np.nan
        ok_mask = frame["status"].eq("ok") & frame["p_value"].notna()
        for _, idx in frame[ok_mask].groupby(["dataset", "model"]).groups.items():
            ordered = frame.loc[idx].sort_values("p_value").index.tolist()
            m = len(ordered)
            adjusted = []
            running = 1.0
            for rank, row_idx in reversed(list(enumerate(ordered, start=1))):
                running = min(running, float(frame.loc[row_idx, "p_value"]) * m / rank)
                adjusted.append((row_idx, running))
            for row_idx, value in adjusted:
                frame.loc[row_idx, "p_value_bh"] = value
    return frame.sort_values(["dataset", "model", "new_selector", "baseline_selector"]).reset_index(drop=True)


def build_run_manifest(runs: list[CompletedRun]) -> dict[str, Any]:
    return {
        "status": "completed",
        "run_count": len(runs),
        "runs": [
            {
                "run_id": run.run_id,
                "dataset": run.dataset,
                "model": run.model,
                "selector": run.selector,
                "run_dir": str(run.run_dir),
                "prediction_path": str(run.prediction_path),
                "completion_status": run.marker["completion_status"],
                "completed_at": run.marker["completed_at"],
                "checkpoint_hash": run.marker["checkpoint_hash"],
                "anchor_hash": run.marker["anchor_hash"],
                "feature_set_hash": run.marker["feature_set_hash"],
                "config_hash": run.marker["config_hash"],
                "prediction_hash": run.marker["prediction_file_hash"],
                "metrics_hash": run.marker["metrics_file_hash"],
                "source_hashes": run.marker["source_hashes"],
                "prediction_row_count": run.marker["prediction_row_count"],
                "selected_feature_count": int(run.metrics["selected_feature_count"]),
                "leakage_audit_status": "passed"
                if run.leakage.get("target_column_excluded")
                and run.leakage.get("temporal_split_disjoint")
                and run.leakage.get("oot_used_in_feature_selection") is False
                else "failed",
            }
            for run in runs
        ],
        "legacy_lendingclub_allowed": False,
        "frozen_baselines_overwritten": False,
        "dev_cv_policy": "diagnostic_only_clip_not_fold_local",
        "aggregation_source": "completed_run_directories",
        "score_psi_recomputed": False,
        "score_psi_recomputation_limitation": "DEV/train score vectors are not persisted; PSI values are cross-checked from run-level outputs.",
    }


def build_all_aggregates(root: Path) -> dict[str, Any]:
    runs = discover_completed_runs(root)
    evaluation = build_evaluation_summary(runs)
    selected_long = build_selected_features_long(runs)
    aggregates: dict[str, Any] = {
        "evaluation_summary.csv": evaluation,
        "evaluation_summary.json": evaluation.to_dict("records"),
        "run_manifest.json": build_run_manifest(runs),
        "selected_features_long.csv": selected_long,
        "selected_feature_summary.csv": build_selected_feature_summary(selected_long),
        "semantic_coverage_summary.csv": build_semantic_summary(selected_long),
        "redundancy_summary.csv": build_redundancy_summary(selected_long, runs),
        "runtime_summary.csv": build_runtime_summary(runs),
        "score_psi_summary.csv": build_score_psi_summary(runs),
    }
    aggregates["comparison_with_frozen_baselines.csv"] = build_baseline_comparison(evaluation)
    aggregates["statistical_significance_summary.csv"] = build_statistical_significance(evaluation)
    validate_aggregates(aggregates)
    return aggregates


def validate_aggregates(aggregates: dict[str, Any]) -> None:
    expected_ids = {expected_run_id(*parts) for parts in EXPECTED_RUNS}
    evaluation = aggregates["evaluation_summary.csv"]
    if len(evaluation) != 8:
        raise ValueError("evaluation_summary must contain exactly 8 rows")
    ids = set(evaluation["run_id"].astype(str))
    if ids != expected_ids or evaluation["run_id"].duplicated().any():
        raise ValueError("evaluation_summary run coverage is invalid")
    if evaluation.astype(str).apply(lambda col: col.str.contains(".in_progress", regex=False)).any().any():
        raise ValueError("aggregate contains .in_progress reference")
    if evaluation[["oot_auc", "oot_gini", "oot_ks", "lift_at_10", "model_score_psi"]].isna().any().any():
        raise ValueError("evaluation_summary has missing core metrics")
    for key in [
        "selected_feature_summary.csv",
        "semantic_coverage_summary.csv",
        "redundancy_summary.csv",
        "runtime_summary.csv",
        "score_psi_summary.csv",
    ]:
        frame = aggregates[key]
        if set(frame["run_id"].astype(str)) != expected_ids:
            raise ValueError(f"{key} does not cover all runs")
    comparison = aggregates["comparison_with_frozen_baselines.csv"]
    if int(comparison["result_origin"].eq("clip_extension").sum()) != 8:
        raise ValueError("comparison table must contain exactly 8 CLIP rows")
    psi = aggregates["score_psi_summary.csv"]
    if not (psi["psi_recomputed"].astype(str).str.lower().eq("false")).all():
        raise ValueError("score PSI limitation must be explicit")


def atomic_write_aggregates(root: Path, aggregates: dict[str, Any]) -> None:
    temp_dir = root / ".aggregate_rebuild_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    try:
        for name, payload in aggregates.items():
            output = temp_dir / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".json"):
                output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            else:
                assert isinstance(payload, pd.DataFrame)
                payload.to_csv(output, index=False)
        for name in AGGREGATE_FILES:
            src = temp_dir / name
            if not src.exists():
                raise ValueError(f"missing aggregate output in temp dir: {name}")
            src.replace(root / name)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def aggregate_status(root: Path) -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        runs = discover_completed_runs(root)
        status["completed_valid_runs"] = len(runs)
    except Exception as exc:
        status["completed_valid_runs"] = 0
        status["run_error"] = repr(exc)
    expected_ids = {expected_run_id(*parts) for parts in EXPECTED_RUNS}
    for name in [
        "evaluation_summary.csv",
        "selected_feature_summary.csv",
        "semantic_coverage_summary.csv",
        "redundancy_summary.csv",
        "runtime_summary.csv",
        "score_psi_summary.csv",
    ]:
        path = root / name
        if not path.exists():
            status[name] = {"rows": 0, "coverage": 0}
            continue
        frame = pd.read_csv(path)
        coverage = int(frame["run_id"].astype(str).isin(expected_ids).sum()) if "run_id" in frame.columns else 0
        status[name] = {"rows": int(len(frame)), "coverage": coverage}
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        status["run_manifest.json"] = {"entries": int(len(manifest.get("runs", [])))}
    else:
        status["run_manifest.json"] = {"entries": 0}
    comparison_path = root / "comparison_with_frozen_baselines.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        origin_col = "result_origin" if "result_origin" in comparison.columns else "source"
        status["comparison_with_frozen_baselines.csv"] = {
            "clip_rows": int(comparison[origin_col].astype(str).isin(["clip_extension", "clip_final_evaluation"]).sum())
        }
    else:
        status["comparison_with_frozen_baselines.csv"] = {"clip_rows": 0}
    status["aggregate_complete"] = (
        status.get("completed_valid_runs") == 8
        and status.get("evaluation_summary.csv", {}).get("rows") == 8
        and status.get("run_manifest.json", {}).get("entries") == 8
        and status.get("selected_feature_summary.csv", {}).get("coverage") == 8
        and status.get("semantic_coverage_summary.csv", {}).get("coverage") == 8
        and status.get("runtime_summary.csv", {}).get("coverage") == 8
        and status.get("score_psi_summary.csv", {}).get("coverage") == 8
    )
    return status
