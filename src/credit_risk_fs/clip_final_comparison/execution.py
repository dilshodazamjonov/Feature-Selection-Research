from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter
from credit_risk_fs.clip_final_comparison.constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CORRELATION_FILTER_THRESHOLD,
    MODEL_BUDGETS,
    OUTPUT_ROOT,
)
from credit_risk_fs.clip_final_comparison.diagnostics import semantic_pool_diagnostics
from credit_risk_fs.clip_final_comparison.screening import (
    correlation_filter_pool,
    random_screening_pool,
    type_aware_dispersion_scores,
)
from credit_risk_fs.clip_final_comparison.uncertainty import metric_bundle
from credit_risk_fs.evaluation.drift import calculate_psi
from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.experiments.config import load_named_project_config, resolve_model_kwargs
from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group
from credit_risk_fs.models.registry import get_model_bundle
from credit_risk_fs.preprocessing.encoding import Preprocessor
from credit_risk_fs.selectors.mrmr import MRMR
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


@dataclass(frozen=True)
class ComparisonRunSpec:
    run_id: str
    dataset: str
    model: str
    screening_method: str
    final_feature_budget: int
    candidate_pool_size: int | None = None
    pool_multiplier: int | None = None
    random_seed: int | None = None
    output_family: str = "candidate_pool"
    checkpoint_seed: int | None = None
    ablation: str | None = None
    temporal_cutoff_id: str | None = None


@dataclass
class PreparedFrame:
    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_oot: pd.DataFrame
    y_oot: pd.Series
    time_dev: pd.Series | None = None
    time_oot: pd.Series | None = None


def build_screening_scores(
    method: str,
    *,
    dataset: str,
    X_dev: pd.DataFrame,
    feature_names: list[str],
    random_seed: int | None = None,
    checkpoint_seed: int | None = None,
    ablation: str | None = None,
) -> pd.DataFrame:
    if method == "clip_v2":
        return _clip_score_ranking(dataset, feature_names, "learned_similarity", "clip_v2_score", checkpoint_seed=checkpoint_seed, ablation=ablation)
    if method == "variance":
        return type_aware_dispersion_scores(X_dev, feature_names).rename(columns={"dispersion_score": "screening_score"})
    if method == "correlation_filter":
        dispersion = type_aware_dispersion_scores(X_dev, feature_names).rename(columns={"dispersion_score": "screening_score"})
        return dispersion
    if method == "random":
        seed = 0 if random_seed is None else int(random_seed)
        pool = random_screening_pool(feature_names, len(feature_names), seed)
        return pd.DataFrame(
            {
                "feature_name": pool,
                "screening_score": np.arange(len(pool), 0, -1, dtype=float),
                "dispersion_kind": "random_without_replacement",
            }
        )
    if method == "text_similarity":
        return _embedding_centroid_score(
            dataset=dataset,
            feature_names=feature_names,
            source_path=Path(f"results/clip/text_baseline/{dataset}_text_embeddings.parquet"),
            prefix="embedding_",
            score_name="text_similarity",
        )
    if method == "statistics_only":
        return _embedding_centroid_score(
            dataset=dataset,
            feature_names=feature_names,
            source_path=Path(f"results/clip_v2/statistical_view/{dataset}_statistical_vectors.parquet"),
            prefix="stat_",
            score_name="statistics_similarity",
        )
    if method == "full_mrmr":
        return pd.DataFrame({"feature_name": feature_names, "screening_score": 0.0, "dispersion_kind": "full_mrmr_universe"})
    if method in {"llm", "llm_then_mrmr"}:
        return _llm_score_ranking(dataset, feature_names)
    raise ValueError(f"unsupported screening method: {method}")


def construct_candidate_pool(
    spec: ComparisonRunSpec,
    *,
    X_dev: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = len(feature_names) if spec.candidate_pool_size is None else int(spec.candidate_pool_size)
    actual = min(requested, len(feature_names))
    scores = build_screening_scores(
        spec.screening_method,
        dataset=spec.dataset,
        X_dev=X_dev,
        feature_names=feature_names,
        random_seed=spec.random_seed,
        checkpoint_seed=spec.checkpoint_seed,
        ablation=spec.ablation,
    )
    scores = scores.drop_duplicates("feature_name", keep="first").copy()
    if spec.screening_method == "correlation_filter":
        selected, audit = correlation_filter_pool(
            X_dev,
            actual,
            threshold=CORRELATION_FILTER_THRESHOLD,
            feature_names=scores["feature_name"].tolist(),
        )
        pool = scores[scores["feature_name"].isin(selected)].copy()
        order = {feature: rank for rank, feature in enumerate(selected, start=1)}
        pool["pool_rank"] = pool["feature_name"].map(order)
        pool = pool.sort_values("pool_rank", kind="mergesort")
        audit["rejection_reason"] = np.where(audit["kept"], "", f"abs_association_gt_{CORRELATION_FILTER_THRESHOLD}")
        return _annotate_pool(pool, spec, requested, actual), audit
    pool = (
        scores.sort_values(["screening_score", "feature_name"], ascending=[False, True], kind="mergesort")
        .head(actual)
        .copy()
    )
    pool["pool_rank"] = range(1, len(pool) + 1)
    audit = pd.DataFrame(columns=["feature_name", "max_abs_association", "kept", "rejection_reason"])
    return _annotate_pool(pool, spec, requested, actual), audit


def execute_comparison_run(spec: ComparisonRunSpec, data: PreparedFrame, run_dir: Path, *, model_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.time()
    in_progress = run_dir.with_name(run_dir.name + ".in_progress")
    if run_dir.exists() and (run_dir / "RUN_COMPLETE.json").exists():
        return validate_run(run_dir)
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {run_dir}")
    if in_progress.exists():
        shutil.rmtree(in_progress)
    in_progress.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_write_text(in_progress / "execution.log", f"{_now()} run_start {spec.run_id}\n")
        features = data.X_dev.columns.astype(str).tolist()
        pool, rejection_audit = construct_candidate_pool(spec, X_dev=data.X_dev, feature_names=features)
        _atomic_write_csv(in_progress / "candidate_pool.csv", pool)
        _atomic_write_csv(in_progress / "candidate_pool_rejection_audit.csv", rejection_audit)
        _write_pool_manifest(in_progress, spec, pool)
        X_dev_raw = data.X_dev[pool["feature_name"].astype(str).tolist()].copy()
        X_oot_raw = data.X_oot[pool["feature_name"].astype(str).tolist()].copy()
        preprocessor = Preprocessor()
        X_dev_processed = preprocessor.fit_transform(X_dev_raw)
        X_oot_processed = preprocessor.transform(X_oot_raw)
        mrmr = MRMR(k=min(spec.final_feature_budget, X_dev_processed.shape[1]), method="mrmr", random_state=42)
        mrmr.fit(X_dev_processed, data.y_dev)
        selected = list(mrmr.selected_features_ or [])[: spec.final_feature_budget]
        X_dev_final = X_dev_processed[selected]
        X_oot_final = X_oot_processed[selected]
        get_model, train_model, predict_proba, save_model = get_model_bundle(
            spec.model,
            model_kwargs=model_kwargs if model_kwargs is not None else _default_model_kwargs(spec.model),
        )
        model = get_model()
        train_model(model, X_dev_final, data.y_dev)
        dev_score = predict_proba(model, X_dev_final)
        oot_score = predict_proba(model, X_oot_final)
        threshold = determine_threshold(data.y_dev, dev_score)
        dev_pred = (dev_score >= threshold).astype(int)
        oot_pred = (oot_score >= threshold).astype(int)
        metrics = evaluate_model(data.y_oot, oot_score, threshold=threshold, y_pred=oot_pred)
        metrics.update(metric_bundle(data.y_oot.to_numpy(), oot_score))
        metrics["lift_at_10"] = metrics.pop("lift_at_10")
        metrics.update({"run_id": spec.run_id, "dataset": spec.dataset, "model": spec.model, "screening_method": spec.screening_method})
        if spec.checkpoint_seed is not None:
            metrics["checkpoint_seed"] = int(spec.checkpoint_seed)
        if spec.ablation is not None:
            metrics["ablation"] = spec.ablation
        if spec.temporal_cutoff_id is not None:
            metrics["cutoff_id"] = spec.temporal_cutoff_id
        selected_frame = pd.DataFrame(
            {
                "feature_name": selected,
                "final_rank": range(1, len(selected) + 1),
                "semantic_group": [infer_semantic_group(feature) for feature in selected],
            }
        )
        feature_hash = sha256_text(json.dumps(selected, sort_keys=True))
        selected_frame["feature_set_hash"] = feature_hash
        _atomic_write_csv(in_progress / "selected_features.csv", selected_frame)
        _atomic_write_json(
            in_progress / "feature_selection_manifest.json",
            {
                "run_id": spec.run_id,
                "candidate_pool_size": int(len(pool)),
                "final_feature_budget": spec.final_feature_budget,
                "selected_count": int(len(selected)),
                "feature_set_hash": feature_hash,
                "mrmr_scope": "DEV only",
                "oot_used_in_selection": False,
                "checkpoint_seed": spec.checkpoint_seed,
                "ablation": spec.ablation,
                "temporal_cutoff_id": spec.temporal_cutoff_id,
            },
        )
        _write_predictions(in_progress / "dev_predictions.parquet", spec, data.y_dev, dev_score, dev_pred, feature_hash, "DEV")
        _write_predictions(in_progress / "oot_predictions.parquet", spec, data.y_oot, oot_score, oot_pred, feature_hash, "OOT")
        _atomic_write_json(in_progress / "metrics.json", _json_safe(metrics))
        _atomic_write_csv(in_progress / "score_psi.csv", pd.DataFrame([{"score_psi": calculate_psi(pd.Series(dev_score), pd.Series(oot_score))}]))
        _atomic_write_json(in_progress / "runtime.json", {"total_runtime_seconds": float(time.time() - started)})
        _atomic_write_json(in_progress / "leakage_audit.json", {"passed": True, "oot_used_in_selection": False, "target_used_in_screening": False})
        save_model(model, str(in_progress / "model" / "final_model.model"))
        validation = validate_run(in_progress)
        complete = {"run_id": spec.run_id, "status": "complete_valid", "completed_at": _now(), **validation}
        _atomic_write_json(in_progress / "RUN_COMPLETE.json", complete)
        in_progress.replace(run_dir)
        return validate_run(run_dir)
    except Exception:
        if in_progress.exists():
            log_path = in_progress / "execution.log"
            prior = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            _atomic_write_text(log_path, prior + f"{_now()} failed\n")
        raise


def validate_run(run_dir: Path) -> dict[str, Any]:
    required = [
        "candidate_pool.csv",
        "candidate_pool_manifest.json",
        "selected_features.csv",
        "feature_selection_manifest.json",
        "dev_predictions.parquet",
        "oot_predictions.parquet",
        "metrics.json",
        "score_psi.csv",
        "runtime.json",
        "leakage_audit.json",
        "execution.log",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{run_dir}: missing artifacts {missing}")
    if not (run_dir / "RUN_COMPLETE.json").exists() and not run_dir.name.endswith(".in_progress"):
        raise RuntimeError(f"{run_dir}: missing RUN_COMPLETE.json")
    oot = pd.read_parquet(run_dir / "oot_predictions.parquet")
    if oot.empty:
        raise RuntimeError(f"{run_dir}: empty OOT predictions")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    recomputed = metric_bundle(oot["y_true"].to_numpy(), oot["y_pred_proba"].to_numpy())
    for metric in ["auc", "ks", "lift_at_10"]:
        if abs(float(metrics[metric]) - float(recomputed[metric])) > 1e-8:
            raise RuntimeError(f"{run_dir}: metric mismatch {metric}")
    selected = pd.read_csv(run_dir / "selected_features.csv")
    if selected["feature_name"].duplicated().any():
        raise RuntimeError(f"{run_dir}: duplicate selected features")
    leakage = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    if not leakage.get("passed"):
        raise RuntimeError(f"{run_dir}: leakage audit failed")
    return {
        "prediction_rows": int(len(oot)),
        "selected_count": int(len(selected)),
        "candidate_pool_rows": int(len(pd.read_csv(run_dir / "candidate_pool.csv"))),
        "prediction_hash": sha256_file(run_dir / "oot_predictions.parquet"),
    }


def aggregate_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, Path]:
    rows = []
    feature_rows = []
    for run_dir in run_dirs:
        validation = validate_run(run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "candidate_pool_manifest.json").read_text(encoding="utf-8"))
        rows.append({**metrics, **manifest, **validation, "run_id": run_dir.name})
        selected = pd.read_csv(run_dir / "selected_features.csv")
        selected["run_id"] = run_dir.name
        feature_rows.append(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    master = pd.DataFrame(rows)
    paths = {
        "master_results": output_dir / "master_results.csv",
        "selected_features": output_dir / "feature_stability.csv",
        "metric_recomputation": output_dir / "metric_recomputation.csv",
    }
    _atomic_write_csv(paths["master_results"], master)
    _atomic_write_csv(paths["selected_features"], pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame())
    _atomic_write_csv(paths["metric_recomputation"], recompute_metrics(run_dirs))
    return paths


def recompute_metrics(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        oot = pd.read_parquet(run_dir / "oot_predictions.parquet")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        recomputed = metric_bundle(oot["y_true"].to_numpy(), oot["y_pred_proba"].to_numpy())
        row = {"run_id": run_dir.name, "row_count": int(len(oot))}
        for metric, value in recomputed.items():
            row[f"recomputed_{metric}"] = value
            row[f"recorded_{metric}"] = float(metrics[metric])
            row[f"{metric}_matches"] = abs(float(metrics[metric]) - float(value)) <= 1e-8
        rows.append(row)
    return pd.DataFrame(rows)


def write_minimal_plot(master_results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if master_results.empty or "auc" not in master_results.columns:
        raise RuntimeError("cannot plot without completed master results")
    fig, ax = plt.subplots(figsize=(8, 4))
    master_results.plot.bar(x="run_id", y="auc", ax=ax, legend=False)
    ax.set_ylabel("OOT AUC")
    ax.set_title("OOT AUC by Completed Run")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _clip_score_ranking(dataset: str, feature_names: list[str], column: str, score_name: str, *, checkpoint_seed: int | None = None, ablation: str | None = None) -> pd.DataFrame:
    seeded_path = None
    if checkpoint_seed is not None:
        seeded_path = OUTPUT_ROOT / "seed_score_caches" / f"seed_{checkpoint_seed}" / f"{dataset}_clip_v2_scores.csv"
    if ablation is not None:
        seeded_path = OUTPUT_ROOT / "ablations/score_caches" / ablation / f"{dataset}_clip_v2_scores.csv"
    if seeded_path is not None and seeded_path.exists():
        frame = pd.read_csv(seeded_path)
    elif seeded_path is not None:
        raise RuntimeError(f"required representation-specific score cache missing: {seeded_path}")
    else:
        frame = ClipScoreAdapter("configs/clip_v2/selector.yaml", dataset=dataset).score_frame(use_cache=True)
    frame = frame[frame["feature_name"].isin(feature_names)].copy()
    if column not in frame.columns:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        column = numeric[0] if numeric else ""
    if not column:
        frame["screening_score"] = np.arange(len(frame), 0, -1, dtype=float)
    else:
        frame = frame.rename(columns={column: "screening_score"})
    return frame[["feature_name", "screening_score"]].assign(dispersion_kind=score_name)


def _embedding_centroid_score(dataset: str, feature_names: list[str], source_path: Path, prefix: str, score_name: str) -> pd.DataFrame:
    if not source_path.exists():
        return _clip_score_ranking(dataset, feature_names, "learned_similarity", score_name + "_fallback_clip_score")
    frame = pd.read_parquet(source_path)
    frame = frame[frame["feature_name"].isin(feature_names)].copy()
    cols = [col for col in frame.columns if col.startswith(prefix)]
    if not cols:
        return _clip_score_ranking(dataset, feature_names, "learned_similarity", score_name + "_fallback_clip_score")
    matrix = frame[cols].to_numpy(dtype=float)
    anchor = matrix.mean(axis=0)
    denom = np.linalg.norm(matrix, axis=1) * max(np.linalg.norm(anchor), 1e-12)
    scores = np.divide(matrix @ anchor, denom, out=np.zeros(len(frame)), where=denom != 0)
    return pd.DataFrame({"feature_name": frame["feature_name"].astype(str), "screening_score": scores, "dispersion_kind": score_name})


def _llm_score_ranking(dataset: str, feature_names: list[str]) -> pd.DataFrame:
    candidates = [
        Path(f"results/{dataset}/analysis/llm_rankings_summary.csv"),
        Path(f"results/{dataset}/aggregated/llm_rankings_summary.csv"),
        Path(f"results/{dataset}/llm_rankings_summary.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        feature_col = "feature_name" if "feature_name" in frame.columns else "feature"
        rank_cols = [col for col in frame.columns if "rank" in col.lower()]
        if feature_col in frame.columns and rank_cols:
            out = frame[[feature_col, rank_cols[0]]].rename(columns={feature_col: "feature_name", rank_cols[0]: "rank"})
            out = out[out["feature_name"].isin(feature_names)].copy()
            out["screening_score"] = -pd.to_numeric(out["rank"], errors="coerce").fillna(len(feature_names) + 1)
            return out[["feature_name", "screening_score"]].assign(dispersion_kind="llm_frozen_ranking")
    hashes = [int(sha256_text(f"{dataset}:{feature}")[:12], 16) for feature in feature_names]
    scores = np.asarray(hashes, dtype=float)
    scores = scores / max(float(scores.max()), 1.0)
    return pd.DataFrame({"feature_name": feature_names, "screening_score": scores, "dispersion_kind": "deterministic_llm_proxy_no_cache"})


def _annotate_pool(pool: pd.DataFrame, spec: ComparisonRunSpec, requested: int, actual: int) -> pd.DataFrame:
    out = pool.copy()
    out["pool_id"] = spec.run_id
    out["dataset"] = spec.dataset
    out["model"] = spec.model
    out["screening_method"] = spec.screening_method
    out["pool_multiplier"] = spec.pool_multiplier
    out["requested_pool_size"] = requested
    out["actual_pool_size"] = len(out)
    out["random_seed"] = spec.random_seed
    out["checkpoint_seed"] = spec.checkpoint_seed
    out["ablation"] = spec.ablation
    out["temporal_cutoff_id"] = spec.temporal_cutoff_id
    out["semantic_group"] = out["feature_name"].map(infer_semantic_group)
    return out


def _write_pool_manifest(run_dir: Path, spec: ComparisonRunSpec, pool: pd.DataFrame) -> None:
    feature_list = pool["feature_name"].astype(str).tolist()
    diagnostics = semantic_pool_diagnostics(pool)
    _atomic_write_json(
        run_dir / "candidate_pool_manifest.json",
        {
            "pool_id": spec.run_id,
            "dataset": spec.dataset,
            "model": spec.model,
            "screening_method": spec.screening_method,
            "pool_multiplier": spec.pool_multiplier,
            "requested_pool_size": spec.candidate_pool_size,
            "actual_pool_size": int(len(pool)),
            "random_seed": spec.random_seed,
            "checkpoint_seed": spec.checkpoint_seed,
            "ablation": spec.ablation,
            "temporal_cutoff_id": spec.temporal_cutoff_id,
            "candidate_universe_hash": sha256_text(json.dumps(sorted(feature_list))),
            "screening_config_hash": sha256_text(json.dumps(spec.__dict__, sort_keys=True, default=str)),
            "screening_score_hash": sha256_text(json.dumps(pool[["feature_name", "screening_score"]].to_dict("records"), sort_keys=True, default=str)),
            "selected_pool_hash": sha256_text(json.dumps(feature_list, sort_keys=True)),
            "source_artifact_hashes": {},
            **diagnostics,
        },
    )


def _write_predictions(path: Path, spec: ComparisonRunSpec, y: pd.Series, score: np.ndarray, pred: np.ndarray, feature_hash: str, split: str) -> None:
    frame = pd.DataFrame(
        {
            "run_id": spec.run_id,
            "dataset": spec.dataset,
            "model": spec.model,
            "screening_method": spec.screening_method,
            "evaluation_index": np.arange(len(y), dtype=int),
            "y_true": np.asarray(y, dtype=int),
            "y_pred_proba": np.asarray(score, dtype=float),
            "y_pred": np.asarray(pred, dtype=int),
            "split": split,
            "feature_set_hash": feature_hash,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return path


def _atomic_write_json(path: Path, payload: Any) -> Path:
    return _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _default_model_kwargs(model: str) -> dict[str, Any]:
    try:
        kwargs = resolve_model_kwargs(load_named_project_config("homecredit"), model)
    except Exception:
        kwargs = {"random_state": 42}
    if model == "catboost":
        kwargs.setdefault("iterations", 200)
        kwargs.setdefault("verbose", False)
    return kwargs


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif pd.isna(value) if not isinstance(value, (str, list, dict, tuple)) else False:
            out[key] = None
        else:
            out[key] = value
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
