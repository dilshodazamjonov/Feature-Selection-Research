from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import json
import random
import time
import hashlib

import joblib
import numpy as np
import pandas as pd

from Models.utils import get_model_bundle, get_selector
from Preprocessing.data_process import DataLoader
from Preprocessing.feature_engineering import (
    build_all_features,
    build_application_time_proxy,
)
from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import determine_threshold, evaluate_model
from evaluation.feature_utils import _feature_score_lookup
from evaluation.feature_stability import selected_feature_psi_frame, selected_feature_psi_summary
from evaluation.stability_scores import calculate_psi
from experiments.config import apply_feature_budget_to_selector_kwargs, apply_random_seed_to_kwargs
from experiments.config import canonical_config_json
from experiments.tracking import build_data_version
from pipelines.comparison import build_experiment_summary_row
from training.kfold_trainer import run_kfold_training
from utils.feature_metadata import infer_semantic_group
from utils.logging_config import setup_logging

DEFAULT_DATA_DIR = "data/inputs"
DEFAULT_DESCRIPTION_PATH = "data/HomeCredit_columns_description.csv"
DEFAULT_TARGET = "TARGET"
DEFAULT_TIME_COL = "recent_decision"
DEFAULT_DROP_ID_COLS = ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
DEFAULT_OUTPUT_DIR = "outputs"
DECISION_TIME_CANDIDATES = ("recent_decision", "PREV_recent_decision_MAX", "DAYS_DECISION")
DEFAULT_EXCLUDED_FEATURE_COLUMNS = (
    DEFAULT_TARGET,
    DEFAULT_TIME_COL,
    "PREV_recent_decision_MAX",
    "DAYS_DECISION",
    "application_time_proxy",
)

logger = setup_logging("pipeline_common", level=logging.INFO)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    selector_name: str
    model_name: str = "lr"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    data_dir: str = DEFAULT_DATA_DIR
    description_path: str = DEFAULT_DESCRIPTION_PATH
    target: str = DEFAULT_TARGET
    time_col: str = DEFAULT_TIME_COL
    drop_id_cols: tuple[str, ...] = DEFAULT_DROP_ID_COLS
    base_output_dir: str = DEFAULT_OUTPUT_DIR
    dev_start_day: int = -600
    oot_start_day: int = -240
    oot_end_day: int = 0
    n_splits: int = 5
    cv_gap_groups: int = 1
    random_state: int = 42
    feature_budget: int = 40
    experiment_output_dir: str | None = None
    excluded_feature_columns: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURE_COLUMNS
    preprocessor_kwargs: dict[str, Any] = field(default_factory=dict)
    selector_kwargs: dict[str, Any] = field(default_factory=dict)
    selector_cls: type | None = None
    experiment_type: str = "single"
    config_hash: str | None = None
    data_fingerprint: dict[str, Any] | None = None


@dataclass(slots=True)
class PreparedExperimentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_oot: pd.DataFrame
    y_oot: pd.Series
    time_col: str
    dropped_older_row_count: int = 0
    dropped_missing_time_row_count: int = 0
    source_row_count: int = 0


@dataclass(slots=True)
class ExperimentRun:
    config: ExperimentConfig
    exp_dir: Path
    summary: dict[str, object]


def create_run_output_dir(base_output_dir: str | Path, run_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_label = run_label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    run_dir = Path(base_output_dir) / f"{safe_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_manifest(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest_path


def resolve_time_col(df: pd.DataFrame, preferred: str, extra_candidates: tuple[str, ...] = ()) -> str:
    candidates = [preferred, preferred.upper(), preferred.lower()]
    for candidate in extra_candidates:
        candidates.extend([candidate, candidate.upper(), candidate.lower()])
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Time column not found. Tried: {candidates}")


def prepare_time_proxy(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    prepared = df.copy()
    if time_col not in prepared.columns:
        raise ValueError(f"{time_col} not found in dataframe.")

    observed_time = prepared[time_col].dropna()
    if observed_time.empty:
        raise ValueError(f"{time_col} has no observed values.")

    missing_count = int(prepared[time_col].isna().sum())
    if missing_count:
        fill_value = float(observed_time.min()) - 30.0
        prepared[time_col] = prepared[time_col].fillna(fill_value)
        logger.info(
            "Filled %s missing %s values with conservative early proxy %.1f",
            f"{missing_count:,}",
            time_col,
            fill_value,
        )

    return prepared


def prepare_modeling_data(config: ExperimentConfig) -> PreparedExperimentData:
    logger.info("Loading datasets from %s", config.data_dir)
    loader = DataLoader(config.data_dir)
    dfs = loader.load_all()

    if "application_train" not in dfs:
        raise ValueError("application_train.csv not found in data directory")

    app_train = dfs["application_train"].copy()

    logger.info("Building application-level time proxy")
    time_proxy_df = build_application_time_proxy(dfs)
    if time_proxy_df is not None:
        logger.info("Application-level time proxy shape: %s", time_proxy_df.shape)

    logger.info("Building feature tables")
    feature_tables = build_all_features(dfs.copy())
    if time_proxy_df is not None:
        feature_tables.append(time_proxy_df)
    if feature_tables:
        feature_shapes = {
            getattr(table, "name", f"table_{idx + 1}"): tuple(table.shape)
            for idx, table in enumerate(feature_tables)
        }
        logger.info("Feature engineering produced %s tables", len(feature_tables))
        logger.info("Feature table shapes: %s", feature_shapes)

    logger.info("Merging feature tables into application_train")
    merged_train = loader.merge_features(app_train, feature_tables, on="SK_ID_CURR")
    logger.info("Merged training shape after feature engineering: %s", merged_train.shape)

    time_col = resolve_time_col(
        merged_train,
        config.time_col,
        extra_candidates=DECISION_TIME_CANDIDATES,
    )

    source_row_count = int(len(merged_train))
    dropped_missing_time_row_count = int(merged_train[time_col].isna().sum())
    merged_train = merged_train[merged_train[time_col].notna()].copy()
    dropped_older_row_count = int((merged_train[time_col] < config.dev_start_day).sum())
    merged_train = merged_train[
        (merged_train[time_col] >= config.dev_start_day)
        & (merged_train[time_col] <= config.oot_end_day)
    ].copy()
    logger.info("Merged training shape after time filtering: %s", merged_train.shape)

    cv_data = merged_train[
        (merged_train[time_col] >= config.dev_start_day)
        & (merged_train[time_col] < config.oot_start_day)
    ].copy()
    oot_data = merged_train[
        (merged_train[time_col] >= config.oot_start_day)
        & (merged_train[time_col] <= config.oot_end_day)
    ].copy()

    if cv_data.empty or oot_data.empty:
        raise ValueError(
            "Temporal split failed: "
            f"cv_rows={len(cv_data)}, oot_rows={len(oot_data)}, "
            f"dev_start_day={config.dev_start_day}, oot_start_day={config.oot_start_day}, "
            f"oot_end_day={config.oot_end_day}"
        )

    X_train_full = cv_data.drop(columns=[config.target])
    y_train_full = cv_data[config.target].copy()
    X_oot_full = oot_data.drop(columns=[config.target])
    y_oot = oot_data[config.target].copy()

    drop_cols = [col for col in config.drop_id_cols if col in X_train_full.columns]
    X_train = X_train_full.drop(columns=drop_cols, errors="ignore")
    X_oot = X_oot_full.drop(
        columns=[col for col in config.drop_id_cols if col in X_oot_full.columns],
        errors="ignore",
    )

    logger.info("Prepared CV split: %s", X_train.shape)
    logger.info("Prepared OOT split: %s", X_oot.shape)

    return PreparedExperimentData(
        X_train=X_train,
        y_train=y_train_full,
        X_oot=X_oot,
        y_oot=y_oot,
        time_col=time_col,
        dropped_older_row_count=dropped_older_row_count,
        dropped_missing_time_row_count=dropped_missing_time_row_count,
        source_row_count=source_row_count,
    )


def _resolve_selector(config: ExperimentConfig) -> tuple[type | None, dict[str, Any]]:
    if config.selector_cls is not None:
        selector_kwargs = apply_random_seed_to_kwargs(
            dict(config.selector_kwargs),
            config.random_state,
        )
        return config.selector_cls, selector_kwargs

    selector_cls, selector_kwargs = get_selector(config.selector_name)
    selector_kwargs = dict(selector_kwargs)
    selector_kwargs.update(config.selector_kwargs)
    selector_kwargs = apply_feature_budget_to_selector_kwargs(
        config.selector_name,
        selector_kwargs,
        config.feature_budget,
    )
    selector_kwargs = apply_random_seed_to_kwargs(selector_kwargs, config.random_state)

    if config.selector_name.lower() in {"llm", "domain_rule_baseline"}:
        if not selector_kwargs.get("description_csv_path"):
            selector_kwargs["description_csv_path"] = config.description_path
        if config.selector_name.lower() == "llm" and not selector_kwargs.get("cache_dir"):
            selector_kwargs["cache_dir"] = str(Path(config.base_output_dir) / "_llm_rankings_cache")

    return selector_cls, selector_kwargs


def drop_excluded_feature_columns(
    X: pd.DataFrame,
    *,
    time_col: str,
    excluded_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Remove configured target/time/leakage columns from model features."""
    columns_to_drop = set(excluded_columns) | {time_col}
    return X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors="ignore")


def write_leakage_report(
    *,
    exp_dir: str | Path,
    config: ExperimentConfig,
    prepared: PreparedExperimentData,
    X_train_model: pd.DataFrame,
    X_oot_model: pd.DataFrame,
) -> Path:
    """
    Persist leakage guardrail checks for the run.

    The checks intentionally fail hard for objective issues such as target/time
    columns inside model features. Scope-based safeguards are recorded from the
    pipeline contract: selectors and preprocessors receive train folds only in
    CV, and the final fit receives DEV only.
    """
    forbidden = set(config.excluded_feature_columns) | {config.target, prepared.time_col}
    train_forbidden = sorted(forbidden.intersection(X_train_model.columns))
    oot_forbidden = sorted(forbidden.intersection(X_oot_model.columns))

    train_time = prepared.X_train[prepared.time_col]
    oot_time = prepared.X_oot[prepared.time_col]
    temporal_split_ok = bool(train_time.max() < oot_time.min())

    report = {
        "target_column_excluded": config.target not in X_train_model.columns
        and config.target not in X_oot_model.columns,
        "temporal_split_disjoint": temporal_split_ok,
        "train_time_max": float(train_time.max()),
        "oot_time_min": float(oot_time.min()),
        "forbidden_columns_in_train_features": train_forbidden,
        "forbidden_columns_in_oot_features": oot_forbidden,
        "oot_used_in_feature_selection": False,
        "llm_metadata_scope": (
            "training_fold_only_in_cv_and_dev_only_for_final_fit"
            if "llm" in config.selector_name.lower()
            else "not_applicable"
        ),
        "preprocessing_fit_scope": "training_fold_only_in_cv_and_dev_only_for_final_fit",
    }

    if train_forbidden or oot_forbidden:
        raise ValueError(
            "Leakage check failed. Forbidden columns reached model features: "
            f"train={train_forbidden}, oot={oot_forbidden}"
        )
    if not temporal_split_ok:
        raise ValueError(
            "Leakage check failed. OOT window is not strictly after DEV: "
            f"train_max={train_time.max()}, oot_min={oot_time.min()}"
        )

    report_path = Path(exp_dir) / "leakage_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report_path


def write_data_split_manifest(
    *,
    exp_dir: str | Path,
    config: ExperimentConfig,
    prepared: PreparedExperimentData,
) -> Path:
    """Persist the temporal DEV/OOT split metadata used by one run."""
    train_time = pd.to_numeric(prepared.X_train[prepared.time_col], errors="coerce")
    oot_time = pd.to_numeric(prepared.X_oot[prepared.time_col], errors="coerce")
    payload = {
        "time_column": prepared.time_col,
        "configured_windows": {
            "dev_start_day": config.dev_start_day,
            "oot_start_day": config.oot_start_day,
            "oot_end_day": config.oot_end_day,
        },
        "DEV_window": {
            "start_day_inclusive": config.dev_start_day,
            "end_day_exclusive": config.oot_start_day,
        },
        "OOT_window": {
            "start_day_inclusive": config.oot_start_day,
            "end_day_inclusive": config.oot_end_day,
        },
        "dev": {
            "row_count": int(len(prepared.X_train)),
            "time_min": float(train_time.min()),
            "time_max": float(train_time.max()),
            "target_rate": float(pd.Series(prepared.y_train).mean()),
        },
        "oot": {
            "row_count": int(len(prepared.X_oot)),
            "time_min": float(oot_time.min()),
            "time_max": float(oot_time.max()),
            "target_rate": float(pd.Series(prepared.y_oot).mean()),
        },
        "dropped_older_row_count": int(prepared.dropped_older_row_count),
        "dropped_missing_time_row_count": int(prepared.dropped_missing_time_row_count),
        "source_row_count": int(prepared.source_row_count),
    }
    path = Path(exp_dir) / "data_split_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def credit_risk_utility(y_true: pd.Series, y_score: np.ndarray, top_fracs: tuple[float, ...] = (0.1, 0.2)) -> dict[str, float]:
    """Compute simple ranking utility metrics for credit-risk scorecards."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "score": np.asarray(y_score)})
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    overall_bad_rate = float(frame["y_true"].mean()) if len(frame) else np.nan
    total_bads = float(frame["y_true"].sum())
    metrics: dict[str, float] = {}
    for frac in top_fracs:
        pct = int(frac * 100)
        n_top = max(1, int(np.ceil(len(frame) * frac)))
        top = frame.head(n_top)
        bad_rate_top = float(top["y_true"].mean()) if len(top) else np.nan
        metrics[f"lift_at_{pct}"] = (
            float(bad_rate_top / overall_bad_rate)
            if overall_bad_rate and pd.notna(overall_bad_rate)
            else np.nan
        )
        metrics[f"bad_rate_capture_at_{pct}"] = (
            float(top["y_true"].sum() / total_bads)
            if total_bads
            else np.nan
        )
    return metrics


def run_experiment(
    config: ExperimentConfig,
    prepared_data: PreparedExperimentData | None = None,
) -> ExperimentRun:
    run_start = time.time()
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    prepared = prepared_data or prepare_modeling_data(config)

    selector_cls, selector_kwargs = _resolve_selector(config)
    get_model, train_model, predict_proba, save_model = get_model_bundle(
        config.model_name,
        model_kwargs=dict(config.model_kwargs),
    )

    logger.info(
        "Starting experiment %s | model=%s | selector=%s",
        config.experiment_name,
        config.model_name,
        config.selector_name,
    )

    results_df = run_kfold_training(
        X=prepared.X_train.copy(),
        y=prepared.y_train.copy(),
        time_col=prepared.time_col,
        get_model=get_model,
        train_model=train_model,
        predict_proba=predict_proba,
        save_model=save_model,
        preprocessor_cls=Preprocessor,
        preprocessor_kwargs=dict(config.preprocessor_kwargs),
        selector_cls=selector_cls,
        selector_kwargs=selector_kwargs,
        model_name=f"{config.model_name}_{config.experiment_name}",
        base_output_dir=config.base_output_dir,
        n_splits=config.n_splits,
        random_state=config.random_state,
        gap_groups=config.cv_gap_groups,
        experiment_output_dir=config.experiment_output_dir,
        selector_name=config.selector_name,
        excluded_feature_columns=config.excluded_feature_columns,
        feature_budget=config.feature_budget,
    )

    exp_dir_attr = results_df.attrs.get("exp_dir")
    if not exp_dir_attr:
        raise ValueError("Experiment directory was not returned by run_kfold_training.")
    exp_dir = Path(exp_dir_attr)

    X_train_model = drop_excluded_feature_columns(
        prepared.X_train,
        time_col=prepared.time_col,
        excluded_columns=config.excluded_feature_columns,
    )
    X_oot_model = drop_excluded_feature_columns(
        prepared.X_oot,
        time_col=prepared.time_col,
        excluded_columns=config.excluded_feature_columns,
    )
    write_leakage_report(
        exp_dir=exp_dir,
        config=config,
        prepared=prepared,
        X_train_model=X_train_model,
        X_oot_model=X_oot_model,
    )
    write_data_split_manifest(exp_dir=exp_dir, config=config, prepared=prepared)

    features_dir = exp_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    final_selector = selector_cls(**selector_kwargs) if selector_cls is not None else None
    if final_selector is not None and hasattr(final_selector, "set_ranking_context"):
        final_selector.set_ranking_context(
            scope="final_dev",
            fold_id=None,
            ranking_artifact_dir=features_dir,
            selector_name=config.selector_name,
        )

    saved_final_features: list[str] | None = None
    final_preprocessor = Preprocessor(**dict(config.preprocessor_kwargs))
    final_preprocessing_time_sec = 0.0
    final_feature_selection_time_sec = 0.0
    if final_selector is not None and getattr(final_selector, "select_before_preprocessing", False):
        selection_start = time.time()
        if getattr(final_selector, "apply_post_preprocessing", False):
            final_selector.fit(X_train_model, prepared.y_train)
            X_train_selected_raw = final_selector.transform(X_train_model)
        else:
            X_train_selected_raw = final_selector.fit_transform(X_train_model, prepared.y_train)
        X_oot_selected_raw = final_selector.transform(X_oot_model)
        final_feature_selection_time_sec += time.time() - selection_start

        preprocessing_start = time.time()
        X_train_processed = final_preprocessor.fit_transform(X_train_selected_raw)
        X_oot_processed = final_preprocessor.transform(X_oot_selected_raw)
        final_preprocessing_time_sec += time.time() - preprocessing_start

        if getattr(final_selector, "apply_post_preprocessing", False):
            selection_start = time.time()
            X_train_final = final_selector.fit_postprocess(X_train_processed, prepared.y_train)
            X_oot_final = final_selector.transform_postprocess(X_oot_processed)
            final_feature_selection_time_sec += time.time() - selection_start
        else:
            X_train_final = X_train_processed
            X_oot_final = X_oot_processed
        saved_final_features = getattr(final_selector, "selected_features_", None)
    elif final_selector is not None:
        preprocessing_start = time.time()
        X_train_processed = final_preprocessor.fit_transform(X_train_model)
        X_oot_processed = final_preprocessor.transform(X_oot_model)
        final_preprocessing_time_sec += time.time() - preprocessing_start
        selection_start = time.time()
        X_train_final = final_selector.fit_transform(X_train_processed, prepared.y_train)
        X_oot_final = final_selector.transform(X_oot_processed)
        final_feature_selection_time_sec += time.time() - selection_start
        saved_final_features = getattr(final_selector, "selected_features_", None)
    else:
        preprocessing_start = time.time()
        X_train_processed = final_preprocessor.fit_transform(X_train_model)
        X_oot_processed = final_preprocessor.transform(X_oot_model)
        final_preprocessing_time_sec += time.time() - preprocessing_start
        X_train_final = X_train_processed
        X_oot_final = X_oot_processed

    if not isinstance(X_train_final, pd.DataFrame) or not isinstance(X_oot_final, pd.DataFrame):
        raise TypeError("Final preprocessing and feature selection must produce pandas DataFrames.")

    final_features = saved_final_features or X_train_final.columns.tolist()
    score_lookup = _feature_score_lookup(final_selector, final_features)
    pd.DataFrame(
        [
            {
                "fold_id": "final_dev",
                "selector": config.selector_name,
                "feature_name": str(feature),
                "feature": str(feature),
                "semantic_group": infer_semantic_group(str(feature)),
                "rank": rank,
                "score": score_lookup.get(str(feature), pd.NA),
            }
            for rank, feature in enumerate(final_features, start=1)
        ]
    ).to_csv(features_dir / "final_selected_features.csv", index=False)

    llm_features = getattr(final_selector, "llm_selected_features_", None)
    if llm_features is not None:
        final_rank = {str(feature): rank for rank, feature in enumerate(final_features, start=1)}
        trace_rows = [
            {
                "scope": "final_dev",
                "fold_id": pd.NA,
                "selector": config.selector_name,
                "llm_rank": rank,
                "feature_name": str(feature),
                "survived_hybrid": str(feature) in final_rank,
                "hybrid_rank": final_rank.get(str(feature), pd.NA),
            }
            for rank, feature in enumerate(llm_features, start=1)
        ]
        if trace_rows:
            trace_path = features_dir / "llm_hybrid_trace.csv"
            trace_df = pd.DataFrame(trace_rows)
            if trace_path.exists():
                trace_df = pd.concat([pd.read_csv(trace_path), trace_df], ignore_index=True)
            trace_df.to_csv(trace_path, index=False)

    final_model = get_model()
    final_training_start = time.time()
    final_model = train_model(final_model, X_train_final, prepared.y_train, None, None)
    final_training_time_sec = time.time() - final_training_start

    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(final_model, str(models_dir / "final_model.model"))
    joblib.dump(final_preprocessor, models_dir / "final_preprocessor.pkl")
    data_fingerprint = config.data_fingerprint or build_data_version(config.data_dir)
    preprocessing_payload = {
        "class": final_preprocessor.__class__.__name__,
        "kwargs": config.preprocessor_kwargs,
    }
    preprocessing_hash = hashlib.sha256(
        canonical_config_json(preprocessing_payload).encode("utf-8")
    ).hexdigest()
    (models_dir / "final_model_metadata.json").write_text(
        json.dumps(
            {
                "model": config.model_name,
                "selector": config.selector_name,
                "experiment_type": config.experiment_type,
                "feature_budget": int(config.feature_budget),
                "selected_features": [str(feature) for feature in final_features],
                "feature_order": [str(feature) for feature in X_train_final.columns.tolist()],
                "model_params": config.model_kwargs,
                "preprocessing": preprocessing_payload,
                "preprocessing_hash": preprocessing_hash,
                "random_seed": int(config.random_state),
                "training_scope": "full_DEV",
                "target_column": config.target,
                "config_hash": config.config_hash,
                "data_fingerprint": data_fingerprint,
                "n_training_rows": int(len(X_train_final)),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    final_evaluation_start = time.time()
    train_proba = predict_proba(final_model, X_train_final)
    oot_proba = predict_proba(final_model, X_oot_final)
    oot_threshold = determine_threshold(prepared.y_train.values, train_proba)
    oot_metrics = evaluate_model(prepared.y_oot.values, oot_proba, threshold=oot_threshold)
    oot_metrics["final_selected_feature_count"] = len(final_features)
    oot_metrics["selected_feature_count"] = len(final_features)
    oot_metrics["total_candidate_feature_count"] = int(X_train_model.shape[1])
    oot_metrics["feature_budget"] = int(config.feature_budget)
    oot_metrics["feature_reduction_ratio"] = (
        1.0 - len(final_features) / X_train_model.shape[1]
        if X_train_model.shape[1]
        else np.nan
    )
    oot_metrics["oot_gini_per_feature"] = oot_metrics["gini"] / len(final_features) if final_features else np.nan
    oot_metrics["oot_auc_per_feature"] = oot_metrics["auc"] / len(final_features) if final_features else np.nan
    oot_metrics["oot_ks_per_feature"] = oot_metrics["ks"] / len(final_features) if final_features else np.nan

    selected_psi_df = selected_feature_psi_frame(X_train_final, X_oot_final)
    oot_metrics.update(selected_feature_psi_summary(selected_psi_df))
    model_score_psi = calculate_psi(pd.Series(train_proba), pd.Series(oot_proba))
    oot_metrics["model_score_psi"] = model_score_psi
    utility_metrics = credit_risk_utility(prepared.y_oot, oot_proba)
    oot_metrics.update(utility_metrics)
    final_evaluation_time_sec = time.time() - final_evaluation_start

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_psi_df.to_csv(results_dir / "selected_feature_psi.csv", index=False)
    pd.DataFrame([{"model_score_psi": model_score_psi}]).to_csv(
        results_dir / "model_score_psi.csv",
        index=False,
    )
    pd.DataFrame([utility_metrics]).to_csv(results_dir / "credit_risk_utility.csv", index=False)
    pd.DataFrame(
        {
            "y_true": prepared.y_oot.values,
            "y_pred_proba": oot_proba,
            "y_pred": (oot_proba >= oot_threshold).astype(int),
        }
    ).to_csv(results_dir / "oot_predictions.csv", index=False)
    pd.DataFrame([oot_metrics]).to_csv(results_dir / "oot_test_results.csv", index=False)

    summary_row = build_experiment_summary_row(
        exp_dir=exp_dir,
        method_name=config.experiment_name,
        model_name=config.model_name,
        selector_name=config.selector_name,
    )
    runtime_payload = {
        "run_id": exp_dir.name,
        "cv_preprocessing_time_sec": float(results_df.attrs.get("cv_preprocessing_time_sec", np.nan)),
        "cv_feature_selection_time_sec": float(results_df.attrs.get("cv_feature_selection_time_sec", np.nan)),
        "cv_training_time_sec": float(results_df.attrs.get("cv_training_time_sec", np.nan)),
        "cv_evaluation_time_sec": float(results_df.attrs.get("cv_evaluation_time_sec", np.nan)),
        "cv_runtime_seconds": float(results_df.attrs.get("cv_runtime_seconds", np.nan)),
        "final_preprocessing_time_sec": float(final_preprocessing_time_sec),
        "final_feature_selection_time_sec": float(final_feature_selection_time_sec),
        "final_training_time_sec": float(final_training_time_sec),
        "final_evaluation_time_sec": float(final_evaluation_time_sec),
        "preprocessing_time_sec": float(results_df.attrs.get("cv_preprocessing_time_sec", 0.0) + final_preprocessing_time_sec),
        "feature_selection_time_sec": float(results_df.attrs.get("cv_feature_selection_time_sec", 0.0) + final_feature_selection_time_sec),
        "training_time_sec": float(results_df.attrs.get("cv_training_time_sec", 0.0) + final_training_time_sec),
        "evaluation_time_sec": float(results_df.attrs.get("cv_evaluation_time_sec", 0.0) + final_evaluation_time_sec),
        "total_runtime_seconds": float(time.time() - run_start),
    }
    pd.DataFrame([runtime_payload]).to_csv(results_dir / "runtime_summary.csv", index=False)
    summary_row["runtime_seconds"] = runtime_payload["total_runtime_seconds"]
    pd.DataFrame([summary_row]).to_csv(results_dir / "experiment_summary.csv", index=False)

    logger.info(
        "Finished experiment %s | exp_dir=%s | oot_auc=%.4f",
        config.experiment_name,
        exp_dir,
        float(summary_row.get("oot_auc", float("nan"))),
    )

    return ExperimentRun(config=config, exp_dir=exp_dir, summary=summary_row)
