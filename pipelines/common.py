from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import json

import pandas as pd

from Models.utils import get_model_bundle, get_selector
from Preprocessing.data_process import DataLoader
from Preprocessing.feature_engineering import (
    build_all_features,
    build_application_time_proxy,
)
from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import determine_threshold, evaluate_model
from pipelines.comparison import build_experiment_summary_row
from training.kfold_trainer import run_kfold_training
from utils.logging_config import setup_logging

DEFAULT_DATA_DIR = "data/inputs"
DEFAULT_DESCRIPTION_PATH = "data/HomeCredit_columns_description.csv"
DEFAULT_TARGET = "TARGET"
DEFAULT_TIME_COL = "recent_decision"
DEFAULT_DROP_ID_COLS = ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
DEFAULT_OUTPUT_DIR = "outputs"
DECISION_TIME_CANDIDATES = ("recent_decision", "PREV_recent_decision_MAX", "DAYS_DECISION")

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
    preprocessor_kwargs: dict[str, Any] = field(default_factory=dict)
    selector_kwargs: dict[str, Any] = field(default_factory=dict)
    selector_cls: type | None = None


@dataclass(slots=True)
class PreparedExperimentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_oot: pd.DataFrame
    y_oot: pd.Series
    time_col: str


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

    merged_train = merged_train[merged_train[time_col].notna()].copy()
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
    )


def _resolve_selector(config: ExperimentConfig) -> tuple[type | None, dict[str, Any]]:
    if config.selector_cls is not None:
        return config.selector_cls, dict(config.selector_kwargs)

    selector_cls, selector_kwargs = get_selector(config.selector_name)
    selector_kwargs = dict(selector_kwargs)
    selector_kwargs.update(config.selector_kwargs)

    if config.selector_name.lower() == "llm":
        if not selector_kwargs.get("description_csv_path"):
            selector_kwargs["description_csv_path"] = config.description_path
        if not selector_kwargs.get("cache_dir"):
            selector_kwargs["cache_dir"] = str(Path(config.base_output_dir) / "llm_selector_cache")

    return selector_cls, selector_kwargs


def run_experiment(
    config: ExperimentConfig,
    prepared_data: PreparedExperimentData | None = None,
) -> ExperimentRun:
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
    )

    exp_dir_attr = results_df.attrs.get("exp_dir")
    if not exp_dir_attr:
        raise ValueError("Experiment directory was not returned by run_kfold_training.")
    exp_dir = Path(exp_dir_attr)

    X_train_model = prepared.X_train.drop(columns=[prepared.time_col], errors="ignore")
    X_oot_model = prepared.X_oot.drop(columns=[prepared.time_col], errors="ignore")

    final_selector = selector_cls(**selector_kwargs) if selector_cls is not None else None
    if final_selector is not None and hasattr(final_selector, "set_artifact_dir"):
        final_selector.set_artifact_dir(exp_dir / "final_selector")

    saved_final_features: list[str] | None = None
    final_preprocessor = Preprocessor(**dict(config.preprocessor_kwargs))
    if final_selector is not None and getattr(final_selector, "select_before_preprocessing", False):
        X_train_selected_raw = final_selector.fit_transform(X_train_model, prepared.y_train)
        X_oot_selected_raw = final_selector.transform(X_oot_model)

        X_train_processed = final_preprocessor.fit_transform(X_train_selected_raw)
        X_oot_processed = final_preprocessor.transform(X_oot_selected_raw)

        if getattr(final_selector, "apply_post_preprocessing", False):
            X_train_final = final_selector.fit_postprocess(X_train_processed, prepared.y_train)
            X_oot_final = final_selector.transform_postprocess(X_oot_processed)
        else:
            X_train_final = X_train_processed
            X_oot_final = X_oot_processed
        saved_final_features = getattr(final_selector, "selected_features_", None)
    elif final_selector is not None:
        X_train_processed = final_preprocessor.fit_transform(X_train_model)
        X_oot_processed = final_preprocessor.transform(X_oot_model)
        X_train_final = final_selector.fit_transform(X_train_processed, prepared.y_train)
        X_oot_final = final_selector.transform(X_oot_processed)
        saved_final_features = getattr(final_selector, "selected_features_", None)
    else:
        X_train_processed = final_preprocessor.fit_transform(X_train_model)
        X_oot_processed = final_preprocessor.transform(X_oot_model)
        X_train_final = X_train_processed
        X_oot_final = X_oot_processed

    if not isinstance(X_train_final, pd.DataFrame) or not isinstance(X_oot_final, pd.DataFrame):
        raise TypeError("Final preprocessing and feature selection must produce pandas DataFrames.")

    final_features = saved_final_features or X_train_final.columns.tolist()
    features_dir = exp_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": final_features}).to_csv(
        features_dir / "final_selected_features.csv",
        index=False,
    )

    final_model = get_model()
    final_model = train_model(final_model, X_train_final, prepared.y_train, None, None)

    train_proba = predict_proba(final_model, X_train_final)
    oot_proba = predict_proba(final_model, X_oot_final)
    oot_threshold = determine_threshold(prepared.y_train.values, train_proba)
    oot_metrics = evaluate_model(prepared.y_oot.values, oot_proba, threshold=oot_threshold)
    oot_metrics["final_selected_feature_count"] = len(final_features)

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([oot_metrics]).to_csv(results_dir / "oot_test_results.csv", index=False)

    summary_row = build_experiment_summary_row(
        exp_dir=exp_dir,
        method_name=config.experiment_name,
        model_name=config.model_name,
        selector_name=config.selector_name,
    )
    pd.DataFrame([summary_row]).to_csv(results_dir / "experiment_summary.csv", index=False)

    logger.info(
        "Finished experiment %s | exp_dir=%s | oot_auc=%.4f",
        config.experiment_name,
        exp_dir,
        float(summary_row.get("oot_auc", float("nan"))),
    )

    return ExperimentRun(config=config, exp_dir=exp_dir, summary=summary_row)
