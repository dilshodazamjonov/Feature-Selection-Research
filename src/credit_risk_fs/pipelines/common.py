from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

from credit_risk_fs.data.loaders import DataLoader
from credit_risk_fs.evaluation._feature_utils import _feature_score_lookup
from credit_risk_fs.evaluation.drift import calculate_psi
from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.evaluation.stability import (
    selected_feature_psi_frame,
    selected_feature_psi_summary,
)
from credit_risk_fs.experiments.compare import build_experiment_summary_row
from credit_risk_fs.experiments.atomic_io import (
    copy_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from credit_risk_fs.experiments.config import (
    apply_feature_budget_to_selector_kwargs,
    apply_random_seed_to_kwargs,
    canonical_config_json,
)
from credit_risk_fs.experiments.result_paths import (
    reject_historical_write,
    sanitize_component,
)
from credit_risk_fs.experiments.resource_policy import apply_estimator_parallelism
from credit_risk_fs.experiments.tracking import (
    build_data_version,
    write_resource_usage,
    write_run_manifest as write_active_run_manifest,
)
from credit_risk_fs.feature_engineering.homecredit.assemble import (
    build_all_features,
    build_application_time_proxy,
)
from credit_risk_fs.feature_engineering.lendingclub.application import (
    build_application_features as build_lendingclub_application_features,
)
from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group
from credit_risk_fs.models.registry import get_model_bundle
from credit_risk_fs.models.training import run_kfold_training
from credit_risk_fs.pipelines.dataset_adapter import resolve_dataset_mode
from credit_risk_fs.preprocessing.encoding import Preprocessor
from credit_risk_fs.preprocessing.lendingclub import (
    prepare_lendingclub_application_frame,
)
from credit_risk_fs.selectors.registry import get_selector
from credit_risk_fs.utils.logging import setup_logging

DEFAULT_DATA_DIR = "data/homecredit/raw"
DEFAULT_DESCRIPTION_PATH = "data/homecredit/metadata/columns_description.csv"
DEFAULT_TARGET = "TARGET"
DEFAULT_TIME_COL = "recent_decision"
DEFAULT_DROP_ID_COLS = ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "loan_id")
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

DERIVED_METRIC_IMPLEMENTATION_VERSION = "saved_prediction_derived_metrics_v1"
SCORE_PSI_IMPLEMENTATION_VERSION = "dev_oof_quantile_psi_v1"
DERIVED_METRIC_VALIDATION_TOLERANCE = 1e-12
SCORE_PSI_REQUESTED_BIN_COUNT = 10
SCORE_PSI_SMOOTHING_EPSILON = 1e-6


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    selector_name: str
    dataset_name: str = "homecredit"
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
    method: str = ""
    source_training_dataset: str = ""
    external_dataset: str = ""
    pairing_policy_version: str = "not_applicable_non_clip"
    stable_row_id_column: str | None = None
    identity_sidecar_path: str | None = None
    identity_manifest_path: str | None = None
    stability_candidate_pool_path: str | None = None
    input_column_projection: dict[str, tuple[str, ...]] | None = None
    required_feature_columns: tuple[str, ...] = ()
    require_full_candidate_projection: bool = True
    csv_chunk_rows: int | None = None
    estimator_threads: int = 1
    stage_callback: Any | None = field(default=None, repr=False)
    cooperative_stop_event: Any | None = field(default=None, repr=False)


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
    dev_stable_row_ids: pd.Series | None = None
    oot_stable_row_ids: pd.Series | None = None
    source_identity: "SourceIdentityProvenance | None" = None
    data_load_report: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceIdentityProvenance:
    manifest: dict[str, Any]
    authenticated_ids: frozenset[str]
    target_by_id: dict[str, Any]


@dataclass(frozen=True)
class PredictionMetadata:
    dataset: str
    split: str
    run_id: str
    method: str
    model: str
    source_training_dataset: str
    external_dataset: str
    configuration_hash: str
    data_manifest_hash: str
    pairing_policy_version: str
    source_identity_manifest_hash: str
    stable_row_id_column: str
    source_stable_id_values_hash: str


def prediction_metadata_from_sources(
    explicit: dict[str, Any], supplemental: dict[str, Any] | None = None
) -> PredictionMetadata:
    supplemental = dict(supplemental or {})
    duplicates = set(explicit) & set(supplemental)
    if duplicates:
        raise ValueError(
            f"prediction metadata fields supplied more than once: {sorted(duplicates)}"
        )
    values = {**explicit, **supplemental}
    allowed = set(PredictionMetadata.__dataclass_fields__)
    unknown = set(values) - allowed
    missing = allowed - set(values)
    if unknown or missing:
        raise ValueError(
            f"prediction metadata schema mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return PredictionMetadata(**values)


@dataclass(slots=True)
class ExperimentRun:
    config: ExperimentConfig
    exp_dir: Path
    summary: dict[str, object]


def _report_execution_stage(
    config: ExperimentConfig,
    stage: str,
    fold_id: str | int | None = None,
) -> None:
    if config.cooperative_stop_event is not None and config.cooperative_stop_event.is_set():
        raise RuntimeError(f"cooperative stop requested before stage {stage}")
    if config.stage_callback is not None:
        config.stage_callback(stage, fold_id)


def create_run_output_dir(base_output_dir: str | Path, run_label: str) -> Path:
    """Compatibility helper for non-active outputs; collisions fail closed."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_label = sanitize_component(run_label, field_name="run label")
    run_dir = reject_historical_write(
        Path(base_output_dir) / f"{safe_label}_{timestamp}"
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"run output directory already exists: {run_dir}") from exc
    return run_dir


def write_run_manifest(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Compatibility wrapper around the canonical active manifest writer."""

    return write_active_run_manifest(run_dir, payload)


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


def calculate_required_columns(
    config: ExperimentConfig,
    loader: DataLoader,
) -> dict[str, list[str]]:
    """Calculate and validate explicit per-table projections before any data load."""

    if config.input_column_projection is not None:
        projections = {
            str(table): [str(column) for column in columns]
            for table, columns in config.input_column_projection.items()
        }
        if not projections:
            raise ValueError("input_column_projection must not be empty")
        for table, columns in projections.items():
            if not columns:
                raise ValueError(f"input projection for {table!r} must not be empty")
            available = loader.inspect_columns(table)
            missing = set(columns) - set(available)
            if missing:
                raise ValueError(
                    f"input projection for {table!r} contains unknown columns: {sorted(missing)}"
                )
        return projections

    tables = loader.available_tables()
    if config.require_full_candidate_projection:
        return {
            table: loader.inspect_columns(table)
            for table in tables
        }
    if not config.required_feature_columns:
        raise ValueError(
            "experiment must calculate required_feature_columns or explicitly request "
            "the full candidate-universe projection"
        )
    if "application_train" not in tables:
        raise ValueError("application_train table is required")
    application_columns = loader.inspect_columns("application_train")
    required = [
        config.target,
        config.time_col,
        *config.drop_id_cols,
        *config.required_feature_columns,
    ]
    projection = []
    for column in required:
        if column in application_columns and column not in projection:
            projection.append(column)
    missing_features = set(config.required_feature_columns) - set(projection)
    if missing_features:
        raise ValueError(
            f"required feature columns are absent from application_train: {sorted(missing_features)}"
        )
    return {"application_train": projection}


def prepare_modeling_data(config: ExperimentConfig) -> PreparedExperimentData:
    logger.info("Loading datasets from %s", config.data_dir)
    loader = DataLoader(config.data_dir)
    projections = calculate_required_columns(config, loader)
    dfs = loader.load_all(
        projections,
        require_projection=True,
        csv_chunk_rows=config.csv_chunk_rows,
    )
    dataset_mode = resolve_dataset_mode(config=config, loaded_frames=dfs)
    if "application_train" not in dfs:
        raise ValueError("application_train.csv not found in data directory")

    app_train = dfs["application_train"]
    source_identity = None
    if config.stable_row_id_column and str(config.dataset_name).lower() == "lendingclub_v2":
        from credit_risk_fs.experiments.lendingclub_identity import (
            load_lendingclub_identity_sidecar,
        )

        if not config.identity_sidecar_path or not config.identity_manifest_path:
            raise ValueError(
                "lendingclub_v2 stable identity requires identity_sidecar_path and "
                "identity_manifest_path"
            )
        identity_bundle = load_lendingclub_identity_sidecar(
            sidecar_path=config.identity_sidecar_path,
            manifest_path=config.identity_manifest_path,
            processed_frame=app_train,
        )
        if config.stable_row_id_column != "loan_id":
            raise ValueError("lendingclub_v2 authenticated identity must use loan_id")
        app_train.insert(0, "loan_id", identity_bundle.frame["loan_id"].to_numpy())
        identity_manifest = dict(identity_bundle.manifest)
        identity_manifest.update(
            {
                "manifest_version": identity_bundle.manifest["schema_version"],
                "dataset": "lendingclub_v2",
                "source_artifact": identity_bundle.manifest["raw_source"]["path"],
                "source_artifact_hash": identity_bundle.manifest["raw_source"]["sha256"],
                "source_identity_manifest_hash": identity_bundle.manifest["manifest_sha256"],
            }
        )
        canonical_ids = identity_bundle.frame["loan_id"].astype(str)
        source_identity = SourceIdentityProvenance(
            manifest=identity_manifest,
            authenticated_ids=frozenset(canonical_ids),
            target_by_id=dict(
                zip(
                    canonical_ids,
                    identity_bundle.frame["target"].astype(int).astype(str),
                    strict=True,
                )
            ),
        )
    elif config.stable_row_id_column:
        source_identity = build_source_identity_provenance(
            app_train,
            dataset=config.dataset_name,
            stable_row_id_column=config.stable_row_id_column,
            target_column=config.target,
            source_artifact_path=Path(config.data_dir) / "application_train.csv",
        )

    if dataset_mode == "homecredit_multitable":
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
    else:
        logger.info("Using single-table dataset mode for %s", config.dataset_name)
        merged_train = app_train
        if str(config.dataset_name).lower() == "lendingclub":
            logger.info("Applying LendingClub application-time cleanup.")
            merged_train = prepare_lendingclub_application_frame(
                merged_train,
                target_col=config.target,
                time_col=config.time_col,
                issue_col="issue_d",
            )
            logger.info("Building LendingClub engineered application features.")
            merged_train = build_lendingclub_application_features(merged_train)
            logger.info(
                "LendingClub modeling frame shape after feature engineering: %s",
                merged_train.shape,
            )

    time_col = resolve_time_col(
        merged_train,
        config.time_col,
        extra_candidates=DECISION_TIME_CANDIDATES,
    )
    if config.stable_row_id_column:
        validate_source_identity_subset(
            merged_train,
            source_identity=source_identity,
            dataset=config.dataset_name,
            stable_row_id_column=config.stable_row_id_column,
            target_column=config.target,
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
    dev_stable_row_ids = (
        _stable_row_ids(
            cv_data,
            dataset=config.dataset_name,
            stable_row_id_column=config.stable_row_id_column,
            source_identity=source_identity,
            target_column=config.target,
        )
        if config.stable_row_id_column
        else None
    )
    oot_stable_row_ids = (
        _stable_row_ids(
            oot_data,
            dataset=config.dataset_name,
            stable_row_id_column=config.stable_row_id_column,
            source_identity=source_identity,
            target_column=config.target,
        )
        if config.stable_row_id_column
        else None
    )
    if dev_stable_row_ids is not None and oot_stable_row_ids is not None:
        validate_authenticated_split_ids(dev_stable_row_ids, oot_stable_row_ids)

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
        dev_stable_row_ids=dev_stable_row_ids.reset_index(drop=True) if dev_stable_row_ids is not None else None,
        oot_stable_row_ids=oot_stable_row_ids.reset_index(drop=True) if oot_stable_row_ids is not None else None,
        source_identity=source_identity,
        data_load_report=dict(loader.last_load_report),
    )


def _resolve_selector(config: ExperimentConfig) -> tuple[type | None, dict[str, Any]]:
    if config.selector_cls is not None:
        selector_kwargs = apply_random_seed_to_kwargs(
            dict(config.selector_kwargs),
            config.random_state,
        )
        _, selector_kwargs = apply_estimator_parallelism(
            config.model_name,
            {},
            selector_kwargs,
            estimator_threads=config.estimator_threads,
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
    _, selector_kwargs = apply_estimator_parallelism(
        config.model_name,
        {},
        selector_kwargs,
        estimator_threads=config.estimator_threads,
    )

    selector_name = config.selector_name.lower()
    metadata_selectors = {
        "llm",
        "llm_then_stat",
        "llm_then_mrmr",
        "llm_then_boruta",
        "stable_core_llm_fill",
        "domain_rule_baseline",
    }
    llm_cache_selectors = {
        "llm",
        "llm_then_stat",
        "llm_then_mrmr",
        "llm_then_boruta",
        "stable_core_llm_fill",
    }

    if selector_name in metadata_selectors:
        if not selector_kwargs.get("description_csv_path"):
            selector_kwargs["description_csv_path"] = config.description_path
        if selector_name in llm_cache_selectors and not selector_kwargs.get("cache_dir"):
            selector_kwargs["cache_dir"] = "artifacts/llm_cache"

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
    write_json_atomic(report_path, report)
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
        "column_projection": prepared.data_load_report,
    }
    path = Path(exp_dir) / "data_split_manifest.json"
    write_json_atomic(path, payload)
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
    _report_execution_stage(config, "experiment_started")
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    prepared = prepared_data or prepare_modeling_data(config)
    if config.experiment_type == "corrected_reverse_transfer":
        if prepared.source_identity is None:
            raise RuntimeError("authenticated raw-source identity manifest is required")
        if prepared.dev_stable_row_ids is None or prepared.oot_stable_row_ids is None:
            raise RuntimeError("authenticated DEV/OOT stable IDs are required")
        dev_identity_frame = pd.DataFrame(
            {
                config.stable_row_id_column: prepared.dev_stable_row_ids,
                config.target: prepared.y_train.reset_index(drop=True),
            }
        )
        oot_identity_frame = pd.DataFrame(
            {
                config.stable_row_id_column: prepared.oot_stable_row_ids,
                config.target: prepared.y_oot.reset_index(drop=True),
            }
        )
        validate_source_identity_subset(
            dev_identity_frame,
            source_identity=prepared.source_identity,
            dataset=config.dataset_name,
            stable_row_id_column=str(config.stable_row_id_column),
            target_column=config.target,
        )
        validate_source_identity_subset(
            oot_identity_frame,
            source_identity=prepared.source_identity,
            dataset=config.dataset_name,
            stable_row_id_column=str(config.stable_row_id_column),
            target_column=config.target,
            verify_source_artifact=False,
        )
        validate_authenticated_split_ids(
            prepared.dev_stable_row_ids, prepared.oot_stable_row_ids
        )
        if not config.experiment_output_dir:
            raise RuntimeError(
                "reverse-transfer identity manifest requires a fixed experiment output directory"
            )
        identity_dir = Path(config.experiment_output_dir) / "data"
        identity_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            identity_dir / "source_identity_manifest.json",
            prepared.source_identity.manifest,
        )

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

    _report_execution_stage(config, "cross_validation")
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
        stable_row_ids=prepared.dev_stable_row_ids,
        stability_candidate_pool_path=config.stability_candidate_pool_path,
        stage_callback=(
            lambda stage, fold_id=None: _report_execution_stage(config, stage, fold_id)
        ),
    )
    _report_execution_stage(config, "cross_validation_completed")

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
    llm_response_dir = exp_dir / "llm_responses" / "final_dev"
    feature_rankings_dir = exp_dir / "feature_rankings"
    selected_feature_sets_dir = exp_dir / "selected_feature_sets"
    final_selector = selector_cls(**selector_kwargs) if selector_cls is not None else None
    if final_selector is not None and hasattr(final_selector, "set_artifact_dir"):
        final_selector.set_artifact_dir(llm_response_dir)
    if final_selector is not None and hasattr(final_selector, "set_ranking_context"):
        final_selector.set_ranking_context(
            scope="final_dev",
            fold_id=None,
            ranking_artifact_dir=features_dir,
            selector_name=config.selector_name,
        )

    saved_final_features: list[str] | None = None
    _report_execution_stage(config, "final_selection")
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
    final_features_frame = pd.DataFrame(
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
    )
    write_csv_atomic(features_dir / "final_selected_features.csv", final_features_frame)
    selected_feature_sets_dir.mkdir(parents=True, exist_ok=True)
    copy_atomic(
        features_dir / "final_selected_features.csv",
        selected_feature_sets_dir / "final_selected_features.csv",
        overwrite=False,
    )

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
            write_csv_atomic(trace_path, trace_df)
    ranking_summary_path = features_dir / "llm_rankings_summary.csv"
    if ranking_summary_path.exists():
        feature_rankings_dir.mkdir(parents=True, exist_ok=True)
        copy_atomic(
            ranking_summary_path,
            feature_rankings_dir / "llm_rankings_summary.csv",
            overwrite=False,
        )

    _report_execution_stage(config, "final_model_fit")
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
    write_json_atomic(
        models_dir / "final_model_metadata.json",
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
    )
    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "selected_features": [str(feature) for feature in final_features],
            "metadata": {
                "model": config.model_name,
                "selector": config.selector_name,
                "experiment_type": config.experiment_type,
                "feature_budget": int(config.feature_budget),
                "config_hash": config.config_hash,
                "data_fingerprint": data_fingerprint,
                "preprocessing_hash": preprocessing_hash,
            },
        },
        models_dir / "final_model_bundle.joblib",
    )

    _report_execution_stage(config, "final_prediction")
    final_prediction_start = time.time()
    train_proba = predict_proba(final_model, X_train_final)
    oot_proba = predict_proba(final_model, X_oot_final)
    final_prediction_time_sec = time.time() - final_prediction_start

    _report_execution_stage(config, "evaluation")
    final_evaluation_start = time.time()
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
    write_csv_atomic(results_dir / "selected_feature_psi.csv", selected_psi_df)
    write_csv_atomic(
        results_dir / "model_score_psi.csv",
        pd.DataFrame([{"model_score_psi": model_score_psi}]),
    )
    write_csv_atomic(
        results_dir / "credit_risk_utility.csv",
        pd.DataFrame([utility_metrics]),
    )
    data_manifest_hash = hashlib.sha256(
        canonical_config_json(config.data_fingerprint or data_fingerprint).encode("utf-8")
    ).hexdigest()
    dev_ids = prepared.dev_stable_row_ids
    oot_ids = prepared.oot_stable_row_ids
    if dev_ids is None or oot_ids is None:
        raise RuntimeError("stable row IDs were not preserved for prediction export")
    identity_manifest = prepared.source_identity.manifest if prepared.source_identity else {}
    metadata_base = {
        "dataset": config.dataset_name,
        "run_id": exp_dir.name,
        "method": config.method or config.experiment_name,
        "model": config.model_name,
        "source_training_dataset": config.source_training_dataset or config.dataset_name,
        "external_dataset": config.external_dataset or config.dataset_name,
        "data_manifest_hash": data_manifest_hash,
        "configuration_hash": config.config_hash or "",
        "pairing_policy_version": config.pairing_policy_version,
        "source_identity_manifest_hash": identity_manifest.get("source_identity_manifest_hash", ""),
        "stable_row_id_column": identity_manifest.get("stable_row_id_column", ""),
        "source_stable_id_values_hash": identity_manifest.get("source_stable_id_values_hash", ""),
    }
    dev_base = pd.DataFrame(
        {
            "stable_row_id": dev_ids.values,
            "target": prepared.y_train.values,
            "prediction_probability": train_proba,
            "predicted_class": (train_proba >= oot_threshold).astype(int),
        }
    )
    oot_base = pd.DataFrame(
        {
            "stable_row_id": oot_ids.values,
            "target": prepared.y_oot.values,
            "prediction_probability": oot_proba,
            "predicted_class": (oot_proba >= oot_threshold).astype(int),
        }
    )
    dev_predictions, dev_prediction_manifest = export_prediction_artifact(
        dev_base,
        metadata=prediction_metadata_from_sources(
            {**metadata_base, "split": "dev"}
        ),
        path=results_dir / "dev_predictions.csv",
        threshold=float(oot_threshold),
        expected_ids=set(dev_ids.astype(str)),
        expected_targets=dict(zip(dev_ids.astype(str), prepared.y_train)),
    )
    oot_predictions, oot_prediction_manifest = export_prediction_artifact(
        oot_base,
        metadata=prediction_metadata_from_sources(
            {**metadata_base, "split": "oot"}
        ),
        path=results_dir / "oot_predictions.csv",
        threshold=float(oot_threshold),
        expected_ids=set(oot_ids.astype(str)),
        expected_targets=dict(zip(oot_ids.astype(str), prepared.y_oot)),
    )
    oof_predictions = results_df.attrs.get("oof_predictions")
    prediction_manifests = [dev_prediction_manifest, oot_prediction_manifest]
    if config.experiment_type == "corrected_reverse_transfer":
        if not isinstance(oof_predictions, pd.DataFrame) or oof_predictions.empty:
            raise RuntimeError("reverse-transfer CV must persist OOF predictions")
        if oof_predictions["stable_row_id"].duplicated().any():
            raise RuntimeError("OOF stable row IDs are not unique")
        eligible_ids = results_df.attrs.get("oof_eligible_stable_row_ids")
        if set(oof_predictions["stable_row_id"].astype(str)) != set(eligible_ids or ()):
            raise RuntimeError("OOF predictions do not cover every eligible DEV row exactly once")
        oof_predictions["predicted_class"] = (
            oof_predictions["prediction_probability"].astype(float)
            >= float(oot_threshold)
        ).astype(int)
        dev_metric_path = results_dir / "dev_oof_predictions.csv"
        fold_manifest = results_df.attrs.get("fold_identity_manifest")
        all_authenticated_dev_targets = dict(
            zip(dev_ids.astype(str), prepared.y_train.to_numpy())
        )
        authenticated_dev_targets = {
            row_id: all_authenticated_dev_targets[row_id]
            for row_id in set(map(str, eligible_ids))
        }
        oof_predictions, oof_prediction_manifest = export_prediction_artifact(
            oof_predictions,
            metadata=prediction_metadata_from_sources(
                {**metadata_base, "split": "DEV_OOF"}
            ),
            path=dev_metric_path,
            threshold=float(oot_threshold),
            expected_ids=set(map(str, eligible_ids)),
            expected_targets=authenticated_dev_targets,
            fold_manifest=fold_manifest,
            forbidden_ids=set(oot_ids.astype(str)),
        )
        prediction_manifests.append(oof_prediction_manifest)
        manifest_dir = exp_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            manifest_dir / "fold_manifest.json",
                {
                    "folds": fold_manifest,
                    "fold_manifest_hash": oof_prediction_manifest["fold_manifest_hash"],
                },
        )
        write_json_atomic(
            manifest_dir / "prediction_manifest.json",
            {"predictions": prediction_manifests},
        )
    else:
        dev_metric_path = results_dir / "dev_predictions.csv"
    expected_dev_targets = (
        authenticated_dev_targets
        if config.experiment_type == "corrected_reverse_transfer"
        else None
    )
    if config.experiment_type == "corrected_reverse_transfer":
        prediction_metrics, _ = generate_metric_provenance_artifacts(
            dev_prediction_path=dev_metric_path,
            oot_prediction_path=results_dir / "oot_predictions.csv",
            prediction_manifests=prediction_manifests,
            metrics_path=results_dir / "prediction_metrics.csv",
            psi_details_path=results_dir / "psi_details.csv",
            metric_manifest_path=exp_dir / "manifests" / "metric_manifest.json",
            threshold=float(oot_threshold),
            run_id=exp_dir.name,
            model=config.model_name,
            configuration_hash=config.config_hash or "",
            data_manifest_hash=data_manifest_hash,
            expected_dev_targets=expected_dev_targets,
        )
    else:
        prediction_metrics = prediction_metrics_from_saved_files(
            dev_metric_path,
            results_dir / "oot_predictions.csv",
            threshold=float(oot_threshold),
        )
    dev_metric_row = prediction_metrics.loc[
        prediction_metrics["split"].eq(
            "DEV_OOF" if config.experiment_type == "corrected_reverse_transfer" else "dev"
        )
    ].iloc[0]
    oot_metric_row = prediction_metrics.loc[prediction_metrics["split"].eq("oot")].iloc[0]
    if config.experiment_type != "corrected_reverse_transfer":
        prediction_metrics["auc_drop"] = float(dev_metric_row["auc"]) - float(
            oot_metric_row["auc"]
        )
        prediction_metrics["auc_drop_convention"] = (
            "DEV_OOF_AUC_MINUS_OOT_AUC"
        )
        prediction_metrics["psi_binning_method"] = "DEV_OOF_quantile"
        prediction_metrics["psi_bin_fit_scope"] = "saved_DEV_predictions"
        write_csv_atomic(results_dir / "prediction_metrics.csv", prediction_metrics)
    oot_row = prediction_metrics.loc[
        prediction_metrics["split"].eq("oot")
    ].iloc[0]
    for key in ("auc", "gini", "ks", "log_loss", "brier"):
        oot_metrics[key] = float(oot_row[key])
    oot_metrics["model_score_psi"] = float(oot_row["score_psi"])
    oot_metrics["prediction_file_hash"] = str(oot_row["prediction_file_hash"])
    oot_metrics["prediction_row_count"] = int(oot_row["row_count"])
    oot_metrics["metric_scope"] = str(oot_row["metric_scope"])
    write_csv_atomic(results_dir / "oot_test_results.csv", pd.DataFrame([oot_metrics]))

    summary_row = build_experiment_summary_row(
        exp_dir=exp_dir,
        method_name=config.experiment_name,
        model_name=config.model_name,
        selector_name=config.selector_name,
    )
    if config.experiment_type == "corrected_reverse_transfer":
        dev_metric = prediction_metrics.loc[
            prediction_metrics["split"].isin(["dev", "DEV_OOF"])
        ].iloc[0]
        for key in list(summary_row):
            if key.startswith("cv_") and key.endswith(("_mean", "_std")):
                summary_row.pop(key, None)
        summary_row.update(
            {
                "dev_auc": float(dev_metric["auc"]),
                "dev_ks": float(dev_metric["ks"]),
                "dev_metric_scope": str(dev_metric["metric_scope"]),
                "dev_prediction_file_hash": str(
                    dev_metric["prediction_file_hash"]
                ),
                "dev_prediction_row_count": int(dev_metric["row_count"]),
                "oot_prediction_file_hash": str(
                    oot_row["prediction_file_hash"]
                ),
                "oot_prediction_row_count": int(oot_row["row_count"]),
            }
        )
    runtime_payload = {
        "run_id": exp_dir.name,
        "cv_preprocessing_time_sec": float(results_df.attrs.get("cv_preprocessing_time_sec", np.nan)),
        "cv_feature_selection_time_sec": float(results_df.attrs.get("cv_feature_selection_time_sec", np.nan)),
        "cv_training_time_sec": float(results_df.attrs.get("cv_training_time_sec", np.nan)),
        "cv_prediction_time_sec": float(results_df.attrs.get("cv_prediction_time_sec", np.nan)),
        "cv_evaluation_time_sec": float(results_df.attrs.get("cv_evaluation_time_sec", np.nan)),
        "cv_runtime_seconds": float(results_df.attrs.get("cv_runtime_seconds", np.nan)),
        "final_preprocessing_time_sec": float(final_preprocessing_time_sec),
        "final_feature_selection_time_sec": float(final_feature_selection_time_sec),
        "final_training_time_sec": float(final_training_time_sec),
        "final_prediction_time_sec": float(final_prediction_time_sec),
        "final_evaluation_time_sec": float(final_evaluation_time_sec),
        "preprocessing_time_sec": float(results_df.attrs.get("cv_preprocessing_time_sec", 0.0) + final_preprocessing_time_sec),
        "feature_selection_time_sec": float(results_df.attrs.get("cv_feature_selection_time_sec", 0.0) + final_feature_selection_time_sec),
        "training_time_sec": float(results_df.attrs.get("cv_training_time_sec", 0.0) + final_training_time_sec),
        "prediction_time_sec": float(results_df.attrs.get("cv_prediction_time_sec", 0.0) + final_prediction_time_sec),
        "evaluation_time_sec": float(results_df.attrs.get("cv_evaluation_time_sec", 0.0) + final_evaluation_time_sec),
        "total_runtime_seconds": float(time.time() - run_start),
    }
    write_csv_atomic(results_dir / "runtime_summary.csv", pd.DataFrame([runtime_payload]))
    write_resource_usage(exp_dir, runtime_payload)
    summary_row["runtime_seconds"] = runtime_payload["total_runtime_seconds"]
    write_csv_atomic(results_dir / "experiment_summary.csv", pd.DataFrame([summary_row]))

    logger.info(
        "Finished experiment %s | exp_dir=%s | oot_auc=%.4f",
        config.experiment_name,
        exp_dir,
        float(summary_row.get("oot_auc", float("nan"))),
    )

    _report_execution_stage(config, "evaluation_completed")

    return ExperimentRun(config=config, exp_dir=exp_dir, summary=summary_row)


SOURCE_IDENTITY_VERSION = "homecredit_raw_source_identity_v1"
GENERIC_SOURCE_IDENTITY_VERSION = "authenticated_raw_source_identity_v1"


def _canonical_identity_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else format(numeric, ".17g")
    return str(value)


def _identity_values_hash(values: pd.Series) -> str:
    canonical = sorted(_canonical_identity_value(value) for value in values)
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _identity_target_hash(ids: pd.Series, targets: pd.Series) -> str:
    pairs = sorted(
        (
            _canonical_identity_value(row_id),
            _canonical_identity_value(target),
        )
        for row_id, target in zip(ids, targets)
    )
    return hashlib.sha256(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _source_identity_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "source_identity_manifest_hash"
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _split_identity_metadata(ids: pd.Series) -> dict[str, Any]:
    return {
        "split_stable_id_values_hash": _identity_values_hash(ids),
        "row_count": int(len(ids)),
        "unique_id_count": int(ids.astype(str).nunique()),
        "null_id_count": int(ids.isna().sum()),
    }


def validate_authenticated_split_ids(
    dev_ids: pd.Series, oot_ids: pd.Series
) -> None:
    if dev_ids.isna().any() or oot_ids.isna().any():
        raise ValueError("authenticated split stable IDs contain null values")
    if dev_ids.astype(str).duplicated().any() or oot_ids.astype(str).duplicated().any():
        raise ValueError("authenticated split stable IDs are duplicated")
    if set(dev_ids.astype(str)) & set(oot_ids.astype(str)):
        raise ValueError("authenticated DEV/OOT stable row IDs overlap")


def build_source_identity_provenance(
    raw_frame: pd.DataFrame,
    *,
    dataset: str,
    stable_row_id_column: str,
    target_column: str,
    source_artifact_path: str | Path,
) -> SourceIdentityProvenance:
    from credit_risk_fs.utils.hashing import sha256_file

    dataset_name = str(dataset).lower()
    path = Path(source_artifact_path)
    if not path.exists() or not path.is_file():
        raise ValueError("source identity artifact is missing")
    if dataset_name == "homecredit" and stable_row_id_column != "SK_ID_CURR":
        raise ValueError("authenticated Home Credit identity must be SK_ID_CURR")
    required = {stable_row_id_column, target_column}
    missing = required - set(raw_frame.columns)
    if missing:
        raise ValueError(
            f"original raw source lacks identity contract columns: {sorted(missing)}"
        )
    ids = raw_frame[stable_row_id_column]
    if ids.isna().any():
        raise ValueError("original raw source stable IDs contain null values")
    canonical_ids = ids.map(_canonical_identity_value)
    if canonical_ids.duplicated().any():
        raise ValueError("original raw source stable IDs are duplicated")
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        original_columns = list(map(str, pq.ParquetFile(path).schema_arrow.names))
    else:
        original_columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    manifest = {
        "manifest_version": (
            SOURCE_IDENTITY_VERSION
            if dataset_name == "homecredit"
            else GENERIC_SOURCE_IDENTITY_VERSION
        ),
        "dataset": dataset_name,
        "stable_row_id_column": stable_row_id_column,
        "source_artifact": str(path.resolve()).replace("\\", "/"),
        "source_artifact_hash": sha256_file(path),
        "original_columns": original_columns,
        "original_column_list_hash": hashlib.sha256(
            json.dumps(original_columns, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_stable_id_values_hash": _identity_values_hash(ids),
        "stable_id_target_alignment_hash": _identity_target_hash(
            ids, raw_frame[target_column]
        ),
        "stable_id_row_count": int(len(ids)),
        "stable_id_unique_count": int(canonical_ids.nunique()),
        "stable_id_uniqueness_status": "unique",
        "stable_id_null_count": int(ids.isna().sum()),
        "creation_stage": "raw_input",
    }
    manifest["source_identity_manifest_hash"] = _source_identity_manifest_hash(
        manifest
    )
    return SourceIdentityProvenance(
        manifest=manifest,
        authenticated_ids=frozenset(canonical_ids),
        target_by_id=dict(
            zip(
                canonical_ids,
                raw_frame[target_column].map(_canonical_identity_value),
            )
        ),
    )


def validate_source_identity_subset(
    frame: pd.DataFrame,
    *,
    source_identity: SourceIdentityProvenance | None,
    dataset: str,
    stable_row_id_column: str,
    target_column: str,
    verify_source_artifact: bool = True,
) -> pd.Series:
    from credit_risk_fs.utils.hashing import sha256_file

    if source_identity is None:
        raise ValueError("source identity manifest is missing")
    if str(dataset).lower() == "lendingclub_v2":
        manifest = source_identity.manifest
        if (
            manifest.get("schema_version")
            != "lendingclub_original_loan_id_sidecar_v1"
            or manifest.get("identity_type") != "authenticated_original_loan_id"
            or manifest.get("stable_row_id_column") != "loan_id"
            or stable_row_id_column != "loan_id"
            or manifest.get("stable_id_uniqueness_status") != "unique"
            or int(manifest.get("stable_id_null_count", -1)) != 0
            or int(manifest.get("dev_oot_overlap_count", -1)) != 0
        ):
            raise ValueError("LendingClub source identity metadata mismatch")
        missing_columns = {stable_row_id_column, target_column} - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"current frame lacks authenticated identity columns: {sorted(missing_columns)}"
            )
        ids = frame[stable_row_id_column]
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise ValueError("authenticated LendingClub loan IDs are null or duplicated")
        canonical_ids = ids.astype(str)
        if set(canonical_ids) - source_identity.authenticated_ids:
            raise ValueError("current LendingClub IDs are not authenticated raw loan IDs")
        observed_targets = frame[target_column].map(_canonical_identity_value)
        if any(
            source_identity.target_by_id.get(row_id) != target
            for row_id, target in zip(canonical_ids, observed_targets, strict=True)
        ):
            raise ValueError("current LendingClub targets are misaligned with raw loan IDs")
        return canonical_ids.astype("string")
    manifest = source_identity.manifest
    required_manifest_fields = {
        "manifest_version",
        "dataset",
        "stable_row_id_column",
        "source_artifact",
        "source_artifact_hash",
        "original_columns",
        "original_column_list_hash",
        "source_stable_id_values_hash",
        "stable_id_target_alignment_hash",
        "stable_id_row_count",
        "stable_id_unique_count",
        "stable_id_uniqueness_status",
        "stable_id_null_count",
        "creation_stage",
        "source_identity_manifest_hash",
    }
    missing_manifest = required_manifest_fields - set(manifest)
    if missing_manifest:
        raise ValueError(
            f"source identity manifest is incomplete: {sorted(missing_manifest)}"
        )
    if manifest["source_identity_manifest_hash"] != _source_identity_manifest_hash(
        manifest
    ):
        raise ValueError("source identity manifest hash mismatch")
    dataset_name = str(dataset).lower()
    expected_metadata = {
        "manifest_version": (
            SOURCE_IDENTITY_VERSION
            if dataset_name == "homecredit"
            else GENERIC_SOURCE_IDENTITY_VERSION
        ),
        "dataset": dataset_name,
        "stable_row_id_column": (
            "SK_ID_CURR" if dataset_name == "homecredit" else stable_row_id_column
        ),
        "stable_id_uniqueness_status": "unique",
        "stable_id_null_count": 0,
        "creation_stage": "raw_input",
    }
    mismatches = [
        key for key, value in expected_metadata.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"source identity metadata mismatch: {mismatches}")
    original_columns = manifest["original_columns"]
    if stable_row_id_column not in original_columns:
        raise ValueError("stable ID was absent from the original source schema")
    observed_column_hash = hashlib.sha256(
        json.dumps(original_columns, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_column_hash != manifest["original_column_list_hash"]:
        raise ValueError("original source column-list hash mismatch")
    if _identity_values_hash(
        pd.Series(sorted(source_identity.authenticated_ids), dtype="string")
    ) != manifest["source_stable_id_values_hash"]:
        raise ValueError("original stable-ID values hash mismatch")
    if len(source_identity.authenticated_ids) != int(
        manifest["stable_id_unique_count"]
    ) or len(source_identity.authenticated_ids) != int(
        manifest["stable_id_row_count"]
    ):
        raise ValueError("authenticated source-ID count mismatch")
    target_ids = pd.Series(list(source_identity.target_by_id), dtype="string")
    target_values = pd.Series(
        [source_identity.target_by_id[row_id] for row_id in target_ids],
        dtype="string",
    )
    if _identity_target_hash(target_ids, target_values) != manifest[
        "stable_id_target_alignment_hash"
    ]:
        raise ValueError("authenticated ID-to-target alignment hash mismatch")
    if verify_source_artifact:
        source_path = Path(manifest["source_artifact"])
        if not source_path.exists() or sha256_file(source_path) != manifest[
            "source_artifact_hash"
        ]:
            raise ValueError("source identity artifact hash mismatch")
        current_columns = pd.read_csv(source_path, nrows=0).columns.astype(str).tolist()
        if current_columns != original_columns:
            raise ValueError("source identity artifact column list changed")

    missing_columns = {
        stable_row_id_column,
        target_column,
    } - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"current frame lacks authenticated identity columns: {sorted(missing_columns)}"
        )
    ids = frame[stable_row_id_column]
    if ids.isna().any():
        raise ValueError("authenticated stable row IDs contain null values")
    canonical_ids = ids.map(_canonical_identity_value)
    if canonical_ids.duplicated().any():
        raise ValueError("authenticated stable row IDs are duplicated")
    unexpected = set(canonical_ids) - source_identity.authenticated_ids
    if unexpected:
        raise ValueError("current stable IDs are not authenticated source IDs")
    observed_targets = frame[target_column].map(_canonical_identity_value)
    misaligned = [
        row_id
        for row_id, target in zip(canonical_ids, observed_targets)
        if source_identity.target_by_id.get(row_id) != target
    ]
    if misaligned:
        raise ValueError("current targets are misaligned with authenticated source IDs")
    return canonical_ids.astype("string")


def _stable_row_ids(
    frame: pd.DataFrame,
    *,
    dataset: str,
    stable_row_id_column: str,
    source_identity: SourceIdentityProvenance | None = None,
    target_column: str = "TARGET",
) -> pd.Series:
    prohibited_generated_names = {
        "index",
        "level_0",
        "row_id",
        "row_number",
        "__index_level_0__",
    }
    if str(stable_row_id_column).strip().lower() in prohibited_generated_names:
        raise ValueError(
            "stable row IDs must come from a persistent source column, not a generated index"
        )
    return validate_source_identity_subset(
        frame,
        source_identity=source_identity,
        dataset=dataset,
        stable_row_id_column=stable_row_id_column,
        target_column=target_column,
        verify_source_artifact=False,
    )


def export_prediction_artifact(
    frame: pd.DataFrame,
    *,
    metadata: PredictionMetadata,
    path: str | Path,
    threshold: float,
    expected_ids: set[str],
    expected_targets: dict[str, Any],
    fold_manifest: list[dict[str, Any]] | None = None,
    forbidden_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from credit_risk_fs.utils.hashing import sha256_file

    required = {
        "stable_row_id",
        "target",
        "prediction_probability",
        "predicted_class",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"prediction export columns missing: {sorted(missing)}")
    output = frame.copy()
    if output["stable_row_id"].isna().any():
        raise ValueError("prediction stable IDs are null or duplicated")
    output["stable_row_id"] = output["stable_row_id"].astype(str)
    if output["stable_row_id"].duplicated().any():
        raise ValueError("prediction stable IDs are null or duplicated")
    probabilities = pd.to_numeric(output["prediction_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError("prediction probabilities must be finite and within [0, 1]")
    expected_classes = (probabilities >= float(threshold)).astype(int)
    if not np.array_equal(expected_classes, output["predicted_class"].astype(int)):
        raise ValueError("predicted classes do not match the declared threshold")
    observed_ids = set(output["stable_row_id"])
    if forbidden_ids and observed_ids & set(forbidden_ids):
        raise ValueError("forbidden OOT IDs occur in DEV OOF predictions")
    if observed_ids != set(expected_ids):
        raise ValueError("prediction stable-ID coverage is incomplete")
    observed_targets = dict(zip(output["stable_row_id"], output["target"]))
    if any(observed_targets[row_id] != expected_targets[row_id] for row_id in expected_ids):
        raise ValueError("prediction targets are misaligned with stable IDs")

    fold_manifest_hash = ""
    if metadata.split == "DEV_OOF":
        if "fold_id" not in output or output["fold_id"].isna().any():
            raise ValueError("DEV_OOF predictions require fold_id")
        if not fold_manifest:
            raise ValueError("DEV_OOF predictions require a fold manifest")
        validation_union: set[str] = set()
        for fold in fold_manifest:
            training_ids = set(map(str, fold["training_ids"]))
            validation_ids = set(map(str, fold["validation_ids"]))
            if training_ids & validation_ids:
                raise ValueError("fold training and validation IDs overlap")
            if validation_union & validation_ids:
                raise ValueError("validation IDs overlap across folds")
            if len(validation_ids) != int(fold["validation_row_count"]):
                raise ValueError("fold validation row count mismatch")
            training_hash = hashlib.sha256(
                json.dumps(sorted(training_ids), separators=(",", ":")).encode()
            ).hexdigest()
            if training_hash != fold["training_id_hash"]:
                raise ValueError("fold training-ID hash mismatch")
            id_hash = hashlib.sha256(
                json.dumps(sorted(validation_ids), separators=(",", ":")).encode()
            ).hexdigest()
            if id_hash != fold["validation_id_hash"]:
                raise ValueError("fold validation-ID hash mismatch")
            observed_fold = set(
                output.loc[
                    output["fold_id"].astype(int).eq(int(fold["fold_id"])),
                    "stable_row_id",
                ]
            )
            if observed_fold != validation_ids:
                raise ValueError("OOF rows do not equal fold validation IDs")
            validation_union.update(validation_ids)
        if validation_union != set(expected_ids):
            raise ValueError("DEV fold coverage is incomplete")
        fold_manifest_hash = hashlib.sha256(
            json.dumps(fold_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif "fold_id" not in output:
        output["fold_id"] = pd.NA

    for key, value in asdict(metadata).items():
        if key in output:
            raise ValueError(f"prediction metadata field already exists: {key}")
        output[key] = value
    output = output.sort_values("stable_row_id", kind="mergesort").reset_index(drop=True)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_integrity = write_csv_atomic(
        target_path,
        output,
        required_columns=required,
        ordered_row_identity_column="stable_row_id",
    )
    saved = pd.read_csv(target_path)
    if len(saved) != len(output) or set(saved["stable_row_id"].astype(str)) != set(expected_ids):
        raise ValueError("persisted prediction artifact differs from validated rows")
    manifest = {
        "prediction_path": str(target_path).replace("\\", "/"),
        "prediction_hash": sha256_file(target_path),
        "row_count": len(saved),
        "unique_stable_id_count": saved["stable_row_id"].astype(str).nunique(),
        "null_stable_id_count": int(saved["stable_row_id"].isna().sum()),
        "size_bytes": artifact_integrity.size_bytes,
        "ordered_row_identity_sha256": artifact_integrity.ordered_row_identity_sha256,
        **asdict(metadata),
        "fold_manifest_hash": fold_manifest_hash,
    }
    return saved, manifest


def _load_saved_prediction_frames(
    dev_path: str | Path,
    oot_path: str | Path,
    *,
    expected_dev_targets: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dev = pd.read_csv(dev_path)
    oot = pd.read_csv(oot_path)
    required = {"stable_row_id", "split", "target", "prediction_probability"}
    dev_split = str(dev["split"].iloc[0]) if len(dev) else ""
    for split, frame in ((dev_split, dev), ("oot", oot)):
        missing = required - set(frame.columns)
        if (
            split not in {"dev", "DEV_OOF", "oot"}
            or missing
            or set(frame["split"].astype(str)) != {split}
        ):
            raise ValueError(
                f"{split} saved prediction scope is invalid: {sorted(missing)}"
            )
        if frame["stable_row_id"].isna().any() or frame[
            "stable_row_id"
        ].astype(str).duplicated().any():
            raise ValueError(f"{split} saved prediction IDs are not unique")
        probabilities = pd.to_numeric(
            frame["prediction_probability"], errors="coerce"
        )
        if (
            probabilities.isna().any()
            or not np.isfinite(probabilities).all()
            or not probabilities.between(0.0, 1.0).all()
        ):
            raise ValueError(
                f"{split} saved prediction probabilities are invalid"
            )
        if split == "DEV_OOF":
            if "fold_id" not in frame:
                raise ValueError("DEV_OOF saved predictions are missing fold IDs")
            folds = pd.to_numeric(frame["fold_id"], errors="coerce")
            if folds.isna().any() or not folds.gt(0).all():
                raise ValueError(
                    "DEV_OOF saved prediction fold IDs are invalid"
                )
    if set(dev["stable_row_id"].astype(str)) & set(
        oot["stable_row_id"].astype(str)
    ):
        raise ValueError("saved DEV/OOT prediction IDs overlap")
    if expected_dev_targets is not None:
        observed_targets = dict(
            zip(dev["stable_row_id"].astype(str), dev["target"])
        )
        if set(observed_targets) != set(expected_dev_targets):
            raise ValueError("saved DEV OOF row coverage is incomplete")
        mismatched_targets = [
            row_id
            for row_id, target in expected_dev_targets.items()
            if observed_targets[row_id] != target
        ]
        if mismatched_targets:
            raise ValueError("saved DEV OOF target alignment is invalid")
    return dev, oot, dev_split


def compute_score_psi(
    reference: pd.Series | np.ndarray,
    comparison: pd.Series | np.ndarray,
    *,
    requested_bin_count: int = SCORE_PSI_REQUESTED_BIN_COUNT,
    smoothing_epsilon: float = SCORE_PSI_SMOOTHING_EPSILON,
    bin_edges: list[float] | np.ndarray | None = None,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Compute reproducible score PSI using quantile bins fitted on DEV OOF."""

    reference_values = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).to_numpy(dtype=float)
    comparison_values = pd.to_numeric(
        pd.Series(comparison), errors="coerce"
    ).to_numpy(dtype=float)
    for scope, values in (
        ("reference", reference_values),
        ("comparison", comparison_values),
    ):
        if (
            len(values) == 0
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                f"PSI {scope} probabilities must be finite and within [0, 1]"
            )
    if int(requested_bin_count) <= 0:
        raise ValueError("PSI requested bin count must be positive")
    if (
        not np.isfinite(float(smoothing_epsilon))
        or float(smoothing_epsilon) <= 0.0
    ):
        raise ValueError("PSI smoothing epsilon must be finite and positive")

    if bin_edges is None:
        candidate_edges = np.percentile(
            reference_values,
            np.linspace(0.0, 100.0, int(requested_bin_count) + 1),
        )
        edges = np.unique(candidate_edges.astype(float))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0], dtype=float)
        else:
            edges[0] = 0.0
            edges[-1] = 1.0
            edges = np.unique(edges)
    else:
        edges = np.asarray(bin_edges, dtype=float)
    if (
        len(edges) < 2
        or not np.isfinite(edges).all()
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError(
            "PSI bin edges must be strictly increasing finite values from 0 to 1"
        )

    effective_bin_count = len(edges) - 1

    def _bin_counts(values: np.ndarray) -> np.ndarray:
        # Internal edges are right-inclusive, matching pd.cut(..., right=True).
        assignments = np.searchsorted(edges[1:-1], values, side="left")
        counts = np.bincount(assignments, minlength=effective_bin_count)
        if int(counts.sum()) != len(values):
            raise ValueError("PSI binning did not assign every probability")
        return counts.astype(int)

    reference_count = _bin_counts(reference_values)
    comparison_count = _bin_counts(comparison_values)
    reference_share = reference_count / float(len(reference_values))
    comparison_share = comparison_count / float(len(comparison_values))
    smoothed_reference_share = reference_share + float(smoothing_epsilon)
    smoothed_comparison_share = comparison_share + float(smoothing_epsilon)
    contributions = (
        smoothed_comparison_share - smoothed_reference_share
    ) * np.log(smoothed_comparison_share / smoothed_reference_share)
    psi = float(np.sum(contributions, dtype=float))
    details = pd.DataFrame(
        {
            "bin_id": np.arange(1, effective_bin_count + 1, dtype=int),
            "lower_bound": edges[:-1],
            "upper_bound": edges[1:],
            "reference_count": reference_count,
            "comparison_count": comparison_count,
            "reference_share": reference_share,
            "comparison_share": comparison_share,
            "smoothed_reference_share": smoothed_reference_share,
            "smoothed_comparison_share": smoothed_comparison_share,
            "psi_contribution": contributions,
        }
    )
    definition = {
        "reference_scope": "DEV_OOF",
        "comparison_scope": "oot",
        "binning_method": "DEV_OOF_quantile",
        "requested_bin_count": int(requested_bin_count),
        "effective_bin_count": int(effective_bin_count),
        "bin_edges": [float(value) for value in edges],
        "duplicate_edge_policy": "sort_unique_candidate_quantile_edges",
        "out_of_range_policy": "reject_outside_closed_probability_interval_0_1",
        "missing_value_policy": "reject",
        "smoothing_epsilon": float(smoothing_epsilon),
        "log_base": "natural",
        "psi_formula": (
            "SUM((smoothed_comparison_share_i - "
            "smoothed_reference_share_i) * "
            "LN(smoothed_comparison_share_i / "
            "smoothed_reference_share_i))"
        ),
        "psi_implementation_version": SCORE_PSI_IMPLEMENTATION_VERSION,
    }
    return psi, details, definition


def _prediction_metric_bundle_from_saved_files(
    dev_path: str | Path,
    oot_path: str | Path,
    *,
    threshold: float,
    expected_dev_targets: dict[str, Any] | None = None,
    psi_bin_edges: list[float] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from credit_risk_fs.utils.hashing import sha256_file

    dev, oot, dev_split = _load_saved_prediction_frames(
        dev_path,
        oot_path,
        expected_dev_targets=expected_dev_targets,
    )
    psi, psi_details, psi_definition = compute_score_psi(
        dev["prediction_probability"],
        oot["prediction_probability"],
        bin_edges=psi_bin_edges,
    )
    rows = []
    for split, frame, path, scope in (
        (
            dev_split,
            dev,
            Path(dev_path),
            "dev_oof_cross_validated"
            if dev_split == "DEV_OOF"
            else "dev_in_sample_final_model",
        ),
        ("oot", oot, Path(oot_path), "oot_holdout_final_model"),
    ):
        metrics = evaluate_model(
            frame["target"].to_numpy(),
            frame["prediction_probability"].to_numpy(),
            threshold=threshold,
        )
        rows.append(
            {
                "split": split,
                "metric_scope": scope,
                "auc": metrics["auc"],
                "gini": metrics["gini"],
                "ks": metrics["ks"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "score_psi": psi,
                "psi_reference_split": dev_split,
                "psi_comparison_split": "oot",
                "row_count": len(frame),
                "prediction_file_hash": sha256_file(path),
                **{
                    column: (
                        str(frame[column].iloc[0])
                        if column in frame
                        and frame[column].nunique(dropna=False) == 1
                        else ""
                    )
                    for column in (
                        "run_id",
                        "method",
                        "model",
                        "source_training_dataset",
                        "external_dataset",
                        "configuration_hash",
                        "data_manifest_hash",
                        "pairing_policy_version",
                        "source_identity_manifest_hash",
                        "fit_scope",
                    )
                },
            }
        )
    metric_frame = pd.DataFrame(rows)
    indexed = metric_frame.set_index("split")
    derived = {
        "dev_oof_auc": float(indexed.loc[dev_split, "auc"]),
        "oot_auc": float(indexed.loc["oot", "auc"]),
        "auc_drop": float(
            indexed.loc[dev_split, "auc"] - indexed.loc["oot", "auc"]
        ),
        "score_psi": float(psi),
        "psi_definition": psi_definition,
    }
    return metric_frame, psi_details, derived


def prediction_metrics_from_saved_files(
    dev_path: str | Path,
    oot_path: str | Path,
    *,
    threshold: float,
    expected_dev_targets: dict[str, Any] | None = None,
) -> pd.DataFrame:
    metrics, _, _ = _prediction_metric_bundle_from_saved_files(
        dev_path,
        oot_path,
        threshold=threshold,
        expected_dev_targets=expected_dev_targets,
    )
    return metrics


def generate_metric_provenance_artifacts(
    *,
    dev_prediction_path: str | Path,
    oot_prediction_path: str | Path,
    prediction_manifests: list[dict[str, Any]],
    metrics_path: str | Path,
    psi_details_path: str | Path,
    metric_manifest_path: str | Path,
    threshold: float,
    run_id: str,
    model: str,
    configuration_hash: str,
    data_manifest_hash: str,
    expected_dev_targets: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate full-precision metric claims from saved predictions and validate them."""

    from credit_risk_fs.utils.hashing import sha256_file

    prediction_by_split = {
        str(item["split"]): item for item in prediction_manifests
    }
    if len(prediction_by_split) != len(prediction_manifests):
        raise ValueError("prediction manifests contain duplicate split claims")
    if {"DEV_OOF", "oot"} - set(prediction_by_split):
        raise ValueError("metric generation requires DEV_OOF and OOT predictions")
    dev_provenance = prediction_by_split["DEV_OOF"]
    oot_provenance = prediction_by_split["oot"]
    metrics, psi_details, derived = _prediction_metric_bundle_from_saved_files(
        dev_prediction_path,
        oot_prediction_path,
        threshold=threshold,
        expected_dev_targets=expected_dev_targets,
    )
    metrics["auc_drop"] = derived["auc_drop"]
    metrics["auc_drop_convention"] = "DEV_OOF_AUC_MINUS_OOT_AUC"
    metrics["psi_binning_method"] = "DEV_OOF_quantile"
    metrics["psi_bin_fit_scope"] = "saved_DEV_OOF_predictions"

    metrics_target = Path(metrics_path)
    psi_target = Path(psi_details_path)
    manifest_target = Path(metric_manifest_path)
    for target in (metrics_target, psi_target, manifest_target):
        target.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(metrics_target, metrics)
    write_csv_atomic(psi_target, psi_details)

    metric_entries = []
    indexed = metrics.set_index("split")
    for split, provenance in (
        ("DEV_OOF", dev_provenance),
        ("oot", oot_provenance),
    ):
        for metric_name in ("auc", "ks", "row_count"):
            metric_entries.append(
                {
                    "metric_name": metric_name,
                    "metric_value": float(indexed.loc[split, metric_name]),
                    "metric_scope": split,
                    "prediction_path": provenance["prediction_path"],
                    "prediction_hash": provenance["prediction_hash"],
                    "prediction_row_count": int(provenance["row_count"]),
                    "configuration_hash": configuration_hash,
                    "data_manifest_hash": data_manifest_hash,
                    "run_id": run_id,
                    "model": model,
                }
            )

    common_derived = {
        "run_id": run_id,
        "model": model,
        "configuration_hash": configuration_hash,
        "data_manifest_hash": data_manifest_hash,
        "source_identity_manifest_hash": dev_provenance[
            "source_identity_manifest_hash"
        ],
        "metric_implementation_version": DERIVED_METRIC_IMPLEMENTATION_VERSION,
        "validation_tolerance": DERIVED_METRIC_VALIDATION_TOLERANCE,
    }
    auc_drop_record = {
        "metric_name": "auc_drop",
        "metric_scope": "DEV_OOF_vs_oot",
        "metric_value": derived["auc_drop"],
        "formula": "DEV_OOF_AUC - OOT_AUC",
        "sign_convention": "DEV_OOF_AUC_MINUS_OOT_AUC",
        "dev_prediction_path": dev_provenance["prediction_path"],
        "dev_prediction_hash": dev_provenance["prediction_hash"],
        "dev_prediction_row_count": int(dev_provenance["row_count"]),
        "oot_prediction_path": oot_provenance["prediction_path"],
        "oot_prediction_hash": oot_provenance["prediction_hash"],
        "oot_prediction_row_count": int(oot_provenance["row_count"]),
        "dev_oof_auc": derived["dev_oof_auc"],
        "oot_auc": derived["oot_auc"],
        **common_derived,
    }
    psi_record = {
        "metric_name": "score_psi",
        "metric_scope": "DEV_OOF_reference_vs_oot_comparison",
        "metric_value": derived["score_psi"],
        **derived["psi_definition"],
        "reference_prediction_path": dev_provenance["prediction_path"],
        "reference_prediction_hash": dev_provenance["prediction_hash"],
        "reference_row_count": int(dev_provenance["row_count"]),
        "comparison_prediction_path": oot_provenance["prediction_path"],
        "comparison_prediction_hash": oot_provenance["prediction_hash"],
        "comparison_row_count": int(oot_provenance["row_count"]),
        "psi_details_path": str(psi_target).replace("\\", "/"),
        "psi_details_hash": sha256_file(psi_target),
        "psi_details_row_count": len(psi_details),
        **common_derived,
    }
    manifest = {
        "run_id": run_id,
        "model": model,
        "configuration_hash": configuration_hash,
        "data_manifest_hash": data_manifest_hash,
        "source_identity_manifest_hash": common_derived[
            "source_identity_manifest_hash"
        ],
        "metrics_path": str(metrics_target).replace("\\", "/"),
        "metrics_hash": sha256_file(metrics_target),
        "metrics": metric_entries,
        "threshold": float(threshold),
        "prediction_manifests": prediction_manifests,
        "auc_drop": derived["auc_drop"],
        "auc_drop_convention": "DEV_OOF_AUC_MINUS_OOT_AUC",
        "score_psi": derived["score_psi"],
        "psi_reference_split": "DEV_OOF",
        "psi_comparison_split": "oot",
        "reference_prediction_hash": dev_provenance["prediction_hash"],
        "comparison_prediction_hash": oot_provenance["prediction_hash"],
        "metric_implementation_version": DERIVED_METRIC_IMPLEMENTATION_VERSION,
        "validation_tolerance": DERIVED_METRIC_VALIDATION_TOLERANCE,
        "derived_metrics": {
            "auc_drop": auc_drop_record,
            "score_psi": psi_record,
        },
    }
    write_json_atomic(manifest_target, manifest)
    validate_metric_provenance(manifest)
    return metrics, manifest


def _require_manifest_fields(
    value: dict[str, Any], required: set[str], *, scope: str
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{scope} is missing required fields: {sorted(missing)}")


def _strict_metric_equal(
    observed: Any, expected: Any, *, field: str, tolerance: float
) -> None:
    try:
        observed_value = float(observed)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if (
        not np.isfinite(observed_value)
        or not np.isfinite(expected_value)
        or not np.isclose(
            observed_value,
            expected_value,
            rtol=0.0,
            atol=float(tolerance),
        )
    ):
        raise ValueError(f"{field} differs from saved-prediction recomputation")


def validate_metric_provenance(manifest: dict[str, Any]) -> None:
    """Fail closed unless every derived metric reproduces from saved predictions."""

    from credit_risk_fs.utils.hashing import sha256_file

    _require_manifest_fields(
        manifest,
        {
            "run_id",
            "model",
            "configuration_hash",
            "data_manifest_hash",
            "source_identity_manifest_hash",
            "metrics_path",
            "metrics_hash",
            "metrics",
            "threshold",
            "prediction_manifests",
            "auc_drop",
            "auc_drop_convention",
            "score_psi",
            "psi_reference_split",
            "psi_comparison_split",
            "reference_prediction_hash",
            "comparison_prediction_hash",
            "metric_implementation_version",
            "validation_tolerance",
            "derived_metrics",
        },
        scope="metric manifest",
    )
    if (
        manifest["metric_implementation_version"]
        != DERIVED_METRIC_IMPLEMENTATION_VERSION
    ):
        raise ValueError("metric implementation version is invalid")
    _strict_metric_equal(
        manifest["validation_tolerance"],
        DERIVED_METRIC_VALIDATION_TOLERANCE,
        field="metric validation tolerance",
        tolerance=0.0,
    )
    tolerance = DERIVED_METRIC_VALIDATION_TOLERANCE

    prediction_rows = list(manifest["prediction_manifests"])
    predictions = {str(row["split"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("metric provenance prediction scopes are duplicated")
    if {"DEV_OOF", "oot"} - set(predictions):
        raise ValueError("metric provenance prediction scopes are incomplete")
    dev, oot = predictions["DEV_OOF"], predictions["oot"]
    provenance_fields = (
        "split",
        "model",
        "run_id",
        "configuration_hash",
        "data_manifest_hash",
        "source_identity_manifest_hash",
    )
    for item in (dev, oot):
        _require_manifest_fields(
            item,
            {
                "prediction_path",
                "prediction_hash",
                "row_count",
                *provenance_fields,
            },
            scope="prediction manifest",
        )
        path = Path(item["prediction_path"])
        if not path.exists() or sha256_file(path) != item["prediction_hash"]:
            raise ValueError("metric provenance prediction hash mismatch")
        frame = pd.read_csv(path)
        if len(frame) != int(item["row_count"]):
            raise ValueError("metric provenance prediction row-count mismatch")
        for field in provenance_fields:
            if field not in frame or frame[field].nunique(dropna=False) != 1:
                raise ValueError(f"metric provenance {field} is not uniform")
            if str(frame[field].iloc[0]) != str(item[field]):
                raise ValueError(f"metric provenance {field} mismatch")
    for field in (
        "model",
        "run_id",
        "configuration_hash",
        "data_manifest_hash",
        "source_identity_manifest_hash",
    ):
        if str(dev[field]) != str(oot[field]):
            raise ValueError(f"metric prediction {field} mismatch")
        if str(manifest[field]) != str(dev[field]):
            raise ValueError(f"metric manifest {field} mismatch")
    if (
        manifest["psi_reference_split"] != "DEV_OOF"
        or manifest["psi_comparison_split"] != "oot"
    ):
        raise ValueError("metric PSI prediction scopes are invalid")
    if (
        manifest["auc_drop_convention"] != "DEV_OOF_AUC_MINUS_OOT_AUC"
    ):
        raise ValueError("metric AUC-drop sign convention is invalid")
    if manifest["reference_prediction_hash"] != dev["prediction_hash"]:
        raise ValueError("metric PSI reference prediction hash mismatch")
    if manifest["comparison_prediction_hash"] != oot["prediction_hash"]:
        raise ValueError("metric PSI comparison prediction hash mismatch")

    derived_metrics = manifest["derived_metrics"]
    if not isinstance(derived_metrics, dict) or set(derived_metrics) != {
        "auc_drop",
        "score_psi",
    }:
        raise ValueError("derived metric records are incomplete")
    auc_record = derived_metrics["auc_drop"]
    psi_record = derived_metrics["score_psi"]
    common_required = {
        "metric_name",
        "metric_scope",
        "metric_value",
        "run_id",
        "model",
        "configuration_hash",
        "data_manifest_hash",
        "source_identity_manifest_hash",
        "metric_implementation_version",
        "validation_tolerance",
    }
    _require_manifest_fields(
        auc_record,
        common_required
        | {
            "formula",
            "sign_convention",
            "dev_prediction_path",
            "dev_prediction_hash",
            "dev_prediction_row_count",
            "oot_prediction_path",
            "oot_prediction_hash",
            "oot_prediction_row_count",
            "dev_oof_auc",
            "oot_auc",
        },
        scope="AUC-drop metric record",
    )
    _require_manifest_fields(
        psi_record,
        common_required
        | {
            "reference_scope",
            "comparison_scope",
            "binning_method",
            "requested_bin_count",
            "effective_bin_count",
            "bin_edges",
            "duplicate_edge_policy",
            "out_of_range_policy",
            "missing_value_policy",
            "smoothing_epsilon",
            "log_base",
            "psi_formula",
            "reference_prediction_path",
            "reference_prediction_hash",
            "reference_row_count",
            "comparison_prediction_path",
            "comparison_prediction_hash",
            "comparison_row_count",
            "psi_implementation_version",
            "psi_details_path",
            "psi_details_hash",
            "psi_details_row_count",
        },
        scope="PSI metric record",
    )
    for record_name, record in (
        ("AUC-drop", auc_record),
        ("PSI", psi_record),
    ):
        for field in (
            "run_id",
            "model",
            "configuration_hash",
            "data_manifest_hash",
            "source_identity_manifest_hash",
            "metric_implementation_version",
        ):
            if str(record[field]) != str(manifest[field]):
                raise ValueError(f"{record_name} {field} mismatch")
        _strict_metric_equal(
            record["validation_tolerance"],
            tolerance,
            field=f"{record_name} validation tolerance",
            tolerance=0.0,
        )

    expected_auc_metadata = {
        "metric_name": "auc_drop",
        "metric_scope": "DEV_OOF_vs_oot",
        "formula": "DEV_OOF_AUC - OOT_AUC",
        "sign_convention": "DEV_OOF_AUC_MINUS_OOT_AUC",
        "dev_prediction_path": dev["prediction_path"],
        "dev_prediction_hash": dev["prediction_hash"],
        "dev_prediction_row_count": int(dev["row_count"]),
        "oot_prediction_path": oot["prediction_path"],
        "oot_prediction_hash": oot["prediction_hash"],
        "oot_prediction_row_count": int(oot["row_count"]),
    }
    for field, expected in expected_auc_metadata.items():
        if auc_record[field] != expected:
            raise ValueError(f"AUC-drop {field} mismatch")

    expected_psi_metadata = {
        "metric_name": "score_psi",
        "metric_scope": "DEV_OOF_reference_vs_oot_comparison",
        "reference_scope": "DEV_OOF",
        "comparison_scope": "oot",
        "binning_method": "DEV_OOF_quantile",
        "requested_bin_count": SCORE_PSI_REQUESTED_BIN_COUNT,
        "duplicate_edge_policy": "sort_unique_candidate_quantile_edges",
        "out_of_range_policy": (
            "reject_outside_closed_probability_interval_0_1"
        ),
        "missing_value_policy": "reject",
        "log_base": "natural",
        "psi_formula": (
            "SUM((smoothed_comparison_share_i - "
            "smoothed_reference_share_i) * "
            "LN(smoothed_comparison_share_i / "
            "smoothed_reference_share_i))"
        ),
        "reference_prediction_path": dev["prediction_path"],
        "reference_prediction_hash": dev["prediction_hash"],
        "reference_row_count": int(dev["row_count"]),
        "comparison_prediction_path": oot["prediction_path"],
        "comparison_prediction_hash": oot["prediction_hash"],
        "comparison_row_count": int(oot["row_count"]),
        "psi_implementation_version": SCORE_PSI_IMPLEMENTATION_VERSION,
    }
    for field, expected in expected_psi_metadata.items():
        if psi_record[field] != expected:
            raise ValueError(f"PSI {field} mismatch")
    _strict_metric_equal(
        psi_record["smoothing_epsilon"],
        SCORE_PSI_SMOOTHING_EPSILON,
        field="PSI smoothing epsilon",
        tolerance=0.0,
    )

    expected_metrics, expected_details, expected_derived = (
        _prediction_metric_bundle_from_saved_files(
            dev["prediction_path"],
            oot["prediction_path"],
            threshold=float(manifest["threshold"]),
        )
    )
    expected_edges = expected_derived["psi_definition"]["bin_edges"]
    claimed_edges = psi_record["bin_edges"]
    if (
        not isinstance(claimed_edges, list)
        or len(claimed_edges) != len(expected_edges)
        or not np.array_equal(
            np.asarray(claimed_edges, dtype=float),
            np.asarray(expected_edges, dtype=float),
        )
    ):
        raise ValueError("PSI bin edges were not fitted on saved DEV OOF predictions")
    if int(psi_record["effective_bin_count"]) != len(expected_edges) - 1:
        raise ValueError("PSI effective bin count mismatch")

    _strict_metric_equal(
        manifest["auc_drop"],
        expected_derived["auc_drop"],
        field="stored AUC drop",
        tolerance=tolerance,
    )
    _strict_metric_equal(
        auc_record["metric_value"],
        expected_derived["auc_drop"],
        field="AUC-drop metric value",
        tolerance=tolerance,
    )
    _strict_metric_equal(
        auc_record["dev_oof_auc"],
        expected_derived["dev_oof_auc"],
        field="pooled DEV OOF AUC",
        tolerance=tolerance,
    )
    _strict_metric_equal(
        auc_record["oot_auc"],
        expected_derived["oot_auc"],
        field="OOT AUC",
        tolerance=tolerance,
    )
    _strict_metric_equal(
        manifest["score_psi"],
        expected_derived["score_psi"],
        field="stored PSI",
        tolerance=tolerance,
    )
    if "psi_value" in manifest:
        _strict_metric_equal(
            manifest["psi_value"],
            expected_derived["score_psi"],
            field="stored PSI alias",
            tolerance=tolerance,
        )
    _strict_metric_equal(
        psi_record["metric_value"],
        expected_derived["score_psi"],
        field="PSI metric value",
        tolerance=tolerance,
    )

    psi_details_path = Path(psi_record["psi_details_path"])
    if (
        not psi_details_path.exists()
        or sha256_file(psi_details_path) != psi_record["psi_details_hash"]
    ):
        raise ValueError("PSI-detail artifact is absent or its hash mismatches")
    actual_details = pd.read_csv(psi_details_path)
    required_detail_columns = list(expected_details.columns)
    if list(actual_details.columns) != required_detail_columns:
        raise ValueError("PSI-detail artifact schema mismatch")
    if (
        len(actual_details) != len(expected_details)
        or len(actual_details) != int(psi_record["psi_details_row_count"])
    ):
        raise ValueError("PSI-detail artifact row-count mismatch")
    for field in (
        "bin_id",
        "reference_count",
        "comparison_count",
    ):
        if not np.array_equal(
            pd.to_numeric(actual_details[field], errors="coerce").to_numpy(),
            expected_details[field].to_numpy(),
        ):
            raise ValueError(f"PSI-detail {field} mismatch")
    for field in (
        "lower_bound",
        "upper_bound",
        "reference_share",
        "comparison_share",
        "smoothed_reference_share",
        "smoothed_comparison_share",
        "psi_contribution",
    ):
        observed = pd.to_numeric(
            actual_details[field], errors="coerce"
        ).to_numpy(dtype=float)
        expected = expected_details[field].to_numpy(dtype=float)
        if (
            not np.isfinite(observed).all()
            or not np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=tolerance,
            )
        ):
            raise ValueError(f"PSI-detail {field} mismatch")
    _strict_metric_equal(
        actual_details["psi_contribution"].sum(),
        psi_record["metric_value"],
        field="PSI-detail contribution sum",
        tolerance=tolerance,
    )

    metrics_path = Path(manifest["metrics_path"])
    if (
        not metrics_path.exists()
        or sha256_file(metrics_path) != manifest["metrics_hash"]
    ):
        raise ValueError("metric artifact is absent or its hash mismatches")
    stored_metrics = pd.read_csv(metrics_path)
    if (
        len(stored_metrics) != 2
        or set(stored_metrics["split"].astype(str)) != {"DEV_OOF", "oot"}
    ):
        raise ValueError("metric artifact prediction scopes are invalid")
    stored_index = stored_metrics.set_index("split")
    expected_index = expected_metrics.set_index("split")
    for split in ("DEV_OOF", "oot"):
        expected_scope = (
            "dev_oof_cross_validated"
            if split == "DEV_OOF"
            else "oot_holdout_final_model"
        )
        if str(stored_index.loc[split, "metric_scope"]) != expected_scope:
            raise ValueError("metric artifact scope mismatch")
        if (
            str(stored_index.loc[split, "prediction_file_hash"])
            != str(expected_index.loc[split, "prediction_file_hash"])
        ):
            raise ValueError("metric artifact prediction hash mismatch")
        for field in ("auc", "ks", "row_count"):
            _strict_metric_equal(
                stored_index.loc[split, field],
                expected_index.loc[split, field],
                field=f"metric artifact {split} {field}",
                tolerance=tolerance,
            )
        for field, expected in (
            ("auc_drop", expected_derived["auc_drop"]),
            ("score_psi", expected_derived["score_psi"]),
        ):
            _strict_metric_equal(
                stored_index.loc[split, field],
                expected,
                field=f"metric artifact {split} {field}",
                tolerance=tolerance,
            )

    base_entries = {}
    for entry in manifest["metrics"]:
        key = (str(entry["metric_scope"]), str(entry["metric_name"]))
        if key in base_entries:
            raise ValueError("metric manifest contains duplicate base metric entries")
        base_entries[key] = entry
    required_base_entries = {
        (split, metric_name)
        for split in ("DEV_OOF", "oot")
        for metric_name in ("auc", "ks", "row_count")
    }
    if set(base_entries) != required_base_entries:
        raise ValueError("metric manifest base metric entries are incomplete")
    expected_index = expected_metrics.set_index("split")
    for (split, metric_name), entry in base_entries.items():
        provenance = dev if split == "DEV_OOF" else oot
        for field, expected in (
            ("prediction_path", provenance["prediction_path"]),
            ("prediction_hash", provenance["prediction_hash"]),
            ("prediction_row_count", int(provenance["row_count"])),
            ("configuration_hash", manifest["configuration_hash"]),
            ("data_manifest_hash", manifest["data_manifest_hash"]),
            ("run_id", manifest["run_id"]),
            ("model", manifest["model"]),
        ):
            if entry.get(field) != expected:
                raise ValueError(f"base metric {field} mismatch")
        _strict_metric_equal(
            entry["metric_value"],
            expected_index.loc[split, metric_name],
            field=f"reported {split} {metric_name}",
            tolerance=tolerance,
        )
