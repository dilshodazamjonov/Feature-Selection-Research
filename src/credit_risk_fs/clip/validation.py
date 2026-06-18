from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from credit_risk_fs.clip.leakage_policy import ALLOWED_DATASETS, LEGACY_DATASETS
from credit_risk_fs.clip.schemas import ClipDatasetRole, ClipFieldRole, ClipFieldSpec


FORBIDDEN_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("target", re.compile(r"(^|_)target($|_)", re.IGNORECASE)),
    ("label", re.compile(r"(^|_)label($|_)", re.IGNORECASE)),
    ("bad_flag", re.compile(r"bad[_-]?flag|(^|_)bad($|_)", re.IGNORECASE)),
    ("default", re.compile(r"(^|_)default($|_)", re.IGNORECASE)),
    ("y_true", re.compile(r"y[_-]?true", re.IGNORECASE)),
    ("y_score", re.compile(r"y[_-]?score", re.IGNORECASE)),
    ("prediction", re.compile(r"(^|_)(pred|prediction|model_score|score_psi|y_score)($|_)", re.IGNORECASE)),
    ("oot", re.compile(r"(^|_)oot($|_)|out[_-]?of[_-]?time", re.IGNORECASE)),
    ("psi", re.compile(r"(^|_)psi($|_)|population[_-]?stability", re.IGNORECASE)),
    ("fold", re.compile(r"(^|_)fold($|_)", re.IGNORECASE)),
    ("split", re.compile(r"(^|_)split($|_)", re.IGNORECASE)),
    ("row_id", re.compile(r"row[_-]?id", re.IGNORECASE)),
    ("customer_id", re.compile(r"customer[_-]?id", re.IGNORECASE)),
    ("loan_id", re.compile(r"loan[_-]?id|member[_-]?id", re.IGNORECASE)),
    ("sk_id", re.compile(r"sk[_-]?id", re.IGNORECASE)),
    ("issue_date", re.compile(r"issue[_-]?d|issue[_-]?date", re.IGNORECASE)),
    ("payment", re.compile(r"payment|pymnt|last[_-]?pymnt|next[_-]?pymnt", re.IGNORECASE)),
    ("recovery", re.compile(r"recover(y|ies)|collection[_-]?recovery", re.IGNORECASE)),
    ("settlement", re.compile(r"settlement|debt[_-]?settlement", re.IGNORECASE)),
    ("hardship", re.compile(r"hardship", re.IGNORECASE)),
    ("charged_off", re.compile(r"charged[_ -]?off|chargeoff", re.IGNORECASE)),
    ("post_origination_status", re.compile(r"post[_-]?origination|loan[_-]?status|future[_-]?outcome", re.IGNORECASE)),
)

FORBIDDEN_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("planning_evidence", re.compile(r"feature_level_evidence_for_clip\.csv", re.IGNORECASE)),
    ("legacy_lendingclub_results", re.compile(r"(^|/)results/lendingclub(/|$)", re.IGNORECASE)),
    ("legacy_results_v1", re.compile(r"(^|/)results_v1(/|$)", re.IGNORECASE)),
)

INPUT_ROLES = {ClipFieldRole.TEXT_INPUT, ClipFieldRole.STATISTICAL_INPUT, ClipFieldRole.ANCHOR_ONLY}


def normalize_path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def forbidden_field_matches(field_name: str) -> list[str]:
    return [name for name, pattern in FORBIDDEN_FIELD_PATTERNS if pattern.search(str(field_name))]


def is_forbidden_field_name(field_name: str) -> bool:
    return bool(forbidden_field_matches(field_name))


def forbidden_path_matches(path: str | Path) -> list[str]:
    text = normalize_path_text(path)
    return [name for name, pattern in FORBIDDEN_PATH_PATTERNS if pattern.search(text)]


def validate_training_source_path(path: str | Path, dataset: str) -> list[str]:
    text = normalize_path_text(path)
    errors = []
    path_matches = forbidden_path_matches(path)
    if path_matches:
        errors.append(f"forbidden source path patterns: {path_matches} in {text}")
    if not text.endswith("/dev_only_clip_training_evidence.csv"):
        errors.append("source path must end with dev_only_clip_training_evidence.csv")
    expected_fragment = f"results/{dataset}/analysis/clip_readiness/dev_only_clip_training_evidence.csv"
    if expected_fragment not in text:
        errors.append(f"source path is not the approved {dataset} DEV-only evidence file")
    return errors


def validate_dataset_roles(train_dataset: str, external_validation_dataset: str) -> list[str]:
    errors = []
    if train_dataset != "homecredit":
        errors.append("Home Credit must be the only training dataset")
    if external_validation_dataset != "lendingclub_v2":
        errors.append("LendingClub v2 must be the external-validation dataset")
    active = {train_dataset, external_validation_dataset}
    legacy = active.intersection(set(LEGACY_DATASETS))
    if legacy:
        errors.append(f"legacy datasets are forbidden for CLIP work: {sorted(legacy)}")
    unknown = active - set(ALLOWED_DATASETS)
    if unknown:
        errors.append(f"unknown CLIP datasets: {sorted(unknown)}")
    if train_dataset == external_validation_dataset:
        errors.append("train and external-validation datasets must be different")
    return errors


def validate_field_role_separation(field_specs: Iterable[ClipFieldSpec]) -> list[str]:
    errors = []
    for spec in field_specs:
        matches = forbidden_field_matches(spec.field_name)
        if matches and spec.field_role in INPUT_ROLES:
            errors.append(
                f"{spec.dataset}.{spec.field_name}: forbidden field pattern used as {spec.field_role.value}: {matches}"
            )
        if spec.field_role in {ClipFieldRole.SUPERVISION_ONLY, ClipFieldRole.EVALUATION_ONLY, ClipFieldRole.FORBIDDEN}:
            if spec.allowed_in_main_training_input:
                errors.append(f"{spec.dataset}.{spec.field_name}: non-input role marked trainable")
    return errors


def validate_evidence_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    role: ClipDatasetRole,
    statistical_fields: Iterable[str],
    extreme_missingness_threshold: float = 0.95,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    required = {
        "dataset",
        "feature",
        "description",
        "semantic_group",
        "source_table",
        "allowed_for_clip_training",
        "clip_training_exclusion_reason",
        "leakage_review_status",
        "leakage_review_action",
    }
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        errors.append(f"{dataset}: missing required evidence columns: {missing_required}")
        return warnings, errors

    observed = set(frame["dataset"].dropna().astype(str))
    if observed != {dataset}:
        errors.append(f"{dataset}: dataset identity mismatch: observed={sorted(observed)}")

    if dataset in LEGACY_DATASETS:
        errors.append(f"{dataset}: legacy dataset contamination")

    duplicate_features = sorted(frame.loc[frame["feature"].duplicated(), "feature"].dropna().astype(str).unique())
    if duplicate_features:
        errors.append(f"{dataset}: duplicate feature names: {duplicate_features[:20]}")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        errors.append(f"{dataset}: duplicate evidence rows: {duplicate_rows}")

    allowed = as_bool(frame["allowed_for_clip_training"])
    allowed_frame = frame[allowed].copy()
    blocked_frame = frame[~allowed].copy()

    if role == ClipDatasetRole.TRAIN and dataset != "homecredit":
        errors.append(f"{dataset}: only homecredit may have train role")
    if role == ClipDatasetRole.EXTERNAL_VALIDATION and dataset != "lendingclub_v2":
        errors.append(f"{dataset}: only lendingclub_v2 may have external_validation role")

    for col in ["description", "semantic_group", "source_table"]:
        blank = ~nonempty(allowed_frame[col])
        if bool(blank.any()):
            errors.append(f"{dataset}: allowed rows contain blank {col}: {int(blank.sum())}")

    unsafe_status = ~allowed_frame["leakage_review_status"].fillna("").astype(str).str.lower().isin({"safe"})
    if bool(unsafe_status.any()):
        errors.append(f"{dataset}: unsafe leakage-review status in allowed rows: {int(unsafe_status.sum())}")

    excluded_action = allowed_frame["leakage_review_action"].fillna("").astype(str).str.lower().eq("exclude")
    if bool(excluded_action.any()):
        errors.append(f"{dataset}: leakage-review excluded rows in allowed set: {int(excluded_action.sum())}")

    blocked_without_reason = ~nonempty(blocked_frame["clip_training_exclusion_reason"])
    if bool(blocked_without_reason.any()):
        errors.append(f"{dataset}: blocked rows missing block reasons: {int(blocked_without_reason.sum())}")

    for field in statistical_fields:
        matches = forbidden_field_matches(field)
        if matches:
            errors.append(f"{dataset}: forbidden statistical input field {field}: {matches}")
        if field not in frame.columns:
            warnings.append(f"{dataset}: configured statistical field missing: {field}")
            continue
        values = pd.to_numeric(allowed_frame[field], errors="coerce")
        if len(values) and values.isna().all():
            errors.append(f"{dataset}: all-null statistical input field: {field}")
        elif len(values) and values.nunique(dropna=True) <= 1:
            warnings.append(f"{dataset}: constant statistical input field: {field}")
        missing_rate = float(values.isna().mean()) if len(values) else 1.0
        if missing_rate > extreme_missingness_threshold:
            warnings.append(f"{dataset}: extreme missingness in statistical input {field}: {missing_rate:.3f}")

    return warnings, errors


def validate_deterministic_order(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return []
    sorted_frame = frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
    current = frame.reset_index(drop=True)
    if current[columns].equals(sorted_frame[columns]):
        return []
    return [f"frame is not sorted by {columns}"]


def scan_forbidden_fields(fields: Iterable[str]) -> dict[str, list[str]]:
    return {field: matches for field in fields if (matches := forbidden_field_matches(field))}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")
