from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.schemas import ClipDatasetRole, ClipEvidenceAudit
from credit_risk_fs.clip.validation import (
    as_bool,
    scan_forbidden_fields,
    validate_evidence_frame,
    validate_training_source_path,
)
from credit_risk_fs.utils.hashing import sha256_file


class ClipEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedClipEvidence:
    dataset: str
    role: ClipDatasetRole
    source_path: Path
    source_hash: str
    frame: pd.DataFrame
    allowed: pd.DataFrame
    blocked: pd.DataFrame
    audit: ClipEvidenceAudit


def load_clip_evidence(
    *,
    dataset: str,
    role: ClipDatasetRole,
    source_path: str | Path,
    statistical_fields: list[str],
) -> LoadedClipEvidence:
    path = Path(source_path)
    errors = validate_training_source_path(path, dataset)
    if errors:
        raise ClipEvidenceError(f"{dataset}: invalid CLIP training evidence path: {'; '.join(errors)}")
    if not path.exists():
        raise ClipEvidenceError(f"{dataset}: CLIP training evidence file does not exist: {path}")

    frame = pd.read_csv(path)
    normalizations: list[str] = []
    expected_columns = set(frame.columns)
    normalized_columns = [col.strip() for col in frame.columns]
    if normalized_columns != list(frame.columns):
        frame = frame.rename(columns=dict(zip(frame.columns, normalized_columns)))
        normalizations.append("stripped_column_whitespace")
    if set(frame.columns) != expected_columns and not normalizations:
        raise ClipEvidenceError(f"{dataset}: unexpected column normalization state")

    if "allowed_for_clip_training" not in frame.columns:
        raise ClipEvidenceError(f"{dataset}: missing allowed_for_clip_training column")

    allowed_mask = as_bool(frame["allowed_for_clip_training"])
    allowed = frame[allowed_mask].copy()
    blocked = frame[~allowed_mask].copy()

    if "clip_training_exclusion_reason" in allowed.columns:
        allowed_reasons = allowed["clip_training_exclusion_reason"].fillna("").astype(str).str.strip()
        if bool(allowed_reasons.ne("").any()):
            raise ClipEvidenceError(f"{dataset}: allowed evidence rows contain block reasons")

    if "leakage_review_status" in allowed.columns:
        unsafe = ~allowed["leakage_review_status"].fillna("").astype(str).str.lower().isin({"safe"})
        if bool(unsafe.any()):
            raise ClipEvidenceError(f"{dataset}: unsafe rows included in allowed evidence")

    if "leakage_review_action" in allowed.columns:
        excluded = allowed["leakage_review_action"].fillna("").astype(str).str.lower().eq("exclude")
        if bool(excluded.any()):
            raise ClipEvidenceError(f"{dataset}: leakage-excluded rows included in allowed evidence")

    warnings, validation_errors = validate_evidence_frame(
        frame,
        dataset=dataset,
        role=role,
        statistical_fields=statistical_fields,
    )
    if validation_errors:
        raise ClipEvidenceError(f"{dataset}: evidence validation failed: {'; '.join(validation_errors)}")

    sorted_frame = frame.sort_values(["dataset", "feature"], kind="mergesort").reset_index(drop=True)
    sorted_allowed = allowed.sort_values(["dataset", "feature"], kind="mergesort").reset_index(drop=True)
    sorted_blocked = blocked.sort_values(["dataset", "feature"], kind="mergesort").reset_index(drop=True)
    source_hash = sha256_file(path)

    audit = ClipEvidenceAudit(
        dataset=dataset,
        role=role,
        source_file=path,
        source_sha256=source_hash,
        row_count=int(len(sorted_frame)),
        allowed_row_count=int(len(sorted_allowed)),
        blocked_row_count=int(len(sorted_blocked)),
        dtypes={col: str(dtype) for col, dtype in sorted_frame.dtypes.items()},
        missingness={col: int(value) for col, value in sorted_frame.isna().sum().items()},
        normalizations=normalizations,
        duplicate_feature_names=sorted(
            sorted_frame.loc[sorted_frame["feature"].duplicated(), "feature"].dropna().astype(str).unique()
        ),
        duplicate_evidence_rows=int(sorted_frame.duplicated().sum()),
        forbidden_fields_detected=scan_forbidden_fields(sorted_frame.columns),
        validation_warnings=warnings,
        validation_errors=[],
    )
    return LoadedClipEvidence(
        dataset=dataset,
        role=role,
        source_path=path,
        source_hash=source_hash,
        frame=sorted_frame,
        allowed=sorted_allowed,
        blocked=sorted_blocked,
        audit=audit,
    )
