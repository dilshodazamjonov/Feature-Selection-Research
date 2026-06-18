from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.clip.leakage_policy import (  # noqa: E402
    ALLOWED_DATASETS,
    REQUIRED_TRAINING_COLUMNS,
    is_forbidden_training_column,
)
from credit_risk_fs.clip.training_manifest import ClipReadinessManifest, load_readiness_manifest  # noqa: E402


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _validate_training_frame(dataset: str, path: Path, min_allowed_rows: int) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{dataset}: missing training evidence file: {path}"]

    frame = pd.read_csv(path)
    missing_required = [col for col in REQUIRED_TRAINING_COLUMNS if col not in frame.columns]
    if missing_required:
        errors.append(f"{dataset}: missing required columns: {missing_required}")

    forbidden_columns = [col for col in frame.columns if is_forbidden_training_column(col)]
    if forbidden_columns:
        errors.append(f"{dataset}: forbidden training columns present: {forbidden_columns}")

    if "dataset" in frame.columns:
        observed = set(frame["dataset"].dropna().astype(str))
        if observed != {dataset}:
            errors.append(f"{dataset}: dataset column mismatch: observed={sorted(observed)}")
        legacy_observed = observed - set(ALLOWED_DATASETS)
        if legacy_observed:
            errors.append(f"{dataset}: legacy or unknown datasets present: {sorted(legacy_observed)}")

    if "allowed_for_clip_training" not in frame.columns:
        return errors

    allowed = _as_bool(frame["allowed_for_clip_training"])
    allowed_frame = frame[allowed].copy()
    if int(allowed.sum()) < min_allowed_rows:
        errors.append(
            f"{dataset}: allowed rows below minimum: allowed={int(allowed.sum())}, minimum={min_allowed_rows}"
        )

    for col in ["description", "semantic_group", "clip_training_text", "leakage_rule"]:
        if col in allowed_frame.columns and not _nonempty(allowed_frame[col]).all():
            errors.append(f"{dataset}: allowed rows contain blank {col}")

    if "clip_training_exclusion_reason" in allowed_frame.columns:
        has_reason = _nonempty(allowed_frame["clip_training_exclusion_reason"])
        if bool(has_reason.any()):
            errors.append(f"{dataset}: allowed rows still have exclusion reasons")

    if "leakage_review_action" in allowed_frame.columns:
        excluded = allowed_frame["leakage_review_action"].fillna("").astype(str).str.lower().eq("exclude")
        if bool(excluded.any()):
            errors.append(f"{dataset}: leakage-review excluded rows are allowed")

    if "leakage_review_status" in allowed_frame.columns:
        unsafe = ~allowed_frame["leakage_review_status"].fillna("safe").astype(str).str.lower().isin({"safe"})
        if bool(unsafe.any()):
            errors.append(f"{dataset}: non-safe leakage-review rows are allowed")

    return errors


def _summary_counts(path: Path) -> dict[str, tuple[int, int, int]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"dataset", "total_rows", "allowed_for_clip_training", "blocked_for_clip_training"}
    if not required.issubset(frame.columns):
        return {}
    return {
        str(row["dataset"]): (
            int(row["total_rows"]),
            int(row["allowed_for_clip_training"]),
            int(row["blocked_for_clip_training"]),
        )
        for row in frame.to_dict("records")
    }


def _actual_counts(path: Path) -> tuple[int, int, int]:
    frame = pd.read_csv(path)
    allowed = _as_bool(frame["allowed_for_clip_training"])
    return int(len(frame)), int(allowed.sum()), int((~allowed).sum())


def validate_manifest(manifest: ClipReadinessManifest) -> list[str]:
    errors: list[str] = []
    datasets = set(manifest.datasets)
    if datasets != set(ALLOWED_DATASETS):
        errors.append(f"active datasets must be {list(ALLOWED_DATASETS)}, got {sorted(datasets)}")

    legacy_active = datasets.intersection(set(manifest.legacy_datasets))
    if legacy_active:
        errors.append(f"legacy datasets cannot be active for CLIP readiness: {sorted(legacy_active)}")

    for dataset in manifest.datasets:
        errors.extend(
            _validate_training_frame(
                dataset,
                manifest.training_evidence[dataset],
                manifest.min_allowed_rows.get(dataset, 1),
            )
        )

    summary = _summary_counts(manifest.cross_dataset_summary)
    if not summary:
        errors.append(f"missing or invalid cross-dataset training summary: {manifest.cross_dataset_summary}")
    else:
        for dataset in manifest.datasets:
            if dataset not in summary:
                errors.append(f"{dataset}: missing from cross-dataset training summary")
                continue
            actual = _actual_counts(manifest.training_evidence[dataset])
            if summary[dataset] != actual:
                errors.append(f"{dataset}: summary counts do not match training evidence: summary={summary[dataset]}, actual={actual}")

    return errors


def _print_result(errors: Iterable[str]) -> int:
    errors = list(errors)
    if not errors:
        print("CLIP readiness validation passed.")
        return 0
    print("CLIP readiness validation failed:")
    for error in errors:
        print(f"- {error}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DEV-only CLIP readiness artifacts.")
    parser.add_argument("--config", default="configs/clip/readiness.yaml")
    args = parser.parse_args()

    manifest = load_readiness_manifest(Path(args.config))
    return _print_result(validate_manifest(manifest))


if __name__ == "__main__":
    raise SystemExit(main())

