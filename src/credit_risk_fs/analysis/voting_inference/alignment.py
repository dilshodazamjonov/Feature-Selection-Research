"""Prediction alignment and leakage auditing for the Prompt 6 evidence package.

Alignment is a hard gate.  Nothing downstream may compute a paired statistic
until every comparison cell reports ``decision == "aligned"``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from credit_risk_fs.analysis.voting_inference.config import (
    AnalysisConfig,
    AuthenticationError,
    read_json,
)
from credit_risk_fs.analysis.voting_inference.inventory import (
    RunRecord,
    read_final_selection,
)

PREDICTION_COLUMNS = (
    "stable_row_id",
    "target",
    "prediction_probability",
    "fold_id",
    "split",
    "probability_orientation",
    "research_eligible",
    "comparison_eligible",
    "coverage_type",
)
EXPECTED_ORIENTATION = "class_1_higher_default_risk"


@dataclass(frozen=True)
class PredictionFrame:
    """One validated applicant-level prediction artifact."""

    run_id: str
    dataset: str
    split: str
    path: Path
    frame: pd.DataFrame
    metadata: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        return len(self.frame)

    @property
    def positive_count(self) -> int:
        return int(self.frame["target"].sum())


def canonical_identity(values: Iterable[Any]) -> pd.Series:
    """Normalise stable row identities to NFC strings without dropping rows."""

    series = pd.Series(list(values), dtype="object")
    if series.isna().any():
        raise AuthenticationError("stable row identity contains a missing value")
    return series.map(lambda value: unicodedata.normalize("NFC", str(value).strip()))


def load_prediction_frame(
    run: RunRecord,
    *,
    split: str,
) -> PredictionFrame:
    """Load and validate one saved prediction artifact without mutating it."""

    split_key = split.upper()
    if split_key == "DEV":
        path, metadata_path = run.dev_predictions, run.dev_metadata
    elif split_key == "OOT":
        path, metadata_path = run.oot_predictions, run.oot_metadata
    else:
        raise AuthenticationError(f"unsupported prediction split: {split!r}")
    if not path.is_file():
        raise AuthenticationError(f"{run.run_id}: missing {split_key} predictions at {path}")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}

    raw = pd.read_csv(path, dtype={"stable_row_id": "string"})
    missing = set(PREDICTION_COLUMNS) - set(raw.columns)
    if missing:
        raise AuthenticationError(
            f"{run.run_id} {split_key}: prediction columns missing {sorted(missing)}"
        )
    frame = pd.DataFrame(
        {
            "stable_row_id": canonical_identity(raw["stable_row_id"]),
            "target": pd.to_numeric(raw["target"], errors="raise").astype("int8"),
            "score": pd.to_numeric(
                raw["prediction_probability"], errors="raise"
            ).astype(float),
            "fold_id": raw["fold_id"].astype("string"),
        }
    )
    orientations = set(raw["probability_orientation"].astype(str).unique())
    if orientations != {EXPECTED_ORIENTATION}:
        raise AuthenticationError(
            f"{run.run_id} {split_key}: probability orientation {sorted(orientations)}"
        )
    for flag in ("research_eligible", "comparison_eligible"):
        values = {str(value).strip().lower() for value in raw[flag].unique()}
        if values != {"true"}:
            raise AuthenticationError(f"{run.run_id} {split_key}: {flag} is not True")
    declared_splits = {str(value).upper() for value in raw["split"].unique()}
    expected_split = {"DEV"} if split_key == "DEV" else {"OOT"}
    if declared_splits != expected_split:
        raise AuthenticationError(
            f"{run.run_id} {split_key}: artifact declares split {sorted(declared_splits)}"
        )
    return PredictionFrame(
        run_id=run.run_id,
        dataset=run.dataset,
        split="DEV_OOF" if split_key == "DEV" else "OOT",
        path=path,
        frame=frame,
        metadata=metadata,
    )


def _score_range_violations(scores: np.ndarray) -> int:
    return int(np.count_nonzero((scores < 0.0) | (scores > 1.0)))


def align_predictions(
    reference: PredictionFrame,
    comparator: PredictionFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align two prediction artifacts one-to-one on the authenticated identity.

    The returned audit row is complete whether or not alignment succeeds, so a
    failure remains visible in the package instead of raising out of the report.
    """

    left = reference.frame
    right = comparator.frame
    left_ids = left["stable_row_id"]
    right_ids = right["stable_row_id"]
    duplicate_count = int(left_ids.duplicated().sum() + right_ids.duplicated().sum())
    left_set = set(left_ids)
    right_set = set(right_ids)
    missing_in_comparator = len(left_set - right_set)
    missing_in_reference = len(right_set - left_set)

    audit: dict[str, Any] = {
        "dataset": reference.dataset,
        "split": reference.split,
        "reference_run_id": reference.run_id,
        "comparator_run_id": comparator.run_id,
        "reference_row_count": len(left),
        "comparator_row_count": len(right),
        "reference_unique_id_count": len(left_set),
        "comparator_unique_id_count": len(right_set),
        "duplicate_id_count": duplicate_count,
        "missing_reference_ids": missing_in_reference,
        "missing_comparator_ids": missing_in_comparator,
        "reference_score_null_count": int(left["score"].isna().sum()),
        "comparator_score_null_count": int(right["score"].isna().sum()),
        "reference_score_range_violations": _score_range_violations(
            left["score"].to_numpy(dtype=float)
        ),
        "comparator_score_range_violations": _score_range_violations(
            right["score"].to_numpy(dtype=float)
        ),
        "positive_class_orientation_status": "verified_class_1_higher_default_risk",
        "reference_identity_target_sha256": reference.metadata.get(
            "identity_target_sha256"
        ),
        "comparator_identity_target_sha256": comparator.metadata.get(
            "identity_target_sha256"
        ),
    }
    audit["row_hash_match_status"] = (
        "match"
        if audit["reference_identity_target_sha256"]
        and audit["reference_identity_target_sha256"]
        == audit["comparator_identity_target_sha256"]
        else "mismatch_or_unavailable"
    )

    if duplicate_count or missing_in_comparator or missing_in_reference:
        audit.update(
            {
                "aligned_row_count": 0,
                "target_mismatch_count": None,
                "unexplained_row_loss": max(len(left), len(right)),
                "decision": "BLOCKED",
                "decision_reason": "identity sets are not one-to-one",
            }
        )
        return pd.DataFrame(columns=["stable_row_id", "target", "score_reference", "score_comparator"]), audit

    merged = (
        left.rename(columns={"target": "target_reference", "score": "score_reference"})
        .merge(
            right.rename(
                columns={"target": "target_comparator", "score": "score_comparator"}
            ),
            on="stable_row_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_comparator"),
        )
    )
    target_mismatch = int(
        merged["target_reference"].ne(merged["target_comparator"]).sum()
    )
    aligned = (
        merged.assign(
            __order__=merged["stable_row_id"].map(
                lambda value: unicodedata.normalize("NFC", value).casefold()
            )
        )
        .sort_values("__order__", kind="mergesort")
        .drop(columns="__order__")
        .reset_index(drop=True)
    )
    if aligned["stable_row_id"].duplicated().any():
        audit.update(
            {
                "aligned_row_count": len(aligned),
                "target_mismatch_count": target_mismatch,
                "unexplained_row_loss": 0,
                "decision": "BLOCKED",
                "decision_reason": "alignment introduced duplicate identities",
            }
        )
        return aligned, audit

    unexplained_loss = max(len(left), len(right)) - len(aligned)
    audit.update(
        {
            "aligned_row_count": len(aligned),
            "aligned_positive_count": int(aligned["target_reference"].sum()),
            "aligned_negative_count": int(
                len(aligned) - aligned["target_reference"].sum()
            ),
            "target_mismatch_count": target_mismatch,
            "unexplained_row_loss": unexplained_loss,
        }
    )
    if target_mismatch:
        audit["decision"] = "BLOCKED"
        audit["decision_reason"] = f"{target_mismatch} target mismatches"
    elif unexplained_loss:
        audit["decision"] = "BLOCKED"
        audit["decision_reason"] = f"{unexplained_loss} rows lost without explanation"
    elif audit["reference_score_range_violations"] or audit[
        "comparator_score_range_violations"
    ]:
        audit["decision"] = "BLOCKED"
        audit["decision_reason"] = "prediction score outside [0, 1]"
    else:
        audit["decision"] = "aligned"
        audit["decision_reason"] = ""

    output = aligned.rename(columns={"target_reference": "target"}).drop(
        columns="target_comparator"
    )
    return output, audit


def dev_oot_disjoint_audit(
    dev: PredictionFrame, oot: PredictionFrame
) -> dict[str, Any]:
    """Verify the DEV and OOT populations of one run share no identity."""

    dev_ids = set(dev.frame["stable_row_id"])
    oot_ids = set(oot.frame["stable_row_id"])
    overlap = len(dev_ids & oot_ids)
    return {
        "run_id": dev.run_id,
        "dataset": dev.dataset,
        "dev_row_count": len(dev.frame),
        "oot_row_count": len(oot.frame),
        "dev_oot_identity_overlap": overlap,
        "dev_oof_fold_ids": ";".join(sorted(set(dev.frame["fold_id"].dropna()))),
        "oot_fold_ids": ";".join(sorted(set(oot.frame["fold_id"].dropna()))),
        "decision": "disjoint" if overlap == 0 else "BLOCKED",
    }


def leakage_audit_row(
    config: AnalysisConfig, run: RunRecord, *, leakage_exclusions: set[str]
) -> dict[str, Any]:
    """Audit one run's final selection against frozen leakage exclusions."""

    selected = read_final_selection(run)
    forbidden = sorted(set(selected) & leakage_exclusions)
    budget = config.final_budget(run.model)
    oot_metadata = read_json(run.oot_metadata) if run.oot_metadata.is_file() else {}
    dev_access = run.directory / "data_access_dev.json"
    dev_opened_oot: list[str] = []
    if dev_access.is_file():
        dev_opened_oot = [
            str(value) for value in read_json(dev_access).get("opened_oot_paths", [])
        ]
    return {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model": run.model,
        "configuration": run.configuration,
        "final_selected_feature_count": len(selected),
        "expected_final_feature_budget": budget,
        "budget_matches": len(selected) == budget,
        "duplicate_selected_features": len(selected) - len(set(selected)),
        "forbidden_features_in_selection": ";".join(forbidden),
        "leakage_exclusion_violation_count": len(forbidden),
        "dev_phase_opened_oot_path_count": len(dev_opened_oot),
        "oot_configuration_frozen_before_scoring": bool(
            (run.manifest.get("summary") or {}).get(
                "configuration_frozen_before_oot", False
            )
        ),
        "oot_embargo_declared": bool(run.manifest.get("oot_embargo", False)),
        "oot_fold_definition": str(oot_metadata.get("fold_definition", "")),
        "dev_coverage_type": _coverage_type(run.dev_metadata),
        "oot_coverage_type": _coverage_type(run.oot_metadata),
        "decision": (
            "BLOCKED"
            if forbidden
            or len(selected) != budget
            or len(selected) != len(set(selected))
            or dev_opened_oot
            else "clean"
        ),
    }


def _coverage_type(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(read_json(path).get("coverage_type", ""))


def frozen_leakage_exclusions(config: AnalysisConfig) -> dict[str, set[str]]:
    """Recover each dataset's frozen leakage/identity/time exclusion set."""

    protocol = config.repository_root / str(
        config.payload["frozen_inputs"]["scientific_protocol"]["path"]
    )
    import yaml

    payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    exclusions: dict[str, set[str]] = {}
    for dataset in config.expected["datasets"]:
        entry = payload["datasets"][dataset]["candidate_universe"]["exclusions"]
        exclusions[dataset] = {str(value) for value in entry}
    return exclusions


__all__ = [
    "EXPECTED_ORIENTATION",
    "PredictionFrame",
    "align_predictions",
    "canonical_identity",
    "dev_oot_disjoint_audit",
    "frozen_leakage_exclusions",
    "leakage_audit_row",
    "load_prediction_frame",
]
