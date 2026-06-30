from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.model import SemanticStatisticalContrastiveEncoder
from credit_risk_fs.utils.hashing import sha256_file, sha256_text


PAIRING_POLICY_VERSION = "identity_equivalence_v2"
SOURCE_DATASET = "lendingclub_v2"
EXTERNAL_DATASET = "homecredit"
REVERSE_METHOD = "lendingclub_clip_to_homecredit_mrmr"
REGISTRY_SCHEMA_VERSION = "reverse_transfer_registry_v2"
REGISTRY_CANONICALIZATION_VERSION = "schema_aware_registry_v2"
ARTIFACT_IDENTITY_VERSION = "canonical_artifact_identity_v1"
TRANSACTION_OUTCOMES = {"NEW_TRANSACTION", "IDEMPOTENT_NO_OP", "CONFLICT"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HASH_PLACEHOLDERS = {
    "unknown", "pending", "not_generated", "not_applicable", "todo", "none", "null"
}


class RegistryConflictError(ValueError):
    transaction_outcome = "CONFLICT"


def canonical_artifact_id(
    *,
    run_id: str,
    artifact_type: str,
    relative_path: str | Path,
    content_hash: str,
    schema_version: str = REGISTRY_SCHEMA_VERSION,
) -> str:
    """Build a deterministic logical artifact ID, not a content-only ID."""
    payload = {
        "identity_version": ARTIFACT_IDENTITY_VERSION,
        "schema_version": str(schema_version),
        "run_id": canonical_registry_value("run_id", run_id),
        "artifact_type": canonical_registry_value(
            "artifact_type", artifact_type
        ),
        "relative_path": canonical_registry_value(
            "relative_path", relative_path, expected_type="path"
        ),
        "content_hash": validate_sha256(content_hash, field="content_hash"),
    }
    return sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


@dataclass(frozen=True)
class RegistrySchema:
    required: frozenset[str]
    primary_key: tuple[str, ...]
    boolean_fields: frozenset[str] = frozenset()
    integer_fields: frozenset[str] = frozenset()
    float_fields: frozenset[str] = frozenset()
    path_fields: frozenset[str] = frozenset()
    hash_fields: frozenset[str] = frozenset()
    json_fields: frozenset[str] = frozenset()
    enum_fields: Mapping[str, frozenset[str]] = None


REGISTRY_SCHEMAS = {
    "run_index.csv": RegistrySchema(
        required=frozenset({
            "run_id", "dataset", "method", "model", "configuration_hash",
            "data_manifest_hash", "metric_artifact_path", "prediction_artifact_path",
            "selected_feature_path", "pairing_policy_version", "reuse_status",
        }),
        primary_key=("run_id",),
        integer_fields=frozenset({"seed", "feature_budget"}),
        path_fields=frozenset({
            "metric_artifact_path", "prediction_artifact_path", "selected_feature_path",
            "checkpoint_path", "manifest_path",
        }),
        hash_fields=frozenset({
            "configuration_hash", "data_manifest_hash", "source_preprocessor_hash",
            "source_raw_dev_statistical_evidence_hash",
        }),
        json_fields=frozenset({"source_checkpoint_hashes", "source_anchor_hashes"}),
        enum_fields={
            "dataset": frozenset({"homecredit", "lendingclub_v2"}),
            "model": frozenset({"lr", "catboost", "semantic_statistical_dual_encoder"}),
            "reuse_status": frozenset({"newly_executed", "reusable_existing", "invalid_pairing_policy"}),
        },
    ),
    "artifact_registry.csv": RegistrySchema(
        required=frozenset({
            "artifact_id", "artifact_type", "relative_path", "file_hash",
            "pairing_policy_version", "reuse_status",
        }),
        primary_key=("artifact_id",),
        boolean_fields=frozenset({"file_exists", "depends_on_clip", "depends_on_old_pairing"}),
        path_fields=frozenset({"relative_path"}),
        hash_fields=frozenset({"artifact_id", "file_hash"}),
        enum_fields={
            "reuse_status": frozenset({"newly_executed", "reusable_existing", "invalid_pairing_policy", "unknown_requires_review"}),
        },
    ),
    "reusable_metrics.csv": RegistrySchema(
        required=frozenset({
            "run_id", "dataset_name", "model", "selector", "experiment_type",
            "oot_auc", "oot_ks", "config_hash", "metric_artifact_path",
            "result_origin", "reuse_status", "pairing_policy_version",
            "data_manifest_hash", "source_identity_manifest_hash",
            "dev_prediction_hash", "oot_prediction_hash",
        }),
        primary_key=("run_id",),
        boolean_fields=frozenset({"llm_shared_ranking_enabled"}),
        integer_fields=frozenset({
            "feature_budget", "llm_ranking_budget", "selected_feature_count",
            "total_candidate_feature_count",
        }),
        float_fields=frozenset({
            "oot_auc", "oot_gini", "oot_ks", "oot_log_loss", "oot_brier",
            "model_score_psi", "runtime_seconds",
        }),
        path_fields=frozenset({"output_folder", "metric_artifact_path"}),
        hash_fields=frozenset({
            "config_hash", "data_manifest_hash", "source_identity_manifest_hash",
            "dev_prediction_hash", "oot_prediction_hash",
        }),
        enum_fields={
            "dataset_name": frozenset({"homecredit", "lendingclub_v2"}),
            "model": frozenset({"lr", "catboost"}),
            "result_origin": frozenset({"newly_executed", "reused_existing"}),
            "reuse_status": frozenset({"newly_executed", "reusable_existing"}),
        },
    ),
    "selected_feature_registry.csv": RegistrySchema(
        required=frozenset({
            "run_id", "dataset", "model", "selector", "selected_feature_path",
            "selected_feature_hash", "pairing_policy_version", "reuse_status",
            "configuration_hash", "data_manifest_hash",
            "source_identity_manifest_hash",
        }),
        primary_key=("run_id",),
        integer_fields=frozenset({"feature_budget", "selected_feature_count"}),
        boolean_fields=frozenset({"depends_on_clip"}),
        path_fields=frozenset({"selected_feature_path"}),
        hash_fields=frozenset({
            "selected_feature_hash", "configuration_hash", "data_manifest_hash",
            "source_identity_manifest_hash",
        }),
        enum_fields={
            "dataset": frozenset({"homecredit", "lendingclub_v2"}),
            "model": frozenset({"lr", "catboost"}),
            "reuse_status": frozenset({"newly_executed", "reusable_existing", "invalid_pairing_policy"}),
        },
    ),
}


def validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA-256 string")
    normalized = value.strip().lower()
    if normalized in HASH_PLACEHOLDERS or normalized == "0" * 64:
        raise ValueError(f"{field} contains a placeholder SHA-256")
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must contain exactly 64 hexadecimal characters")
    return normalized
ALLOWED_IDENTITY_REASONS = {
    "same_feature",
    "verified_alias",
    "exact_dev_duplicate",
    "documented_identity_transform",
}
DIAGNOSTIC_ONLY_RELATIONS = {
    "same_source_table",
    "same_broad_family",
    "high_text_similarity",
    "equal_statistical_descriptor",
    "high_correlation",
    "similar_suffix",
    "same_business_domain",
}
FORBIDDEN_REPRESENTATION_TOKENS = {
    "target",
    "label",
    "oot",
    "prediction",
    "psi",
    "post_origination",
}
PREDICTION_COLUMNS = [
    "stable_row_id",
    "dataset",
    "split",
    "target",
    "prediction_probability",
    "predicted_class",
    "run_id",
    "method",
    "model",
    "source_training_dataset",
    "external_dataset",
    "data_manifest_hash",
    "configuration_hash",
    "pairing_policy_version",
    "fit_scope",
]


@dataclass(frozen=True)
class DatasetRoles:
    training_dataset: str
    external_dataset: str
    training_feature_manifest: str
    external_feature_manifest: str
    training_raw_statistical_source: str
    external_raw_statistical_source: str
    training_statistical_fit_scope: str
    external_statistical_transform_scope: str

    def validate(self) -> None:
        if not self.training_dataset or not self.external_dataset:
            raise ValueError("training_dataset and external_dataset are required")
        if self.training_dataset == self.external_dataset:
            raise ValueError("training and external datasets must be different")
        if "oot" in self.training_statistical_fit_scope.lower():
            raise ValueError("source statistical fit scope must exclude OOT")
        if "transform_only" not in self.external_statistical_transform_scope.lower():
            raise ValueError("external statistical scope must be transform_only")

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "pairing_policy_version": PAIRING_POLICY_VERSION,
            "source_domain": self.training_dataset,
            "external_domain": self.external_dataset,
        }


def reconcile_feature_universe(
    evidence: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    required = {"feature_name"}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"feature evidence missing columns: {sorted(missing)}")
    frame = evidence.copy()
    frame["feature_name"] = frame["feature_name"].fillna("").astype(str).str.strip()
    if frame["feature_name"].eq("").any():
        raise ValueError("feature evidence contains empty feature names")
    if frame["feature_name"].duplicated().any():
        duplicate = frame.loc[frame["feature_name"].duplicated(False), "feature_name"].tolist()
        raise ValueError(f"feature names must map deterministically: {duplicate[:20]}")
    frame["source_table"] = _series_or_default(frame, "source_table", "unknown")
    frame["semantic_group"] = _series_or_default(frame, "semantic_group", "unknown")
    if "text_available" in frame:
        text_available = frame["text_available"].fillna(False).astype(bool)
    else:
        descriptions = _series_or_default(frame, "description", "")
        text_available = descriptions.astype(str).str.strip().ne("")
    if "raw_statistical_evidence_available" in frame:
        stat_available = frame["raw_statistical_evidence_available"].fillna(False).astype(bool)
    else:
        stat_available = _series_or_default(
            frame, "saved_dev_training_signal_available", False
        ).fillna(False).astype(bool)
    frame["dataset"] = dataset
    frame["feature_id"] = [
        sha256_text(f"{dataset}|{name}|{table}")
        for name, table in zip(frame["feature_name"], frame["source_table"])
    ]
    if frame["feature_id"].duplicated().any():
        raise ValueError("feature IDs are not unique")
    frame["text_available"] = text_available
    frame["raw_statistical_evidence_available"] = stat_available
    frame["eligible_for_text_view"] = text_available
    frame["eligible_for_statistical_view"] = stat_available
    frame["eligible_for_pairing"] = text_available & stat_available
    frame["eligible_for_training"] = frame["eligible_for_pairing"]
    frame["eligible_for_validation"] = frame["eligible_for_pairing"]
    frame["excluded"] = ~frame["eligible_for_pairing"]
    frame["split_assignment"] = "unassigned"
    frame["exclusion_reason"] = [
        (
            ""
            if text and stat
            else "missing_description_and_approved_dev_statistical_evidence"
            if not text and not stat
            else "missing_description"
            if not text
            else "missing_approved_dev_statistical_evidence"
        )
        for text, stat in zip(text_available, stat_available)
    ]
    columns = [
        "feature_id",
        "feature_name",
        "source_table",
        "semantic_group",
        "text_available",
        "raw_statistical_evidence_available",
        "eligible_for_text_view",
        "eligible_for_statistical_view",
        "eligible_for_pairing",
        "eligible_for_training",
        "eligible_for_validation",
        "split_assignment",
        "excluded",
        "exclusion_reason",
        "dataset",
    ]
    return frame[columns].sort_values("feature_id", kind="mergesort").reset_index(drop=True)


def load_identity_evidence(
    path: str | Path,
    *,
    reconciled: pd.DataFrame,
    source_dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence_path = Path(path)
    if not evidence_path.exists():
        raise FileNotFoundError(f"identity evidence is missing: {evidence_path}")
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = {
        "source_dataset": source_dataset,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "identity_evidence_version": "explicit_identity_evidence_v1",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"identity evidence {key} mismatch")
    if payload.get("external_dataset") == source_dataset:
        raise ValueError("identity evidence source/external roles are identical")
    by_name = reconciled.set_index("feature_name")["feature_id"].astype(str).to_dict()
    rows: list[dict[str, str]] = []
    for field, reason in (
        ("verified_aliases", "verified_alias"),
        ("documented_identity_transforms", "documented_identity_transform"),
    ):
        values = payload.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"identity evidence {field} must be a list")
        for item in values:
            if not isinstance(item, Mapping):
                raise ValueError(f"identity evidence {field} entries must be objects")
            left_name = str(item.get("feature_name_a", ""))
            right_name = str(item.get("feature_name_b", ""))
            if not left_name or not right_name or left_name == right_name:
                raise ValueError("identity evidence requires two distinct feature names")
            if left_name not in by_name or right_name not in by_name:
                raise ValueError("identity evidence references a feature outside the source universe")
            left_id, right_id = by_name[left_name], by_name[right_name]
            declared = {str(item.get("feature_id_a", "")), str(item.get("feature_id_b", ""))}
            if declared != {left_id, right_id}:
                raise ValueError("identity evidence stable feature IDs do not match source reconciliation")
            rows.append(
                {
                    "feature_id_a": left_id,
                    "feature_id_b": right_id,
                    "feature_name_a": left_name,
                    "feature_name_b": right_name,
                    "reason": reason,
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "feature_id_a",
            "feature_id_b",
            "feature_name_a",
            "feature_name_b",
            "reason",
        ],
    ).drop_duplicates()
    manifest = {
        **expected,
        "external_dataset": payload.get("external_dataset"),
        "identity_relation_count": len(frame),
        "identity_evidence_hash": sha256_file(evidence_path),
        "identity_relation_table_hash": sha256_text(frame.to_csv(index=False)),
    }
    return frame, manifest


def deterministic_feature_split(
    reconciled: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    validation_fraction: float,
    identity_relations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    eligible = reconciled[reconciled["eligible_for_pairing"].astype(bool)].copy()
    if eligible["feature_id"].duplicated().any():
        raise ValueError("duplicate feature IDs")
    if set(eligible["dataset"].astype(str)) != {dataset}:
        raise ValueError("split input does not match the declared training dataset")
    parents = {feature_id: feature_id for feature_id in eligible["feature_id"].astype(str)}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    if identity_relations is not None and len(identity_relations):
        required = {"feature_id_a", "feature_id_b", "reason"}
        missing = required - set(identity_relations.columns)
        if missing:
            raise ValueError(f"identity relations missing columns: {sorted(missing)}")
        unsupported = set(identity_relations["reason"].astype(str)) - ALLOWED_IDENTITY_REASONS
        if unsupported:
            raise ValueError(f"non-identity relations cannot constrain the split: {sorted(unsupported)}")
        for row in identity_relations.itertuples(index=False):
            left, right = str(row.feature_id_a), str(row.feature_id_b)
            if left in parents and right in parents:
                union(left, right)

    eligible["identity_group"] = eligible["feature_id"].astype(str).map(find)
    group_meta = (
        eligible.groupby("identity_group", as_index=False)
        .agg(
            row_count=("feature_id", "size"),
            semantic_group=("semantic_group", lambda values: sorted(map(str, values))[0]),
        )
    )
    group_meta["order"] = [
        sha256_text(f"{seed}|{semantic}|{identity}")
        for identity, semantic in zip(group_meta["identity_group"], group_meta["semantic_group"])
    ]
    validation_groups: set[str] = set()
    target = max(1, round(len(eligible) * validation_fraction))
    for _, semantic_rows in group_meta.groupby("semantic_group", sort=True):
        ordered = semantic_rows.sort_values(["order", "identity_group"], kind="mergesort")
        semantic_target = round(int(ordered["row_count"].sum()) * validation_fraction)
        selected = 0
        for row in ordered.itertuples(index=False):
            if selected >= semantic_target:
                break
            validation_groups.add(str(row.identity_group))
            selected += int(row.row_count)
    if not validation_groups:
        validation_groups.add(str(group_meta.sort_values("order").iloc[0]["identity_group"]))
    current = int(
        eligible["identity_group"].astype(str).isin(validation_groups).sum()
    )
    if current < target:
        for row in group_meta.sort_values(["order", "identity_group"]).itertuples(index=False):
            validation_groups.add(str(row.identity_group))
            current += int(row.row_count)
            if current >= target:
                break
    eligible["split_assignment"] = np.where(
        eligible["identity_group"].astype(str).isin(validation_groups),
        "validation",
        "train",
    )
    train_ids = set(eligible.loc[eligible["split_assignment"].eq("train"), "feature_id"])
    validation_ids = set(
        eligible.loc[eligible["split_assignment"].eq("validation"), "feature_id"]
    )
    if train_ids & validation_ids:
        raise RuntimeError("feature train/validation overlap")
    group_splits = eligible.groupby("identity_group")["split_assignment"].nunique()
    if bool(group_splits.gt(1).any()):
        raise RuntimeError("identity-equivalent features cross splits")
    split = eligible[
        [
            "feature_id",
            "feature_name",
            "dataset",
            "semantic_group",
            "identity_group",
            "split_assignment",
        ]
    ].sort_values("feature_id", kind="mergesort").reset_index(drop=True)
    distribution = (
        split.groupby(["split_assignment", "semantic_group"]).size().rename("count").reset_index()
    )
    manifest = {
        "dataset": dataset,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "train_feature_ids": sorted(train_ids),
        "validation_feature_ids": sorted(validation_ids),
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "identity_group_overlap_count": 0,
        "semantic_group_distribution": distribution.to_dict("records"),
        "split_hash": sha256_text(split.to_csv(index=False)),
        "target_used": False,
        "oot_used": False,
        "external_evidence_used": False,
    }
    return split, manifest


def build_feature_positive_pairs(
    *,
    split: pd.DataFrame,
    text_view: pd.DataFrame,
    statistical_view: pd.DataFrame,
    dataset: str,
    source_manifest_hash: str,
) -> pd.DataFrame:
    _reject_forbidden_columns(text_view.columns)
    _reject_forbidden_columns(statistical_view.columns)
    required = {"feature_id", "feature_name"}
    for label, frame in (("split", split), ("text", text_view), ("statistical", statistical_view)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} view missing columns: {sorted(missing)}")
        if frame["feature_id"].duplicated().any():
            raise ValueError(f"{label} view contains duplicate feature IDs")
    merged = (
        split.merge(
            text_view[["feature_id", "feature_name", "embedding_cache_key"]],
            on=["feature_id", "feature_name"],
            how="inner",
        )
        .merge(
            statistical_view[
                ["feature_id", "feature_name", "stable_row_id", "statistical_vector_hash"]
            ],
            on=["feature_id", "feature_name"],
            how="inner",
        )
        .sort_values("feature_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(merged) != len(split):
        missing_ids = set(split["feature_id"]) - set(merged["feature_id"])
        raise ValueError(f"missing paired views; no vectors were fabricated: {sorted(missing_ids)[:20]}")
    if set(merged["dataset"].astype(str)) != {dataset}:
        raise ValueError("pair dataset mismatch")
    merged["split"] = merged["split_assignment"]
    merged["pair_role"] = merged["split"].map(
        {"train": "train_positive", "validation": "validation_positive"}
    )
    merged["allowed_for_training"] = merged["split"].eq("train")
    merged["allowed_for_validation"] = merged["split"].eq("validation")
    merged["allowed_for_external_evaluation"] = False
    merged["text_embedding_row_id"] = merged["embedding_cache_key"].astype(str)
    merged["statistical_vector_row_id"] = merged["stable_row_id"].astype(str)
    merged["source_manifest_hash"] = source_manifest_hash
    merged["pair_id"] = [
        sha256_text(f"{dataset}|{feature_id}|{split_name}|{source_manifest_hash}")
        for feature_id, split_name in zip(merged["feature_id"], merged["split"])
    ]
    merged["positive_pair_index"] = range(len(merged))
    merged["feature_order_hash"] = feature_order_hash(
        merged["feature_name"].astype(str).tolist()
    )
    return merged[
        [
            "feature_id",
            "feature_name",
            "dataset",
            "semantic_group",
            "identity_group",
            "split",
            "pair_role",
            "pair_id",
            "positive_pair_index",
            "feature_order_hash",
            "text_embedding_row_id",
            "statistical_vector_row_id",
            "statistical_vector_hash",
            "source_manifest_hash",
            "allowed_for_training",
            "allowed_for_validation",
            "allowed_for_external_evaluation",
        ]
    ]


def validate_raw_descriptors(
    frame: pd.DataFrame,
    *,
    dataset: str,
    allowed_scope: str = "dev",
) -> None:
    _reject_forbidden_columns(frame.columns)
    if "dataset" not in frame or set(frame["dataset"].astype(str)) != {dataset}:
        raise ValueError("raw descriptor dataset mismatch")
    if "descriptor_state" not in frame or set(frame["descriptor_state"].astype(str)) != {
        "raw_descriptor"
    }:
        raise ValueError("expected raw_descriptor input; pre-transformed vectors are forbidden")
    if "evidence_scope" not in frame or set(frame["evidence_scope"].astype(str)) != {
        allowed_scope
    }:
        raise ValueError("descriptor evidence must be approved DEV-only")


def validate_frozen_external_transform(
    frame: pd.DataFrame,
    *,
    source_dataset: str,
    external_dataset: str,
    preprocessor_hash: str,
) -> None:
    expected = {
        "dataset": external_dataset,
        "descriptor_state": "external_frozen_transformed_descriptor",
        "preprocessor_fit_dataset": source_dataset,
        "preprocessor_hash": preprocessor_hash,
    }
    for column, value in expected.items():
        if column not in frame or set(frame[column].astype(str)) != {str(value)}:
            raise ValueError(f"external transformed descriptor mismatch for {column}")


def frozen_project(
    *,
    model: SemanticStatisticalContrastiveEncoder,
    features: pd.DataFrame,
    text_values: np.ndarray,
    statistical_values: np.ndarray,
    source_dataset: str,
    external_dataset: str,
    checkpoint_hash: str,
    anchor: np.ndarray,
    anchor_hash: str,
    preprocessor_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(features["dataset"].astype(str)) != {external_dataset}:
        raise ValueError("projection input is not the declared external dataset")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("projection heads must be frozen before external projection")
    model.eval()
    with torch.no_grad():
        text_tensor = torch.as_tensor(text_values, dtype=torch.float32)
        stat_tensor = torch.as_tensor(statistical_values, dtype=torch.float32)
        text_projection, stat_projection = model(text_tensor, stat_tensor)
        joint = torch.nn.functional.normalize(
            (text_projection + stat_projection) / 2.0, p=2, dim=-1
        ).cpu().numpy()
    meta = features[["feature_id", "feature_name", "dataset"]].reset_index(drop=True).copy()
    meta["pairing_policy_version"] = PAIRING_POLICY_VERSION
    meta["source_dataset"] = source_dataset
    meta["external_dataset"] = external_dataset
    meta["checkpoint_hash"] = checkpoint_hash
    meta["anchor_hash"] = anchor_hash
    meta["statistical_preprocessor_hash"] = preprocessor_hash
    vectors = pd.DataFrame(
        joint, columns=[f"joint_{index:04d}" for index in range(joint.shape[1])]
    )
    embeddings = pd.concat([meta, vectors], axis=1)
    scores = meta.copy()
    scores["learned_similarity"] = joint @ np.asarray(anchor, dtype=float)
    scores = scores.sort_values(
        ["learned_similarity", "feature_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scores["learned_rank"] = range(1, len(scores) + 1)
    return embeddings, scores


def align_external_feature_views(
    semantic: pd.DataFrame,
    statistical: pd.DataFrame,
    *,
    external_dataset: str,
    semantic_hash: str,
    statistical_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"feature_id", "feature_name", "dataset"}
    for label, frame in (("semantic", semantic), ("statistical", statistical)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} view missing identity columns: {sorted(missing)}")
        if frame["feature_id"].isna().any() or frame["feature_id"].astype(str).eq("").any():
            raise ValueError(f"{label} view contains missing feature IDs")
        if frame["feature_id"].duplicated().any():
            raise ValueError(f"{label} view contains duplicate feature IDs")
        if set(frame["dataset"].astype(str)) != {external_dataset}:
            raise ValueError(f"{label} view dataset metadata mismatch")
    semantic_ids = set(semantic["feature_id"].astype(str))
    statistical_ids = set(statistical["feature_id"].astype(str))
    exclusion_rows = [
        {
            "feature_id": feature_id,
            "exclusion_reason": "missing_statistical_view",
        }
        for feature_id in sorted(semantic_ids - statistical_ids)
    ] + [
        {
            "feature_id": feature_id,
            "exclusion_reason": "missing_semantic_view",
        }
        for feature_id in sorted(statistical_ids - semantic_ids)
    ]
    exclusions = pd.DataFrame(
        exclusion_rows, columns=["feature_id", "exclusion_reason"]
    )
    if len(exclusions):
        raise ValueError(
            "external semantic/statistical identities differ; exclusions must be "
            "resolved before projection: "
            + exclusions.to_json(orient="records")
        )
    semantic_ordered = semantic.assign(
        feature_id=semantic["feature_id"].astype(str)
    ).sort_values("feature_id", kind="mergesort").reset_index(drop=True)
    statistical_ordered = statistical.assign(
        feature_id=statistical["feature_id"].astype(str)
    ).sort_values("feature_id", kind="mergesort").reset_index(drop=True)
    if semantic_ordered["feature_id"].tolist() != statistical_ordered["feature_id"].tolist():
        raise ValueError("external feature identity alignment failed")
    if not semantic_ordered["feature_name"].astype(str).equals(
        statistical_ordered["feature_name"].astype(str)
    ):
        raise ValueError("external feature names conflict after feature-ID join")
    identity = semantic_ordered[["feature_id", "feature_name", "dataset"]].copy()
    manifest = {
        "alignment_version": "feature_id_one_to_one_v1",
        "external_dataset": external_dataset,
        "semantic_input_hash": semantic_hash,
        "statistical_input_hash": statistical_hash,
        "aligned_feature_count": len(identity),
        "excluded_feature_count": 0,
        "semantic_feature_ids_equal_statistical_feature_ids": True,
        "joined_identity_hash": sha256_text(identity.to_csv(index=False)),
    }
    manifest["alignment_manifest_hash"] = manifest_hash(manifest)
    return semantic_ordered, statistical_ordered, exclusions, manifest


def aggregate_seed_embeddings(
    seed_embeddings: Mapping[int, pd.DataFrame],
    *,
    seed_list: Iterable[int],
    reference_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seeds = [int(seed) for seed in seed_list]
    if seeds != [11, 22, 33, 44, 55]:
        raise ValueError("reverse-transfer consensus requires fixed seeds 11,22,33,44,55")
    if set(seed_embeddings) != set(seeds) or reference_seed not in seed_embeddings:
        raise ValueError("seed embedding set is incomplete")
    reference = seed_embeddings[reference_seed].sort_values("feature_id").reset_index(drop=True)
    columns = _vector_columns(reference, "joint_")
    reference_ids = reference["feature_id"].astype(str).tolist()
    ref_values = _row_normalize(reference[columns].to_numpy(dtype=float))
    aligned_values = []
    score_frames = []
    for seed in seeds:
        current = seed_embeddings[seed].sort_values("feature_id").reset_index(drop=True)
        if current["feature_id"].astype(str).tolist() != reference_ids:
            raise ValueError("seed feature identities are not aligned")
        values = _row_normalize(current[columns].to_numpy(dtype=float))
        if seed != reference_seed:
            left, _, right = np.linalg.svd(values.T @ ref_values, full_matrices=False)
            values = values @ (left @ right)
        aligned_values.append(_row_normalize(values))
        if "learned_similarity" in current:
            score_frames.append(current["learned_similarity"].to_numpy(dtype=float))
    consensus = _row_normalize(np.mean(np.stack(aligned_values), axis=0))
    output = reference[["feature_id", "feature_name", "dataset"]].copy()
    output = pd.concat(
        [
            output,
            pd.DataFrame(consensus, columns=columns),
        ],
        axis=1,
    )
    if score_frames:
        output["consensus_score"] = np.mean(np.stack(score_frames), axis=0)
        output["consensus_rank"] = (
            output["consensus_score"].rank(method="first", ascending=False).astype(int)
        )
    manifest = {
        "seed_list": seeds,
        "reference_seed": int(reference_seed),
        "alignment_method": "orthogonal_procrustes_svd",
        "embedding_aggregation": "l2_normalize_align_mean_l2_normalize",
        "score_aggregation": "arithmetic_mean_all_fixed_seeds",
        "rank_aggregation": "rank_of_consensus_score",
    }
    return output, manifest


def fixed_candidate_pool(
    ranking: pd.DataFrame,
    *,
    model: str,
    pool_size: int,
    final_budget: int,
) -> pd.DataFrame:
    if pool_size < final_budget:
        raise ValueError("candidate pool is smaller than final feature budget")
    rank_column = (
        "consensus_clip_rank"
        if "consensus_clip_rank" in ranking
        else "consensus_rank"
        if "consensus_rank" in ranking
        else "learned_rank"
    )
    required = {"feature_id", "feature_name", rank_column}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"ranking missing columns: {sorted(missing)}")
    ordered = ranking.sort_values(
        [rank_column, "feature_id"], kind="mergesort"
    ).head(int(pool_size)).copy()
    if len(ordered) != pool_size:
        raise ValueError(f"configured pool requires {pool_size} eligible features")
    ordered["model"] = model
    ordered["candidate_pool_size"] = int(pool_size)
    ordered["final_feature_budget"] = int(final_budget)
    ordered["candidate_pool_frozen_before_mrmr"] = True
    ordered["pairing_policy_version"] = PAIRING_POLICY_VERSION
    return ordered


def build_prediction_frame(
    *,
    stable_row_ids: Iterable[Any],
    target: Iterable[Any],
    probability: Iterable[float],
    dataset: str,
    split: str,
    run_id: str,
    model: str,
    data_manifest_hash: str,
    configuration_hash: str,
    threshold: float,
    source_training_dataset: str = SOURCE_DATASET,
    external_dataset: str = EXTERNAL_DATASET,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "stable_row_id": list(stable_row_ids),
            "target": list(target),
            "prediction_probability": list(probability),
        }
    )
    if frame["stable_row_id"].isna().any() or frame["stable_row_id"].duplicated().any():
        raise ValueError(f"{split}: stable row IDs must be present and unique")
    frame["dataset"] = dataset
    frame["split"] = split
    frame["predicted_class"] = (
        frame["prediction_probability"].astype(float) >= float(threshold)
    ).astype(int)
    frame["run_id"] = run_id
    frame["method"] = REVERSE_METHOD
    frame["model"] = model
    frame["source_training_dataset"] = source_training_dataset
    frame["external_dataset"] = external_dataset
    frame["data_manifest_hash"] = data_manifest_hash
    frame["configuration_hash"] = configuration_hash
    frame["pairing_policy_version"] = PAIRING_POLICY_VERSION
    frame["fit_scope"] = f"{external_dataset}_dev_model_fit"
    return frame[PREDICTION_COLUMNS]


def validate_prediction_splits(dev: pd.DataFrame, oot: pd.DataFrame) -> None:
    for split_name, frame in (("dev", dev), ("oot", oot)):
        missing = set(PREDICTION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name} prediction provenance missing: {sorted(missing)}")
        if frame["stable_row_id"].duplicated().any():
            raise ValueError(f"{split_name} stable row IDs are not unique")
    overlap = set(dev["stable_row_id"].astype(str)) & set(oot["stable_row_id"].astype(str))
    if overlap:
        raise ValueError("DEV/OOT stable row ID overlap detected")


def checkpoint_provenance(
    *,
    source_dataset: str,
    configuration_hash: str,
    data_manifest_hash: str,
    statistical_preprocessor_hash: str,
    source_anchor_hash: str,
) -> dict[str, str]:
    return {
        "source_dataset": source_dataset,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "configuration_hash": configuration_hash,
        "data_manifest_hash": data_manifest_hash,
        "statistical_preprocessor_hash": statistical_preprocessor_hash,
        "source_anchor_hash": source_anchor_hash,
    }


def validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"checkpoint incompatible for {key}: expected {value!r}, "
                f"observed {manifest.get(key)!r}"
            )
    if manifest.get("pairing_policy_version") != PAIRING_POLICY_VERSION:
        raise ValueError("old invalid checkpoint pairing policy")


def _artifact_identity_diagnostic(
    frame: pd.DataFrame,
    *,
    invariant: str,
    key_name: str,
    key_value: str,
) -> str:
    def values(column: str) -> list[str]:
        if column not in frame:
            return []
        result = {
            str(value)
            for value in frame[column]
            if canonical_registry_value(column, value) != "<NULL>"
        }
        return sorted(result)

    return (
        f"artifact identity conflict; invariant={invariant}; "
        f"{key_name}={key_value!r}; "
        f"artifact_ids={values('artifact_id')}; "
        f"canonical_paths={values('relative_path')}; "
        f"hashes={values('file_hash')}; "
        f"owning_run_ids={values('_logical_owner')}; "
        f"artifact_types={values('artifact_type')}; "
        f"origins={values('_registry_origin')}"
    )


def validate_artifact_identity(
    frame: pd.DataFrame,
    *,
    origins: Iterable[str] | None = None,
) -> None:
    """Enforce stable bidirectional artifact ID/path identity."""

    if frame.empty:
        return
    required = {"artifact_id", "relative_path", "file_hash", "artifact_type"}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"artifact identity fields are missing: {sorted(missing)}"
        )
    checked = frame.copy()
    checked["artifact_id"] = checked["artifact_id"].map(
        lambda value: canonical_registry_value("artifact_id", value)
    )
    checked["relative_path"] = checked["relative_path"].map(
        lambda value: canonical_registry_value(
            "relative_path", value, expected_type="path"
        )
    )
    checked["file_hash"] = checked["file_hash"].map(
        lambda value: str(value).strip().lower()
    )
    checked["artifact_type"] = checked["artifact_type"].map(
        lambda value: canonical_registry_value("artifact_type", value)
    )
    if origins is None:
        checked["_registry_origin"] = "registry"
    else:
        origin_values = list(origins)
        if len(origin_values) != len(checked):
            raise ValueError("artifact identity origin count mismatch")
        checked["_registry_origin"] = origin_values

    owners = (
        checked["created_by_run_id"]
        if "created_by_run_id" in checked
        else pd.Series([""] * len(checked), index=checked.index)
    )
    reuse = (
        checked["reuse_status"]
        if "reuse_status" in checked
        else pd.Series([""] * len(checked), index=checked.index)
    )
    checked["_logical_owner"] = [
        (
            "REUSABLE"
            if canonical_registry_value("reuse_status", status)
            == "reusable_existing"
            and canonical_registry_value("created_by_run_id", owner) == "<NULL>"
            else canonical_registry_value("created_by_run_id", owner)
        )
        for owner, status in zip(owners, reuse)
    ]

    id_consistency_fields = {
        "relative_path": "artifact_id->canonical_path",
        "file_hash": "artifact_id->file_hash",
        "artifact_type": "artifact_id->artifact_type",
        "_logical_owner": "artifact_id->owning_run_id",
        "dataset": "artifact_id->dataset",
        "dataset_name": "artifact_id->dataset",
        "model": "artifact_id->model",
        "method": "artifact_id->method",
        "configuration_hash": "artifact_id->configuration_hash",
        "config_hash": "artifact_id->configuration_hash",
        "data_manifest_hash": "artifact_id->data_manifest_hash",
        "pairing_policy_version": "artifact_id->pairing_policy_version",
        "scientific_stage": "artifact_id->scientific_stage",
        "stage": "artifact_id->scientific_stage",
    }
    for artifact_id, group in checked.groupby("artifact_id", dropna=False):
        for field, invariant in id_consistency_fields.items():
            if field not in group:
                continue
            observed = {
                canonical_registry_value(field, value)
                for value in group[field]
                if canonical_registry_value(field, value) != "<NULL>"
            }
            if len(observed) > 1:
                raise RegistryConflictError(
                    _artifact_identity_diagnostic(
                        group,
                        invariant=invariant,
                        key_name="artifact_id",
                        key_value=str(artifact_id),
                    )
                )

    path_consistency_fields = {
        "artifact_id": "canonical_path->artifact_id",
        "file_hash": "canonical_path->file_hash",
        "artifact_type": "canonical_path->artifact_type",
        "_logical_owner": "canonical_path->owning_run_id",
    }
    for canonical_path, group in checked.groupby("relative_path", dropna=False):
        for field, invariant in path_consistency_fields.items():
            observed = {
                canonical_registry_value(field, value)
                for value in group[field]
                if canonical_registry_value(field, value) != "<NULL>"
            }
            if len(observed) > 1:
                raise RegistryConflictError(
                    _artifact_identity_diagnostic(
                        group,
                        invariant=invariant,
                        key_name="canonical_path",
                        key_value=str(canonical_path),
                    )
                )


def append_registry_rows(
    *,
    registry_path: str | Path,
    rows: pd.DataFrame,
    equivalence_columns: Iterable[str],
    schema: RegistrySchema | None = None,
) -> pd.DataFrame:
    path = Path(registry_path)
    schema = schema or REGISTRY_SCHEMAS.get(path.name)
    if schema is None:
        raise ValueError(f"no explicit registry schema for {path.name}")
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    incoming = rows.copy()
    if "result_origin" in incoming:
        incoming["result_origin"] = "newly_executed"
    keys = list(equivalence_columns)
    missing = set(keys) - set(incoming.columns)
    if missing:
        raise ValueError(f"registry rows missing equivalence fields: {sorted(missing)}")
    validate_registry_frame(
        existing, schema=schema, strict=False, origin="current"
    )
    incoming = validate_registry_frame(
        incoming, schema=schema, strict=True, origin="proposed"
    )
    if path.name == "artifact_registry.csv" and not existing.empty:
        validate_artifact_identity(
            pd.concat([existing, incoming], ignore_index=True, sort=False),
            origins=["current"] * len(existing) + ["proposed"] * len(incoming),
        )
    if existing.empty:
        combined = incoming
    else:
        existing_by_key = {
            tuple(canonical_registry_value(column, row[column]) for column in keys): row
            for _, row in existing.iterrows()
        }
        keep = []
        for _, row in incoming.iterrows():
            key = tuple(canonical_registry_value(column, row[column]) for column in keys)
            if key not in existing_by_key:
                keep.append(True)
                continue
            prior = existing_by_key[key]
            shared = [
                column
                for column in incoming.columns
                if column in existing.columns and column not in keys
            ]
            conflicts = [
                column
                for column in shared
                if canonical_registry_value(column, prior.get(column))
                != canonical_registry_value(column, row.get(column))
            ]
            if conflicts:
                raise RegistryConflictError(
                    f"conflicting existing registry key {key}: {conflicts}"
                )
            keep.append(False)
        combined = pd.concat([existing, incoming.loc[keep]], ignore_index=True, sort=False)
    combined.attrs["registry_changed"] = bool(existing.empty or len(combined) > len(existing))
    validate_registry_frame(
        combined, schema=schema, strict=False, origin="combined"
    )
    return combined


def validate_registry_frame(
    frame: pd.DataFrame,
    *,
    schema: RegistrySchema,
    strict: bool,
    origin: str = "registry",
) -> pd.DataFrame:
    if frame.empty:
        if strict:
            raise ValueError("new registry rows are empty")
        return frame.copy()
    required_columns = schema.required if strict else frozenset(schema.primary_key)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"registry schema missing required columns: {sorted(missing)}")
    normalized = frame.copy()
    for column in normalized.columns:
        if column in schema.boolean_fields:
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(column, value, expected_type="boolean")
            )
        elif column in schema.integer_fields:
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(column, value, expected_type="integer")
            )
        elif column in schema.float_fields:
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(column, value, expected_type="float")
            )
        elif column in schema.path_fields:
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(column, value, expected_type="path")
            )
        elif column in schema.hash_fields and strict:
            normalized[column] = normalized[column].map(
                lambda value: validate_sha256(value, field=column)
            )
        elif column in schema.json_fields:
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(column, value, expected_type="json")
            )
            if strict and column in {"source_checkpoint_hashes", "source_anchor_hashes"}:
                for payload in normalized[column]:
                    parsed = json.loads(payload)
                    if not isinstance(parsed, dict) or set(parsed) != {"11", "22", "33", "44", "55"}:
                        raise ValueError(f"{column} must contain exactly five seed hashes")
                    for seed, hash_value in parsed.items():
                        validate_sha256(hash_value, field=f"{column}.{seed}")
        elif schema.enum_fields and column in schema.enum_fields:
            allowed = schema.enum_fields[column]
            normalized[column] = normalized[column].map(
                lambda value: canonical_registry_value(
                    column, value, expected_type="enum", allowed=allowed
                )
            )
    if schema is REGISTRY_SCHEMAS["artifact_registry.csv"]:
        validate_artifact_identity(
            normalized, origins=[origin] * len(normalized)
        )
    keys = normalized[list(schema.primary_key)].apply(
        lambda row: tuple(
            canonical_registry_value(column, row[column])
            for column in schema.primary_key
        ),
        axis=1,
    )
    if keys.duplicated().any():
        raise ValueError(f"registry primary key is not unique: {schema.primary_key}")
    if strict:
        for required in schema.required:
            if normalized[required].map(lambda value: canonical_registry_value(required, value)).eq("<NULL>").any():
                raise ValueError(f"required registry field is empty: {required}")
    return normalized


def canonical_registry_value(
    column: str,
    value: Any,
    *,
    expected_type: str | None = None,
    allowed: frozenset[str] | None = None,
) -> str:
    """Normalize CSV round-trip type drift without weakening scientific identity."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "<NULL>"
    name = str(column).strip().lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = str(value).strip()
    if not text:
        return "<NULL>"
    if expected_type == "json" or text[:1] in {"{", "["}:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if expected_type == "json":
                raise ValueError(f"registry field {column} contains malformed JSON")
        else:
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    boolean_columns = {
        "file_exists",
        "depends_on_clip",
        "depends_on_old_pairing",
        "llm_shared_ranking_enabled",
    }
    if expected_type == "boolean" or name in boolean_columns:
        if text.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(f"registry field {column} is not boolean")
        return "true" if text.lower() in {"true", "1"} else "false"
    numeric_tokens = (
        "seed",
        "budget",
        "count",
        "seconds",
        "auc",
        "gini",
        "ks",
        "loss",
        "brier",
        "psi",
    )
    if expected_type in {"integer", "float"} or isinstance(value, (int, float, np.integer, np.floating)) or any(
        token in name for token in numeric_tokens
    ):
        try:
            number = Decimal(text)
            if number.is_finite():
                if expected_type == "integer" and number != number.to_integral_value():
                    raise ValueError(f"registry field {column} must be integral")
                normalized = format(number.normalize(), "f")
                return "0" if normalized in {"-0", ""} else normalized
        except InvalidOperation:
            if expected_type in {"integer", "float"}:
                raise ValueError(f"registry field {column} is not finite numeric")
    if expected_type == "path" or "path" in name or name.endswith("_folder"):
        normalized_path = text.replace("\\", "/")
        while normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]
        candidate = Path(normalized_path)
        if candidate.is_absolute():
            try:
                normalized_path = candidate.resolve().relative_to(
                    Path.cwd().resolve()
                ).as_posix()
            except ValueError:
                raise ValueError(f"registry path escapes repository root: {column}")
        else:
            if ".." in candidate.parts:
                raise ValueError(
                    f"registry path contains unresolved parent traversal: {column}"
                )
            normalized_path = Path(
                *(part for part in candidate.parts if part not in {"", "."})
            ).as_posix()
        if normalized_path in {"", "."}:
            raise ValueError(f"registry path is empty after normalization: {column}")
        return normalized_path.rstrip("/")
    if "hash" in name:
        return validate_sha256(text, field=column)
    if "timestamp" in name or name.endswith("_at"):
        try:
            timestamp = pd.Timestamp(text)
        except (TypeError, ValueError):
            return text
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    enum_columns = {
        "dataset",
        "dataset_name",
        "model",
        "selector",
        "experiment_type",
        "method",
        "artifact_type",
        "result_origin",
        "reuse_status",
        "source_training_dataset",
        "external_dataset",
    }
    if expected_type == "enum" or name in enum_columns:
        normalized_enum = text.lower()
        if allowed is not None and normalized_enum not in allowed:
            raise ValueError(f"registry field {column} has invalid enum {text!r}")
        return normalized_enum
    return text


RegistryFailureInjector = Callable[[str, Mapping[str, Any]], None]


def _inject_registry_failure(
    injector: RegistryFailureInjector | None,
    point: str,
    **context: Any,
) -> None:
    if injector is not None:
        injector(point, context)


def _validate_registry_payload_content(
    target: Path, content_path: Path
) -> None:
    if target.name in REGISTRY_SCHEMAS:
        frame = pd.read_csv(io.BytesIO(content_path.read_bytes()))
        validate_registry_frame(
            frame,
            schema=REGISTRY_SCHEMAS[target.name],
            strict=False,
            origin="staged",
        )
    elif target.name == "summary_manifest.json":
        validate_summary_manifest(
            json.loads(content_path.read_text(encoding="utf-8"))
        )


def _restore_registry_path(path: Path, original: bytes | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{path.name}.rollback.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    rollback = Path(name)
    try:
        rollback.write_bytes(original)
        os.replace(rollback, path)
    finally:
        rollback.unlink(missing_ok=True)


def atomic_registry_transaction(
    payloads: Mapping[str | Path, bytes],
    *,
    transaction_manifest_path: str | Path,
    metadata: Mapping[str, Any],
    failure_injector: RegistryFailureInjector | None = None,
) -> dict[str, Any]:
    """Atomically replace registry files; manifest replacement is the commit point."""

    if not payloads:
        raise ValueError("registry transaction has no files")
    targets = [Path(path) for path in payloads]
    content_by_target = dict(zip(targets, payloads.values()))
    if len({str(path.resolve()) for path in targets}) != len(targets):
        raise ValueError("registry transaction contains duplicate targets")
    manifest_path = Path(transaction_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("registry transaction lock is already held") from exc
    os.close(lock_fd)

    originals: dict[Path, bytes | None] = {}
    original_hashes: dict[Path, str | None] = {}
    temp_paths: dict[Path, Path] = {}
    temp_manifest: Path | None = None
    originals_captured = False
    committed = False
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    rollback_failure: BaseException | None = None
    cleanup_failures: list[str] = []
    try:
        _inject_registry_failure(
            failure_injector,
            "after_lock_acquired",
            lock_path=str(lock_path),
        )
        for path in [*targets, manifest_path]:
            original = path.read_bytes() if path.exists() else None
            originals[path] = original
            original_hashes[path] = (
                hashlib.sha256(original).hexdigest()
                if original is not None
                else None
            )
        originals_captured = True
        _inject_registry_failure(
            failure_injector,
            "after_originals_captured",
            original_hashes={
                str(path): value for path, value in original_hashes.items()
            },
        )

        if all(
            originals[path] == content_by_target[path] for path in targets
        ):
            if originals[manifest_path] is None:
                raise ValueError(
                    "idempotent registry state lacks a transaction manifest"
                )
            manifest = json.loads(
                originals[manifest_path].decode("utf-8")
            )
            validate_transaction_manifest(manifest)
            manifest["idempotent_noop"] = True
            manifest["transaction_outcome"] = "IDEMPOTENT_NO_OP"
            result = manifest
        else:
            for index, target in enumerate(targets):
                content = content_by_target[target]
                target.parent.mkdir(parents=True, exist_ok=True)
                _inject_registry_failure(
                    failure_injector,
                    "during_temp_create",
                    target=str(target),
                    target_index=index,
                )
                handle, name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    dir=target.parent,
                )
                os.close(handle)
                temp = Path(name)
                temp_paths[target] = temp
                with temp.open("wb") as stream:
                    _inject_registry_failure(
                        failure_injector,
                        "during_temp_write",
                        target=str(target),
                        target_index=index,
                    )
                    stream.write(content)
                    _inject_registry_failure(
                        failure_injector,
                        "during_temp_flush",
                        target=str(target),
                        target_index=index,
                    )
                    stream.flush()
                    _inject_registry_failure(
                        failure_injector,
                        "during_temp_fsync",
                        target=str(target),
                        target_index=index,
                    )
                    os.fsync(stream.fileno())
                _inject_registry_failure(
                    failure_injector,
                    "after_temp_write",
                    target=str(target),
                    target_index=index,
                )
                _inject_registry_failure(
                    failure_injector,
                    "during_temp_schema_validation",
                    target=str(target),
                    target_index=index,
                )
                _validate_registry_payload_content(target, temp)
                _inject_registry_failure(
                    failure_injector,
                    "during_temp_hash_validation",
                    target=str(target),
                    target_index=index,
                )
                if (
                    hashlib.sha256(temp.read_bytes()).hexdigest()
                    != hashlib.sha256(content).hexdigest()
                ):
                    raise IOError(
                        f"temporary registry hash validation failed: {target}"
                    )
                _inject_registry_failure(
                    failure_injector,
                    "after_temp_validation",
                    target=str(target),
                    target_index=index,
                )

            _inject_registry_failure(
                failure_injector,
                "before_first_replace",
                target=str(targets[0]),
            )
            non_summary_targets = [
                target
                for target in targets
                if target.name != "summary_manifest.json"
            ]
            final_registry_target = non_summary_targets[-1]
            for index, target in enumerate(targets):
                os.replace(temp_paths[target], target)
                if index == 0:
                    _inject_registry_failure(
                        failure_injector,
                        "after_first_replace",
                        target=str(target),
                        target_index=index,
                    )
                if (
                    index > 0
                    and target != final_registry_target
                    and target.name != "summary_manifest.json"
                ):
                    _inject_registry_failure(
                        failure_injector,
                        "after_middle_replace",
                        target=str(target),
                        target_index=index,
                    )
                if target == final_registry_target:
                    _inject_registry_failure(
                        failure_injector,
                        "after_final_replace",
                        target=str(target),
                        target_index=index,
                    )
                if target.name == "summary_manifest.json":
                    _inject_registry_failure(
                        failure_injector,
                        "after_summary_replace",
                        target=str(target),
                        target_index=index,
                    )

            _inject_registry_failure(
                failure_injector,
                "during_post_write_schema_validation",
            )
            for target in targets:
                _validate_registry_payload_content(target, target)
            _inject_registry_failure(
                failure_injector,
                "during_post_write_hash_validation",
            )
            for target in targets:
                if (
                    hashlib.sha256(target.read_bytes()).hexdigest()
                    != hashlib.sha256(content_by_target[target]).hexdigest()
                ):
                    raise IOError(
                        f"post-write registry hash validation failed: {target}"
                    )
            _inject_registry_failure(
                failure_injector,
                "after_post_commit_validation",
            )

            _inject_registry_failure(
                failure_injector,
                "during_cleanup",
                phase="pre_commit_staged_file_cleanup",
            )
            for temp in temp_paths.values():
                temp.unlink(missing_ok=True)

            manifest = {
                **dict(metadata),
                "status": "committed",
                "transaction_status": "NEW_TRANSACTION",
                "transaction_outcome": "NEW_TRANSACTION",
                "transaction_version": "atomic_registry_transaction_v2",
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "canonicalization_version": REGISTRY_CANONICALIZATION_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "commit_boundary": (
                    "validated_transaction_manifest_atomic_replacement"
                ),
                "transaction_id": sha256_text(
                    json.dumps(
                        {
                            "targets": sorted(
                                str(path.resolve()) for path in targets
                            ),
                            "post_hashes": {
                                str(path): sha256_file(path)
                                for path in targets
                            },
                            "metadata": dict(metadata),
                        },
                        sort_keys=True,
                        default=str,
                    )
                ),
                "registry_paths": [
                    str(path).replace("\\", "/") for path in targets
                ],
                "original_file_existence": {
                    str(path).replace("\\", "/"): originals[path] is not None
                    for path in targets
                },
                "pre_transaction_hashes": {
                    str(path).replace("\\", "/"): original_hashes[path]
                    for path in targets
                },
                "updated_files": {
                    str(path).replace("\\", "/"): sha256_file(path)
                    for path in targets
                },
            }
            _inject_registry_failure(
                failure_injector,
                "before_transaction_manifest",
            )
            _inject_registry_failure(
                failure_injector,
                "during_transaction_manifest_validation",
                phase="in_memory",
            )
            validate_transaction_manifest(manifest)
            handle, name = tempfile.mkstemp(
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                dir=manifest_path.parent,
            )
            os.close(handle)
            temp_manifest = Path(name)
            _inject_registry_failure(
                failure_injector,
                "during_transaction_manifest_write",
            )
            with temp_manifest.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            _inject_registry_failure(
                failure_injector,
                "during_transaction_manifest_validation",
                phase="persisted",
            )
            persisted_manifest = json.loads(
                temp_manifest.read_text(encoding="utf-8")
            )
            validate_transaction_manifest(persisted_manifest)
            _inject_registry_failure(
                failure_injector,
                "during_transaction_manifest_replace",
            )
            os.replace(temp_manifest, manifest_path)
            temp_manifest = None
            committed = True
            result = manifest
            try:
                _inject_registry_failure(
                    failure_injector,
                    "after_transaction_manifest",
                    commit_boundary_reached=True,
                )
            except Exception as exc:
                cleanup_failures.append(
                    f"post-commit hook: {type(exc).__name__}: {exc}"
                )
    except BaseException as exc:
        failure = exc
        if not committed and originals_captured:
            try:
                for path in [*targets, manifest_path]:
                    original = originals.get(path)
                    current = path.read_bytes() if path.exists() else None
                    if current != original:
                        _restore_registry_path(path, original)
                    restored = path.read_bytes() if path.exists() else None
                    if restored != original:
                        raise RuntimeError(
                            f"registry rollback verification failed: {path}"
                        )
                    expected_hash = original_hashes.get(path)
                    restored_hash = (
                        hashlib.sha256(restored).hexdigest()
                        if restored is not None
                        else None
                    )
                    if restored_hash != expected_hash:
                        raise RuntimeError(
                            f"registry rollback hash verification failed: {path}"
                        )
            except BaseException as exc:
                rollback_failure = exc
    finally:
        for temp in temp_paths.values():
            try:
                temp.unlink(missing_ok=True)
            except Exception as exc:
                cleanup_failures.append(
                    f"temporary cleanup: {type(exc).__name__}: {exc}"
                )
        if temp_manifest is not None:
            try:
                temp_manifest.unlink(missing_ok=True)
            except Exception as exc:
                cleanup_failures.append(
                    f"manifest cleanup: {type(exc).__name__}: {exc}"
                )
        try:
            lock_path.unlink(missing_ok=True)
        except Exception as exc:
            cleanup_failures.append(
                f"lock release: {type(exc).__name__}: {exc}"
            )

    if rollback_failure is not None:
        raise RuntimeError(
            "registry transaction failed with "
            f"{type(failure).__name__}: {failure}; rollback failed with "
            f"{type(rollback_failure).__name__}: {rollback_failure}"
        ) from failure
    if failure is not None:
        if hasattr(failure, "add_note"):
            failure.add_note("registry rollback succeeded with byte verification")
            for cleanup_failure in cleanup_failures:
                failure.add_note(cleanup_failure)
        raise failure
    if result is None:
        raise RuntimeError("registry transaction produced no result")
    if cleanup_failures:
        result = {
            **result,
            "post_commit_warnings": cleanup_failures,
        }
    return result


def validate_registry_bundle(
    frames: Mapping[str, pd.DataFrame],
    *,
    verify_artifacts: bool = False,
    repository_root: str | Path = ".",
    enforced_run_ids: set[str] | None = None,
) -> None:
    normalized = {}
    for name, schema in REGISTRY_SCHEMAS.items():
        if name not in frames:
            raise ValueError(f"registry bundle is missing {name}")
        normalized[name] = validate_registry_frame(
            frames[name], schema=schema, strict=False
        )
    runs = set(normalized["run_index.csv"]["run_id"].astype(str))
    for name in ("reusable_metrics.csv", "selected_feature_registry.csv"):
        missing_runs = set(normalized[name]["run_id"].astype(str)) - runs
        if missing_runs:
            raise ValueError(f"{name} references missing run IDs: {sorted(missing_runs)}")
    artifacts = normalized["artifact_registry.csv"]
    owned = artifacts["created_by_run_id"].dropna().astype(str)
    owned = owned[owned.str.strip().ne("")]
    if set(owned) - runs:
        raise ValueError("artifact registry references missing owning runs")
    path_hashes = {}
    root = Path(repository_root).resolve()
    for row in artifacts.itertuples(index=False):
        path = canonical_registry_value("relative_path", row.relative_path, expected_type="path")
        file_hash = str(row.file_hash).strip().lower()
        if path in path_hashes and path_hashes[path] != file_hash:
            raise ValueError("artifact path has conflicting hashes")
        path_hashes[path] = file_hash
        owner = str(getattr(row, "created_by_run_id", "") or "")
        should_verify = (
            verify_artifacts
            and (enforced_run_ids is None or owner in enforced_run_ids)
            and canonical_registry_value(
                "file_exists", getattr(row, "file_exists", False), expected_type="boolean"
            )
            == "true"
        )
        if should_verify:
            artifact = (root / path).resolve()
            if root not in artifact.parents and artifact != root:
                raise ValueError("artifact path escapes repository")
            if not artifact.exists() or sha256_file(artifact) != validate_sha256(
                file_hash, field="file_hash"
            ):
                raise ValueError(f"artifact hash mismatch: {path}")
    selected = normalized["selected_feature_registry.csv"]
    for row in selected.itertuples(index=False):
        if enforced_run_ids is not None and str(row.run_id) not in enforced_run_ids:
            continue
        path = canonical_registry_value(
            "selected_feature_path", row.selected_feature_path, expected_type="path"
        )
        if path not in path_hashes or path_hashes[path] != str(row.selected_feature_hash).lower():
            raise ValueError("selected-feature artifact reference mismatch")
    run_index = normalized["run_index.csv"].set_index("run_id")
    for row in normalized["reusable_metrics.csv"].itertuples(index=False):
        if enforced_run_ids is not None and str(row.run_id) not in enforced_run_ids:
            continue
        run = run_index.loc[str(row.run_id)]
        if str(row.dataset_name).lower() != str(run.dataset).lower():
            raise ValueError("metric/run dataset mismatch")
        if str(row.model).lower() != str(run.model).lower():
            raise ValueError("metric/run model mismatch")
        if str(row.metric_artifact_path).replace("\\", "/") != str(run.metric_artifact_path).replace("\\", "/"):
            raise ValueError("metric artifact path mismatch")
        if str(row.selector) != str(run.method):
            raise ValueError("metric/run method mismatch")
        if str(row.config_hash).lower() != str(run.configuration_hash).lower():
            raise ValueError("metric/run configuration-hash mismatch")
        if hasattr(row, "data_manifest_hash") and hasattr(
            run, "data_manifest_hash"
        ):
            metric_data_hash = canonical_registry_value(
                "data_manifest_hash", row.data_manifest_hash
            )
            run_data_hash = canonical_registry_value(
                "data_manifest_hash", run.data_manifest_hash
            )
            if (
                metric_data_hash != "<NULL>"
                and run_data_hash != "<NULL>"
                and metric_data_hash != run_data_hash
            ):
                raise ValueError("metric/run data-manifest-hash mismatch")
        if str(row.pairing_policy_version) != str(run.pairing_policy_version):
            raise ValueError("metric/run pairing-policy mismatch")
        if hasattr(row, "dev_prediction_hash") and hasattr(
            row, "oot_prediction_hash"
        ):
            artifact_hashes = set(
                artifacts["file_hash"].astype(str).str.lower()
            )
            dev_hash = canonical_registry_value(
                "dev_prediction_hash", row.dev_prediction_hash
            )
            oot_hash = canonical_registry_value(
                "oot_prediction_hash", row.oot_prediction_hash
            )
            if (
                dev_hash != "<NULL>"
                and oot_hash != "<NULL>"
                and (
                    dev_hash.lower() not in artifact_hashes
                    or oot_hash.lower() not in artifact_hashes
                )
            ):
                raise ValueError(
                    "metric prediction hash has no artifact reference"
                )
    for row in selected.itertuples(index=False):
        if enforced_run_ids is not None and str(row.run_id) not in enforced_run_ids:
            continue
        run = run_index.loc[str(row.run_id)]
        if str(row.model).lower() != str(run.model).lower():
            raise ValueError("selection/run model mismatch")
        if hasattr(row, "configuration_hash"):
            selection_config = canonical_registry_value(
                "configuration_hash", row.configuration_hash
            )
            if (
                selection_config != "<NULL>"
                and selection_config.lower()
                != str(run.configuration_hash).lower()
            ):
                raise ValueError(
                    "selection/run configuration-hash mismatch"
                )
        if hasattr(row, "data_manifest_hash") and hasattr(
            run, "data_manifest_hash"
        ):
            selection_data_hash = canonical_registry_value(
                "data_manifest_hash", row.data_manifest_hash
            )
            run_data_hash = canonical_registry_value(
                "data_manifest_hash", run.data_manifest_hash
            )
            if (
                selection_data_hash != "<NULL>"
                and run_data_hash != "<NULL>"
                and selection_data_hash != run_data_hash
            ):
                raise ValueError(
                    "selection/run data-manifest-hash mismatch"
                )


def registry_bundle_dry_run(
    frames: Mapping[str, pd.DataFrame],
    *,
    verify_artifacts: bool = False,
    repository_root: str | Path = ".",
    enforced_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a proposed bundle without touching active registries."""

    try:
        validate_registry_bundle(
            frames,
            verify_artifacts=verify_artifacts,
            repository_root=repository_root,
            enforced_run_ids=enforced_run_ids,
        )
    except RegistryConflictError as exc:
        return {
            "transaction_outcome": "CONFLICT",
            "conflict_diagnostic": str(exc),
            "affected_active_files": [],
            "writes_performed": False,
            "success_transaction_manifest_written": False,
        }
    return {
        "transaction_outcome": "NEW_TRANSACTION",
        "conflict_diagnostic": "",
        "affected_active_files": [],
        "writes_performed": False,
        "success_transaction_manifest_written": False,
    }


def validate_summary_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "registry_version",
        "run_counts",
        "artifact_counts",
        "registry_file_hashes",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"summary manifest schema missing: {sorted(missing)}")
    if not isinstance(manifest["run_counts"], dict) or not isinstance(
        manifest["artifact_counts"], dict
    ):
        raise ValueError("summary manifest counts must be JSON objects")
    if not isinstance(manifest["registry_file_hashes"], dict):
        raise ValueError("summary manifest registry hashes must be a JSON object")
    for path, hash_value in manifest["registry_file_hashes"].items():
        canonical_registry_value("registry_path", path, expected_type="path")
        validate_sha256(hash_value, field=f"registry_file_hashes[{path}]")


def build_summary_manifest(
    base_manifest: Mapping[str, Any],
    *,
    registry_root: str | Path,
    payloads: Mapping[str | Path, bytes],
) -> dict[str, Any]:
    """Deterministically refresh summary counts and hashes from final registry bytes."""
    root = Path(registry_root)
    content_by_name = {Path(path).name: content for path, content in payloads.items()}
    table_names = (
        "run_index.csv",
        "artifact_registry.csv",
        "reusable_metrics.csv",
        "selected_feature_registry.csv",
    )
    required_names = {*table_names, "results_access_guide.md"}
    missing = required_names - set(content_by_name)
    if missing:
        raise ValueError(f"summary inputs missing: {sorted(missing)}")
    frames = {
        name: pd.read_csv(io.BytesIO(content_by_name[name]))
        for name in table_names
    }

    def status_counts(
        frame: pd.DataFrame, existing: Mapping[str, Any]
    ) -> dict[str, int]:
        if "reuse_status" not in frame:
            raise ValueError("summary source table lacks reuse_status")
        observed = frame["reuse_status"].fillna("").astype(str).value_counts().to_dict()
        ordered_keys = [key for key in existing if key != "total"]
        ordered_keys.extend(sorted(set(observed) - set(ordered_keys)))
        counts = {key: int(observed.get(key, 0)) for key in ordered_keys}
        counts["total"] = int(len(frame))
        return counts

    summary = dict(base_manifest)
    summary["run_counts"] = status_counts(
        frames["run_index.csv"], base_manifest.get("run_counts", {})
    )
    summary["artifact_counts"] = status_counts(
        frames["artifact_registry.csv"], base_manifest.get("artifact_counts", {})
    )
    summary["reusable_metric_rows"] = int(len(frames["reusable_metrics.csv"]))
    summary["selected_feature_artifact_rows"] = int(
        len(frames["selected_feature_registry.csv"])
    )
    summary["registry_file_hashes"] = {
        (root / name).as_posix(): hashlib.sha256(content_by_name[name]).hexdigest()
        for name in (*table_names, "results_access_guide.md")
    }
    validate_summary_manifest(summary)
    return summary


def validate_summary_manifest_payloads(
    manifest: Mapping[str, Any],
    *,
    registry_root: str | Path,
    payloads: Mapping[str | Path, bytes],
) -> None:
    """Reject stale counts or hashes against a final registry payload set."""
    expected = build_summary_manifest(
        manifest,
        registry_root=registry_root,
        payloads=payloads,
    )
    fields = (
        "run_counts",
        "artifact_counts",
        "reusable_metric_rows",
        "selected_feature_artifact_rows",
        "registry_file_hashes",
    )
    stale = [field for field in fields if manifest.get(field) != expected.get(field)]
    if stale:
        raise ValueError(f"summary manifest is stale: {stale}")


def validate_transaction_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "transaction_id",
        "transaction_status",
        "timestamp",
        "registry_paths",
        "pre_transaction_hashes",
        "updated_files",
        "canonicalization_version",
        "schema_version",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"transaction manifest schema missing: {sorted(missing)}")
    validate_sha256(manifest["transaction_id"], field="transaction_id")
    if manifest["transaction_status"] != "NEW_TRANSACTION":
        raise ValueError("transaction manifest has invalid status")
    pd.Timestamp(manifest["timestamp"])
    for hash_value in manifest["pre_transaction_hashes"].values():
        if hash_value is not None:
            validate_sha256(hash_value, field="pre_transaction_hash")
    for hash_value in manifest["updated_files"].values():
        validate_sha256(hash_value, field="post_transaction_hash")
    if manifest.get("transaction_version") == "atomic_registry_transaction_v2":
        v2_required = {"original_file_existence", "commit_boundary"}
        missing_v2 = v2_required - set(manifest)
        if missing_v2:
            raise ValueError(
                f"transaction v2 manifest schema missing: {sorted(missing_v2)}"
            )
        if (
            manifest["commit_boundary"]
            != "validated_transaction_manifest_atomic_replacement"
        ):
            raise ValueError("transaction manifest commit boundary is invalid")
        if set(manifest["original_file_existence"]) != set(
            manifest["registry_paths"]
        ):
            raise ValueError(
                "transaction manifest original-existence paths mismatch"
            )
        if not all(
            isinstance(value, bool)
            for value in manifest["original_file_existence"].values()
        ):
            raise ValueError(
                "transaction manifest original-existence values are invalid"
            )


def implementation_contract(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    return {
        "status": "implementation_only_not_executed",
        "output_root": str(root).replace("\\", "/"),
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "source_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "required_stages": ["prepare", "train", "project", "evaluate", "register"],
        "scientific_outputs": {
            "feature_reconciliation": str(root / "feature_universe/feature_reconciliation.csv"),
            "source_train_pairs": str(
                root / "pairing/lendingclub_v2_train_positive_pairs.parquet"
            ),
            "source_validation_pairs": str(
                root / "pairing/lendingclub_v2_validation_positive_pairs.parquet"
            ),
            "identity_evidence": str(
                root / "pairing/identity_evidence_manifest.json"
            ),
            "training": str(root / "training/seeds"),
            "source_anchor": str(root / "source_anchor/source_anchor_manifest.json"),
            "reverse_embeddings": str(
                root / "reverse_projection/homecredit_reverse_embeddings.parquet"
            ),
            "reverse_scores": str(
                root / "reverse_projection/homecredit_reverse_scores.csv"
            ),
            "reverse_reconciliation": str(
                root
                / "reverse_projection/homecredit_reverse_feature_reconciliation.csv"
            ),
            "projection_manifest": str(
                root / "reverse_projection/reverse_projection_manifest.json"
            ),
            "alignment_manifest": str(
                root / "reverse_projection/alignment_manifest.json"
            ),
            "candidate_pools": str(root / "candidate_pools"),
            "downstream": str(root / "downstream"),
        },
    }


def manifest_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def file_manifest(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return {
        "path": str(value).replace("\\", "/"),
        "exists": value.exists(),
        "sha256": sha256_file(value) if value.exists() and value.is_file() else None,
    }


def _reject_forbidden_columns(columns: Iterable[Any]) -> None:
    bad = [
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_REPRESENTATION_TOKENS)
    ]
    if bad:
        raise ValueError(f"forbidden target/OOT representation columns: {bad}")


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame:
        return frame[column].fillna(default)
    return pd.Series([default] * len(frame), index=frame.index)


def _vector_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = sorted(column for column in frame.columns if str(column).startswith(prefix))
    if not columns:
        raise ValueError(f"no {prefix} vector columns")
    return columns


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if bool((norms <= 0).any()):
        raise ValueError("embedding contains zero-norm rows")
    return values / norms
