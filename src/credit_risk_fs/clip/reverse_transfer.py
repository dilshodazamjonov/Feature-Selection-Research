from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def append_registry_rows(
    *,
    registry_path: str | Path,
    rows: pd.DataFrame,
    equivalence_columns: Iterable[str],
) -> pd.DataFrame:
    path = Path(registry_path)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    incoming = rows.copy()
    if "result_origin" in incoming:
        incoming["result_origin"] = "newly_executed"
    keys = list(equivalence_columns)
    missing = set(keys) - set(incoming.columns)
    if missing:
        raise ValueError(f"registry rows missing equivalence fields: {sorted(missing)}")
    if existing.empty:
        combined = incoming
    else:
        existing_keys = {
            tuple(str(row[column]) for column in keys)
            for _, row in existing.iterrows()
        }
        keep = [
            tuple(str(row[column]) for column in keys) not in existing_keys
            for _, row in incoming.iterrows()
        ]
        combined = pd.concat([existing, incoming.loc[keep]], ignore_index=True, sort=False)
    return combined


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
            "source_pairs": str(root / "pairing/lendingclub_v2_positive_pairs.parquet"),
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
