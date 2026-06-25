from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.utils.hashing import sha256_text


NEGATIVE_POLICY_VERSION = "identity_equivalence_v2"
MASK_PRODUCING_REASONS = {
    "verified_alias",
    "exact_dev_duplicate",
    "documented_identity_transform",
}


@dataclass(frozen=True)
class NegativePolicyResult:
    exclusion_pairs: pd.DataFrame
    candidate_audit: pd.DataFrame
    near_duplicate_audit: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    manifest: dict[str, Any]


def build_negative_policy(
    *,
    train_pairs: pd.DataFrame,
    all_homecredit_pairs: pd.DataFrame,
    text_embeddings: pd.DataFrame | None = None,
    exact_dev_duplicates: pd.DataFrame | None = None,
    verified_aliases: Iterable[Iterable[str]] = (),
    documented_identity_transforms: Iterable[Iterable[str]] = (),
    near_duplicate_text_threshold: float = 0.95,
    threshold_diagnostics: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99),
    min_safe_negative_count: int = 25,
    float_tolerance: float = 1e-8,
) -> NegativePolicyResult:
    train = train_pairs.copy().reset_index(drop=True)
    _validate_training_boundary(train, all_homecredit_pairs)
    features = train["feature_name"].astype(str).tolist()
    feature_set = set(features)
    order_hash = feature_order_hash(features)

    diagnostic_pairs, near_audit, sensitivity = _diagnostic_relations(
        train=train,
        text_embeddings=text_embeddings,
        threshold=float(near_duplicate_text_threshold),
        thresholds=threshold_diagnostics,
        tolerance=float(float_tolerance),
    )
    exclusions: list[dict[str, Any]] = []
    exclusions.extend(
        _explicit_identity_rows(
            train,
            verified_aliases,
            reason="verified_alias",
            evidence="explicitly verified duplicate metadata/source alias",
            order_hash=order_hash,
        )
    )
    exclusions.extend(
        _explicit_identity_rows(
            train,
            documented_identity_transforms,
            reason="documented_identity_transform",
            evidence="documented deterministic identity-preserving transformation",
            order_hash=order_hash,
        )
    )
    exclusions.extend(_exact_duplicate_rows(train, exact_dev_duplicates, order_hash=order_hash))
    exclusion_frame = _exclusion_frame(exclusions)

    exclusion_sets = {feature: set() for feature in features}
    for row in exclusion_frame.to_dict("records"):
        exclusion_sets[str(row["anchor_feature_name"])].add(str(row["excluded_feature_name"]))
    audit_rows = []
    for feature in features:
        relation_counts = {
            relation: len(values.get(feature, set()))
            for relation, values in diagnostic_pairs.items()
        }
        remaining = len(features) - 1 - len(exclusion_sets[feature])
        audit_rows.append(
            {
                "feature_name": feature,
                "candidate_negative_count": len(features) - 1,
                "excluded_verified_alias_count": _reason_count(exclusion_frame, feature, "verified_alias"),
                "excluded_exact_dev_duplicate_count": _reason_count(exclusion_frame, feature, "exact_dev_duplicate"),
                "excluded_documented_identity_transform_count": _reason_count(
                    exclusion_frame, feature, "documented_identity_transform"
                ),
                "diagnostic_same_family_count": relation_counts["diagnostic_same_family"],
                "diagnostic_text_similarity_count": relation_counts["diagnostic_text_similarity"],
                "diagnostic_statistical_similarity_count": relation_counts["diagnostic_statistical_similarity"],
                "same_source_table_count": relation_counts["same_source_table"],
                "remaining_safe_negative_count": remaining,
                "warning": "safe_negative_count_below_threshold" if remaining < min_safe_negative_count else "",
            }
        )
    audit_frame = pd.DataFrame(audit_rows).sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    manifest = {
        "policy_version": NEGATIVE_POLICY_VERSION,
        "policy": "all in-batch Home Credit train features are negatives unless verified identity-equivalent",
        "mask_producing_relations": [
            "same_feature",
            "verified_alias",
            "exact_dev_duplicate",
            "documented_identity_transform",
        ],
        "diagnostic_only_relations": [
            "diagnostic_same_family",
            "diagnostic_text_similarity",
            "diagnostic_statistical_similarity",
            "same_source_table",
        ],
        "same_feature_implemented_by_diagonal_positive": True,
        "explicit_hard_negatives_enabled": False,
        "cross_dataset_negatives_enabled": False,
        "validation_as_training_negative": False,
        "near_duplicate_text_threshold": float(near_duplicate_text_threshold),
        "float_tolerance": float(float_tolerance),
        "excluded_reason_counts": exclusion_frame["exclusion_reason"].value_counts().sort_index().to_dict(),
        "diagnostic_relation_counts": {
            relation: int(sum(len(values) for values in mapping.values()))
            for relation, mapping in diagnostic_pairs.items()
        },
        "threshold_sensitivity": sensitivity.to_dict("records"),
        "min_safe_negative_count": int(min_safe_negative_count),
        "feature_count": int(len(train)),
        "feature_order_hash": order_hash,
        "warning_count": int(audit_frame["warning"].astype(str).ne("").sum()),
    }
    manifest["negative_policy_hash"] = sha256_text(
        exclusion_frame.to_csv(index=False)
        + audit_frame.to_csv(index=False)
        + near_audit.to_csv(index=False)
        + json.dumps({key: manifest[key] for key in sorted(manifest) if key != "negative_policy_hash"}, sort_keys=True)
    )
    return NegativePolicyResult(exclusion_frame, audit_frame, near_audit, sensitivity, manifest)


def _validate_training_boundary(train: pd.DataFrame, all_homecredit_pairs: pd.DataFrame) -> None:
    if train["feature_name"].duplicated().any():
        raise ValueError("training pairs contain duplicate feature identities")
    if not train["split"].astype(str).eq("train").all() or not train["dataset"].astype(str).eq("homecredit").all():
        raise ValueError("negative policy only accepts Home Credit training rows")
    if len(all_homecredit_pairs):
        unexpected = all_homecredit_pairs[
            all_homecredit_pairs["split"].astype(str).eq("train")
            & ~all_homecredit_pairs["feature_name"].astype(str).isin(set(train["feature_name"].astype(str)))
        ]
        if len(unexpected):
            raise ValueError("all_homecredit_pairs contains unexpected training features")


def _explicit_identity_rows(
    train: pd.DataFrame,
    pairs: Iterable[Iterable[str]],
    *,
    reason: str,
    evidence: str,
    order_hash: str,
) -> list[dict[str, Any]]:
    by_feature = train.set_index(train["feature_name"].astype(str), drop=False)
    rows = []
    for pair in pairs:
        values = [str(value) for value in pair]
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError(f"{reason} entries must contain two distinct feature names")
        if not set(values).issubset(set(by_feature.index)):
            raise ValueError(f"{reason} references a feature outside the training split: {values}")
        for anchor, excluded in ((values[0], values[1]), (values[1], values[0])):
            rows.append(_identity_row(by_feature.loc[anchor], by_feature.loc[excluded], reason, evidence, order_hash))
    return rows


def _exact_duplicate_rows(
    train: pd.DataFrame,
    exact_dev_duplicates: pd.DataFrame | None,
    *,
    order_hash: str,
) -> list[dict[str, Any]]:
    if exact_dev_duplicates is None or exact_dev_duplicates.empty:
        return []
    required = {"anchor_feature_name", "excluded_feature_name", "exclusion_reason", "dataset", "split", "evidence"}
    missing = required - set(exact_dev_duplicates.columns)
    if missing:
        raise ValueError(f"exact DEV duplicate evidence missing columns: {sorted(missing)}")
    if set(exact_dev_duplicates["dataset"].astype(str)) != {"homecredit"}:
        raise ValueError("exact duplicate evidence is not Home Credit-only")
    if set(exact_dev_duplicates["split"].astype(str)) != {"train"}:
        raise ValueError("exact duplicate evidence is not DEV train-only")
    if set(exact_dev_duplicates["exclusion_reason"].astype(str)) != {"exact_dev_duplicate"}:
        raise ValueError("exact duplicate evidence contains an unsupported reason")
    by_feature = train.set_index(train["feature_name"].astype(str), drop=False)
    rows = []
    for record in exact_dev_duplicates.to_dict("records"):
        anchor = str(record["anchor_feature_name"])
        excluded = str(record["excluded_feature_name"])
        if anchor not in by_feature.index or excluded not in by_feature.index or anchor == excluded:
            continue
        rows.append(
            _identity_row(
                by_feature.loc[anchor],
                by_feature.loc[excluded],
                "exact_dev_duplicate",
                str(record["evidence"]),
                order_hash,
            )
        )
    return rows


def _identity_row(anchor: pd.Series, excluded: pd.Series, reason: str, evidence: str, order_hash: str) -> dict[str, Any]:
    return {
        "anchor_feature_name": str(anchor["feature_name"]),
        "excluded_feature_name": str(excluded["feature_name"]),
        "anchor_pair_id": str(anchor["pair_id"]),
        "excluded_pair_id": str(excluded["pair_id"]),
        "exclusion_reason": reason,
        "evidence": evidence,
        "policy_version": NEGATIVE_POLICY_VERSION,
        "feature_order_hash": order_hash,
    }


def _exclusion_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "anchor_feature_name",
        "excluded_feature_name",
        "anchor_pair_id",
        "excluded_pair_id",
        "exclusion_reason",
        "evidence",
        "policy_version",
        "feature_order_hash",
    ]
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates()
    if frame.empty:
        return frame
    return frame.sort_values(
        ["anchor_feature_name", "excluded_feature_name", "exclusion_reason"], kind="mergesort"
    ).reset_index(drop=True)


def _diagnostic_relations(
    *,
    train: pd.DataFrame,
    text_embeddings: pd.DataFrame | None,
    threshold: float,
    thresholds: tuple[float, ...],
    tolerance: float,
) -> tuple[dict[str, dict[str, set[str]]], pd.DataFrame, pd.DataFrame]:
    features = train["feature_name"].astype(str).tolist()
    relations = {
        relation: {feature: set() for feature in features}
        for relation in [
            "diagnostic_same_family",
            "diagnostic_text_similarity",
            "diagnostic_statistical_similarity",
            "same_source_table",
        ]
    }
    for relation, column in [
        ("diagnostic_same_family", "base_feature_family"),
        ("diagnostic_statistical_similarity", "statistical_vector_hash"),
        ("same_source_table", "source_table_or_formula"),
    ]:
        if column not in train.columns:
            continue
        for _, group in train.groupby(column, dropna=False):
            members = group["feature_name"].astype(str).tolist()
            for feature in members:
                relations[relation][feature].update(other for other in members if other != feature)

    near_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    if text_embeddings is not None:
        matrix = _aligned_embedding_matrix(train, text_embeddings)
        normalized = matrix / np.where(np.linalg.norm(matrix, axis=1) == 0, 1.0, np.linalg.norm(matrix, axis=1))[:, None]
        similarities = normalized @ normalized.T
        for i, feature_a in enumerate(features):
            for j in range(i + 1, len(features)):
                similarity = float(similarities[i, j])
                if similarity + tolerance < threshold:
                    continue
                feature_b = features[j]
                relations["diagnostic_text_similarity"][feature_a].add(feature_b)
                relations["diagnostic_text_similarity"][feature_b].add(feature_a)
                near_rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "split": "train",
                        "cosine_similarity": similarity,
                        "threshold": threshold,
                        "relation": "diagnostic_text_similarity",
                        "mask_producing": False,
                    }
                )
        for diagnostic_threshold in thresholds:
            count = int(np.triu(similarities + tolerance >= diagnostic_threshold, k=1).sum())
            threshold_rows.append(
                {
                    "threshold": float(diagnostic_threshold),
                    "diagnostic_pair_count": count,
                    "mask_producing_pair_count": 0,
                }
            )
    else:
        threshold_rows = [
            {"threshold": float(value), "diagnostic_pair_count": 0, "mask_producing_pair_count": 0}
            for value in thresholds
        ]
    near_audit = pd.DataFrame(
        near_rows,
        columns=["feature_a", "feature_b", "split", "cosine_similarity", "threshold", "relation", "mask_producing"],
    )
    return relations, near_audit, pd.DataFrame(threshold_rows)


def _aligned_embedding_matrix(train: pd.DataFrame, text_embeddings: pd.DataFrame) -> np.ndarray:
    columns = sorted(col for col in text_embeddings.columns if str(col).startswith("embedding_") and len(str(col)) == 14)
    if not columns:
        raise ValueError("text embeddings contain no embedding columns")
    if text_embeddings["embedding_cache_key"].duplicated().any():
        raise ValueError("text embedding cache keys are not unique")
    by_key = text_embeddings.set_index("embedding_cache_key", drop=False)
    rows = []
    for row in train.itertuples(index=False):
        key = str(row.text_embedding_row_id)
        if key not in by_key.index:
            raise ValueError(f"missing text embedding for training pair: {row.feature_name}")
        embedding = by_key.loc[key]
        if str(embedding["feature_name"]) != str(row.feature_name):
            raise ValueError(f"text embedding feature mismatch for {row.feature_name}")
        rows.append(embedding[columns].to_numpy(dtype=np.float32))
    matrix = np.vstack(rows).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("text embeddings contain non-finite values")
    return matrix


def _reason_count(frame: pd.DataFrame, feature: str, reason: str) -> int:
    if frame.empty:
        return 0
    return int(
        frame["anchor_feature_name"].astype(str).eq(feature)
        .mul(frame["exclusion_reason"].astype(str).eq(reason))
        .sum()
    )
