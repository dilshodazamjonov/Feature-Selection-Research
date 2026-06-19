from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text


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
    near_duplicate_text_threshold: float = 0.95,
    threshold_diagnostics: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99),
    min_safe_negative_count: int = 25,
    float_tolerance: float = 1e-8,
) -> NegativePolicyResult:
    train = train_pairs.copy().sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    if not train["split"].astype(str).eq("train").all():
        raise ValueError("negative policy only accepts Home Credit training rows")
    if not train["dataset"].astype(str).eq("homecredit").all():
        raise ValueError("negative policy only accepts Home Credit rows")
    if len(all_homecredit_pairs):
        non_train = all_homecredit_pairs[~all_homecredit_pairs["feature_name"].astype(str).isin(set(train["feature_name"].astype(str)))]
        if len(non_train) and non_train["split"].astype(str).eq("train").any():
            raise ValueError("all_homecredit_pairs contains unexpected duplicate train features")
    near_duplicate_map, exact_text_map, near_audit, sensitivity = _near_duplicate_text_policy(
        train=train,
        text_embeddings=text_embeddings,
        threshold=float(near_duplicate_text_threshold),
        diagnostics=threshold_diagnostics,
        tolerance=float(float_tolerance),
        min_safe_negative_count=min_safe_negative_count,
    )
    exclusions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in train.to_dict("records"):
        feature = str(row["feature_name"])
        same_feature = train[train["feature_name"].astype(str).eq(feature)]
        same_family = train[
            train["base_feature_family"].astype(str).eq(str(row["base_feature_family"]))
            & ~train["feature_name"].astype(str).eq(feature)
        ]
        duplicate_formula = train[
            train["source_table_or_formula"].astype(str).eq(str(row["source_table_or_formula"]))
            & ~train["feature_name"].astype(str).eq(feature)
        ]
        duplicate_stats = train[
            train["statistical_vector_hash"].astype(str).eq(str(row["statistical_vector_hash"]))
            & ~train["feature_name"].astype(str).eq(feature)
        ]
        exact_text = train[train["feature_name"].astype(str).isin(exact_text_map.get(feature, set()))]
        near_duplicate_text = train[train["feature_name"].astype(str).isin(near_duplicate_map.get(feature, set()))]
        reason_frames = [
            ("same_feature", same_feature),
            ("same_canonical_family", same_family),
            ("exact_text_duplicate", exact_text),
            ("near_duplicate_text_embedding", near_duplicate_text),
            ("duplicate_formula", duplicate_formula),
            ("duplicate_statistical_vector", duplicate_stats),
        ]
        excluded_features: set[str] = set()
        for reason, frame in reason_frames:
            for other in frame.to_dict("records"):
                other_feature = str(other["feature_name"])
                excluded_features.add(other_feature)
                exclusions.append(
                    {
                        "anchor_feature_name": feature,
                        "excluded_feature_name": other_feature,
                        "anchor_pair_id": row["pair_id"],
                        "excluded_pair_id": other["pair_id"],
                        "exclusion_reason": reason,
                        "anchor_group_key": row["group_key"],
                        "excluded_group_key": other["group_key"],
                        "anchor_base_feature_family": row["base_feature_family"],
                        "excluded_base_feature_family": other["base_feature_family"],
                        "anchor_canonical_feature_family": row.get("canonical_feature_family", ""),
                        "excluded_canonical_feature_family": other.get("canonical_feature_family", ""),
                    }
                )
        candidate_count = max(0, len(train) - 1)
        same_feature_count = int(len(same_feature))
        same_family_count = int(len(set(same_family["feature_name"].astype(str))))
        exact_text_count = int(len(set(exact_text["feature_name"].astype(str))))
        near_text_count = int(len(set(near_duplicate_text["feature_name"].astype(str))))
        formula_count = int(len(set(duplicate_formula["feature_name"].astype(str))))
        stats_count = int(len(set(duplicate_stats["feature_name"].astype(str))))
        remaining = max(0, candidate_count - len((excluded_features - {feature}).intersection(set(train["feature_name"].astype(str)))))
        audits.append(
            {
                "feature_name": feature,
                "candidate_negative_count": candidate_count,
                "excluded_same_feature_count": same_feature_count,
                "excluded_same_family_count": same_family_count,
                "excluded_exact_text_duplicate_count": exact_text_count,
                "excluded_near_duplicate_text_embedding_count": near_text_count,
                "excluded_duplicate_formula_count": formula_count,
                "excluded_duplicate_statistical_vector_count": stats_count,
                "remaining_safe_negative_count": remaining,
                "warning": "safe_negative_count_below_threshold" if remaining < min_safe_negative_count else "",
            }
        )
    exclusion_frame = pd.DataFrame(exclusions).drop_duplicates().sort_values(
        ["anchor_feature_name", "excluded_feature_name", "exclusion_reason"], kind="mergesort"
    )
    audit_frame = pd.DataFrame(audits).sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    manifest = {
        "policy": "in-batch negatives within Home Credit training only",
        "explicit_hard_negatives_enabled": False,
        "cross_dataset_negatives_enabled": False,
        "validation_as_training_negative": False,
        "near_duplicate_text_policy": "cosine_similarity_over_frozen_normalized_text_embeddings",
        "near_duplicate_text_threshold": float(near_duplicate_text_threshold),
        "float_tolerance": float(float_tolerance),
        "excluded_reason_counts": exclusion_frame["exclusion_reason"].value_counts().sort_index().to_dict()
        if len(exclusion_frame)
        else {},
        "threshold_sensitivity": sensitivity.to_dict("records"),
        "min_safe_negative_count": int(min_safe_negative_count),
        "feature_count": int(len(train)),
        "warning_count": int(audit_frame["warning"].astype(str).ne("").sum()) if len(audit_frame) else 0,
        "negative_policy_hash": sha256_text(
            exclusion_frame.to_csv(index=False) + audit_frame.to_csv(index=False) + near_audit.to_csv(index=False)
        ),
    }
    return NegativePolicyResult(
        exclusion_pairs=exclusion_frame,
        candidate_audit=audit_frame,
        near_duplicate_audit=near_audit,
        threshold_sensitivity=sensitivity,
        manifest=manifest,
    )


def _near_duplicate_text_policy(
    *,
    train: pd.DataFrame,
    text_embeddings: pd.DataFrame | None,
    threshold: float,
    diagnostics: tuple[float, ...],
    tolerance: float,
    min_safe_negative_count: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    exact_map = _exact_text_duplicate_map(train)
    near_map: dict[str, set[str]] = {feature: set() for feature in train["feature_name"].astype(str)}
    if text_embeddings is None:
        near_audit = _empty_near_duplicate_audit()
        sensitivity = _empty_threshold_sensitivity(diagnostics)
        return near_map, exact_map, near_audit, sensitivity

    matrix = _aligned_embedding_matrix(train, text_embeddings)
    norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms[:, None]
    similarities = normalized @ normalized.T
    features = train["feature_name"].astype(str).tolist()
    records: list[dict[str, Any]] = []
    for i, feature_a in enumerate(features):
        for j in range(i + 1, len(features)):
            feature_b = features[j]
            similarity = float(similarities[i, j])
            exact_text_duplicate = feature_b in exact_map.get(feature_a, set())
            same_canonical_family = str(train.iloc[i]["base_feature_family"]) == str(train.iloc[j]["base_feature_family"])
            if similarity + tolerance < threshold and not exact_text_duplicate:
                continue
            reason = "exact_text_duplicate" if exact_text_duplicate else "near_duplicate_text_embedding"
            if reason == "near_duplicate_text_embedding":
                near_map[feature_a].add(feature_b)
                near_map[feature_b].add(feature_a)
            records.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "split": "train",
                    "cosine_similarity": similarity,
                    "threshold": float(threshold),
                    "exact_text_duplicate": bool(exact_text_duplicate),
                    "same_canonical_family": bool(same_canonical_family),
                    "semantic_group_a": str(train.iloc[i].get("semantic_group", "")),
                    "semantic_group_b": str(train.iloc[j].get("semantic_group", "")),
                    "source_a": str(train.iloc[i].get("source_table_or_formula", "")),
                    "source_b": str(train.iloc[j].get("source_table_or_formula", "")),
                    "exclusion_reason": reason,
                    "pair_hash": sha256_text(f"{feature_a}|{feature_b}|{similarity:.12g}|{threshold:.12g}|{reason}"),
                }
            )
    near_audit = pd.DataFrame(records)
    if len(near_audit):
        near_audit = near_audit.sort_values(["feature_a", "feature_b", "exclusion_reason"], kind="mergesort").reset_index(drop=True)
    else:
        near_audit = _empty_near_duplicate_audit()
    sensitivity = _threshold_sensitivity(
        train=train,
        similarities=similarities,
        thresholds=diagnostics,
        exact_map=exact_map,
        tolerance=tolerance,
        min_safe_negative_count=min_safe_negative_count,
    )
    return near_map, exact_map, near_audit, sensitivity


def _exact_text_duplicate_map(train: pd.DataFrame) -> dict[str, set[str]]:
    key = "normalized_text_hash" if "normalized_text_hash" in train.columns else "text_hash"
    mapping: dict[str, set[str]] = {feature: set() for feature in train["feature_name"].astype(str)}
    for _, frame in train.groupby(key, dropna=False):
        features = frame["feature_name"].astype(str).tolist()
        if len(features) < 2:
            continue
        for feature in features:
            mapping[feature].update(other for other in features if other != feature)
    return mapping


def _aligned_embedding_matrix(train: pd.DataFrame, text_embeddings: pd.DataFrame) -> np.ndarray:
    columns = sorted([col for col in text_embeddings.columns if str(col).startswith("embedding_") and len(str(col)) == 14])
    if not columns:
        raise ValueError("text embeddings contain no embedding columns")
    by_key = text_embeddings.set_index("embedding_cache_key", drop=False)
    rows = []
    for row in train.itertuples(index=False):
        key = str(row.text_embedding_row_id)
        if key not in by_key.index:
            raise ValueError(f"missing text embedding for training pair: {row.feature_name}")
        emb = by_key.loc[key]
        if str(emb["feature_name"]) != str(row.feature_name):
            raise ValueError(f"text embedding feature mismatch for {row.feature_name}")
        rows.append(emb[columns].to_numpy(dtype=np.float32))
    matrix = np.vstack(rows).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("text embeddings contain non-finite values")
    return matrix


def _threshold_sensitivity(
    *,
    train: pd.DataFrame,
    similarities: np.ndarray,
    thresholds: tuple[float, ...],
    exact_map: dict[str, set[str]],
    tolerance: float,
    min_safe_negative_count: int,
) -> pd.DataFrame:
    features = train["feature_name"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    base_exclusions = _base_exclusion_sets(train, exact_map)
    for threshold in thresholds:
        near_pairs: list[tuple[str, str, float]] = []
        near_sets = {feature: set(values) for feature, values in base_exclusions.items()}
        semantic_matches = 0
        family_matches = 0
        for i, feature_a in enumerate(features):
            for j in range(i + 1, len(features)):
                feature_b = features[j]
                if feature_b in exact_map.get(feature_a, set()):
                    continue
                similarity = float(similarities[i, j])
                if similarity + tolerance >= float(threshold):
                    near_pairs.append((feature_a, feature_b, similarity))
                    near_sets[feature_a].add(feature_b)
                    near_sets[feature_b].add(feature_a)
                    semantic_matches += int(str(train.iloc[i].get("semantic_group", "")) == str(train.iloc[j].get("semantic_group", "")))
                    family_matches += int(str(train.iloc[i].get("base_feature_family", "")) == str(train.iloc[j].get("base_feature_family", "")))
        safe_counts = [max(0, len(features) - 1 - len(near_sets[feature])) for feature in features]
        examples = [
            {"feature_a": a, "feature_b": b, "cosine_similarity": round(sim, 6)}
            for a, b, sim in sorted(near_pairs, key=lambda item: (-item[2], item[0], item[1]))[:10]
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "excluded_pair_count": int(len(near_pairs)),
                "affected_feature_count": int(len({feature for pair in near_pairs for feature in pair[:2]})),
                "minimum_safe_negative_count": int(min(safe_counts) if safe_counts else 0),
                "median_safe_negative_count": float(np.median(safe_counts) if safe_counts else 0),
                "maximum_safe_negative_count": int(max(safe_counts) if safe_counts else 0),
                "example_pairs": str(examples),
                "semantic_group_agreement_rate": float(semantic_matches / len(near_pairs)) if near_pairs else 0.0,
                "canonical_family_agreement_rate": float(family_matches / len(near_pairs)) if near_pairs else 0.0,
                "warning": "safe_negative_count_below_threshold" if safe_counts and min(safe_counts) < min_safe_negative_count else "",
            }
        )
    return pd.DataFrame(rows).sort_values("threshold", kind="mergesort").reset_index(drop=True)


def _base_exclusion_sets(train: pd.DataFrame, exact_map: dict[str, set[str]]) -> dict[str, set[str]]:
    features = train["feature_name"].astype(str).tolist()
    exclusions: dict[str, set[str]] = {feature: set(exact_map.get(feature, set())) for feature in features}
    for column in ["base_feature_family", "source_table_or_formula", "statistical_vector_hash"]:
        for _, frame in train.groupby(column, dropna=False):
            group_features = frame["feature_name"].astype(str).tolist()
            if len(group_features) < 2:
                continue
            for feature in group_features:
                exclusions[feature].update(other for other in group_features if other != feature)
    return exclusions


def _empty_near_duplicate_audit() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature_a",
            "feature_b",
            "split",
            "cosine_similarity",
            "threshold",
            "exact_text_duplicate",
            "same_canonical_family",
            "semantic_group_a",
            "semantic_group_b",
            "source_a",
            "source_b",
            "exclusion_reason",
            "pair_hash",
        ]
    )


def _empty_threshold_sensitivity(thresholds: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "threshold": float(threshold),
                "excluded_pair_count": 0,
                "affected_feature_count": 0,
                "minimum_safe_negative_count": 0,
                "median_safe_negative_count": 0.0,
                "maximum_safe_negative_count": 0,
                "example_pairs": "[]",
                "semantic_group_agreement_rate": 0.0,
                "canonical_family_agreement_rate": 0.0,
                "warning": "",
            }
            for threshold in thresholds
        ]
    )
