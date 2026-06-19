from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.negative_policy import build_negative_policy


def test_negative_policy_excludes_same_feature_same_family_and_duplicate_statistics(tmp_path):
    train = pd.read_parquet("results/clip/contrastive_data/homecredit_train_positive_pairs.parquet")
    validation = pd.read_parquet("results/clip/contrastive_data/homecredit_validation_positive_pairs.parquet")
    text = pd.read_parquet("results/clip/text_baseline/homecredit_text_embeddings.parquet")

    negative = build_negative_policy(
        train_pairs=train,
        all_homecredit_pairs=pd.concat([train, validation], ignore_index=True),
        text_embeddings=text,
    )

    reasons = set(negative.exclusion_pairs["exclusion_reason"])
    assert "same_feature" in reasons
    assert "same_canonical_family" in reasons
    assert "duplicate_statistical_vector" in reasons
    assert "duplicate_formula" in reasons
    assert "near_duplicate_text_embedding" in reasons
    assert negative.manifest["cross_dataset_negatives_enabled"] is False
    assert negative.manifest["explicit_hard_negatives_enabled"] is False
    assert negative.manifest["validation_as_training_negative"] is False
    assert negative.manifest["near_duplicate_text_threshold"] == 0.95
    assert negative.candidate_audit["remaining_safe_negative_count"].min() >= 25
    assert not negative.near_duplicate_audit.empty


def test_same_base_family_is_excluded_as_negative():
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "dataset": "homecredit",
                "feature_name": "A_MEAN",
                "base_feature_family": "family:A",
                "canonical_feature_family": "A",
                "text_hash": "t1",
                "normalized_text_hash": "nt1",
                "source_table_or_formula": "formula1",
                "statistical_vector_hash": "s1",
                "group_key": "family:A",
                "split": "train",
            },
            {
                "pair_id": "p2",
                "dataset": "homecredit",
                "feature_name": "A_MAX",
                "base_feature_family": "family:A",
                "canonical_feature_family": "A",
                "text_hash": "t2",
                "normalized_text_hash": "nt2",
                "source_table_or_formula": "formula2",
                "statistical_vector_hash": "s2",
                "group_key": "family:A",
                "split": "train",
            },
            {
                "pair_id": "p3",
                "dataset": "homecredit",
                "feature_name": "B",
                "base_feature_family": "name:B",
                "canonical_feature_family": "B",
                "text_hash": "t3",
                "normalized_text_hash": "nt3",
                "source_table_or_formula": "formula3",
                "statistical_vector_hash": "s3",
                "group_key": "name:B",
                "split": "train",
            },
        ]
    )

    result = build_negative_policy(train_pairs=pairs, all_homecredit_pairs=pairs, min_safe_negative_count=0)
    subset = result.exclusion_pairs[
        (result.exclusion_pairs["anchor_feature_name"].eq("A_MEAN"))
        & (result.exclusion_pairs["excluded_feature_name"].eq("A_MAX"))
    ]
    assert "same_canonical_family" in set(subset["exclusion_reason"])


def test_near_duplicate_text_threshold_is_used_and_exact_duplicates_are_distinct():
    pairs = _pair_frame()
    embeddings = _embedding_frame(pairs)

    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        text_embeddings=embeddings,
        near_duplicate_text_threshold=0.95,
        min_safe_negative_count=0,
    )

    reasons = set(result.exclusion_pairs["exclusion_reason"])
    assert "exact_text_duplicate" in reasons
    assert "near_duplicate_text_embedding" in reasons
    exact_rows = result.near_duplicate_audit[result.near_duplicate_audit["exclusion_reason"].eq("exact_text_duplicate")]
    near_rows = result.near_duplicate_audit[result.near_duplicate_audit["exclusion_reason"].eq("near_duplicate_text_embedding")]
    assert not exact_rows.empty
    assert not near_rows.empty
    assert exact_rows["exact_text_duplicate"].all()
    assert not near_rows["exact_text_duplicate"].any()


def test_cosine_below_threshold_is_allowed_when_no_other_exclusion_applies():
    pairs = _pair_frame()
    embeddings = _embedding_frame(pairs)

    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        text_embeddings=embeddings,
        near_duplicate_text_threshold=0.99,
        min_safe_negative_count=0,
    )
    subset = result.exclusion_pairs[
        (result.exclusion_pairs["anchor_feature_name"].eq("NEAR_A"))
        & (result.exclusion_pairs["excluded_feature_name"].eq("NEAR_B"))
    ]

    assert "near_duplicate_text_embedding" not in set(subset["exclusion_reason"])


def test_validation_and_lendingclub_do_not_enter_training_negative_candidates():
    train = pd.read_parquet("results/clip/contrastive_data/homecredit_train_positive_pairs.parquet")
    validation = pd.read_parquet("results/clip/contrastive_data/homecredit_validation_positive_pairs.parquet")
    text = pd.read_parquet("results/clip/text_baseline/homecredit_text_embeddings.parquet")

    result = build_negative_policy(
        train_pairs=train,
        all_homecredit_pairs=pd.concat([train, validation], ignore_index=True),
        text_embeddings=text,
    )

    validation_features = set(validation["feature_name"].astype(str))
    excluded_features = set(result.exclusion_pairs["excluded_feature_name"].astype(str))
    assert not validation_features.intersection(excluded_features)


def _pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("EXACT_A", "p1", "k1", "same_text", "formula1", "s1", "family:EXACT_A"),
            _row("EXACT_B", "p2", "k2", "same_text", "formula2", "s2", "family:EXACT_B"),
            _row("NEAR_A", "p3", "k3", "near_a", "formula3", "s3", "family:NEAR_A"),
            _row("NEAR_B", "p4", "k4", "near_b", "formula4", "s4", "family:NEAR_B"),
            _row("FAR", "p5", "k5", "far", "formula5", "s5", "family:FAR"),
        ]
    )


def _row(feature: str, pair_id: str, key: str, normalized_hash: str, formula: str, stat_hash: str, family: str) -> dict:
    return {
        "pair_id": pair_id,
        "dataset": "homecredit",
        "feature_name": feature,
        "base_feature_family": family,
        "canonical_feature_family": family.replace("family:", ""),
        "text_hash": f"text_{feature}",
        "normalized_text_hash": normalized_hash,
        "source_table_or_formula": formula,
        "statistical_vector_hash": stat_hash,
        "group_key": family,
        "split": "train",
        "text_embedding_row_id": key,
        "semantic_group": "test_group",
    }


def _embedding_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    values = {
        "k1": [1.0, 0.0, 0.0],
        "k2": [1.0, 0.0, 0.0],
        "k3": [1.0, 0.0, 0.0],
        "k4": [0.96, 0.28, 0.0],
        "k5": [0.0, 1.0, 0.0],
    }
    rows = []
    for row in pairs.to_dict("records"):
        vector = values[row["text_embedding_row_id"]]
        rows.append(
            {
                "dataset": "homecredit",
                "feature_name": row["feature_name"],
                "embedding_cache_key": row["text_embedding_row_id"],
                "embedding_0000": vector[0],
                "embedding_0001": vector[1],
                "embedding_0002": vector[2],
            }
        )
    return pd.DataFrame(rows)
