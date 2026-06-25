from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.exact_duplicates import feature_order_hash, find_exact_dev_duplicate_pairs
from credit_risk_fs.clip.negative_policy import NEGATIVE_POLICY_VERSION, build_negative_policy
from credit_risk_fs.clip.training_validation import false_negative_mask


def test_identity_only_policy_keeps_diagnostic_relations_as_negatives():
    pairs = _pairs()
    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        text_embeddings=_embeddings(pairs),
        min_safe_negative_count=0,
    )

    assert result.exclusion_pairs.empty
    assert result.manifest["policy_version"] == NEGATIVE_POLICY_VERSION
    assert result.manifest["diagnostic_relation_counts"]["same_source_table"] > 0
    assert result.manifest["diagnostic_relation_counts"]["diagnostic_same_family"] > 0
    assert result.manifest["diagnostic_relation_counts"]["diagnostic_text_similarity"] > 0
    assert result.manifest["diagnostic_relation_counts"]["diagnostic_statistical_similarity"] > 0
    assert false_negative_mask(pairs, result.exclusion_pairs).sum().item() == 0


def test_unrelated_same_table_and_ext_sources_remain_negatives():
    pairs = _pairs()
    result = build_negative_policy(train_pairs=pairs, all_homecredit_pairs=pairs, min_safe_negative_count=0)
    mask = false_negative_mask(pairs, result.exclusion_pairs)
    index = {name: idx for idx, name in enumerate(pairs["feature_name"])}

    assert not mask[index["AMT_ANNUITY"], index["DAYS_BIRTH"]]
    assert not mask[index["EXT_SOURCE_1"], index["EXT_SOURCE_2"]]
    assert not mask[index["EXT_SOURCE_2"], index["EXT_SOURCE_3"]]


def test_verified_alias_and_exact_dev_duplicates_are_masked_symmetrically():
    pairs = _pairs()
    dev = pd.DataFrame(
        {
            name: [1, None, 3, 4] if name in {"ALIAS_A", "ALIAS_B"} else range(4)
            for name in pairs["feature_name"]
        }
    )
    exact = find_exact_dev_duplicate_pairs(
        dev,
        feature_names=pairs["feature_name"].tolist(),
        dataset="homecredit",
        split="train",
    )
    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        exact_dev_duplicates=exact,
        verified_aliases=[["EXT_SOURCE_1", "EXT_SOURCE_2"]],
        min_safe_negative_count=0,
    )
    mask = false_negative_mask(pairs, result.exclusion_pairs)
    index = {name: idx for idx, name in enumerate(pairs["feature_name"])}

    assert mask[index["ALIAS_A"], index["ALIAS_B"]]
    assert mask[index["ALIAS_B"], index["ALIAS_A"]]
    assert mask[index["EXT_SOURCE_1"], index["EXT_SOURCE_2"]]
    assert mask[index["EXT_SOURCE_2"], index["EXT_SOURCE_1"]]
    assert not mask.diag().any()


def test_mask_rejects_stale_order_hash_and_zero_valid_negatives():
    pairs = _pairs().iloc[:3].copy().reset_index(drop=True)
    stale = pd.DataFrame(
        [
            _exclusion("AMT_ANNUITY", "DAYS_BIRTH", pairs, "stale"),
            _exclusion("DAYS_BIRTH", "AMT_ANNUITY", pairs, "stale"),
        ]
    )
    try:
        false_negative_mask(pairs, stale)
    except ValueError as error:
        assert "order hash" in str(error)
    else:
        raise AssertionError("stale mask order hash was accepted")

    order_hash = feature_order_hash(pairs["feature_name"].tolist())
    complete = []
    for a in pairs["feature_name"]:
        for b in pairs["feature_name"]:
            if a != b:
                complete.append(_exclusion(a, b, pairs, order_hash))
    try:
        false_negative_mask(pairs, pd.DataFrame(complete))
    except ValueError as error:
        assert "zero valid negatives" in str(error)
    else:
        raise AssertionError("zero-valid-negative mask was accepted")


def test_mask_is_deterministic_and_batch_order_invariant():
    pairs = _pairs()
    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        verified_aliases=[["ALIAS_A", "ALIAS_B"]],
        min_safe_negative_count=0,
    )
    first = false_negative_mask(pairs, result.exclusion_pairs)
    second = false_negative_mask(pairs, result.exclusion_pairs)
    assert first.equal(second)

    shuffled = pairs.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled["positive_pair_index"] = range(len(shuffled))
    shuffled["feature_order_hash"] = feature_order_hash(shuffled["feature_name"].tolist())
    remapped = result.exclusion_pairs.copy()
    remapped["feature_order_hash"] = feature_order_hash(shuffled["feature_name"].tolist())
    shuffled_mask = false_negative_mask(shuffled, remapped)
    original_index = {name: idx for idx, name in enumerate(pairs["feature_name"])}
    shuffled_index = {name: idx for idx, name in enumerate(shuffled["feature_name"])}
    assert first[original_index["ALIAS_A"], original_index["ALIAS_B"]]
    assert shuffled_mask[shuffled_index["ALIAS_A"], shuffled_index["ALIAS_B"]]


def _pairs() -> pd.DataFrame:
    features = [
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "APARTMENTS_AVG",
        "APARTMENTS_MEDI",
        "ALIAS_A",
        "ALIAS_B",
    ]
    order_hash = feature_order_hash(features)
    rows = []
    for index, feature in enumerate(features):
        source = "application_train"
        family = "family:EXT_SOURCE" if feature.startswith("EXT_SOURCE") else f"family:{feature.split('_')[0]}"
        stat_hash = "same_stat" if feature in {"EXT_SOURCE_1", "EXT_SOURCE_2"} else f"stat_{feature}"
        normalized_text = "same_text" if feature in {"APARTMENTS_AVG", "APARTMENTS_MEDI"} else f"text_{feature}"
        rows.append(
            {
                "feature_id": f"feature_{index}",
                "positive_pair_index": index,
                "feature_order_hash": order_hash,
                "pair_id": f"pair_{index}",
                "dataset": "homecredit",
                "feature_name": feature,
                "base_feature_family": family,
                "canonical_feature_family": family,
                "text_hash": f"text_hash_{feature}",
                "normalized_text_hash": normalized_text,
                "source_table_or_formula": source,
                "statistical_vector_hash": stat_hash,
                "group_key": family,
                "split": "train",
                "text_embedding_row_id": f"embedding_{index}",
                "statistical_vector_row_id": f"statistical_{index}",
                "source_manifest_hash": "source_hash",
                "semantic_group": "application",
            }
        )
    return pd.DataFrame(rows)


def _embeddings(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in pairs.iterrows():
        vector = [1.0, 0.0] if row["feature_name"] in {"APARTMENTS_AVG", "APARTMENTS_MEDI"} else [0.0, 1.0]
        rows.append(
            {
                "feature_name": row["feature_name"],
                "embedding_cache_key": row["text_embedding_row_id"],
                "embedding_0000": vector[0],
                "embedding_0001": vector[1],
            }
        )
    return pd.DataFrame(rows)


def _exclusion(a: str, b: str, pairs: pd.DataFrame, order_hash: str) -> dict:
    by_feature = pairs.set_index("feature_name")
    return {
        "anchor_feature_name": a,
        "excluded_feature_name": b,
        "anchor_pair_id": by_feature.loc[a, "pair_id"],
        "excluded_pair_id": by_feature.loc[b, "pair_id"],
        "exclusion_reason": "verified_alias",
        "evidence": "test alias",
        "policy_version": NEGATIVE_POLICY_VERSION,
        "feature_order_hash": order_hash,
    }
