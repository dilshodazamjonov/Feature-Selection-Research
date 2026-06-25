from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.exact_duplicates import feature_order_hash, find_exact_dev_duplicate_pairs
from credit_risk_fs.clip.training_validation import tensors_for_pairs


def test_exact_dev_duplicates_require_equal_values_and_missingness():
    frame = pd.DataFrame(
        {
            "NUM_A": [1.0, None, 3.0],
            "NUM_B": [1, None, 3],
            "DIFFERENT_MISSING": [1, 2, 3],
            "BOOL_A": pd.Series([True, False, None], dtype="boolean"),
            "BOOL_B": pd.Series([True, False, None], dtype="boolean"),
            "CAT_A": pd.Series(["x", None, "y"], dtype="category"),
            "CAT_B": pd.Series(["x", None, "y"], dtype="category"),
        }
    )
    duplicates = find_exact_dev_duplicate_pairs(frame, feature_names=list(frame.columns))
    pairs = {
        (row.anchor_feature_name, row.excluded_feature_name)
        for row in duplicates.itertuples(index=False)
    }
    assert ("NUM_A", "NUM_B") in pairs
    assert ("BOOL_A", "BOOL_B") in pairs
    assert ("CAT_A", "CAT_B") in pairs
    assert ("NUM_A", "DIFFERENT_MISSING") not in pairs


def test_positive_views_follow_the_same_shuffle_and_reject_stale_order():
    features = ["A", "B", "C"]
    order_hash = feature_order_hash(features)
    pairs = pd.DataFrame(
        {
            "feature_id": ["id_a", "id_b", "id_c"],
            "dataset": ["homecredit"] * 3,
            "feature_name": features,
            "source_manifest_hash": ["source"] * 3,
            "text_embedding_row_id": ["ta", "tb", "tc"],
            "statistical_vector_row_id": ["sa", "sb", "sc"],
            "positive_pair_index": range(3),
            "feature_order_hash": [order_hash] * 3,
        }
    )
    text = pd.DataFrame(
        {
            "feature_name": ["C", "A", "B"],
            "embedding_cache_key": ["tc", "ta", "tb"],
            "embedding_0000": [3.0, 1.0, 2.0],
        }
    )
    stat = pd.DataFrame(
        {
            "feature_name": ["B", "C", "A"],
            "stable_row_id": ["sb", "sc", "sa"],
            "stat_0000": [2.0, 3.0, 1.0],
        }
    )
    text_tensor, stat_tensor = tensors_for_pairs(pairs, text, stat)
    assert text_tensor[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert stat_tensor[:, 0].tolist() == [1.0, 2.0, 3.0]

    stale = pairs.iloc[[2, 0, 1]].reset_index(drop=True)
    try:
        tensors_for_pairs(stale, text, stat)
    except RuntimeError as error:
        assert "positive-pair indices" in str(error)
    else:
        raise AssertionError("stale positive-pair order was accepted")
