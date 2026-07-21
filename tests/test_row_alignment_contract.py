from __future__ import annotations

from decimal import Decimal

from credit_risk_fs.experiments.row_alignment import (
    ordered_values_sha256,
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
    split_alignment_summary,
    split_id_overlap_count,
)


def test_identical_ordered_inputs_have_identical_hashes():
    assert ordered_row_id_sha256([1, 2, "three"]) == ordered_row_id_sha256(
        [1, 2, "three"]
    )


def test_row_reordering_changes_the_ordered_hash():
    assert ordered_row_id_sha256([1, 2, 3]) != ordered_row_id_sha256([3, 2, 1])


def test_target_modification_changes_the_pair_hash():
    assert ordered_row_id_target_sha256([1, 2], [0, 1]) != (
        ordered_row_id_target_sha256([1, 2], [1, 1])
    )


def test_numeric_type_normalization_is_deliberate():
    assert ordered_row_id_sha256([1, 2.0, Decimal("3.00")]) == (
        ordered_row_id_sha256([1.0, Decimal("2"), 3])
    )
    assert ordered_row_id_sha256([1]) != ordered_row_id_sha256(["1"])


def test_summary_counts_and_dev_oot_overlap_detection():
    summary = split_alignment_summary([1, 2, 2, None], [0, 1, 0, 1])
    assert summary["row_count"] == 4
    assert summary["unique_id_count"] == 2
    assert summary["missing_id_count"] == 1
    assert summary["duplicate_id_count"] == 1
    assert summary["positive_count"] == 2
    assert split_id_overlap_count([1, 2], [2, 3]) == 1


def test_arbitrary_ordered_tuple_hash_is_length_and_order_sensitive():
    assert ordered_values_sha256([(1, "a"), (2, "b")]) != ordered_values_sha256(
        [(2, "b"), (1, "a")]
    )
