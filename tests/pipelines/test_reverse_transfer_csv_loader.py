from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    load_lendingclub_required_columns,
)


TIME = "recent_decision"
TARGET = "TARGET"


def _write_csv(path: Path, columns: dict[str, list[object]]) -> Path:
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def _valid_csv(tmp_path: Path) -> Path:
    return _write_csv(
        tmp_path / "lendingclub.csv",
        {
            "feature_a": [1, 2, 3],
            TARGET: [0, 1, 0],
            TIME: [-3, -2, -1],
            "feature_b": ["x", "2", "z"],
        },
    )


def test_time_column_is_excluded_from_requested_features(tmp_path: Path) -> None:
    frame, features = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=[TIME, "feature_a"],
    )
    assert features == ["feature_a"]
    assert frame.columns.tolist() == [TIME, TARGET, "feature_a"]


def test_target_column_is_excluded_from_requested_features(tmp_path: Path) -> None:
    frame, features = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=[TARGET, "feature_a"],
    )
    assert features == ["feature_a"]
    assert frame.columns.tolist() == [TIME, TARGET, "feature_a"]


def test_repeated_feature_names_are_deduplicated_in_order(tmp_path: Path) -> None:
    frame, features = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=["feature_b", "feature_a", "feature_b", "feature_a"],
    )
    assert features == ["feature_b", "feature_a"]
    assert frame.columns.tolist() == [TIME, TARGET, "feature_b", "feature_a"]


def test_missing_requested_feature_fails_with_context(tmp_path: Path) -> None:
    path = _valid_csv(tmp_path)
    with pytest.raises(
        TransferStageError, match="missing required columns.*missing_feature"
    ) as exc_info:
        load_lendingclub_required_columns(
            path,
            time_column=TIME,
            target_column=TARGET,
            feature_columns=["feature_a", "missing_feature", "feature_a"],
        )
    message = str(exc_info.value)
    assert str(path) in message
    assert "required_column_count=4" in message
    assert "duplicate_requested_columns=['feature_a']" in message
    assert f"time_column='{TIME}'" in message
    assert f"target_column='{TARGET}'" in message


def test_missing_time_column_fails_clearly(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "missing_time.csv",
        {TARGET: [0, 1], "feature_a": [1, 2]},
    )
    with pytest.raises(TransferStageError, match="missing required columns.*recent_decision"):
        load_lendingclub_required_columns(
            path,
            time_column=TIME,
            target_column=TARGET,
            feature_columns=["feature_a"],
        )


def test_missing_target_column_fails_clearly(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "missing_target.csv",
        {TIME: [-2, -1], "feature_a": [1, 2]},
    )
    with pytest.raises(TransferStageError, match="missing required columns.*TARGET"):
        load_lendingclub_required_columns(
            path,
            time_column=TIME,
            target_column=TARGET,
            feature_columns=["feature_a"],
        )


def test_mixed_type_csv_loads_under_installed_pandas(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "mixed.csv",
        {
            TIME: [-4, -3, -2, -1],
            TARGET: [0, 1, 0, 1],
            "mixed_feature": [1, "category", 3.5, "other"],
        },
    )
    frame, features = load_lendingclub_required_columns(
        path,
        time_column=TIME,
        target_column=TARGET,
        feature_columns=["mixed_feature"],
    )
    assert len(frame) == 4
    assert features == ["mixed_feature"]
    assert frame["mixed_feature"].astype(str).tolist() == [
        "1",
        "category",
        "3.5",
        "other",
    ]


def test_loaded_column_order_is_requested_order_not_source_order(
    tmp_path: Path,
) -> None:
    frame, _ = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=["feature_b", "feature_a"],
    )
    assert frame.columns.tolist() == [TIME, TARGET, "feature_b", "feature_a"]


def test_target_and_time_are_absent_from_feature_matrix(tmp_path: Path) -> None:
    frame, features = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=[TIME, TARGET, "feature_a", "feature_b"],
    )
    model_matrix = frame.loc[:, features]
    assert TIME not in model_matrix
    assert TARGET not in model_matrix
    assert model_matrix.columns.tolist() == ["feature_a", "feature_b"]


def test_valid_csv_preserves_all_rows_and_features(tmp_path: Path) -> None:
    frame, features = load_lendingclub_required_columns(
        _valid_csv(tmp_path),
        time_column=TIME,
        target_column=TARGET,
        feature_columns=["feature_a", "feature_b"],
    )
    assert len(frame) == 3
    assert frame.columns.tolist() == [TIME, TARGET, "feature_a", "feature_b"]
    assert features == ["feature_a", "feature_b"]


def test_duplicate_required_source_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_header.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([TIME, TARGET, "feature_a", "feature_a"])
        writer.writerow([-2, 0, 1, 1])
    with pytest.raises(
        TransferStageError, match="ambiguous duplicate required header columns"
    ):
        load_lendingclub_required_columns(
            path,
            time_column=TIME,
            target_column=TARGET,
            feature_columns=["feature_a"],
        )
