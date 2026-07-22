from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.data.loaders import DataLoader
from credit_risk_fs.pipelines.common import ExperimentConfig, calculate_required_columns


def test_csv_projection_and_chunked_load(tmp_path):
    frame = pd.DataFrame({"id": [1, 2], "keep": [1.5, 2.5], "drop": ["x", "y"]})
    frame.to_csv(tmp_path / "application_train.csv", index=False)
    loader = DataLoader(tmp_path)
    loaded = loader.load_table(
        "application_train",
        columns=["id", "keep"],
        require_projection=True,
        csv_chunk_rows=1,
    )
    assert list(loaded.columns) == ["id", "keep"]
    assert loader.last_load_report["application_train"]["requested_columns"] == ["id", "keep"]


def test_parquet_projection_uses_explicit_columns(tmp_path):
    pd.DataFrame({"id": [1], "keep": [2.0], "drop": [3]}).to_parquet(
        tmp_path / "application_train.parquet", index=False
    )
    loaded = DataLoader(tmp_path).load_table(
        "application_train", columns=["keep"], require_projection=True
    )
    assert list(loaded.columns) == ["keep"]


def test_accidental_unprojected_experiment_load_is_rejected(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "application_train.csv", index=False)
    loader = DataLoader(tmp_path)
    with pytest.raises(ValueError, match="explicit columns projection"):
        loader.load_table("application_train", columns=None, require_projection=True)
    with pytest.raises(ValueError, match="calculated column projections"):
        loader.load_all(require_projection=True)


def test_explicit_full_candidate_projection_is_accepted(tmp_path):
    frame = pd.DataFrame({"TARGET": [0], "recent_decision": [-1], "f1": [1.0]})
    frame.to_csv(tmp_path / "application_train.csv", index=False)
    loader = DataLoader(tmp_path)
    config = ExperimentConfig(
        experiment_name="synthetic",
        selector_name="none",
        dataset_name="synthetic",
        data_dir=str(tmp_path),
        require_full_candidate_projection=True,
    )
    projections = calculate_required_columns(config, loader)
    assert projections == {"application_train": ["TARGET", "recent_decision", "f1"]}
    loaded = loader.load_all(projections, require_projection=True)
    assert list(loaded["application_train"].columns) == projections["application_train"]


def test_selected_feature_projection_requires_declared_features(tmp_path):
    pd.DataFrame(
        {
            "record_id": ["a"],
            "TARGET": [0],
            "recent_decision": [-1],
            "f1": [1.0],
            "f2": [2.0],
        }
    ).to_csv(tmp_path / "application_train.csv", index=False)
    loader = DataLoader(tmp_path)
    config = ExperimentConfig(
        experiment_name="synthetic",
        selector_name="none",
        dataset_name="synthetic",
        data_dir=str(tmp_path),
        drop_id_cols=("record_id",),
        required_feature_columns=("f2",),
        require_full_candidate_projection=False,
    )
    assert calculate_required_columns(config, loader) == {
        "application_train": ["TARGET", "recent_decision", "record_id", "f2"]
    }


def test_loader_does_not_silently_downcast(tmp_path):
    frame = pd.DataFrame({"value": pd.Series([1.25, 2.5], dtype="float64")})
    frame.to_parquet(tmp_path / "application_train.parquet", index=False)
    loaded = DataLoader(tmp_path).load_table(
        "application_train", columns=["value"], require_projection=True
    )
    assert str(loaded["value"].dtype) == "float64"
