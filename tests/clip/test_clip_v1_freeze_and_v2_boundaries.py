from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.clip.statistical_preprocessor_v2 import RobustStatisticalPreprocessorV2
from credit_risk_fs.clip.statistical_schema_v2 import (
    DESCRIPTOR_COLUMNS_V2,
    SCALED_DESCRIPTOR_COLUMNS_V2,
    UNSCALED_INDICATOR_COLUMNS_V2,
    descriptor_order_v2,
)
from credit_risk_fs.clip.statistical_view_v2 import (
    TYPE_BINARY,
    TYPE_CATEGORICAL,
    TYPE_NUMERIC,
    build_statistical_view_frame,
    compute_feature_descriptors,
    resolve_feature_type,
    validate_allowed_input_columns,
)
from credit_risk_fs.clip.versioning import (
    CLIP_V1,
    CLIP_V1_STATISTICAL_VIEW,
    CLIP_V2,
    CLIP_V2_STATISTICAL_VIEW,
    assert_version_output_root,
    versioned_cache_namespace,
)


def test_clip_v1_freeze_manifest_is_complete_and_missingness_only():
    manifest = json.loads(Path("results/clip_versions/v1/freeze_manifest.json").read_text(encoding="utf-8"))

    assert manifest["version_name"] == "clip_v1"
    assert manifest["scientific_name"] == "missingness_only"
    assert manifest["statistical_vector_dimension"] == 1
    assert manifest["statistical_field_list"] == ["missing_rate_dev"]
    assert manifest["training_dataset"] == "homecredit"
    assert manifest["external_validation_dataset"] == "lendingclub_v2"
    assert len(manifest["evaluation_run_ids"]) == 8
    assert len(manifest["prediction_hashes"]) == 8
    assert manifest["artifact_count"] >= 20


def test_clip_v2_descriptor_order_is_exact_and_deterministic():
    assert len(DESCRIPTOR_COLUMNS_V2) == 13
    assert descriptor_order_v2() == [
        "missing_rate",
        "unique_ratio",
        "concentration_share",
        "signed_log_mean",
        "log_standard_deviation",
        "clipped_skewness",
        "normalized_entropy",
        "is_numeric",
        "is_categorical",
        "is_binary",
        "numeric_stats_valid",
        "skewness_valid",
        "entropy_valid",
    ]
    assert SCALED_DESCRIPTOR_COLUMNS_V2 == DESCRIPTOR_COLUMNS_V2[:7]
    assert UNSCALED_INDICATOR_COLUMNS_V2 == DESCRIPTOR_COLUMNS_V2[7:]


def test_numeric_descriptor_math_and_constant_validity_masks():
    row = compute_feature_descriptors(pd.Series([0.0, 2.0, 4.0, np.nan]), feature_name="amount", metadata_type="numeric")

    assert row["resolved_type"] == TYPE_NUMERIC
    assert row["missing_rate"] == pytest.approx(0.25)
    assert row["unique_ratio"] == pytest.approx(1.0)
    assert row["concentration_share"] == pytest.approx(1 / 3)
    assert row["signed_log_mean"] == pytest.approx(np.log1p(2.0))
    assert row["log_standard_deviation"] > 0
    assert row["numeric_stats_valid"] == 1
    assert row["entropy_valid"] == 0

    constant = compute_feature_descriptors(pd.Series([5.0, 5.0, 5.0]), feature_name="constant", metadata_type="numeric")
    assert constant["clipped_skewness"] == 0.0
    assert constant["skewness_valid"] == 0


def test_categorical_entropy_binary_concentration_and_all_missing():
    categorical = compute_feature_descriptors(pd.Series(["a", "a", "b", "c"]), feature_name="cat", metadata_type="categorical")
    assert categorical["resolved_type"] == TYPE_CATEGORICAL
    assert categorical["concentration_share"] == pytest.approx(0.5)
    assert 0.0 < categorical["normalized_entropy"] <= 1.0
    assert categorical["entropy_valid"] == 1

    binary = compute_feature_descriptors(pd.Series([1, 1, 0, 1]), feature_name="flag", metadata_type="binary")
    assert binary["resolved_type"] == TYPE_BINARY
    assert binary["concentration_share"] == pytest.approx(0.75)
    assert binary["is_binary"] == 1

    missing = compute_feature_descriptors(pd.Series([np.nan, np.nan]), feature_name="missing", metadata_type="numeric")
    assert missing["missing_rate"] == 1.0
    assert missing["concentration_share"] == 0.0
    assert missing["numeric_stats_valid"] == 0


def test_integer_coded_category_uses_metadata_type_resolution():
    values = pd.Series([1, 2, 3, 1])
    resolution = resolve_feature_type(values, feature_name="coded", metadata_type="categorical")

    assert resolution.resolved_type == TYPE_CATEGORICAL
    assert resolution.resolution_rule == "metadata_categorical"


def test_forbidden_target_oot_prediction_and_llm_inputs_are_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        validate_allowed_input_columns(["feature_a", "TARGET"])
    with pytest.raises(ValueError, match="forbidden"):
        build_statistical_view_frame(pd.DataFrame({"good": [1, 2], "oot_score_psi": [0.1, 0.2]}))
    with pytest.raises(ValueError, match="forbidden"):
        validate_allowed_input_columns(["llm_rank", "prediction_score", "post_origination_flag"])


def test_clip_v2_scaler_fits_only_homecredit_train_and_does_not_scale_indicators():
    frame = pd.DataFrame(
        {
            "missing_rate": [0.0, 0.2, 1.0],
            "unique_ratio": [0.5, 0.5, 1.0],
            "concentration_share": [0.0, 0.5, 1.0],
            "signed_log_mean": [0.0, 1.0, 2.0],
            "log_standard_deviation": [0.0, 0.5, 1.0],
            "clipped_skewness": [0.0, 10.0, -10.0],
            "normalized_entropy": [0.0, 0.5, 1.0],
            "is_numeric": [1, 0, 0],
            "is_categorical": [0, 1, 0],
            "is_binary": [0, 0, 1],
            "numeric_stats_valid": [1, 0, 0],
            "skewness_valid": [0, 0, 0],
            "entropy_valid": [0, 1, 1],
        }
    )
    preprocessor = RobustStatisticalPreprocessorV2()

    with pytest.raises(ValueError, match="Home Credit"):
        preprocessor.fit(frame, dataset="lendingclub_v2", split="train")
    with pytest.raises(ValueError, match="Home Credit"):
        preprocessor.fit(frame, dataset="homecredit", split="validation")

    transformed = preprocessor.fit_transform(frame, dataset="homecredit", split="train")
    assert transformed.shape[1] == 13
    assert transformed[UNSCALED_INDICATOR_COLUMNS_V2].equals(frame[UNSCALED_INDICATOR_COLUMNS_V2].astype("float32"))
    assert preprocessor.fit_feature_count_ == 3
    assert preprocessor.fit_split_hash_

    with pytest.raises(ValueError, match="must not refit"):
        preprocessor.transform(frame, allow_refit=True)


def test_versioned_paths_and_cache_namespaces_are_isolated():
    assert_version_output_root(experiment_version=CLIP_V1, output_root="results/clip")
    assert_version_output_root(experiment_version=CLIP_V2, output_root="results/clip_v2")
    with pytest.raises(ValueError):
        assert_version_output_root(experiment_version=CLIP_V2, output_root="results/clip")

    v1 = versioned_cache_namespace(experiment_version=CLIP_V1, statistical_view_version=CLIP_V1_STATISTICAL_VIEW)
    v2 = versioned_cache_namespace(experiment_version=CLIP_V2, statistical_view_version=CLIP_V2_STATISTICAL_VIEW)
    assert v1 != v2


def test_clip_v2_dry_run_scripts_do_not_require_generated_artifacts():
    commands = [
        ["scripts/build_clip_v2_statistical_view.py", "--dry-run"],
        ["scripts/build_clip_v2_contrastive_data.py", "--dry-run"],
        ["scripts/train_clip_v2_encoder.py", "--dry-run"],
        ["scripts/validate_clip_v2_selector_integration.py", "--dry-run"],
        ["scripts/run_clip_v2_final_evaluation.py", "--plan"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, *command], check=False, capture_output=True, text=True)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
        payload = json.loads(result.stdout)
        assert payload.get("model_trained") is False or payload.get("expensive_model_training") is False
