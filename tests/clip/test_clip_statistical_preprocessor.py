from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_preprocessor import StatisticalPreprocessor, build_vector_frame, input_field_hash


def test_preprocessor_fits_medians_and_scaling_on_training_only():
    train = pd.DataFrame({"missing_rate_dev": [0.0, 0.2, np.nan]})
    validation = pd.DataFrame({"missing_rate_dev": [100.0]})
    external = pd.DataFrame({"missing_rate_dev": [-100.0]})

    preprocessor = StatisticalPreprocessor(field_order=["missing_rate_dev"], scaling_strategy="standard")
    preprocessor.fit(train)

    assert preprocessor.imputation_values_["missing_rate_dev"] == 0.1
    assert preprocessor.center_["missing_rate_dev"] != validation["missing_rate_dev"].mean()
    assert preprocessor.center_["missing_rate_dev"] != external["missing_rate_dev"].mean()
    transformed = preprocessor.transform(pd.concat([train, validation, external], ignore_index=True))
    assert np.isfinite(transformed.to_numpy()).all()


def test_preprocessor_rejects_unexpected_string_fields():
    preprocessor = StatisticalPreprocessor(field_order=["missing_rate_dev"])
    frame = pd.DataFrame({"missing_rate_dev": ["not-a-number"]})

    try:
        preprocessor.fit(frame)
    except ValueError as exc:
        assert "non-numeric" in str(exc)
    else:
        raise AssertionError("non-numeric statistical field should fail")


def test_field_order_and_vector_alignment_are_explicit():
    metadata = pd.DataFrame(
        {
            "dataset": ["homecredit", "homecredit"],
            "feature_name": ["A", "B"],
            "split": ["train", "validation"],
            "group_key": ["g1", "g2"],
            "semantic_group": ["s", "s"],
            "source_table_or_formula": ["src", "src"],
            "source_manifest_hash": ["hash", "hash"],
        }
    )
    transformed = pd.DataFrame({"missing_rate_dev": [0.0, 1.0]})
    preprocessor = StatisticalPreprocessor(field_order=["missing_rate_dev"]).fit(pd.DataFrame({"missing_rate_dev": [0.0, 1.0]}))

    vectors = build_vector_frame(metadata=metadata, transformed=transformed, preprocessor=preprocessor)

    assert vectors["stable_row_index"].tolist() == [0, 1]
    assert vectors["input_field_hash"].nunique() == 1
    assert vectors["input_field_hash"].iloc[0] == input_field_hash(["missing_rate_dev"])
    assert vectors["vector_dimension"].eq(1).all()
    assert vectors["statistical_vector_hash"].nunique() == 2
