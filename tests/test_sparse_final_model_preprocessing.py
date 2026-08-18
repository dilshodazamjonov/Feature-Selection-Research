from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.svm import _liblinear

from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.models.registry import get_model_bundle
from credit_risk_fs.preprocessing.encoding import Preprocessor, SparsePreprocessor


ZIP_FEATURE = "d1__person__registaddr_zipcode_184M__last_by_num_group1"


def _bounded_reference_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    boundary = ["nine"] * 9 + ["ten"] * 10 + ["eleven"] * 11
    zip_values = [f"zip_{index:04d}" for index in range(45) for _ in range(2)]
    rows = len(zip_values)
    boundary.extend(["common"] * (rows - len(boundary)))
    train = pd.DataFrame(
        {
            "numeric": np.r_[np.linspace(-3.0, 4.0, rows - 2), np.nan, np.inf],
            "numeric_constant": np.full(rows, 7.5),
            "numeric_all_missing": np.full(rows, np.nan),
            ZIP_FEATURE: pd.Series(zip_values, dtype="string"),
            "frequency_boundary": pd.Series(boundary, dtype="string"),
            "categorical_missing": pd.Series(
                [None if index % 13 == 0 else f"group_{index % 4}" for index in range(rows)],
                dtype="category",
            ),
        }
    )
    validation = train.iloc[[7, 3, 50, 1, 89]].copy()
    validation.index = pd.Index([501, 307, 211, 109, 907])
    validation.loc[307, ZIP_FEATURE] = "never_seen_zip"
    validation.loc[211, "frequency_boundary"] = "never_seen_boundary"
    validation.loc[109, "categorical_missing"] = None
    target = pd.Series(
        ((np.arange(rows) * 7 + np.arange(rows) // 3) % 11 < 4).astype("int64")
    )
    return train, validation, target


def test_sparse_encoding_is_exactly_equivalent_to_dense_reference() -> None:
    train, validation, _ = _bounded_reference_frames()
    dense = Preprocessor(
        num_strategy="mean",
        num_scaler="standard",
        cat_max_card=7,
        cat_missing="Missing",
        cat_min_frequency=10,
    )
    sparse_preprocessor = SparsePreprocessor(
        num_strategy="mean",
        num_scaler="standard",
        cat_max_card=7,
        cat_missing="Missing",
        cat_min_frequency=10,
    )

    dense_train = dense.fit_transform(train)
    sparse_train = sparse_preprocessor.fit_transform(train)
    dense_validation = dense.transform(validation)
    sparse_validation = sparse_preprocessor.transform(validation)

    assert sparse.isspmatrix_csr(sparse_train)
    assert sparse.isspmatrix_csr(sparse_validation)
    assert sparse_train.dtype == np.dtype("float32")
    assert sparse_train.has_canonical_format
    assert sparse_train.has_sorted_indices
    assert dense_train.columns.tolist() == sparse_preprocessor.get_feature_names_out()
    assert dense_validation.columns.tolist() == sparse_preprocessor.get_feature_names_out()
    assert dense_train.shape == sparse_train.shape
    assert dense_validation.shape == sparse_validation.shape
    np.testing.assert_array_equal(dense_train.to_numpy(), sparse_train.toarray())
    np.testing.assert_array_equal(
        dense_validation.to_numpy(), sparse_validation.toarray()
    )
    assert validation.index.tolist() == [501, 307, 211, 109, 907]

    dense_ohe = dense.cat_encoder.ohe
    sparse_ohe = sparse_preprocessor.cat_encoder.ohe
    assert dense_ohe is not None and sparse_ohe is not None
    for dense_categories, sparse_categories in zip(
        dense_ohe.categories_, sparse_ohe.categories_, strict=True
    ):
        np.testing.assert_array_equal(dense_categories, sparse_categories)
    for dense_infrequent, sparse_infrequent in zip(
        dense_ohe.infrequent_categories_,
        sparse_ohe.infrequent_categories_,
        strict=True,
    ):
        if dense_infrequent is None:
            assert sparse_infrequent is None
        else:
            np.testing.assert_array_equal(dense_infrequent, sparse_infrequent)

    schema = sparse_preprocessor.schema_metadata()
    assert schema["numeric"]["with_mean"] is True
    assert schema["categorical"]["min_frequency"] == 10
    assert schema["categorical"]["missing_value"] == "Missing"
    assert schema["categorical"]["handle_unknown"] == "ignore"
    assert schema["encoded_feature_names"] == dense_train.columns.tolist()


def test_frozen_logistic_dense_and_sparse_models_are_technically_equivalent() -> None:
    train, validation, target = _bounded_reference_frames()
    dense_preprocessor = Preprocessor(cat_min_frequency=10)
    sparse_preprocessor = SparsePreprocessor(cat_min_frequency=10)
    dense_train = dense_preprocessor.fit_transform(train)
    sparse_train = sparse_preprocessor.fit_transform(train)
    dense_validation = dense_preprocessor.transform(validation)
    sparse_validation = sparse_preprocessor.transform(validation)
    kwargs = {
        "solver": "liblinear",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    get_model, _, predict_proba, _ = get_model_bundle("lr", kwargs)
    dense_model = get_model().fit(dense_train, target)
    sparse_model = get_model().fit(sparse_train, target)

    dense_dev = np.asarray(predict_proba(dense_model, dense_train), dtype=np.float64)
    sparse_dev = np.asarray(predict_proba(sparse_model, sparse_train), dtype=np.float64)
    dense_validation_score = np.asarray(
        predict_proba(dense_model, dense_validation), dtype=np.float64
    )
    sparse_validation_score = np.asarray(
        predict_proba(sparse_model, sparse_validation), dtype=np.float64
    )
    np.testing.assert_allclose(
        dense_model.model.coef_, sparse_model.model.coef_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        dense_model.model.intercept_, sparse_model.model.intercept_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(dense_dev, sparse_dev, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        dense_validation_score, sparse_validation_score, rtol=0.0, atol=1e-12
    )
    dense_threshold = determine_threshold(target.to_numpy(), dense_dev)
    sparse_threshold = determine_threshold(target.to_numpy(), sparse_dev)
    assert dense_threshold == pytest.approx(sparse_threshold, rel=0.0, abs=1e-12)
    dense_metrics = evaluate_model(
        target.iloc[: len(validation)].to_numpy(),
        dense_validation_score,
        threshold=dense_threshold,
    )
    sparse_metrics = evaluate_model(
        target.iloc[: len(validation)].to_numpy(),
        sparse_validation_score,
        threshold=sparse_threshold,
    )
    assert dense_metrics.keys() == sparse_metrics.keys()
    for key in dense_metrics:
        assert dense_metrics[key] == pytest.approx(
            sparse_metrics[key], rel=0.0, abs=1e-12, nan_ok=True
        )


def test_liblinear_accepts_float32_csr_without_a_full_input_conversion(monkeypatch) -> None:
    train, _, target = _bounded_reference_frames()
    encoded = SparsePreprocessor(cat_min_frequency=10).fit_transform(train)
    observed: dict[str, object] = {}
    original = _liblinear.train_wrap

    def inspected_train_wrap(X, *args, **kwargs):
        observed["format"] = X.format
        observed["dtype"] = str(X.dtype)
        observed["same_data_owner"] = np.shares_memory(X.data, encoded.data)
        return original(X, *args, **kwargs)

    monkeypatch.setattr(_liblinear, "train_wrap", inspected_train_wrap)
    get_model, _, _, _ = get_model_bundle(
        "lr",
        {
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
    )
    model = get_model().fit(encoded, target)
    assert observed == {
        "format": "csr",
        "dtype": "float32",
        "same_data_owner": True,
    }
    assert model.model.coef_.dtype == np.dtype("float64")


def test_catboost_frozen_representation_accepts_csr_without_native_categories() -> None:
    train, _, target = _bounded_reference_frames()
    encoded = SparsePreprocessor(cat_min_frequency=10).fit_transform(train)
    get_model, _, predict_proba, _ = get_model_bundle(
        "catboost",
        {
            "depth": 3,
            "learning_rate": 0.01,
            "l2_leaf_reg": 95,
            "min_data_in_leaf": 20,
            "colsample_bylevel": 0.9,
            "random_strength": 0.125,
            "grow_policy": "Depthwise",
            "one_hot_max_size": 21,
            "leaf_estimation_method": "Newton",
            "bootstrap_type": "Bernoulli",
            "subsample": 0.55,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "iterations": 5,
            "early_stopping_rounds": 150,
            "verbose": False,
            "random_state": 42,
            "allow_writing_files": False,
            "thread_count": 1,
        },
    )
    model = get_model().fit(encoded, target, eval_set=None)
    scores = np.asarray(predict_proba(model, encoded), dtype=np.float64)
    assert scores.shape == (len(train),)
    assert np.isfinite(scores).all()


def test_sparse_production_transform_contains_no_dense_materialization_call() -> None:
    source = inspect.getsource(SparsePreprocessor.transform)
    assert ".toarray(" not in source
    assert ".todense(" not in source
    assert "pd.concat" not in source
    assert "sparse.hstack" in source
