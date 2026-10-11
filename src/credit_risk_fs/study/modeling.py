"""Preprocessing and the two fixed backbones.

Preprocessing is fitted on the training partition and applied unchanged to the
evaluation rows: numeric infinities are set to missing, numerics are mean-imputed and
standardised, categoricals receive an explicit missing token and are one-hot encoded,
levels unseen in training are ignored and levels rarer than ``cat_min_frequency``
observations are dropped. Selection operates on original variables, so the one-hot
expansion follows selection.

Backbones (configurations fixed and identical across selectors): logistic regression
(``liblinear``, L2, ``C`` = 1, 1,000 iterations, balanced class weights, seed 42) and
CatBoost (depth 10, learning rate 0.01, L2 leaf regularisation 95, 290 observations per
leaf, ``rsm`` 0.90, random strength 0.125, depthwise growth, Bernoulli subsample 0.55,
Logloss with AUC evaluation, balanced class weights, seed 42, four threads, a fixed
1,500 iterations without early stopping).
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

INFREQUENT_SUFFIX = "_infrequent_sklearn"


def _preprocessor(cfg: dict[str, Any], *, sparse_output: bool = False):
    from credit_risk_fs.preprocessing.encoding import Preprocessor, SparsePreprocessor

    settings = cfg.get("preprocessing", {})
    kwargs = {
        "num_strategy": settings.get("num_strategy", "mean"),
        "num_scaler": settings.get("num_scaler", "standard"),
        "cat_missing": settings.get("cat_missing", "Missing"),
        "cat_min_frequency": int(cfg["cat_min_frequency"]),
    }
    return SparsePreprocessor(**kwargs) if sparse_output else Preprocessor(**kwargs)


def _feature_names(preprocessor: Any) -> list[str]:
    numeric = list(preprocessor.num_scaler.num_cols)
    encoder = preprocessor.cat_encoder
    categorical = (
        encoder.ohe.get_feature_names_out(encoder.cat_cols).astype(str).tolist() if encoder.ohe is not None else []
    )
    return numeric + categorical


class FittedPreprocessing:
    """A training-partition preprocessing map with rare-level columns removed."""

    def __init__(self, cfg: dict[str, Any], X_train: pd.DataFrame, *, sparse_output: bool = False) -> None:
        self.sparse_output = bool(sparse_output)
        self.preprocessor = _preprocessor(cfg, sparse_output=sparse_output)
        self.preprocessor.fit(X_train)
        names = _feature_names(self.preprocessor)
        self.keep = np.asarray([not name.endswith(INFREQUENT_SUFFIX) for name in names], dtype=bool)
        self.names = [name for name, kept in zip(names, self.keep) if kept]
        self.numeric = list(self.preprocessor.num_scaler.num_cols)
        self.categorical = list(self.preprocessor.cat_encoder.cat_cols)

    def transform(self, X: pd.DataFrame):
        output = self.preprocessor.transform(X)
        if self.sparse_output:
            matrix = sparse.csr_matrix(output)
            return matrix[:, np.flatnonzero(self.keep)] if not self.keep.all() else matrix
        frame = output if isinstance(output, pd.DataFrame) else pd.DataFrame(output)
        if not self.keep.all():
            frame = frame.loc[:, [column for column, kept in zip(frame.columns, self.keep) if kept]]
        return frame.astype("float32")

    @property
    def n_columns(self) -> int:
        return len(self.names)

    def source_of(self, column: str) -> str:
        """Original variable behind an output column (itself for a numeric column)."""

        if column in self.numeric:
            return column
        candidates = [name for name in self.categorical if column.startswith(f"{name}_")]
        return max(candidates, key=len) if candidates else column


def make_model(cfg: dict[str, Any], backbone: str):
    params = dict(cfg["models"][backbone])
    if backbone == "lr":
        from sklearn.linear_model import LogisticRegression

        # L2 is the estimator default; scikit-learn 1.8 deprecates passing ``penalty``.
        penalty = params.pop("penalty", "l2")
        if penalty != "l2":
            params["penalty"] = penalty
        return LogisticRegression(**params)
    if backbone == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)
    raise ValueError(f"unknown backbone {backbone!r}")


def positive_scores(model: Any, Z: Any) -> np.ndarray:
    return np.asarray(model.predict_proba(Z))[:, 1].astype(float)


def _as_model_input(Z: Any) -> Any:
    if sparse.issparse(Z):
        return Z
    return np.ascontiguousarray(np.asarray(Z, dtype=np.float32))


def fit_score(
    cfg: dict[str, Any],
    backbone: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_evals: Sequence[pd.DataFrame],
    *,
    mode: str = "subset",
    columns: Sequence[str] | None = None,
    pca_components: int | None = None,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Fit the preprocessing and the backbone on training rows; score every evaluation frame.

    ``mode``: ``subset`` (one-hot expansion of the selected original variables),
    ``expanded`` (``columns`` are output columns of the preprocessing map, selected on
    the expanded scale), ``pca`` (the leading ``pca_components`` components of the
    preprocessed candidates) or ``full`` (all supplied variables, sparse map).
    Returns the score vectors, the number of model input columns and their names.
    """

    preprocessing = FittedPreprocessing(cfg, X_train, sparse_output=(mode == "full"))
    Z_train = preprocessing.transform(X_train)
    Z_evals = [preprocessing.transform(frame) for frame in X_evals]
    names = list(preprocessing.names)
    if mode == "expanded":
        wanted = [str(column) for column in columns or []]
        missing = [column for column in wanted if column not in names]
        if missing:
            raise ValueError(f"expanded columns absent from the preprocessing map: {missing[:5]}")
        Z_train = Z_train.loc[:, wanted]
        Z_evals = [frame.loc[:, wanted] for frame in Z_evals]
        names = wanted
    elif mode == "pca":
        from sklearn.decomposition import PCA

        options: dict[str, Any] = {"n_components": int(pca_components), "random_state": int(cfg["seed"])}
        if _supports_covariance_eigh():
            # Exact eigendecomposition of the covariance matrix: deterministic and
            # memory-light when rows far outnumber columns.
            options["svd_solver"] = "covariance_eigh"
        pca = PCA(**options).fit(np.asarray(Z_train, dtype=np.float32))
        Z_train = pca.transform(np.asarray(Z_train, dtype=np.float32))
        Z_evals = [pca.transform(np.asarray(frame, dtype=np.float32)) for frame in Z_evals]
        names = [f"PC{index}" for index in range(1, int(pca_components) + 1)]
    model = make_model(cfg, backbone)
    model.fit(_as_model_input(Z_train), np.asarray(y_train, dtype=int))
    scores = [positive_scores(model, _as_model_input(Z)) for Z in Z_evals]
    return scores, len(names), names


def _supports_covariance_eigh() -> bool:
    import sklearn

    try:
        major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover
        return False
    return (major, minor) >= (1, 5)


def fit_score_native(
    cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    """CatBoost with native categorical handling: same numerics, categoricals as cat_features."""

    from catboost import CatBoostClassifier

    from credit_risk_fs.preprocessing.encoding import NumericalScaler

    settings = cfg.get("preprocessing", {})
    missing_token = settings.get("cat_missing", "Missing")
    scaler = NumericalScaler(strategy=settings.get("num_strategy", "mean"), scaler=settings.get("num_scaler", "standard"))
    scaler.fit(X_train)
    categorical = [
        name for name in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[name].dtype) or pd.api.types.is_bool_dtype(X_train[name].dtype)
    ]

    def frame(X: pd.DataFrame) -> pd.DataFrame:
        numeric = scaler.transform(X)
        parts = [numeric.reset_index(drop=True)]
        if categorical:
            parts.append(
                pd.DataFrame(
                    {name: X[name].astype("string").fillna(missing_token).astype(str).to_numpy() for name in categorical}
                )
            )
        return pd.concat(parts, axis=1)

    model = CatBoostClassifier(**dict(cfg["models"]["catboost"]))
    model.fit(frame(X_train), np.asarray(y_train, dtype=int), cat_features=categorical)
    return positive_scores(model, frame(X_eval))


__all__ = ["FittedPreprocessing", "INFREQUENT_SUFFIX", "fit_score", "fit_score_native", "make_model", "positive_scores"]
