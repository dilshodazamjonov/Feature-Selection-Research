# preprocessing/preprocessing.py

import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional

import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


class OriginalFeatureNumericEncoder:
    """Training-only numeric encoding with one output per original feature.

    This encoder is intentionally limited to supervised feature-selection
    stages.  Unlike the final-model preprocessor it never expands categorical
    columns, so rankings and RFE selections retain a one-to-one relationship
    with the frozen original candidate names.
    """

    missing_token = "<MISSING>"

    def __init__(self) -> None:
        self.feature_names_: list[str] | None = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.fill_values_: dict[str, float] = {}
        self.category_maps_: dict[str, dict[str, int]] = {}

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
            raise ValueError("OriginalFeatureNumericEncoder requires a non-empty DataFrame")
        names = [str(column) for column in X.columns]
        if len(names) != len(set(names)):
            raise ValueError("selection encoding requires unique original feature names")
        self.feature_names_ = names
        self.numeric_columns_ = [
            name for name in names if pd.api.types.is_numeric_dtype(X[name])
        ]
        self.categorical_columns_ = [
            name for name in names if name not in self.numeric_columns_
        ]
        self.fill_values_ = {}
        for name in self.numeric_columns_:
            values = pd.to_numeric(X[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = values.median()
            self.fill_values_[name] = 0.0 if pd.isna(median) else float(median)
        self.category_maps_ = {}
        for name in self.categorical_columns_:
            values = X[name].astype("string").fillna(self.missing_token).astype(str)
            categories = sorted(set(values), key=lambda value: value.casefold())
            self.category_maps_[name] = {
                value: index for index, value in enumerate(categories)
            }
        return self

    def _encode_column(self, name: str, source: pd.Series) -> pd.Series:
        if name in self.numeric_columns_:
            return (
                pd.to_numeric(source, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.fill_values_[name])
                .astype("float32")
            )
        values = source.astype("string").fillna(self.missing_token).astype(str)
        return values.map(self.category_maps_[name]).fillna(-1).astype("float32")

    def _validate_transform_input(self, X: pd.DataFrame) -> list[str]:
        if self.feature_names_ is None:
            raise ValueError("OriginalFeatureNumericEncoder has not been fitted")
        observed = [str(column) for column in X.columns]
        if observed != self.feature_names_:
            raise ValueError(
                "selection encoding column order mismatch: "
                f"expected={self.feature_names_}, observed={observed}"
            )
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_names = self._validate_transform_input(X)
        # A single C-contiguous float32 owner preserves the existing effective
        # dtype while avoiding a simultaneously-live dictionary of 675 encoded
        # Series plus the consolidated DataFrame block.  Each column is still
        # produced by the exact same pandas conversion/mapping expression.
        output = np.empty((len(X), len(feature_names)), dtype=np.float32)
        for position, name in enumerate(feature_names):
            encoded = self._encode_column(name, X[name])
            output[:, position] = encoded.to_numpy(dtype=np.float32, copy=False)
        return pd.DataFrame(
            output,
            index=X.index,
            columns=feature_names,
            copy=False,
        )

    def transform_releasing_source(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform while destructively releasing each consumed source column.

        This opt-in path is for memory-bounded training-only workflows that no
        longer need ``X``. Values and order are identical to :meth:`transform`,
        but a source column is removed before its encoded values are copied into
        the single output buffer. Wide float64/object source blocks therefore do
        not remain resident while the complete float32 matrix is populated.
        """

        feature_names = self._validate_transform_input(X)
        output_index = X.index.copy(deep=True)
        output = np.empty((len(X), len(feature_names)), dtype=np.float32)
        for position, name in enumerate(feature_names):
            source = X.pop(name)
            encoded = self._encode_column(name, source)
            del source
            output[:, position] = encoded.to_numpy(dtype=np.float32, copy=False)
            del encoded
        return pd.DataFrame(
            output,
            index=output_index,
            columns=feature_names,
            copy=False,
        )

    def transform_releasing_source_to_memmap(
        self,
        X: pd.DataFrame,
        path: str | Path,
        *,
        feature_split_size: int = 64,
        before_split: Callable[[int, int, list[str]], None] | None = None,
    ) -> tuple[pd.DataFrame, np.memmap]:
        """Encode exact values into a batched, disk-backed float32 matrix.

        The dense in-memory implementation allocates the complete output owner
        before all wide source blocks have been returned to the operating
        system.  For the Prompt-16 full-DEV projection that transient overlap is
        larger than the frozen RAM ceiling.  This method instead writes one
        bounded feature split at a time to a Fortran-ordered ``.npy`` memmap.
        Each mapping is flushed and closed before the next split, allowing
        Windows to reclaim file-backed pages under memory pressure.

        Encoding expressions, learned fill values, category maps, row order,
        feature order, and float32 values are identical to :meth:`transform`.
        ``X`` is destructively emptied exactly as in
        :meth:`transform_releasing_source`.
        """

        feature_names = self._validate_transform_input(X)
        split_size = int(feature_split_size)
        if split_size <= 0:
            raise ValueError("feature_split_size must be positive")
        output_path = Path(path)
        if output_path.exists():
            raise FileExistsError(f"disk-backed selector encoding already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_index = X.index.copy(deep=True)
        shape = (len(X), len(feature_names))

        # Create only the NPY header and file extent here. Each subsequent
        # feature split gets its own short-lived mapping, so no complete mapped
        # working set overlaps the still-resident source frame.
        mapping = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
            fortran_order=True,
        )
        mapping.flush()
        mapping._mmap.close()
        del mapping

        split_count = (len(feature_names) + split_size - 1) // split_size
        try:
            for split_index, start in enumerate(
                range(0, len(feature_names), split_size), start=1
            ):
                stop = min(start + split_size, len(feature_names))
                split_names = feature_names[start:stop]
                if before_split is not None:
                    before_split(split_index, split_count, list(split_names))
                split = np.empty(
                    (len(X), len(split_names)),
                    dtype=np.float32,
                    order="F",
                )
                for offset, name in enumerate(split_names):
                    source = X.pop(name)
                    encoded = self._encode_column(name, source)
                    del source
                    split[:, offset] = encoded.to_numpy(dtype=np.float32, copy=False)
                    del encoded
                mapping = np.lib.format.open_memmap(output_path, mode="r+")
                mapping[:, start:stop] = split
                mapping.flush()
                mapping._mmap.close()
                del mapping, split
        except BaseException:
            # Leave the explicitly named partial artifact to the authenticated
            # cache owner. It decides whether to archive or rebuild it.
            raise

        read_only = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(read_only, np.memmap):
            raise RuntimeError("disk-backed selector encoding did not reopen as a memmap")
        if read_only.shape != shape or read_only.dtype != np.dtype("float32"):
            read_only._mmap.close()
            raise RuntimeError("disk-backed selector encoding shape or dtype changed")
        frame = pd.DataFrame(
            read_only,
            index=output_index,
            columns=feature_names,
            copy=False,
        )
        if not np.shares_memory(frame.to_numpy(copy=False), read_only):
            read_only._mmap.close()
            raise RuntimeError("pandas copied the disk-backed selector encoding")
        return frame, read_only

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class NumericalScaler:
    """
    Preprocess numerical features: missing value filling and scaling.
    """

    def __init__(self, strategy: str = "mean", scaler: str = "standard"):
        self.strategy = strategy  # 'mean', 'median', 'zero', 'constant'
        self.scaler_type = scaler  # 'standard', 'minmax', None
        self.scaler = None
        self.num_cols: List[str] = []

    def fit(self, X: pd.DataFrame):
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        X_num = X[self.num_cols].copy()
        X_num = X_num.replace([np.inf, -np.inf], np.nan)

        # handle missing values
        if self.strategy == "mean":
            self.fill_values_ = X_num.mean()
        elif self.strategy == "median":
            self.fill_values_ = X_num.median()
        elif self.strategy == "zero":
            self.fill_values_ = pd.Series(0, index=self.num_cols)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        self.fill_values_ = self.fill_values_.fillna(0)
        X_num = X_num.fillna(self.fill_values_)
        X_num = X_num.fillna(0)

        # fit scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler().fit(X_num)
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler().fit(X_num)
        else:
            self.scaler = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X[self.num_cols].copy()
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_)
        X_num = X_num.fillna(0)

        if self.scaler:
            X_num = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=self.num_cols,
                index=X.index
            )
        X_num = X_num.astype("float32")

        return X_num

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class CategoricalEncoder:
    """
    Preprocess categorical features with consistent one-hot encoding.
    """

    def __init__(
        self,
        max_cardinality: int = 7,
        missing_value: str = "Missing",
        min_frequency: int | float | None = 10,
        sparse_output: bool = False,
    ):
        self.max_cardinality = max_cardinality
        self.missing_value = missing_value
        self.min_frequency = min_frequency
        self.sparse_output = bool(sparse_output)
        self.cat_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None

    def _prepare_categorical_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared: dict[str, pd.Series] = {}
        for column in self.cat_cols:
            series = X[column].astype("string")
            prepared[column] = series.fillna(self.missing_value).astype(str)
        return pd.DataFrame(prepared, index=X.index)

    def fit(self, X: pd.DataFrame):
        self.cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        if self.cat_cols:
            X_cat = self._prepare_categorical_frame(X)
            self.ohe = OneHotEncoder(
                sparse_output=self.sparse_output,
                handle_unknown="ignore",
                min_frequency=self.min_frequency,
                dtype=np.float32,
            )
            self.ohe.fit(X_cat)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | sparse.csr_matrix:
        if self.cat_cols and self.ohe is not None:
            X_cat = self._prepare_categorical_frame(X)
            X_encoded = self.ohe.transform(X_cat)
            if self.sparse_output:
                encoded = sparse.csr_matrix(X_encoded, dtype=np.float32, copy=False)
                encoded.sum_duplicates()
                encoded.eliminate_zeros()
                encoded.sort_indices()
                return encoded
            return pd.DataFrame(
                X_encoded,
                columns=self.ohe.get_feature_names_out(self.cat_cols),
                index=X.index
            ).astype("float32")

        if self.sparse_output:
            return sparse.csr_matrix((len(X), 0), dtype=np.float32)
        return pd.DataFrame(index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class Preprocessor:
    """
    Full preprocessing pipeline combining numerical scaling and categorical encoding.
    """

    def __init__(
        self,
        num_strategy="mean",
        num_scaler="standard",
        cat_max_card=7,
        cat_missing="Missing",
        cat_min_frequency=10,
    ):
        self.num_scaler = NumericalScaler(strategy=num_strategy, scaler=num_scaler)
        self.cat_encoder = CategoricalEncoder(
            max_cardinality=cat_max_card,
            missing_value=cat_missing,
            min_frequency=cat_min_frequency,
        )

    def fit(self, X: pd.DataFrame):
        self.num_scaler.fit(X)
        self.cat_encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = self.num_scaler.transform(X)
        X_cat = self.cat_encoder.transform(X)
        X_final = pd.concat([X_num, X_cat], axis=1)
        return X_final.astype("float32")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar without changing fitted values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Return a canonical, sorted CSR matrix without dense materialization."""

    output = sparse.csr_matrix(matrix, copy=False)
    output.sum_duplicates()
    output.eliminate_zeros()
    output.sort_indices()
    if not output.has_canonical_format or not output.has_sorted_indices:
        raise RuntimeError("CSR canonicalization failed")
    return output


def csr_audit_metadata(matrix: sparse.spmatrix) -> dict[str, Any]:
    """Describe and hash the three CSR buffers for auditable preprocessing."""

    output = canonical_csr(matrix)

    def digest(values: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(values)
        return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()

    return {
        "format": "csr",
        "shape": [int(output.shape[0]), int(output.shape[1])],
        "dtype": str(output.dtype),
        "nnz": int(output.nnz),
        "has_sorted_indices": bool(output.has_sorted_indices),
        "has_canonical_format": bool(output.has_canonical_format),
        "data_dtype": str(output.data.dtype),
        "indices_dtype": str(output.indices.dtype),
        "indptr_dtype": str(output.indptr.dtype),
        "estimated_csr_bytes": int(
            output.data.nbytes + output.indices.nbytes + output.indptr.nbytes
        ),
        "data_sha256": digest(output.data),
        "indices_sha256": digest(output.indices),
        "indptr_sha256": digest(output.indptr),
    }


class SparsePreprocessor:
    """Final-model preprocessing with exact legacy semantics and CSR output.

    The bounded numeric block is transformed by :class:`NumericalScaler`
    exactly as in :class:`Preprocessor`, including centered
    ``StandardScaler`` behavior and the final float32 cast. Only that already
    transformed block is converted to CSR. Categorical values use the same
    missing token, frequency threshold, category ordering, infrequent bucket,
    and unknown handling, but ``OneHotEncoder`` emits sparse output directly.
    """

    matrix_dtype = np.dtype("float32")

    def __init__(
        self,
        num_strategy: str = "mean",
        num_scaler: str = "standard",
        cat_max_card: int = 7,
        cat_missing: str = "Missing",
        cat_min_frequency: int | float | None = 10,
    ) -> None:
        self.num_strategy = num_strategy
        self.num_scaler_type = num_scaler
        self.cat_max_card = cat_max_card
        self.cat_missing = cat_missing
        self.cat_min_frequency = cat_min_frequency
        self.num_scaler = NumericalScaler(strategy=num_strategy, scaler=num_scaler)
        self.cat_encoder = CategoricalEncoder(
            max_cardinality=cat_max_card,
            missing_value=cat_missing,
            min_frequency=cat_min_frequency,
            sparse_output=True,
        )
        self.input_feature_names_: list[str] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
            raise ValueError("SparsePreprocessor requires a non-empty DataFrame")
        self.input_feature_names_ = [str(column) for column in X.columns]
        if len(self.input_feature_names_) != len(set(self.input_feature_names_)):
            raise ValueError("SparsePreprocessor requires unique feature names")
        self.num_scaler.fit(X)
        self.cat_encoder.fit(X)
        categorical_names = (
            self.cat_encoder.ohe.get_feature_names_out(
                self.cat_encoder.cat_cols
            ).astype(str).tolist()
            if self.cat_encoder.ohe is not None
            else []
        )
        self.feature_names_ = list(self.num_scaler.num_cols) + categorical_names
        return self

    def _validate_input(self, X: pd.DataFrame) -> None:
        if self.input_feature_names_ is None or self.feature_names_ is None:
            raise ValueError("SparsePreprocessor has not been fitted")
        observed = [str(column) for column in X.columns]
        if observed != self.input_feature_names_:
            raise ValueError(
                "final-model preprocessing column order mismatch: "
                f"expected={self.input_feature_names_}, observed={observed}"
            )

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        self._validate_input(X)
        numeric_frame = self.num_scaler.transform(X)
        numeric = sparse.csr_matrix(
            numeric_frame.to_numpy(dtype=np.float32, copy=False),
            dtype=np.float32,
            copy=False,
        )
        categorical = self.cat_encoder.transform(X)
        if not sparse.isspmatrix_csr(categorical):
            raise RuntimeError("sparse categorical encoder returned a dense matrix")
        output = sparse.hstack((numeric, categorical), format="csr", dtype=np.float32)
        output = canonical_csr(output)
        if output.dtype != self.matrix_dtype:
            raise RuntimeError(f"final-model CSR dtype changed: {output.dtype}")
        if output.shape[1] != len(self.get_feature_names_out()):
            raise RuntimeError("final-model CSR feature metadata length changed")
        return output

    def fit_transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        if self.feature_names_ is None:
            raise ValueError("SparsePreprocessor has not been fitted")
        return list(self.feature_names_)

    def schema_metadata(self) -> dict[str, Any]:
        if self.input_feature_names_ is None or self.feature_names_ is None:
            raise ValueError("SparsePreprocessor has not been fitted")
        ohe = self.cat_encoder.ohe
        categories = [] if ohe is None else [
            [_json_scalar(value) for value in values] for values in ohe.categories_
        ]
        infrequent = []
        if ohe is not None:
            for values in ohe.infrequent_categories_:
                infrequent.append(
                    None
                    if values is None
                    else [_json_scalar(value) for value in values]
                )
        scaler = self.num_scaler.scaler
        return {
            "implementation": (
                "credit_risk_fs.preprocessing.encoding.SparsePreprocessor"
            ),
            "matrix_format": "csr",
            "matrix_dtype": str(self.matrix_dtype),
            "input_feature_names": list(self.input_feature_names_),
            "numeric_columns": list(self.num_scaler.num_cols),
            "categorical_columns": list(self.cat_encoder.cat_cols),
            "encoded_feature_names": list(self.feature_names_),
            "numeric": {
                "imputation_strategy": self.num_strategy,
                "fill_values": {
                    str(name): _json_scalar(value)
                    for name, value in self.num_scaler.fill_values_.items()
                },
                "scaler": self.num_scaler_type,
                "with_mean": (
                    bool(scaler.with_mean)
                    if isinstance(scaler, StandardScaler)
                    else None
                ),
                "with_std": (
                    bool(scaler.with_std)
                    if isinstance(scaler, StandardScaler)
                    else None
                ),
                "mean": (
                    _json_scalar(scaler.mean_)
                    if scaler is not None and hasattr(scaler, "mean_")
                    else None
                ),
                "scale": (
                    _json_scalar(scaler.scale_)
                    if scaler is not None and hasattr(scaler, "scale_")
                    else None
                ),
            },
            "categorical": {
                "missing_value": self.cat_missing,
                "min_frequency": self.cat_min_frequency,
                "handle_unknown": "ignore",
                "sparse_output": True,
                "dtype": "float32",
                "categories": categories,
                "infrequent_categories": infrequent,
            },
        }
