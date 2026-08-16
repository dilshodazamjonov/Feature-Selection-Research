# preprocessing/preprocessing.py

import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
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
    ):
        self.max_cardinality = max_cardinality
        self.missing_value = missing_value
        self.min_frequency = min_frequency
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
                sparse_output=False,
                handle_unknown="ignore",
                min_frequency=self.min_frequency,
                dtype=np.float32,
            )
            self.ohe.fit(X_cat)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cat_cols and self.ohe is not None:
            X_cat = self._prepare_categorical_frame(X)
            X_encoded = self.ohe.transform(X_cat)
            return pd.DataFrame(
                X_encoded,
                columns=self.ohe.get_feature_names_out(self.cat_cols),
                index=X.index
            ).astype("float32")

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
