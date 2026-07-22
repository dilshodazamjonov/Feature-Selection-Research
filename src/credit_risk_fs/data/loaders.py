# data.py
import logging
from pathlib import Path
from typing import List, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from credit_risk_fs.utils.logging import setup_logging

# Setup module logger
logger = setup_logging("data_loader", level=logging.INFO)

HOME_CREDIT_DAY_SENTINEL = 365243

def normalize_home_credit_sentinel_dates(
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """
    Replace Home Credit's day-based missing-value sentinel with NaN.

    The raw competition tables use ``365243`` in multiple ``DAYS_*`` columns to
    represent missing or not-applicable dates. Leaving that value untouched
    distorts aggregates, IV calculations, and model coefficients.
    """
    day_cols = [
        col
        for col in df.columns
        if "DAYS" in col.upper() and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not day_cols:
        return df

    cleaned = df.copy()
    replaced_total = 0

    for col in day_cols:
        sentinel_mask = cleaned[col] == HOME_CREDIT_DAY_SENTINEL
        if sentinel_mask.any():
            replaced_total += int(sentinel_mask.sum())
            cleaned.loc[sentinel_mask, col] = np.nan

    if replaced_total:
        dataset_label = dataset_name or "dataframe"
        logger.info(
            "Replaced %s sentinel day values in %s",
            f"{replaced_total:,}",
            dataset_label,
        )

    return cleaned

class DataLoader:
    """
    Class for loading, merging, and preparing datasets for modeling.
    """

    def __init__(self, data_dir: str | Path):
        """
        Parameters:
            data_dir: path to folder containing all CSVs
        """
        self.data_dir = Path(data_dir)
        self.dataframes = {}
        self.load_errors = []
        self.last_load_report: dict[str, dict[str, object]] = {}

    def available_tables(self) -> dict[str, Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        tables: dict[str, Path] = {}
        for path in sorted(self.data_dir.iterdir(), key=lambda item: item.name):
            if path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            if path.stem in tables:
                raise ValueError(f"ambiguous table formats for {path.stem!r}")
            tables[path.stem] = path
        if not tables:
            raise ValueError(f"No CSV or Parquet files found in {self.data_dir}")
        return tables

    def inspect_columns(self, table_name: str) -> list[str]:
        tables = self.available_tables()
        if table_name not in tables:
            raise KeyError(f"unknown data table: {table_name}")
        path = tables[table_name]
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            return list(map(str, pq.ParquetFile(path).schema_arrow.names))
        try:
            return list(map(str, pd.read_csv(path, nrows=0, encoding="utf-8").columns))
        except UnicodeDecodeError:
            return list(map(str, pd.read_csv(path, nrows=0, encoding="latin1").columns))

    def load_table(
        self,
        table_name: str,
        *,
        columns: Sequence[str] | None,
        require_projection: bool = True,
        csv_chunk_rows: int | None = None,
    ) -> pd.DataFrame:
        """Load exactly one explicitly projected table without dtype-changing casts."""

        if require_projection and columns is None:
            raise ValueError(
                f"experiment data load for {table_name!r} requires an explicit columns projection"
            )
        requested = None if columns is None else [str(column) for column in columns]
        if requested is not None:
            if not requested or len(requested) != len(set(requested)):
                raise ValueError(f"projection for {table_name!r} must be non-empty and unique")
            available = self.inspect_columns(table_name)
            missing = set(requested) - set(available)
            if missing:
                raise ValueError(
                    f"projection for {table_name!r} contains unknown columns: {sorted(missing)}"
                )

        path = self.available_tables()[table_name]
        if csv_chunk_rows is not None and int(csv_chunk_rows) <= 0:
            raise ValueError("csv_chunk_rows must be positive when supplied")
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path, columns=requested)
        else:
            read_kwargs = {"usecols": requested, "encoding": "utf-8"}
            try:
                if csv_chunk_rows is None:
                    frame = pd.read_csv(path, **read_kwargs)
                else:
                    chunks = pd.read_csv(path, chunksize=int(csv_chunk_rows), **read_kwargs)
                    frame = pd.concat(chunks, ignore_index=True)
            except UnicodeDecodeError:
                read_kwargs["encoding"] = "latin1"
                if csv_chunk_rows is None:
                    frame = pd.read_csv(path, **read_kwargs)
                else:
                    chunks = pd.read_csv(path, chunksize=int(csv_chunk_rows), **read_kwargs)
                    frame = pd.concat(chunks, ignore_index=True)

        frame = normalize_home_credit_sentinel_dates(frame, table_name)
        dtype_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        self.last_load_report[table_name] = {
            "path": str(path),
            "requested_columns": requested,
            "loaded_columns": list(map(str, frame.columns)),
            "row_count": int(len(frame)),
            "dtype_bytes": dtype_bytes,
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "projection_enforced": bool(require_projection),
            "csv_chunk_rows": csv_chunk_rows,
        }
        self.dataframes[table_name] = frame
        return frame

    def load_all(
        self,
        projections: Mapping[str, Sequence[str]] | None = None,
        *,
        require_projection: bool = False,
        csv_chunk_rows: int | None = None,
    ):
        """
        Loads all CSV files in the data directory into a dictionary.
        Handles encoding issues automatically.
        """
        tables = self.available_tables()
        if require_projection and projections is None:
            raise ValueError("experiment data loading requires calculated column projections")
        selected_names = list(tables) if projections is None else list(projections)
        unknown_tables = set(selected_names) - set(tables)
        if unknown_tables:
            raise ValueError(f"projection contains unknown tables: {sorted(unknown_tables)}")
        pbar = tqdm(selected_names, desc="Loading projected tables")
        for name in pbar:
            pbar.set_description(f"Reading {tables[name].name}")
            try:
                self.load_table(
                    name,
                    columns=None if projections is None else projections[name],
                    require_projection=require_projection,
                    csv_chunk_rows=csv_chunk_rows,
                )
            except Exception as exc:
                self.load_errors.append((tables[name].name, str(exc)))
                if require_projection:
                    raise
                logger.warning("Failed to load %s: %s", tables[name].name, exc)

        if self.load_errors:
            logger.warning(f"{len(self.load_errors)} files failed to load")

        return self.dataframes

    def get(self, name: str) -> pd.DataFrame:
        """
        Retrieve a loaded dataframe by name (without '.csv').
        """
        return self.dataframes.get(name)

    def merge_left(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        Perform a left merge between two dataframes on a given column.
        """
        return df1.merge(df2, on=on, how="left")

    def merge_features(self, base_df: pd.DataFrame, feature_dfs: List[pd.DataFrame], on: str) -> pd.DataFrame:
        """
        Sequentially merge multiple feature tables onto base_df.
        """
        df = base_df.copy()
        for feat_df in feature_dfs:
            df = df.merge(feat_df, on=on, how="left")
        return df

    def prepare_dataset(self, raw_df: pd.DataFrame, feature_tables: List[pd.DataFrame], target_col: str = "TARGET"):
        """
        Merge feature tables into the base application dataframe and split X, y.
        """
        df = self.merge_features(raw_df, feature_tables, on="SK_ID_CURR")
        y = df[target_col] if target_col in df.columns else None
        X = df.drop(columns=[target_col, "SK_ID_CURR"], errors='ignore')
        return X, y, df

# -----------------------------
# Aggregation helper
# -----------------------------
def build_aggregations(df: pd.DataFrame, groupby_col: str, agg_config: Dict) -> pd.DataFrame:
    """
    Generic aggregation builder.

    Example agg_config:
        {
            "avg_credit": ("AMT_CREDIT", "mean"),
            "total_credit": ("AMT_CREDIT", "sum")
        }
    """
    return df.groupby(groupby_col).agg(**agg_config).reset_index()
