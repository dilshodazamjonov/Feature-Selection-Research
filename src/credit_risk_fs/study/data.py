"""Dataset loading, chronological ordering, folds and the candidate screen.

Home Credit and LendingClub v2 are assembled by the repository loaders
(``credit_risk_fs.pipelines.common``), which validate the DEV and HO row counts and
the ordered row-identity hashes, and are cached as parquet under the study output
directory. Stability 2024 is read column by column from the matrix built by
``scripts/build_homecredit_model_stability_2024.py``.

DEV rows are kept in chronological order (time, then identifier); HO rows are kept in
the canonical identifier order used for paired comparisons.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.study.config import FOLDS, FULL_DEV, StudyError, output_dir, resolve
from credit_risk_fs.study.utils import ensure_dir, log, read_json

ID, TIME, TARGET = "__id__", "__time__", "__target__"
STRING_LIKE = ("string", "large_string", "dictionary")


def id_keys(values: Sequence[Any]) -> np.ndarray:
    """Canonical identifier order key (NFC, case-folded string)."""

    return np.asarray(
        [unicodedata.normalize("NFC", str(value)).casefold() for value in values], dtype=object
    )


def chronological_order(times: Sequence[Any], ids: Sequence[Any]) -> np.ndarray:
    """Row positions sorted by time, every tie resolved by the canonical identifier."""

    from credit_risk_fs.experiments.lendingclub_identity import stable_chronological_order

    frame = pd.DataFrame(
        {"time": np.asarray(times), "identity": np.asarray(ids), "position": np.arange(len(ids))}
    )
    ordered = stable_chronological_order(frame, time_column="time", identity_column="identity")
    return ordered["position"].to_numpy()


def identifier_order(ids: Sequence[Any]) -> np.ndarray:
    keys = id_keys(ids)
    if len(set(keys.tolist())) != len(keys):
        raise StudyError("held-out identifiers are not unique after normalisation")
    return np.argsort(keys, kind="mergesort")


def make_folds(times_sorted: np.ndarray, n_splits: int, gap: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds on chronologically sorted DEV rows."""

    from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit

    splitter = GroupedTimeSeriesSplit(n_splits=int(n_splits), gap=int(gap))
    folds = {}
    for index, (train, valid) in enumerate(splitter.split(np.asarray(times_sorted)), start=1):
        folds[f"fold{index}"] = (np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64))
    if list(folds) != list(FOLDS):
        raise StudyError(f"expected five folds, obtained {list(folds)}")
    return folds


def missing_mask(values: pd.Series) -> np.ndarray:
    missing = values.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        numeric = values.to_numpy(dtype="float64", na_value=np.nan)
        missing = missing | np.isinf(numeric)
    return missing


# --------------------------------------------------------------------- Home Credit, LendingClub
def _repository_split(cfg: dict[str, Any], split: str, universe: Sequence[str] | None) -> tuple[pd.DataFrame, list[str]]:
    from credit_risk_fs.pipelines.common import (
        prepare_voting_pilot_dev_data,
        prepare_voting_research_oot_data,
    )

    root = resolve(cfg, ".")
    chunk = int(cfg.get("csv_chunk_rows", 25_000))
    if split == "dev":
        prepared = prepare_voting_pilot_dev_data(root, dataset=cfg["loader"], csv_chunk_rows=chunk)
    else:
        prepared = prepare_voting_research_oot_data(
            root,
            dataset=cfg["loader"],
            projected_candidate_features=list(universe or ()),
            csv_chunk_rows=chunk,
        )
    head = pd.DataFrame(
        {
            ID: prepared.stable_row_ids.astype(str).to_numpy(),
            TIME: prepared.time_values.to_numpy(),
            TARGET: prepared.y.to_numpy().astype("int8"),
        }
    )
    frame = pd.concat([head, prepared.X.reset_index(drop=True)], axis=1)
    return frame, [str(name) for name in prepared.candidate_features]


def build_tabular_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load DEV and HO through the repository loaders and cache them as parquet."""

    cache = ensure_dir(output_dir(cfg) / "data")
    log(f"{cfg['dataset']}: loading DEV through the repository loader")
    dev, universe = _repository_split(cfg, "dev", None)
    dev = dev.take(chronological_order(dev[TIME].to_numpy(), dev[ID].to_numpy())).reset_index(drop=True)
    dev.to_parquet(cache / "dev.parquet", index=False)
    n_dev = len(dev)
    del dev
    log(f"{cfg['dataset']}: loading HO through the repository loader")
    ho, ho_universe = _repository_split(cfg, "ho", universe)
    if ho_universe != universe:
        raise StudyError("HO feature order differs from DEV")
    ho = ho.take(identifier_order(ho[ID].to_numpy())).reset_index(drop=True)
    ho.to_parquet(cache / "ho.parquet", index=False)
    meta: dict[str, Any] = {"universe": universe, "n_dev": n_dev, "n_ho": len(ho)}
    if cfg.get("release_table"):
        from credit_risk_fs.data.loaders import DataLoader

        loader = DataLoader(resolve(cfg, cfg["release_table"]["data_dir"]))
        ids = loader.load_table(
            cfg["release_table"]["table"],
            columns=[cfg["release_table"]["id_column"]],
            csv_chunk_rows=int(cfg.get("csv_chunk_rows", 25_000)),
        )
        meta["release_rows"] = int(len(ids))
    return meta


def _read_tabular(cfg: dict[str, Any], split: str, columns: Sequence[str] | None) -> pd.DataFrame:
    path = output_dir(cfg) / "data" / f"{split}.parquet"
    if not path.exists():
        raise StudyError(f"{path} is missing; run --stage prepare first")
    wanted = None if columns is None else [ID, TIME, TARGET, *[c for c in columns if c not in (ID, TIME, TARGET)]]
    return pd.read_parquet(path, columns=wanted)


# --------------------------------------------------------------------- Stability 2024
def stability_parts(cfg: dict[str, Any]) -> list[Path]:
    matrix = resolve(cfg, cfg["matrix_root"]) / "matrix"
    parts = sorted(matrix.glob("part-*.parquet"))
    if not parts:
        raise StudyError(
            f"no matrix parts under {matrix}; build the Stability 2024 matrix first (see README)"
        )
    return parts


def stability_predictors(cfg: dict[str, Any]) -> list[str]:
    metadata = read_json(resolve(cfg, cfg["matrix_root"]) / "metadata.json")
    return [str(name) for name in metadata["predictor_columns"]]


def _stability_base(cfg: dict[str, Any]) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(part, columns=["case_id", "date_decision", "target"]) for part in stability_parts(cfg)]
    frame = pa.concat_tables(tables).to_pandas()
    frame["date_decision"] = frame["date_decision"].astype(str).str.slice(0, 10)
    return frame


def build_stability_index(cfg: dict[str, Any]) -> dict[str, Any]:
    """Split the matrix rows into DEV and HO and fix their order."""

    base = _stability_base(cfg)
    dates = base["date_decision"].to_numpy(dtype=str).astype("U10")
    dev_mask = (dates >= cfg["dev_start"]) & (dates < cfg["ho_start"])
    ho_mask = (dates >= cfg["ho_start"]) & (dates <= cfg["ho_end"])
    dev_rows = np.flatnonzero(dev_mask)
    ho_rows = np.flatnonzero(ho_mask)
    ids = base["case_id"].astype(str).to_numpy()
    dev_rows = dev_rows[chronological_order(dates[dev_rows], ids[dev_rows])]
    ho_rows = ho_rows[identifier_order(ids[ho_rows])]
    cache = ensure_dir(output_dir(cfg) / "data")
    np.savez(
        cache / "stability_index.npz",
        dev_rows=dev_rows,
        ho_rows=ho_rows,
        dev_ids=ids[dev_rows].astype(str),
        ho_ids=ids[ho_rows].astype(str),
        dev_dates=dates[dev_rows].astype(str),
        ho_dates=dates[ho_rows].astype(str),
        dev_target=base["target"].to_numpy()[dev_rows].astype("int8"),
        ho_target=base["target"].to_numpy()[ho_rows].astype("int8"),
    )
    return {
        "universe": stability_predictors(cfg),
        "n_dev": int(len(dev_rows)),
        "n_ho": int(len(ho_rows)),
        "rows_outside_windows": int(len(base) - len(dev_rows) - len(ho_rows)),
        "release_rows": int(len(base)),
    }


def _stability_index(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    path = output_dir(cfg) / "data" / "stability_index.npz"
    if not path.exists():
        raise StudyError(f"{path} is missing; run --stage prepare first")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stability_index(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Matrix row positions, identifiers, dates and targets of DEV (chronological) and HO."""

    return _stability_index(cfg)


def stability_categorical(cfg: dict[str, Any], columns: Sequence[str]) -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(stability_parts(cfg)[0])
    return [str(name) for name in columns if _string_like(schema.field(str(name)).type)]


def _string_like(field_type: Any) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_string(field_type)
        or pa.types.is_large_string(field_type)
        or pa.types.is_dictionary(field_type)
    )


def read_stability_columns(
    cfg: dict[str, Any],
    columns: Sequence[str],
    row_sets: Sequence[np.ndarray],
    *,
    batch_size: int = 100,
) -> list[pd.DataFrame]:
    """Read ``columns`` for each set of matrix row positions (float32 numerics, categories)."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = stability_parts(cfg)
    schema = pq.read_schema(parts[0])
    names = [str(name) for name in columns]
    categorical = [name for name in names if _string_like(schema.field(name).type)]
    numeric = [name for name in names if name not in set(categorical)]
    position = {name: index for index, name in enumerate(numeric)}
    blocks = [np.empty((len(rows), len(numeric)), dtype=np.float32, order="F") for rows in row_sets]
    categories: list[dict[str, pd.Categorical]] = [{} for _ in row_sets]
    for start in range(0, len(names), batch_size):
        batch = names[start : start + batch_size]
        table = pa.concat_tables([pq.read_table(part, columns=batch) for part in parts])
        for name in batch:
            column = table.column(name)
            if name in position:
                values = column.cast(pa.float32(), safe=False).to_numpy()
                for block, rows in zip(blocks, row_sets):
                    block[:, position[name]] = values[rows]
            else:
                values = column.to_pandas().to_numpy(dtype=object)
                for store, rows in zip(categories, row_sets):
                    store[name] = pd.Categorical(values[rows])
        del table
    frames = []
    for block, store in zip(blocks, categories):
        frame = pd.DataFrame(block, columns=numeric, copy=False)
        if categorical:
            frame = pd.concat([frame, pd.DataFrame(store, columns=categorical)], axis=1)
        if list(frame.columns) != names:
            frame = frame.loc[:, names]
        frames.append(frame)
    return frames


def stability_missing_rates(cfg: dict[str, Any], columns: Sequence[str], rows: np.ndarray, *, batch_size: int = 100) -> pd.Series:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = stability_parts(cfg)
    rates: dict[str, float] = {}
    names = [str(name) for name in columns]
    for start in range(0, len(names), batch_size):
        batch = names[start : start + batch_size]
        table = pa.concat_tables([pq.read_table(part, columns=batch) for part in parts])
        for name in batch:
            column = table.column(name)
            if _string_like(column.type):
                values = column.to_pandas().to_numpy(dtype=object)[rows]
                rates[name] = float(pd.isna(values).mean())
            else:
                values = column.cast(pa.float64(), safe=False).to_numpy()[rows]
                rates[name] = float((~np.isfinite(values)).mean())
        del table
    return pd.Series(rates)


# --------------------------------------------------------------------- unified access
@dataclass
class Split:
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray
    times: np.ndarray


def load_split(cfg: dict[str, Any], split: str, columns: Sequence[str]) -> Split:
    """DEV (chronological order) or HO (identifier order) restricted to ``columns``."""

    if split not in {"dev", "ho"}:
        raise ValueError("split must be 'dev' or 'ho'")
    names = [str(name) for name in columns]
    if cfg["loader"] == "stability_2024":
        index = _stability_index(cfg)
        rows = index[f"{split}_rows"]
        (X,) = read_stability_columns(cfg, names, [rows])
        return Split(
            X=X,
            y=index[f"{split}_target"].astype("int8"),
            ids=index[f"{split}_ids"].astype(str),
            times=index[f"{split}_dates"].astype(str),
        )
    frame = _read_tabular(cfg, split, names)
    y = frame.pop(TARGET).to_numpy().astype("int8")
    ids = frame.pop(ID).astype(str).to_numpy()
    times = frame.pop(TIME).to_numpy()
    if list(frame.columns) != names:
        frame = frame.loc[:, names]
    return Split(X=frame, y=y, ids=ids, times=times)


def load_targets(cfg: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y, ids, times)`` of a split without feature columns."""

    if cfg["loader"] == "stability_2024":
        index = _stability_index(cfg)
        return (
            index[f"{split}_target"].astype("int8"),
            index[f"{split}_ids"].astype(str),
            index[f"{split}_dates"].astype(str),
        )
    frame = _read_tabular(cfg, split, [])
    return frame[TARGET].to_numpy().astype("int8"), frame[ID].astype(str).to_numpy(), frame[TIME].to_numpy()


# --------------------------------------------------------------------- prepared artefacts
@dataclass
class Prepared:
    universe: list[str]
    candidates: list[str]
    folds: dict[str, tuple[np.ndarray, np.ndarray | None]]
    n_dev: int
    n_ho: int
    categorical: list[str]
    families: dict[str, str]
    n_families: int
    records: list[dict[str, Any]]

    def train_rows(self, partition: str) -> np.ndarray:
        if partition == FULL_DEV:
            return np.arange(self.n_dev, dtype=np.int64)
        return self.folds[partition][0]

    def valid_rows(self, partition: str) -> np.ndarray | None:
        if partition == FULL_DEV:
            return None
        return self.folds[partition][1]


def load_prepared(cfg: dict[str, Any]) -> Prepared:
    directory = output_dir(cfg) / "prepare"
    if not (directory / "candidates.json").exists():
        raise StudyError(f"{directory} is incomplete; run --stage prepare first")
    candidates = read_json(directory / "candidates.json")
    folds: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    with np.load(directory / "folds.npz", allow_pickle=False) as data:
        for name in FOLDS:
            folds[name] = (data[f"{name}_train"], data[f"{name}_valid"])
    families = read_json(directory / "families.json")
    return Prepared(
        universe=list(candidates["universe"]),
        candidates=list(candidates["candidates"]),
        folds=folds,
        n_dev=int(candidates["n_dev"]),
        n_ho=int(candidates["n_ho"]),
        categorical=list(candidates["categorical"]),
        families=dict(families["families"]),
        n_families=int(families["n_families"]),
        records=read_json(directory / "records.json"),
    )


__all__ = [
    "ID",
    "TARGET",
    "TIME",
    "Prepared",
    "Split",
    "build_stability_index",
    "build_tabular_cache",
    "chronological_order",
    "id_keys",
    "identifier_order",
    "load_prepared",
    "load_split",
    "load_targets",
    "make_folds",
    "missing_mask",
    "read_stability_columns",
    "stability_categorical",
    "stability_index",
    "stability_missing_rates",
    "stability_predictors",
]
