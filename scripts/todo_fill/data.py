"""Dataset access that reproduces the frozen protocol exactly.

* Home Credit and LendingClub v2 use the split-aware, hash-authenticated voting
  loaders (``prepare_voting_pilot_dev_data`` / ``prepare_voting_research_oot_data``)
  and the canonical fold projection (``GroupedTimeSeriesSplit(n_splits=5, gap=1)``
  over the stable chronological order).
* The third dataset is built once with the frozen adapter (``build_modeling_matrix``)
  and sliced by the protocol lock's date boundaries, mirroring ``_read_date_slice``.

Loaded DEV frames are cached as parquet under the work directory so re-runs do
not repeat the chunked aggregation of the raw CSVs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.todo_fill.common import (
    FULL_DEV,
    HOMECREDIT,
    LENDINGCLUB,
    REPO_ROOT,
    THIRD,
    FillError,
    Timer,
    available_ram_gb,
    frame_gib,
    heartbeat,
    log,
    require_ram,
)

THIRD_INPUT_ROOT = REPO_ROOT / "data/homecredit_model_stability_2024"
THIRD_LOCK = REPO_ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"
THIRD_NON_PREDICTORS = ("case_id", "date_decision", "MONTH", "WEEK_NUM", "target")


@dataclass
class DevBundle:
    dataset: str
    X: pd.DataFrame
    y: pd.Series
    time_values: pd.Series
    stable_row_ids: pd.Series
    candidate_features: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(len(self.y))


class DataContext:
    """Lazy, cached access to DEV/HO frames and fold projections."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)
        self.frames_dir = self.work_dir / "frames"
        self._dev: dict[str, DevBundle] = {}
        self._projections: dict[tuple[str, int], dict[str, Any]] = {}
        self._third: ThirdDataset | None = None

    # ------------------------------------------------------------- DEV frames
    def dev(self, dataset: str) -> DevBundle:
        if dataset == THIRD:
            raise FillError("the third dataset is sliced by date; use DataContext.third()")
        if dataset in self._dev:
            return self._dev[dataset]
        bundle = self._load_dev_cached(dataset)
        self._dev[dataset] = bundle
        return bundle

    def release_dev(self, dataset: str) -> None:
        self._dev.pop(dataset, None)
        self._projections = {key: value for key, value in self._projections.items() if key[0] != dataset}

    def _load_dev_cached(self, dataset: str) -> DevBundle:
        cache = self.frames_dir / dataset
        x_path, meta_path, universe_path = cache / "dev_X.parquet", cache / "dev_meta.parquet", cache / "universe.json"
        if x_path.exists() and meta_path.exists() and universe_path.exists():
            log(f"[data] loading cached DEV frame for {dataset} from {cache}")
            X = pd.read_parquet(x_path)
            meta = pd.read_parquet(meta_path)
            universe = tuple(json.loads(universe_path.read_text(encoding="utf-8")))
            if list(X.columns) != list(universe):
                raise FillError(f"cached DEV frame for {dataset} does not match its candidate universe")
            dtypes_path = cache / "dev_dtypes.json"
            if dtypes_path.exists():
                X = _restore_dtypes(X, json.loads(dtypes_path.read_text(encoding="utf-8")))
            return DevBundle(
                dataset=dataset,
                X=X,
                y=meta["target"].astype("int8"),
                time_values=meta["time_value"],
                stable_row_ids=meta["stable_row_id"].astype(str),
                candidate_features=universe,
            )
        from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data

        log(f"[data] loading DEV rows for {dataset} with the canonical voting loader (this can take a while)")
        timer = Timer()
        with heartbeat(f"loading {dataset} DEV rows"):
            prepared = prepare_voting_pilot_dev_data(
                REPO_ROOT, dataset=dataset, csv_chunk_rows=25_000, csv_low_memory=False
            )
        if any(int(item.get("oot_rows_retained", -1)) != 0 for item in prepared.data_access_log):
            raise FillError("DEV loader retained OOT rows")
        universe = tuple(prepared.candidate_universe or prepared.candidate_features)
        X = prepared.X.loc[:, list(universe)].reset_index(drop=True)
        bundle = DevBundle(
            dataset=dataset,
            X=X,
            y=pd.Series(prepared.y.to_numpy(), name="target").astype("int8"),
            time_values=pd.Series(prepared.time_values.to_numpy(), name="time_value"),
            stable_row_ids=pd.Series(prepared.stable_row_ids.astype(str).to_numpy(), name="stable_row_id"),
            candidate_features=universe,
        )
        log(f"[data] {dataset}: {bundle.n_rows} DEV rows x {len(universe)} candidates in {timer.seconds():.0f}s; caching")
        cache.mkdir(parents=True, exist_ok=True)
        try:
            X.to_parquet(x_path, index=False)
            pd.DataFrame(
                {"target": bundle.y, "time_value": bundle.time_values, "stable_row_id": bundle.stable_row_ids}
            ).to_parquet(meta_path, index=False)
            universe_path.write_text(json.dumps(list(universe)), encoding="utf-8")
            (cache / "dev_dtypes.json").write_text(json.dumps({name: str(dtype) for name, dtype in X.dtypes.items()}), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - caching is best effort
            log(f"[data] could not cache {dataset} DEV frame: {exc}")
        return bundle

    # -------------------------------------------------------------- folds
    def projection(self, dataset: str, fold_id: int) -> dict[str, Any]:
        key = (dataset, fold_id)
        if key not in self._projections:
            from credit_risk_fs.experiments.rank_voting import canonical_fold_projection

            bundle = self.dev(dataset)
            self._projections[key] = canonical_fold_projection(
                y=bundle.y,
                stable_row_ids=bundle.stable_row_ids,
                time_values=bundle.time_values,
                fold_id=fold_id,
            )
        return self._projections[key]

    def training_frame(self, dataset: str, partition: str) -> tuple[pd.DataFrame, pd.Series]:
        """Raw training slice for a partition: fold training rows (chronological) or all DEV rows."""

        if dataset == THIRD:
            return self.third().training_frame(partition)
        bundle = self.dev(dataset)
        if partition == FULL_DEV:
            return bundle.X, bundle.y
        fold_id = int(partition.replace("fold", ""))
        projection = self.projection(dataset, fold_id)
        positions = projection["source_positions"][projection["training_indices"]]
        X = bundle.X.iloc[positions].reset_index(drop=True)
        y = bundle.y.iloc[positions].reset_index(drop=True)
        return X, y

    # ----------------------------------------------------------------- HO
    def holdout(self, dataset: str, features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
        """Locked HO rows projected to ``features``; cached per dataset and grown column-wise."""

        if dataset == THIRD:
            return self.third().holdout_frame(features)
        cache = self.frames_dir / dataset / "ho.parquet"
        frame = pd.read_parquet(cache) if cache.exists() else None
        missing = [feature for feature in features if frame is None or feature not in frame.columns]
        if missing:
            from credit_risk_fs.pipelines.common import prepare_voting_research_oot_data

            bundle = self.dev(dataset)
            # Every miss loads the complete candidate universe once, so the locked HO
            # population is materialised a single time per dataset instead of per subset.
            present = set(frame.columns) if frame is not None else set()
            ordered_missing = [feature for feature in bundle.candidate_features if feature not in present]
            log(f"[data] loading HO rows for {dataset} projected to {len(ordered_missing)} columns")
            with heartbeat(f"loading {dataset} HO rows"):
                oot = prepare_voting_research_oot_data(
                    REPO_ROOT, dataset=dataset, projected_candidate_features=ordered_missing, csv_chunk_rows=25_000
                )
            fresh = oot.X.loc[:, ordered_missing].reset_index(drop=True)
            fresh.insert(0, "__stable_row_id__", oot.stable_row_ids.astype(str).to_numpy())
            fresh.insert(1, "__target__", oot.y.to_numpy().astype("int8"))
            if frame is None:
                frame = fresh
            else:
                fresh = fresh.set_index("__stable_row_id__").loc[frame["__stable_row_id__"].to_numpy()]
                frame = pd.concat([frame.reset_index(drop=True), fresh.loc[:, ordered_missing].reset_index(drop=True)], axis=1)
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                frame.to_parquet(cache, index=False)
            except Exception as exc:  # pragma: no cover - caching is best effort
                log(f"[data] could not cache {dataset} HO frame: {exc}")
        X = frame.loc[:, list(features)].reset_index(drop=True)
        y = frame["__target__"].astype("int8").reset_index(drop=True)
        return X, y

    # ------------------------------------------------------------ third
    def third(self) -> "ThirdDataset":
        if self._third is None:
            self._third = ThirdDataset(self.work_dir / "hcms2024_matrix")
        return self._third


def _restore_dtypes(X: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Best-effort cast back to the loader's dtypes (parquet may widen or stringify columns)."""

    for column, dtype in dtypes.items():
        if column not in X.columns or str(X[column].dtype) == dtype:
            continue
        try:
            if dtype == "object":
                X[column] = X[column].astype(object).where(X[column].notna(), None)
            elif dtype == "category":
                X[column] = X[column].astype("category")
            else:
                X[column] = X[column].astype(dtype)
        except Exception:  # pragma: no cover - keep the parquet dtype when casting fails
            continue
    return X


# ---------------------------------------------------------------- third dataset


class ThirdDataset:
    """Frozen-adapter matrix for ``homecredit_model_stability_2024`` sliced by protocol dates."""

    def __init__(self, matrix_root: Path) -> None:
        self.matrix_root = Path(matrix_root)
        self._lock: dict[str, Any] | None = None
        self._predictors: tuple[str, ...] | None = None

    @property
    def lock(self) -> dict[str, Any]:
        if self._lock is None:
            self._lock = json.loads(THIRD_LOCK.read_text(encoding="utf-8"))
        return self._lock

    @property
    def boundaries(self) -> dict[str, Any]:
        return self.lock["approved_protocol"]["split_and_fold_boundaries"]

    @property
    def matrix_settings(self) -> dict[str, Any]:
        return self.lock["approved_protocol"]["method_and_evaluation_matrix"]

    def ensure_matrix(self) -> Path:
        """Build the 1,959-predictor matrix once (idempotent, authenticated by the adapter)."""

        success = self.matrix_root / "_SUCCESS"
        if success.exists() and (self.matrix_root / "metadata.json").exists():
            return self.matrix_root
        if not THIRD_INPUT_ROOT.exists():
            raise FillError(f"third dataset raw input missing: {THIRD_INPUT_ROOT}")
        from credit_risk_fs.data.homecredit_model_stability_2024 import load_adapter_contract
        from credit_risk_fs.data.homecredit_model_stability_2024.adapter import build_modeling_matrix

        require_ram(4.0, "third dataset matrix build")
        log(f"[data] building the frozen third-dataset matrix under {self.matrix_root} (about an hour on the reference machine)")
        timer = Timer()
        contract = load_adapter_contract(THIRD_LOCK)
        self.matrix_root.parent.mkdir(parents=True, exist_ok=True)
        with heartbeat("building the third-dataset matrix"):
            result = build_modeling_matrix(
                input_root=THIRD_INPUT_ROOT,
                output_root=self.matrix_root,
                contract=contract,
                mode="research",
                shard_rows=50_000,
            )
        log(
            f"[data] third-dataset matrix ready: {result.row_count} rows x {result.predictor_count} predictors "
            f"in {timer.seconds():.0f}s (reused={result.reused_completed_build})"
        )
        return self.matrix_root

    @property
    def predictors(self) -> tuple[str, ...]:
        if self._predictors is None:
            self.ensure_matrix()
            metadata = json.loads((self.matrix_root / "metadata.json").read_text(encoding="utf-8"))
            predictors = tuple(str(item) for item in metadata["predictor_columns"])
            if len(predictors) != 1959:
                raise FillError(f"third dataset matrix has {len(predictors)} predictors; expected 1959")
            try:
                from credit_risk_fs.experiments.prompt_16_llm_supplement import EXPECTED_UNIVERSE_SHA256
                from scripts.todo_fill.common import canonical_sha256

                if canonical_sha256(list(predictors)) != EXPECTED_UNIVERSE_SHA256:
                    log("[data] WARNING: third dataset predictor order digest differs from the frozen universe digest")
            except Exception:  # pragma: no cover - verification is best effort
                pass
            self._predictors = predictors
        return self._predictors

    def _part_paths(self) -> list[Path]:
        self.ensure_matrix()
        return sorted((self.matrix_root / "matrix").glob("part-*.parquet"))

    def validation_range(self, partition: str) -> tuple[str, str, int]:
        """(date_min, date_max, expected_rows) of a fold's frozen validation slice."""

        fold_id = int(partition.replace("fold", ""))
        fold = next(item for item in self.boundaries["folds"] if int(item["fold_id"]) == fold_id)
        validation = fold["validation"]
        return str(validation["date_min"]), str(validation["date_max"]), int(validation["rows"])

    def validation_frame(self, partition: str, predictors: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
        date_min, date_max, expected_rows = self.validation_range(partition)
        frame = self.read_slice(date_min, date_max, list(predictors))
        if len(frame) != expected_rows:
            raise FillError(f"third dataset {partition} validation has {len(frame)} rows; protocol lock expects {expected_rows}")
        return frame.loc[:, list(predictors)].reset_index(drop=True), frame["target"].astype("int8").reset_index(drop=True)

    def holdout_frame_with_ids(self, features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Locked OOT rows projected to ``features`` plus the ``case_id`` identity column."""

        date_min, date_max, expected_rows = self.date_range("holdout")
        frame = self.read_slice(date_min, date_max, list(features))
        if len(frame) != expected_rows:
            raise FillError(f"third dataset holdout has {len(frame)} rows; protocol lock expects {expected_rows}")
        return (
            frame.loc[:, list(features)].reset_index(drop=True),
            frame["target"].astype("int8").reset_index(drop=True),
            frame["case_id"].astype(str).reset_index(drop=True),
        )

    def date_range(self, partition: str) -> tuple[str, str, int]:
        """(date_min, date_max, expected_rows) for a training partition or ``holdout``."""

        if partition == FULL_DEV:
            dev = self.boundaries["dev"]
            return str(dev["date_min"]), str(dev["date_max"]), int(dev["rows"])
        if partition == "holdout":
            oot = self.boundaries["oot"]
            return str(oot["date_min"]), str(oot["date_max"]), int(oot["rows"])
        fold_id = int(partition.replace("fold", ""))
        fold = next(item for item in self.boundaries["folds"] if int(item["fold_id"]) == fold_id)
        train = fold["train"]
        return str(train["date_min"]), str(train["date_max"]), int(train["rows"])

    def read_slice(self, date_min: str, date_max: str, columns: Sequence[str]) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from datetime import date

        wanted = [*THIRD_NON_PREDICTORS, *[column for column in columns if column not in THIRD_NON_PREDICTORS]]
        tables: list[pa.Table] = []
        for path in self._part_paths():
            boundary = pq.read_table(path, columns=["date_decision"])
            kind = boundary["date_decision"].type
            if pa.types.is_date(kind):
                lower, upper = pa.scalar(date.fromisoformat(date_min), type=kind), pa.scalar(date.fromisoformat(date_max), type=kind)
            elif pa.types.is_timestamp(kind):
                lower, upper = pa.scalar(pd.Timestamp(date_min), type=kind), pa.scalar(pd.Timestamp(date_max), type=kind)
            else:
                lower, upper = pa.scalar(date_min, type=kind), pa.scalar(date_max, type=kind)
            mask = pc.and_(pc.greater_equal(boundary["date_decision"], lower), pc.less_equal(boundary["date_decision"], upper))
            indices = pc.indices_nonzero(mask)
            if len(indices) == 0:
                continue
            tables.append(pq.read_table(path, columns=wanted).take(indices))
        if not tables:
            raise FillError(f"no third-dataset rows between {date_min} and {date_max}")
        table = pa.concat_tables(tables)
        del tables
        frame = table.to_pandas(split_blocks=True, self_destruct=True)
        del table
        pa.default_memory_pool().release_unused()
        for name in columns:
            if name in frame.columns and pd.api.types.is_bool_dtype(frame[name].dtype):
                frame[name] = frame[name].astype("Float32")
        frame.sort_values(["date_decision", "case_id"], kind="mergesort", inplace=True, ignore_index=True)
        return frame

    def training_frame(self, partition: str, predictors: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
        columns = list(predictors) if predictors is not None else list(self.predictors)
        date_min, date_max, expected_rows = self.date_range(partition)
        needed = frame_gib(expected_rows, len(columns), 8) * 1.5
        require_ram(needed, f"third dataset {partition} training slice ({expected_rows} rows x {len(columns)} columns)")
        log(f"[data] reading third-dataset {partition}: {date_min}..{date_max} ({expected_rows} rows, {len(columns)} columns)")
        frame = self.read_slice(date_min, date_max, columns)
        if len(frame) != expected_rows:
            raise FillError(f"third dataset {partition} has {len(frame)} rows; protocol lock expects {expected_rows}")
        y = frame["target"].astype("int8").reset_index(drop=True)
        X = frame.loc[:, columns].reset_index(drop=True)
        del frame
        return X, y

    def holdout_frame(self, features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
        date_min, date_max, expected_rows = self.date_range("holdout")
        frame = self.read_slice(date_min, date_max, list(features))
        if len(frame) != expected_rows:
            raise FillError(f"third dataset holdout has {len(frame)} rows; protocol lock expects {expected_rows}")
        return frame.loc[:, list(features)].reset_index(drop=True), frame["target"].astype("int8").reset_index(drop=True)

    def dev_frame(self, features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
        X, y = self.training_frame(FULL_DEV, predictors=list(features))
        return X, y


def encode_for_selection(X: pd.DataFrame, *, release_source: bool = False) -> pd.DataFrame:
    """One numeric column per original candidate, exactly as the frozen selector boundary."""

    from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder

    encoder = OriginalFeatureNumericEncoder()
    if release_source:
        encoder.fit(X)
        return encoder.transform_releasing_source(X)
    return encoder.fit_transform(X)


def dense_preprocess(X: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """The legacy matrix's dense model preprocessor (scaled numerics + one-hot categoricals)."""

    from credit_risk_fs.preprocessing.encoding import Preprocessor
    from scripts.todo_fill.common import PREPROCESSOR_KWARGS

    preprocessor = Preprocessor(**PREPROCESSOR_KWARGS.get(dataset, {}))
    transformed = preprocessor.fit_transform(X)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(np.asarray(transformed), index=X.index)
    return transformed
