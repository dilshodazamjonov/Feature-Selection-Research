"""Frozen backbone refits (fold validation AUC or full-DEV -> HO scores), run in worker processes.

Every fit is the paper's pipeline: dataset-specific ``Preprocessor`` (mean imputation +
standard scaling, ``Missing`` token + one-hot with the dataset's ``cat_min_frequency``),
the frozen ``configs/models/*.yaml`` backbone, seed 42, single CatBoost thread.  The
``native_cat`` variant (step 39b) keeps the numeric block identical and hands the raw
categorical columns to CatBoost through ``cat_features`` instead of one-hot columns.

A fit is identified by the hash of (dataset, backbone, partition, variant, features) and
checkpointed under ``<work_dir>/gate4/fits``; HO score vectors are stored as ``.npy`` next
to the JSON so the paired inference of steps 37/38/44 never refits.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.todo_fill.common import (
    FULL_DEV,
    PREPROCESSOR_KWARGS,
    REPO_ROOT,
    THIRD,
    FillError,
    Manifest,
    log,
    read_json,
    write_json,
)

VARIANTS = ("onehot", "native_cat")


@dataclass(frozen=True)
class FitSpec:
    dataset: str
    backbone: str
    partition: str
    features: tuple[str, ...]
    variant: str = "onehot"
    label: str = ""

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown fit variant {self.variant}")
        if not self.features:
            raise ValueError("a fit needs at least one feature")

    @property
    def fit_id(self) -> str:
        payload = json.dumps([self.dataset, self.backbone, self.partition, self.variant, list(self.features)], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "backbone": self.backbone, "partition": self.partition, "features": list(self.features), "variant": self.variant, "label": self.label}


def fits_dir(work_dir: Path) -> Path:
    path = Path(work_dir) / "gate4" / "fits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(work_dir: Path, spec: FitSpec) -> Path:
    return fits_dir(work_dir) / f"{spec.fit_id}.json"


def scores_path(work_dir: Path, spec_or_id: FitSpec | str) -> Path:
    fit_id = spec_or_id if isinstance(spec_or_id, str) else spec_or_id.fit_id
    return fits_dir(work_dir) / f"{fit_id}.npy"


def identity_path(work_dir: Path, dataset: str) -> Path:
    return fits_dir(work_dir) / f"{dataset}__ho_identity.parquet"


def stored(work_dir: Path, spec: FitSpec) -> dict[str, Any] | None:
    path = result_path(work_dir, spec)
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("status") != "ok":
        return None
    if spec.partition == FULL_DEV and not scores_path(work_dir, spec).exists():
        return None
    return payload


# ------------------------------------------------------------------ data access (worker side)


def _model_params(backbone: str) -> dict[str, Any]:
    import yaml

    params = dict(yaml.safe_load((REPO_ROOT / f"configs/models/{backbone}.yaml").read_text(encoding="utf-8"))["model_params"][backbone])
    if backbone == "catboost":
        params["verbose"] = 0  # logging only; the frozen configuration is otherwise untouched
    return params


def _restore(X: pd.DataFrame, work_dir: Path, dataset: str) -> pd.DataFrame:
    from scripts.todo_fill.data import _restore_dtypes

    dtypes_path = Path(work_dir) / "frames" / dataset / "dev_dtypes.json"
    if dtypes_path.exists():
        dtypes = json.loads(dtypes_path.read_text(encoding="utf-8"))
        X = _restore_dtypes(X, {name: dtypes[name] for name in X.columns if name in dtypes})
    return X


def fold_positions(work_dir: Path, dataset: str, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Training / validation source positions (into the cached DEV frame) for one canonical fold."""

    cache = Path(work_dir) / "frames" / dataset / f"fold_positions_{fold_id}.npz"
    if cache.exists():
        payload = np.load(cache)
        return payload["train"], payload["validation"]
    from credit_risk_fs.experiments.rank_voting import canonical_fold_projection

    meta = pd.read_parquet(Path(work_dir) / "frames" / dataset / "dev_meta.parquet")
    projection = canonical_fold_projection(
        y=meta["target"].astype("int8"),
        stable_row_ids=meta["stable_row_id"].astype(str),
        time_values=meta["time_value"],
        fold_id=fold_id,
    )
    train = np.asarray(projection["source_positions"][projection["training_indices"]], dtype=np.int64)
    validation = np.asarray(projection["source_positions"][projection["validation_indices"]], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train=train, validation=validation)
    return train, validation


def load_training(work_dir: Path, dataset: str, partition: str, raw_features: Sequence[str]):
    """(X_train, y_train, X_validation | None, y_validation | None) for one partition."""

    raw = list(raw_features)
    if dataset == THIRD:
        from scripts.todo_fill.data import ThirdDataset

        third = ThirdDataset(Path(work_dir) / "hcms2024_matrix")
        X, y = third.training_frame(partition, predictors=raw)
        if partition == FULL_DEV:
            return X, y, None, None
        X_val, y_val = third.validation_frame(partition, raw)
        return X, y, X_val, y_val
    frames = Path(work_dir) / "frames" / dataset
    X = _restore(pd.read_parquet(frames / "dev_X.parquet", columns=raw), work_dir, dataset)
    y = pd.read_parquet(frames / "dev_meta.parquet", columns=["target"])["target"].astype("int8")
    if partition == FULL_DEV:
        return X, y, None, None
    train, validation = fold_positions(work_dir, dataset, int(partition.replace("fold", "")))
    return (
        X.iloc[train].reset_index(drop=True),
        y.iloc[train].reset_index(drop=True),
        X.iloc[validation].reset_index(drop=True),
        y.iloc[validation].reset_index(drop=True),
    )


def load_holdout(work_dir: Path, dataset: str, raw_features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = list(raw_features)
    if dataset == THIRD:
        from scripts.todo_fill.data import ThirdDataset

        return ThirdDataset(Path(work_dir) / "hcms2024_matrix").holdout_frame_with_ids(raw)
    cache = Path(work_dir) / "frames" / dataset / "ho.parquet"
    if not cache.exists():
        raise FillError(f"HO frame for {dataset} is not cached; the driver must load it before dispatching fits")
    import pyarrow.parquet as pq

    available = set(pq.read_schema(cache).names)
    missing = [name for name in raw if name not in available]
    if missing:
        raise FillError(f"HO frame for {dataset} lacks {len(missing)} columns (e.g. {missing[:3]}); the driver must extend it first")
    frame = pd.read_parquet(cache, columns=["__stable_row_id__", "__target__", *raw])
    X = _restore(frame.loc[:, raw], work_dir, dataset)
    return X, frame["__target__"].astype("int8"), frame["__stable_row_id__"].astype(str)


def raw_bases(features: Sequence[str], universe: Sequence[str]) -> list[str]:
    from scripts.todo_fill.frozen import collapse_model_matrix_columns

    return collapse_model_matrix_columns(features, universe)


def universe_of(work_dir: Path, dataset: str) -> list[str]:
    if dataset == THIRD:
        metadata = json.loads((Path(work_dir) / "hcms2024_matrix" / "metadata.json").read_text(encoding="utf-8"))
        return [str(item) for item in metadata["predictor_columns"]]
    return json.loads((Path(work_dir) / "frames" / dataset / "universe.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------- the fit itself


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [name for name in X.columns if not pd.api.types.is_numeric_dtype(X[name])]


class _NativeCatPipeline:
    """Numeric block exactly as ``Preprocessor``; categoricals as CatBoost ``cat_features``."""

    def __init__(self) -> None:
        from credit_risk_fs.preprocessing.encoding import NumericalScaler

        self.scaler = NumericalScaler(strategy="mean", scaler="standard")
        self.cat_cols: list[str] = []

    def fit(self, X: pd.DataFrame) -> "_NativeCatPipeline":
        self.scaler.fit(X)
        self.cat_cols = _categorical_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = self.scaler.transform(X)
        parts = [numeric]
        if self.cat_cols:
            cats = pd.DataFrame({name: X[name].astype("string").fillna("Missing").astype(str) for name in self.cat_cols}, index=X.index)
            parts.append(cats)
        return pd.concat(parts, axis=1)


def _fit_and_score(spec: FitSpec, X_train: pd.DataFrame, y_train: pd.Series, evaluations: list[pd.DataFrame]) -> tuple[list[np.ndarray], dict[str, Any]]:
    params = _model_params(spec.backbone)
    info: dict[str, Any] = {"n_categorical_raw": len(_categorical_columns(X_train)), "raw_features": list(X_train.columns)}
    dense_selection = [name for name in spec.features if name not in set(X_train.columns)]
    if spec.variant == "onehot":
        from credit_risk_fs.models.registry import get_model_bundle
        from credit_risk_fs.preprocessing.encoding import Preprocessor

        preprocessor = Preprocessor(**PREPROCESSOR_KWARGS.get(spec.dataset, {}))
        transformed = preprocessor.fit_transform(X_train)
        if dense_selection:
            wanted = [name for name in spec.features if name in transformed.columns]
            info["dense_selection"] = True
            info["missing_dense_columns"] = [name for name in spec.features if name not in transformed.columns]
            columns = wanted
        else:
            columns = list(transformed.columns)
        info["n_model_columns"] = len(columns)
        get_model, train_model, predict_proba, _ = get_model_bundle(spec.backbone, model_kwargs=params)
        model = train_model(get_model(), transformed.loc[:, columns], y_train, None, None)
        outputs = [np.asarray(predict_proba(model, preprocessor.transform(frame).loc[:, columns]), dtype=float) for frame in evaluations]
        return outputs, info
    if spec.backbone != "catboost":
        raise FillError("native categorical handling is a CatBoost-only variant")
    from catboost import CatBoostClassifier, Pool

    pipeline = _NativeCatPipeline().fit(X_train)
    train_frame = pipeline.transform(X_train)
    info["native_cat_features"] = list(pipeline.cat_cols)
    info["n_model_columns"] = int(train_frame.shape[1])
    if dense_selection:
        info["dense_selection_collapsed_to_raw"] = True
    model = CatBoostClassifier(**params, thread_count=1)
    model.fit(Pool(train_frame, y_train, cat_features=pipeline.cat_cols))
    outputs = [model.predict_proba(Pool(pipeline.transform(frame), cat_features=pipeline.cat_cols))[:, 1].astype(float) for frame in evaluations]
    return outputs, info


def run_fit(spec_payload: dict[str, Any], work_dir: str) -> dict[str, Any]:
    """Worker entry point: fit one spec, checkpoint, return the result dictionary."""

    from sklearn.metrics import roc_auc_score

    spec = FitSpec(**{key: (tuple(value) if key == "features" else value) for key, value in spec_payload.items()})
    root = Path(work_dir)
    done = stored(root, spec)
    if done is not None:
        return done
    started = time.perf_counter()
    try:
        universe = universe_of(root, spec.dataset)
        raw = raw_bases(spec.features, universe)
        unknown = [name for name in raw if name not in set(universe)]
        if unknown:
            raise FillError(f"features outside the {spec.dataset} candidate universe: {unknown[:5]}")
        X_train, y_train, X_val, y_val = load_training(root, spec.dataset, spec.partition, raw)
        if spec.partition == FULL_DEV:
            X_ho, y_ho, ids = load_holdout(root, spec.dataset, raw)
            (scores,), info = _fit_and_score(spec, X_train, y_train, [X_ho])
            if len(scores) != len(y_ho) or not np.isfinite(scores).all():
                raise FillError("HO score vector is incomplete or non-finite")
            np.save(scores_path(root, spec), scores)
            identity = identity_path(root, spec.dataset)
            if not identity.exists():
                pd.DataFrame({"stable_row_id": ids.to_numpy(), "target": y_ho.to_numpy(dtype=int)}).to_parquet(identity, index=False)
            payload = {"status": "ok", **spec.to_dict(), "fit_id": spec.fit_id, "ho_auc": float(roc_auc_score(y_ho, scores)), "n_ho": int(len(y_ho)), "n_train": int(len(y_train)), **info}
        else:
            (scores,), info = _fit_and_score(spec, X_train, y_train, [X_val])
            payload = {"status": "ok", **spec.to_dict(), "fit_id": spec.fit_id, "auc": float(roc_auc_score(y_val, scores)), "n_train": int(len(y_train)), "n_validation": int(len(y_val)), **info}
        payload["seconds"] = time.perf_counter() - started
        payload["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(result_path(root, spec), payload)
        return payload
    except Exception as exc:  # noqa: BLE001 - reported to the driver, never fatal
        return {"status": "failed", **spec.to_dict(), "fit_id": spec.fit_id, "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started}


# --------------------------------------------------------------------- executor


@dataclass
class FitExecutor:
    work_dir: Path
    manifest: Manifest
    jobs: int = 6
    third_jobs: int = 3
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def lookup(self, spec: FitSpec) -> dict[str, Any] | None:
        if spec.fit_id in self.results and self.results[spec.fit_id].get("status") == "ok":
            return self.results[spec.fit_id]
        found = stored(self.work_dir, spec)
        if found is not None:
            self.results[spec.fit_id] = found
        return found

    def run(self, specs: Sequence[FitSpec], step: str) -> dict[str, dict[str, Any]]:
        """Fit every spec not yet checkpointed; return fit_id -> result for all of them."""

        unique = list(dict.fromkeys(specs))
        pending = [spec for spec in unique if self.lookup(spec) is None]
        if pending:
            regular = [spec for spec in pending if spec.dataset != THIRD]
            third = [spec for spec in pending if spec.dataset == THIRD]
            log(f"[{step}] {len(pending)} backbone fits pending ({len(unique) - len(pending)} already checkpointed); jobs={self.jobs}")
            for batch, workers in ((regular, self.jobs), (third, self.third_jobs)):
                if batch:
                    self._run_batch(batch, max(1, workers), step)
        return {spec.fit_id: self.results.get(spec.fit_id, {"status": "failed", "reason": "no result"}) for spec in unique}

    def _run_batch(self, batch: list[FitSpec], workers: int, step: str) -> None:
        # Heaviest first keeps the pool busy at the end; CatBoost dominates.
        ordered = sorted(batch, key=lambda spec: (spec.backbone != "catboost", spec.partition != FULL_DEV, spec.partition), reverse=False)
        ordered.sort(key=lambda spec: 0 if spec.backbone == "catboost" else 1)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        started = time.perf_counter()
        done = 0
        if workers == 1:
            for spec in ordered:
                self._record(run_fit(spec.to_dict(), str(self.work_dir)), spec, step)
                done += 1
                log(f"[{step}] fit {done}/{len(ordered)} finished ({(time.perf_counter() - started) / 60:.1f} min elapsed)")
            return
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(run_fit, spec.to_dict(), str(self.work_dir)): spec for spec in ordered}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001
                    payload = {"status": "failed", **spec.to_dict(), "fit_id": spec.fit_id, "reason": f"worker crashed: {type(exc).__name__}: {exc}"}
                self._record(payload, spec, step)
                done += 1
                log(f"[{step}] fit {done}/{len(ordered)} finished: {spec.dataset}/{spec.backbone}/{spec.partition}/{spec.variant} {spec.label} ({(time.perf_counter() - started) / 60:.1f} min elapsed)")

    def _record(self, payload: dict[str, Any], spec: FitSpec, step: str) -> None:
        self.results[spec.fit_id] = payload
        if payload.get("status") == "ok":
            self.manifest.timing(f"fit:{spec.dataset}/{spec.backbone}/{spec.partition}/{spec.variant}:{spec.label}", float(payload.get("seconds", 0.0)), n_features=len(spec.features))
        else:
            self.manifest.skip(step, {"fit": spec.label or spec.fit_id, "dataset": spec.dataset, "backbone": spec.backbone, "partition": spec.partition}, f"fit failed: {payload.get('reason')}")


def ho_scores(work_dir: Path, spec: FitSpec) -> np.ndarray:
    return np.load(scores_path(work_dir, spec))


def ho_identity(work_dir: Path, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(identity_path(work_dir, dataset))
