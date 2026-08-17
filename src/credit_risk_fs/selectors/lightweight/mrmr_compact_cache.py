"""Atomic disk-backed execution cache for canonical mutual-information mRMR.

This module changes storage and restart behavior only.  Discretization is
supplied by the canonical selector, and every MI value is still calculated by
``sklearn.metrics.mutual_info_score`` in the same argument order as the frozen
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    canonical_sha256,
    file_sha256,
)
from credit_risk_fs.experiments.atomic_io import (
    atomic_publish,
    write_json_atomic,
    write_text_atomic,
)


COMPACT_MRMR_CACHE_SCHEMA_VERSION = "mrmr_compact_checkpoint_v1"
COMPACT_MRMR_MEMORY_STRATEGY = (
    "batched_disk_backed_compact_integer_codes_and_atomic_mi_v1"
)
DEFAULT_FEATURE_BATCH_SIZE = 32


class CompactMRMRCheckpointError(RuntimeError):
    """Raised when a completed compact mRMR checkpoint fails authentication."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _close_memmap(mapping: np.memmap | None) -> None:
    if mapping is None:
        return
    owner = getattr(mapping, "_mmap", None)
    if owner is not None and not owner.closed:
        owner.close()


def _array_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "byte_size": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _archive_incomplete(path: Path, root: Path) -> None:
    if not path.exists() or (path / "_SUCCESS").is_file():
        return
    archive = root / "incomplete"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / (
        path.name
        + f"-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(path, destination)


def _sealed_array(
    path: Path,
    *,
    identity: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> tuple[np.memmap, dict[str, Any]] | None:
    success_path = path / "_SUCCESS"
    if not success_path.is_file():
        return None
    manifest_path = path / "manifest.json"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint metadata is unreadable: {path}"
        ) from exc
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint marker changed: {path}"
        )
    if manifest.get("identity") != dict(identity):
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint identity changed: {path}"
        )
    artifact = manifest.get("artifact", {})
    array_path = path / str(artifact.get("path", ""))
    if (
        not array_path.is_file()
        or array_path.stat().st_size != int(artifact.get("byte_size", -1))
        or file_sha256(array_path) != artifact.get("sha256")
    ):
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint artifact changed: {path}"
        )
    inventory = sorted(item.name for item in path.iterdir() if item.is_file())
    expected_inventory = sorted(
        ["_SUCCESS", "manifest.json", str(artifact["path"])]
    )
    if inventory != expected_inventory:
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint inventory changed: {path}"
        )
    mapping = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(mapping, np.memmap):
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint is not memory mapped: {path}"
        )
    if mapping.shape != expected_shape or mapping.dtype != expected_dtype:
        _close_memmap(mapping)
        raise CompactMRMRCheckpointError(
            f"completed mRMR checkpoint shape or dtype changed: {path}"
        )
    return mapping, manifest


def _write_sealed_array(
    path: Path,
    *,
    identity: Mapping[str, Any],
    filename: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    populate: Callable[[np.memmap], None],
) -> tuple[np.memmap, dict[str, Any]]:
    path.mkdir(parents=True, exist_ok=False)
    target = path / filename

    def writer(partial: Path) -> None:
        mapping = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=dtype,
            shape=shape,
            fortran_order=False,
        )
        try:
            populate(mapping)
            mapping.flush()
        finally:
            _close_memmap(mapping)

    metadata = atomic_publish(
        target,
        writer,
        artifact_format="npy",
        overwrite=False,
    )
    artifact = {
        "path": filename,
        "byte_size": int(metadata.size_bytes),
        "sha256": metadata.sha256,
    }
    manifest = {
        "schema_version": COMPACT_MRMR_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "identity": dict(identity),
        "artifact": artifact,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(path / "manifest.json", manifest, overwrite=False)
    write_text_atomic(
        path / "_SUCCESS",
        json.dumps(
            {"manifest_sha256": file_sha256(path / "manifest.json")},
            sort_keys=True,
        )
        + "\n",
        overwrite=False,
    )
    loaded = _sealed_array(
        path,
        identity=identity,
        expected_shape=shape,
        expected_dtype=dtype,
    )
    if loaded is None:  # pragma: no cover - marker was just written
        raise CompactMRMRCheckpointError(
            f"new mRMR checkpoint did not authenticate: {path}"
        )
    return loaded


def _storage_dtype(X: pd.DataFrame, n_bins: int) -> np.dtype[Any]:
    numeric_only = all(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        for dtype in X.dtypes
    )
    if not numeric_only:
        return np.dtype("int32")
    maximum = max(0, int(n_bins) - 1)
    if maximum <= np.iinfo(np.int8).max:
        return np.dtype("int8")
    if maximum <= np.iinfo(np.int16).max:
        return np.dtype("int16")
    return np.dtype("int32")


class CompactMRMRCheckpointStore:
    """Batched compact code store plus independently sealed MI vectors."""

    def __init__(
        self,
        root: str | Path,
        *,
        execution_identity: Mapping[str, Any],
        candidate_order: Sequence[str],
        row_count: int,
        n_bins: int,
        storage_dtype: np.dtype[Any],
        feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.candidate_order = [str(value) for value in candidate_order]
        self.row_count = int(row_count)
        self.n_bins = int(n_bins)
        self.storage_dtype = np.dtype(storage_dtype)
        self.feature_batch_size = int(feature_batch_size)
        self.progress_callback = progress_callback
        if self.row_count <= 0 or not self.candidate_order:
            raise ValueError("compact mRMR cache requires rows and candidates")
        if self.feature_batch_size <= 0:
            raise ValueError("compact mRMR feature batch size must be positive")
        self.identity = {
            "schema_version": COMPACT_MRMR_CACHE_SCHEMA_VERSION,
            "memory_strategy": COMPACT_MRMR_MEMORY_STRATEGY,
            "execution_identity": dict(execution_identity),
            "candidate_order_sha256": canonical_sha256(self.candidate_order),
            "candidate_count": len(self.candidate_order),
            "row_count": self.row_count,
            "n_bins": self.n_bins,
            "missing_code": -1,
            "storage_dtype": self.storage_dtype.name,
            "feature_batch_size": self.feature_batch_size,
            "mi_estimator": "sklearn.metrics.mutual_info_score",
        }
        self._feature_positions = {
            name: index for index, name in enumerate(self.candidate_order)
        }
        self._batch_paths: list[Path] = []
        self._batch_shapes: list[tuple[int, int]] = []
        self._root_manifest_sha256: str | None = None
        self._active_batch_index: int | None = None
        self._active_mapping: np.memmap | None = None
        self._relevance_manifest_sha256: str | None = None
        self._pair_manifest_sha256s: dict[str, str] = {}

    @classmethod
    def prepare(
        cls,
        root: str | Path,
        *,
        X: pd.DataFrame,
        execution_identity: Mapping[str, Any],
        candidate_order: Sequence[str],
        n_bins: int,
        discretize: Callable[[pd.Series, int], np.ndarray],
        feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
        before_batch: Callable[[int, int, Sequence[str]], None] | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> CompactMRMRCheckpointStore:
        dtype = _storage_dtype(X, n_bins)
        store = cls(
            root,
            execution_identity=execution_identity,
            candidate_order=candidate_order,
            row_count=len(X),
            n_bins=n_bins,
            storage_dtype=dtype,
            feature_batch_size=feature_batch_size,
            progress_callback=progress_callback,
        )
        store._prepare_codes(X, discretize=discretize, before_batch=before_batch)
        return store

    def _progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(str(event), details)

    def _batch_identity(
        self, batch_index: int, names: Sequence[str]
    ) -> dict[str, Any]:
        return {
            "operation": "compact_mrmr_discrete_code_batch",
            "root_identity_sha256": canonical_sha256(self.identity),
            "batch_index": int(batch_index),
            "ordered_features": list(names),
            "ordered_features_sha256": canonical_sha256(list(names)),
            "shape": [len(names), self.row_count],
            "dtype": self.storage_dtype.name,
        }

    def _prepare_codes(
        self,
        X: pd.DataFrame,
        *,
        discretize: Callable[[pd.Series, int], np.ndarray],
        before_batch: Callable[[int, int, Sequence[str]], None] | None,
    ) -> None:
        if list(X.columns) != self.candidate_order or len(X) != self.row_count:
            raise CompactMRMRCheckpointError(
                "compact mRMR source shape or candidate order changed"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "identity.json"
        if identity_path.is_file():
            try:
                observed = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CompactMRMRCheckpointError(
                    "compact mRMR root identity is unreadable"
                ) from exc
            if observed != self.identity:
                raise CompactMRMRCheckpointError(
                    "compact mRMR root identity changed"
                )
        else:
            write_json_atomic(identity_path, self.identity, overwrite=False)

        batches_root = self.root / "code_batches"
        batches_root.mkdir(exist_ok=True)
        ranges = list(
            enumerate(
                range(0, len(self.candidate_order), self.feature_batch_size),
                start=1,
            )
        )
        batch_manifest_rows: list[dict[str, Any]] = []
        for batch_index, start in ranges:
            names = self.candidate_order[
                start : start + self.feature_batch_size
            ]
            path = batches_root / f"batch_{batch_index:03d}"
            identity = self._batch_identity(batch_index, names)
            loaded = _sealed_array(
                path,
                identity=identity,
                expected_shape=(len(names), self.row_count),
                expected_dtype=self.storage_dtype,
            )
            if loaded is None:
                _archive_incomplete(path, self.root)
                if before_batch is not None:
                    before_batch(batch_index, len(ranges), names)

                def populate(mapping: np.memmap, names: list[str] = names) -> None:
                    for local_index, name in enumerate(names):
                        codes = np.asarray(
                            discretize(X[name].reset_index(drop=True), self.n_bins),
                            dtype="int64",
                        )
                        if codes.shape != (self.row_count,):
                            raise CompactMRMRCheckpointError(
                                f"mRMR discretizer shape changed for {name}"
                            )
                        compact = codes.astype(self.storage_dtype, copy=False)
                        if not np.array_equal(
                            compact.astype("int64", copy=False), codes
                        ):
                            raise CompactMRMRCheckpointError(
                                f"compact mRMR code range overflow for {name}"
                            )
                        mapping[local_index, :] = compact

                loaded = _write_sealed_array(
                    path,
                    identity=identity,
                    filename="codes.npy",
                    shape=(len(names), self.row_count),
                    dtype=self.storage_dtype,
                    populate=populate,
                )
                self._progress(
                    "code_batch_completed",
                    batch_index=batch_index,
                    batch_count=len(ranges),
                    feature_count=len(names),
                )
            else:
                self._progress(
                    "code_batch_reused",
                    batch_index=batch_index,
                    batch_count=len(ranges),
                    feature_count=len(names),
                )
            mapping, manifest = loaded
            _close_memmap(mapping)
            manifest_sha = file_sha256(path / "manifest.json")
            batch_manifest_rows.append(
                {
                    "batch_index": batch_index,
                    "path": path.relative_to(self.root).as_posix(),
                    "manifest_sha256": manifest_sha,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "feature_count": len(names),
                }
            )
            self._batch_paths.append(path)
            self._batch_shapes.append((len(names), self.row_count))

        root_manifest_identity = {
            "operation": "compact_mrmr_discrete_code_store",
            "root_identity_sha256": canonical_sha256(self.identity),
            "batch_count": len(batch_manifest_rows),
            "batches": batch_manifest_rows,
        }
        manifest_path = self.root / "manifest.json"
        success_path = self.root / "_SUCCESS"
        if success_path.is_file():
            success = json.loads(success_path.read_text(encoding="utf-8"))
            if success.get("manifest_sha256") != file_sha256(manifest_path):
                raise CompactMRMRCheckpointError(
                    "compact mRMR root completion marker changed"
                )
            if json.loads(manifest_path.read_text(encoding="utf-8")) != (
                root_manifest_identity
            ):
                raise CompactMRMRCheckpointError(
                    "compact mRMR root manifest changed"
                )
        else:
            write_json_atomic(
                manifest_path, root_manifest_identity, overwrite=manifest_path.exists()
            )
            write_text_atomic(
                success_path,
                json.dumps(
                    {"manifest_sha256": file_sha256(manifest_path)},
                    sort_keys=True,
                )
                + "\n",
                overwrite=False,
            )
        self._root_manifest_sha256 = file_sha256(manifest_path)

    def _activate_batch(self, batch_index: int) -> np.memmap:
        if self._active_batch_index == batch_index and self._active_mapping is not None:
            return self._active_mapping
        self.close_active_batch()
        path = self._batch_paths[batch_index]
        array_path = path / "codes.npy"
        if not array_path.is_file():
            raise CompactMRMRCheckpointError(f"mRMR code batch disappeared: {path}")
        mapping = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(mapping, np.memmap)
            or mapping.shape != self._batch_shapes[batch_index]
            or mapping.dtype != self.storage_dtype
        ):
            _close_memmap(mapping if isinstance(mapping, np.memmap) else None)
            raise CompactMRMRCheckpointError(
                f"mRMR code batch shape or dtype changed after authentication: {path}"
            )
        self._active_mapping = mapping
        self._active_batch_index = batch_index
        return self._active_mapping

    def codes(self, name: str) -> np.ndarray:
        position = self._feature_positions[str(name)]
        batch_index, local_index = divmod(position, self.feature_batch_size)
        mapping = self._activate_batch(batch_index)
        return mapping[local_index]

    def close_active_batch(self) -> None:
        _close_memmap(self._active_mapping)
        self._active_mapping = None
        self._active_batch_index = None

    def close(self) -> None:
        self.close_active_batch()

    def _target_digest(self, target_codes: np.ndarray) -> str:
        values = np.ascontiguousarray(target_codes, dtype="int64")
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    def relevance(self, target_codes: np.ndarray) -> dict[str, float]:
        values = np.asarray(target_codes, dtype="int64")
        if values.shape != (self.row_count,):
            raise CompactMRMRCheckpointError("mRMR target row count changed")
        path = self.root / "mi" / "relevance"
        identity = {
            "operation": "compact_mrmr_relevance_vector",
            "code_store_manifest_sha256": self._root_manifest_sha256,
            "target_int64_sha256": self._target_digest(values),
            "candidate_order_sha256": canonical_sha256(self.candidate_order),
            "mi_estimator": "sklearn.metrics.mutual_info_score",
            "shape": [len(self.candidate_order)],
            "dtype": "float64",
        }
        loaded = _sealed_array(
            path,
            identity=identity,
            expected_shape=(len(self.candidate_order),),
            expected_dtype=np.dtype("float64"),
        )
        if loaded is None:
            _archive_incomplete(path, self.root)

            def populate(mapping: np.memmap) -> None:
                for index, name in enumerate(self.candidate_order):
                    if index % self.feature_batch_size == 0:
                        self._progress(
                            "relevance_batch_started",
                            feature_start=index,
                            feature_count=len(self.candidate_order),
                        )
                    mapping[index] = float(
                        mutual_info_score(self.codes(name), values)
                    )

            loaded = _write_sealed_array(
                path,
                identity=identity,
                filename="values.npy",
                shape=(len(self.candidate_order),),
                dtype=np.dtype("float64"),
                populate=populate,
            )
        mapping, _ = loaded
        result = {
            name: float(mapping[index])
            for index, name in enumerate(self.candidate_order)
        }
        _close_memmap(mapping)
        self.close_active_batch()
        self._relevance_manifest_sha256 = file_sha256(path / "manifest.json")
        return result

    def pair_vector(self, selected_feature: str) -> np.memmap:
        selected = str(selected_feature)
        selected_index = self._feature_positions[selected]
        slug = hashlib.sha256(selected.encode("utf-8")).hexdigest()[:16]
        path = self.root / "mi" / "pair_vectors" / (
            f"feature_{selected_index:04d}_{slug}"
        )
        identity = {
            "operation": "compact_mrmr_pair_mi_vector",
            "code_store_manifest_sha256": self._root_manifest_sha256,
            "selected_feature": selected,
            "selected_feature_index": selected_index,
            "candidate_order_sha256": canonical_sha256(self.candidate_order),
            "argument_order": "ascending_feature_name",
            "mi_estimator": "sklearn.metrics.mutual_info_score",
            "shape": [len(self.candidate_order)],
            "dtype": "float64",
        }
        loaded = _sealed_array(
            path,
            identity=identity,
            expected_shape=(len(self.candidate_order),),
            expected_dtype=np.dtype("float64"),
        )
        if loaded is None:
            _archive_incomplete(path, self.root)
            selected_codes = np.array(self.codes(selected), copy=True)
            self.close_active_batch()

            def populate(mapping: np.memmap) -> None:
                for index, candidate in enumerate(self.candidate_order):
                    if index % self.feature_batch_size == 0:
                        self._progress(
                            "pair_mi_batch_started",
                            selected_feature=selected,
                            feature_start=index,
                            feature_count=len(self.candidate_order),
                        )
                    candidate_codes = self.codes(candidate)
                    if candidate <= selected:
                        left_codes, right_codes = candidate_codes, selected_codes
                    else:
                        left_codes, right_codes = selected_codes, candidate_codes
                    mapping[index] = float(
                        mutual_info_score(left_codes, right_codes)
                    )

            loaded = _write_sealed_array(
                path,
                identity=identity,
                filename="values.npy",
                shape=(len(self.candidate_order),),
                dtype=np.dtype("float64"),
                populate=populate,
            )
            del selected_codes
            self._progress(
                "pair_vector_completed",
                selected_feature=selected,
                selected_feature_index=selected_index,
            )
        else:
            self._progress(
                "pair_vector_reused",
                selected_feature=selected,
                selected_feature_index=selected_index,
            )
        self.close_active_batch()
        self._pair_manifest_sha256s[selected] = file_sha256(
            path / "manifest.json"
        )
        return loaded[0]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": COMPACT_MRMR_CACHE_SCHEMA_VERSION,
            "memory_strategy": COMPACT_MRMR_MEMORY_STRATEGY,
            "root": str(self.root),
            "root_identity_sha256": canonical_sha256(self.identity),
            "code_store_manifest_sha256": self._root_manifest_sha256,
            "candidate_count": len(self.candidate_order),
            "row_count": self.row_count,
            "storage_dtype": self.storage_dtype.name,
            "feature_batch_size": self.feature_batch_size,
            "relevance_manifest_sha256": self._relevance_manifest_sha256,
            "pair_vector_manifest_sha256s": dict(
                sorted(self._pair_manifest_sha256s.items())
            ),
            "scientific_semantics_changed": False,
        }


__all__ = [
    "COMPACT_MRMR_CACHE_SCHEMA_VERSION",
    "COMPACT_MRMR_MEMORY_STRATEGY",
    "DEFAULT_FEATURE_BATCH_SIZE",
    "CompactMRMRCheckpointError",
    "CompactMRMRCheckpointStore",
]
