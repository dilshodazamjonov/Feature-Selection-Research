"""Validated atomic publication for canonical experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

PARTIAL_SUFFIX = ".partial"


class ArtifactIntegrityError(ValueError):
    """Raised when a temporary or finalized artifact fails validation."""


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    path: str
    size_bytes: int
    sha256: str
    artifact_format: str
    schema: dict[str, str] | None = None
    row_count: int | None = None
    ordered_row_identity_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _validate_columns(
    observed: Iterable[str],
    *,
    required_columns: Iterable[str] | None,
    expected_columns: Iterable[str] | None,
) -> None:
    observed_list = [str(value) for value in observed]
    missing = set(map(str, required_columns or ())) - set(observed_list)
    if missing:
        raise ArtifactIntegrityError(f"artifact is missing required columns: {sorted(missing)}")
    if expected_columns is not None and observed_list != list(map(str, expected_columns)):
        raise ArtifactIntegrityError(
            "artifact column order/schema mismatch: "
            f"expected={list(map(str, expected_columns))}, observed={observed_list}"
        )


def _ordered_identity_hash_csv(path: Path, column: str) -> str:
    digest = hashlib.sha256()
    for chunk in pd.read_csv(path, usecols=[column], dtype={column: "string"}, chunksize=100_000):
        for value in chunk[column]:
            encoded = json.dumps(None if pd.isna(value) else str(value), ensure_ascii=False)
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _ordered_identity_hash_parquet(path: Path, column: str) -> str:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    digest = hashlib.sha256()
    for batch in parquet.iter_batches(columns=[column], batch_size=100_000):
        values = batch.column(0).to_pylist()
        for value in values:
            encoded = json.dumps(None if value is None else str(value), ensure_ascii=False)
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def inspect_artifact(
    path: str | Path,
    *,
    artifact_format: str | None = None,
    required_columns: Iterable[str] | None = None,
    expected_columns: Iterable[str] | None = None,
    expected_row_count: int | None = None,
    ordered_row_identity_column: str | None = None,
) -> ArtifactMetadata:
    """Validate one finalized artifact and return integrity metadata."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ArtifactIntegrityError(f"artifact is missing: {candidate}")
    fmt = (artifact_format or candidate.suffix.lstrip(".") or "binary").lower()
    schema: dict[str, str] | None = None
    row_count: int | None = None
    row_identity_hash: str | None = None

    if fmt in {"json", "checkpoint"}:
        try:
            json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"JSON artifact is unreadable: {candidate}") from exc
    elif fmt in {"yaml", "yml"}:
        try:
            import yaml

            yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactIntegrityError(f"YAML artifact is unreadable: {candidate}") from exc
    elif fmt == "csv":
        try:
            header = pd.read_csv(candidate, nrows=0)
            _validate_columns(
                header.columns,
                required_columns=required_columns,
                expected_columns=expected_columns,
            )
            schema = _schema_from_frame(header)
            row_count = sum(len(chunk) for chunk in pd.read_csv(candidate, chunksize=100_000))
            if ordered_row_identity_column is not None:
                if ordered_row_identity_column not in header.columns:
                    raise ArtifactIntegrityError(
                        f"row identity column is missing: {ordered_row_identity_column}"
                    )
                row_identity_hash = _ordered_identity_hash_csv(
                    candidate, ordered_row_identity_column
                )
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactIntegrityError(f"CSV artifact is unreadable: {candidate}") from exc
    elif fmt in {"parquet", "pq"}:
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(candidate)
            names = parquet.schema_arrow.names
            _validate_columns(
                names,
                required_columns=required_columns,
                expected_columns=expected_columns,
            )
            schema = {
                field.name: str(field.type) for field in parquet.schema_arrow
            }
            row_count = int(parquet.metadata.num_rows)
            if ordered_row_identity_column is not None:
                if ordered_row_identity_column not in names:
                    raise ArtifactIntegrityError(
                        f"row identity column is missing: {ordered_row_identity_column}"
                    )
                row_identity_hash = _ordered_identity_hash_parquet(
                    candidate, ordered_row_identity_column
                )
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactIntegrityError(f"Parquet artifact is unreadable: {candidate}") from exc

    if expected_row_count is not None and row_count != int(expected_row_count):
        raise ArtifactIntegrityError(
            f"artifact row-count mismatch: expected={expected_row_count}, observed={row_count}"
        )
    return ArtifactMetadata(
        path=str(candidate),
        size_bytes=int(candidate.stat().st_size),
        sha256=sha256_file(candidate),
        artifact_format=fmt,
        schema=schema,
        row_count=row_count,
        ordered_row_identity_sha256=row_identity_hash,
    )


def _partial_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}{PARTIAL_SUFFIX}")


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def atomic_publish(
    path: str | Path,
    writer: Callable[[Path], None],
    *,
    artifact_format: str | None = None,
    required_columns: Iterable[str] | None = None,
    expected_columns: Iterable[str] | None = None,
    expected_row_count: int | None = None,
    ordered_row_identity_column: str | None = None,
    overwrite: bool = True,
    before_replace: Callable[[Path, Path], None] | None = None,
) -> ArtifactMetadata:
    """Write, validate, and atomically replace one artifact on its destination volume.

    Failed or interrupted publications retain a uniquely named ``.partial`` file.
    The caller may quarantine it, but it is never interpreted as final output.
    """

    from credit_risk_fs.experiments.result_paths import reject_historical_write

    target = reject_historical_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {target}")
    partial = reject_historical_write(_partial_path(target))
    writer(partial)
    if not partial.is_file():
        raise ArtifactIntegrityError(f"artifact writer did not create: {partial}")
    _fsync_file(partial)
    inspect_artifact(
        partial,
        artifact_format=artifact_format,
        required_columns=required_columns,
        expected_columns=expected_columns,
        expected_row_count=expected_row_count,
        ordered_row_identity_column=ordered_row_identity_column,
    )
    if before_replace is not None:
        before_replace(partial, target)
    os.replace(partial, target)
    return inspect_artifact(
        target,
        artifact_format=artifact_format,
        required_columns=required_columns,
        expected_columns=expected_columns,
        expected_row_count=expected_row_count,
        ordered_row_identity_column=ordered_row_identity_column,
    )


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any] | list[Any],
    *,
    overwrite: bool = True,
) -> ArtifactMetadata:
    def writer(partial: Path) -> None:
        partial.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    return atomic_publish(path, writer, artifact_format="json", overwrite=overwrite)


def write_yaml_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> ArtifactMetadata:
    def writer(partial: Path) -> None:
        import yaml

        partial.write_text(
            yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    return atomic_publish(path, writer, artifact_format="yaml", overwrite=overwrite)


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = True,
) -> ArtifactMetadata:
    return atomic_publish(
        path,
        lambda partial: partial.write_text(text, encoding="utf-8"),
        artifact_format="text",
        overwrite=overwrite,
    )


def write_csv_atomic(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] | None = None,
    ordered_row_identity_column: str | None = None,
    overwrite: bool = True,
) -> ArtifactMetadata:
    columns = [str(value) for value in frame.columns]
    return atomic_publish(
        path,
        lambda partial: frame.to_csv(partial, index=False),
        artifact_format="csv",
        required_columns=required_columns,
        expected_columns=columns,
        expected_row_count=len(frame),
        ordered_row_identity_column=ordered_row_identity_column,
        overwrite=overwrite,
    )


def write_parquet_atomic(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] | None = None,
    ordered_row_identity_column: str | None = None,
    overwrite: bool = True,
) -> ArtifactMetadata:
    columns = [str(value) for value in frame.columns]
    return atomic_publish(
        path,
        lambda partial: frame.to_parquet(partial, index=False),
        artifact_format="parquet",
        required_columns=required_columns,
        expected_columns=columns,
        expected_row_count=len(frame),
        ordered_row_identity_column=ordered_row_identity_column,
        overwrite=overwrite,
    )


def copy_atomic(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> ArtifactMetadata:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return atomic_publish(
        destination,
        lambda partial: shutil.copyfile(source_path, partial),
        artifact_format=source_path.suffix.lstrip(".") or "binary",
        overwrite=overwrite,
    )


def partial_artifacts(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(path for path in root.rglob(f"*{PARTIAL_SUFFIX}") if path.is_file())


def quarantine_partial_artifacts(run_directory: str | Path) -> list[Path]:
    """Move only known partial files into the run-local incomplete directory."""

    run_root = Path(run_directory).resolve()
    incomplete = run_root / "incomplete"
    moved: list[Path] = []
    for source in partial_artifacts(run_root):
        if source.is_relative_to(incomplete):
            continue
        incomplete.mkdir(parents=True, exist_ok=True)
        destination = incomplete / source.name
        if destination.exists():
            destination = incomplete / f"{source.stem}.{uuid.uuid4().hex}{source.suffix}"
        os.replace(source, destination)
        moved.append(destination)
    return moved
