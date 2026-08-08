"""Deterministic, resource-bounded adapter for the frozen 2024 protocol.

Nothing in this module executes at import time.  Data access is possible only
through functions that require an explicit input root and authenticated
``AdapterContract``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
import unicodedata

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contract import (
    AdapterContract,
    FeatureRule,
    ProtocolContractError,
    TableSpec,
    canonical_sha256,
    file_sha256,
    load_adapter_contract,
)


BuildMode = Literal["fixture", "research"]
ResourceHook = Callable[[Mapping[str, Any]], None]
INTERNAL_SOURCE_PART = "__protocol_source_part"
INTERNAL_ROW_OFFSET = "__physical_row_offset"
CHECKPOINT_SCHEMA_VERSION = "homecredit_model_stability_2024_checkpoint_v1"
MANIFEST_SCHEMA_VERSION = "homecredit_model_stability_2024_manifest_v1"


class AdapterError(ValueError):
    """Base exception for fail-closed adapter validation."""


class InventoryError(AdapterError):
    """Raised for missing, changed, unknown, or forbidden input files."""


class SchemaMismatchError(AdapterError):
    """Raised before reading rows when an Arrow schema is not exact."""


class DataValidationError(AdapterError):
    """Raised when relational or value invariants do not hold."""


class CheckpointMismatchError(AdapterError):
    """Raised when a checkpoint cannot be authenticated for reuse."""


class ManifestAuthenticationError(AdapterError):
    """Raised when a completed build artifact is missing or changed."""


class LeakageBoundaryError(AdapterError):
    """Raised when a downstream fit/transform scope would leak OOT data."""


@dataclass(frozen=True, slots=True)
class InputArtifact:
    relative_path: str
    size_bytes: int
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class InputInventory:
    mode: BuildMode
    artifacts: tuple[InputArtifact, ...]
    identity_sha256: str
    excluded_depth_2_present: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "artifacts": [asdict(item) for item in self.artifacts],
            "identity_sha256": self.identity_sha256,
            "excluded_depth_2_present": list(self.excluded_depth_2_present),
        }


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    output_feature: str
    source_family: str
    source_feature: str | None
    aggregation: str
    logical_type: str
    protocol_action: str
    protocol_output_prefix: str


@dataclass(frozen=True, slots=True)
class FitScopeToken:
    fitted_scope: Literal["dev_fold", "full_dev"]
    fitted_case_identity_sha256: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_root: Path
    manifest_path: Path
    manifest_sha256: str
    matrix_parts: tuple[Path, ...]
    predictor_count: int
    row_count: int
    reused_completed_build: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(hook: ResourceHook | None, event: str, **values: Any) -> None:
    if hook is not None:
        hook({"event": event, **values})


def _require_mode(mode: str) -> BuildMode:
    if mode not in {"fixture", "research"}:
        raise AdapterError("mode must be explicitly 'fixture' or 'research'")
    return mode  # type: ignore[return-value]


def _safe_input_path(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    if not candidate.is_relative_to(root):
        raise InventoryError(f"input path escapes explicit root: {relative_path}")
    return candidate


def authenticate_contract_instance(contract: AdapterContract) -> None:
    """Re-authenticate the instance's explicit lock before any data-root access."""

    authoritative = load_adapter_contract(contract.lock_path)
    if canonical_sha256(contract.to_dict()) != canonical_sha256(authoritative.to_dict()):
        raise ProtocolContractError(
            "adapter contract instance differs from its authenticated canonical lock"
        )


def inspect_input_inventory(
    input_root: str | Path,
    contract: AdapterContract,
    *,
    mode: str,
    requested_families: Iterable[str] | None = None,
) -> InputInventory:
    """Authenticate explicit included files and reject unknown Parquet inputs.

    Known depth-2 files are recorded by path only and are never opened.  They
    are part of the official download layout but cannot be requested or read.
    """

    resolved_mode = _require_mode(mode)
    authenticate_contract_instance(contract)
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise InventoryError(f"explicit input root is missing: {root}")
    if requested_families is not None:
        contract.validate_requested_tables(requested_families)

    included = set(contract.included_input_paths)
    excluded = set(contract.excluded_depth_2_paths)
    observed_parquets: set[str] = set()
    for path in root.rglob("*.parquet"):
        relative = path.relative_to(root).as_posix()
        observed_parquets.add(relative)
    unknown = observed_parquets - included - excluded
    if unknown:
        raise InventoryError(f"unregistered Parquet input(s): {sorted(unknown)}")

    artifacts: list[InputArtifact] = []
    specs = [*contract.support_files]
    specs.extend(part for table in contract.tables for part in table.partitions)
    for spec in specs:
        path = _safe_input_path(root, spec.relative_path)
        if not path.is_file():
            raise InventoryError(f"registered input is missing: {spec.relative_path}")
        size = int(path.stat().st_size)
        digest = file_sha256(path)
        if resolved_mode == "research" and (
            size != spec.expected_size_bytes or digest != spec.expected_sha256
        ):
            raise InventoryError(
                f"research input identity mismatch: {spec.relative_path}"
            )
        artifacts.append(
            InputArtifact(
                relative_path=spec.relative_path,
                size_bytes=size,
                sha256=digest,
                role="support" if spec in contract.support_files else "included_parquet",
            )
        )
    identity = canonical_sha256([asdict(item) for item in artifacts])
    if resolved_mode == "research" and identity != contract.included_raw_input_digest:
        # The lock digest authenticates its canonical inventory projection,
        # which may differ from this richer fixture-safe projection. Verify the
        # canonical projection separately instead of weakening either identity.
        canonical_projection = [
            {
                "relative_path": spec.protocol_relative_path,
                "byte_size": spec.expected_size_bytes,
                "sha256": spec.expected_sha256,
            }
            for spec in sorted(specs, key=lambda item: item.protocol_relative_path)
        ]
        if canonical_sha256(canonical_projection) != contract.included_raw_input_digest:
            raise InventoryError("canonical included-input digest reconstruction failed")
    return InputInventory(
        mode=resolved_mode,
        artifacts=tuple(artifacts),
        identity_sha256=identity,
        excluded_depth_2_present=tuple(sorted(observed_parquets & excluded)),
    )


def validate_requested_partition(
    contract: AdapterContract, family: str, relative_path: str
) -> None:
    """Reject depth-2 and unregistered partitions without opening them."""

    if relative_path in contract.excluded_depth_2_paths:
        raise InventoryError(f"depth-2 partition execution is forbidden: {relative_path}")
    table = contract.table(family)
    if relative_path not in {item.relative_path for item in table.partitions}:
        raise InventoryError(
            f"partition is not registered for {family}: {relative_path}"
        )


def _expected_schema(table: TableSpec) -> pa.Schema:
    types = {
        "int64": pa.int64(),
        "double": pa.float64(),
        "string": pa.string(),
        "bool": pa.bool_(),
    }
    try:
        return pa.schema([(name, types[dtype]) for name, dtype in table.schema_fields])
    except KeyError as exc:
        raise ProtocolContractError(f"unsupported locked Arrow dtype: {exc.args[0]}") from exc


def validate_partition_schema(path: str | Path, table: TableSpec) -> None:
    observed = pq.ParquetFile(path).schema_arrow
    expected = _expected_schema(table)
    if observed != expected:
        raise SchemaMismatchError(
            f"schema mismatch for {table.family}: expected={expected}, observed={observed}"
        )


def _required_columns(table: TableSpec) -> tuple[str, ...]:
    return tuple(
        rule.feature_name
        for rule in table.feature_rules
        if rule.action == "include"
        or rule.logical_type in {
            "identifier",
            "order_identifier",
            "target",
            "split_control",
        }
    )


def read_registered_table(
    input_root: str | Path,
    table: TableSpec,
    *,
    case_ids: Sequence[int] | None = None,
    batch_size: int = 65_536,
    resource_hook: ResourceHook | None = None,
) -> pa.Table:
    """Read projected columns in locked partition order using bounded batches."""

    root = Path(input_root).resolve()
    columns = _required_columns(table)
    batches: list[pa.RecordBatch] = []
    case_set = None if case_ids is None else pa.array(case_ids, type=pa.int64())
    for source_part, partition in enumerate(table.partitions):
        validate_requested_partition_for_table = partition.relative_path
        if validate_requested_partition_for_table is None:  # pragma: no cover
            raise InventoryError("empty partition path")
        path = _safe_input_path(root, partition.relative_path)
        validate_partition_schema(path, table)
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            original_rows = batch.num_rows
            offsets = pa.array(range(offset, offset + original_rows), type=pa.int64())
            offset += original_rows
            if case_set is not None:
                mask = pc.is_in(batch.column(batch.schema.get_field_index("case_id")), value_set=case_set)
                selected = pc.indices_nonzero(mask)
                batch = batch.take(selected)
                offsets = offsets.take(selected)
            if table.depth == "1" and batch.num_rows:
                batch = batch.append_column(
                    INTERNAL_SOURCE_PART,
                    pa.array([source_part] * batch.num_rows, type=pa.int32()),
                )
                batch = batch.append_column(INTERNAL_ROW_OFFSET, offsets)
            if batch.num_rows:
                batches.append(batch)
            _emit(
                resource_hook,
                "input_batch",
                family=table.family,
                source_part=source_part,
                rows_read=original_rows,
                rows_selected=batch.num_rows,
            )
    schema = _expected_schema(table)
    if table.depth == "1":
        schema = schema.append(pa.field(INTERNAL_SOURCE_PART, pa.int32())).append(
            pa.field(INTERNAL_ROW_OFFSET, pa.int64())
        )
    if not batches:
        selected_names = columns + (
            (INTERNAL_SOURCE_PART, INTERNAL_ROW_OFFSET) if table.depth == "1" else ()
        )
        selected_schema = pa.schema([schema.field(name) for name in selected_names])
        return pa.Table.from_batches([], schema=selected_schema)
    return pa.Table.from_batches(batches)


def validate_base(base: pa.Table) -> pa.Table:
    required = {"case_id", "date_decision", "MONTH", "WEEK_NUM", "target"}
    if not required.issubset(base.column_names):
        raise DataValidationError(f"base is missing columns: {sorted(required - set(base.column_names))}")
    keys = base["case_id"].to_pylist()
    if any(value is None for value in keys):
        raise DataValidationError("base case_id must be non-null")
    if len(set(keys)) != len(keys):
        raise DataValidationError("base case_id must be unique")
    targets = base["target"].to_pylist()
    if any(value is None for value in targets) or not set(targets).issubset({0, 1}):
        raise DataValidationError("base target must be non-null binary 0/1")
    dates = base["date_decision"].to_pylist()
    if any(value is None for value in dates):
        raise DataValidationError("base date_decision must be non-null")
    for value in dates:
        _parse_date(value, field="base.date_decision")
    order = pc.sort_indices(base, sort_keys=[("date_decision", "ascending"), ("case_id", "ascending")])
    return base.take(order)


def _parse_date(value: Any, *, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataValidationError(f"{field} must be an ISO date string or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DataValidationError(f"{field} is not a strict ISO date: {value!r}") from exc
    if parsed.isoformat() != value:
        raise DataValidationError(f"{field} is not canonical YYYY-MM-DD: {value!r}")
    return parsed


def _signed_days(source: Any, reference: Any, *, field: str) -> int | None:
    source_date = _parse_date(source, field=field)
    reference_date = _parse_date(reference, field="base.date_decision")
    if source_date is None:
        return None
    if reference_date is None:  # validate_base prevents this, retained fail-closed.
        raise DataValidationError("base.date_decision must be non-null")
    return (source_date - reference_date).days


def _ensure_related_keys(
    base: pa.Table, related: pa.Table, *, family: str
) -> dict[int, int]:
    base_keys = [int(value) for value in base["case_id"].to_pylist()]
    base_index = {value: index for index, value in enumerate(base_keys)}
    related_keys = related["case_id"].to_pylist()
    if any(value is None for value in related_keys):
        raise DataValidationError(f"{family} case_id must be non-null")
    orphans = sorted(set(map(int, related_keys)) - set(base_index))
    if orphans:
        raise DataValidationError(
            f"{family} contains orphan case_id values; count={len(orphans)}, first={orphans[:5]}"
        )
    return base_index


def join_depth_0(base: pa.Table, related: pa.Table, table: TableSpec) -> tuple[pa.Table, tuple[FeatureLineage, ...]]:
    """Apply one-to-one validation and a stable base-left join."""

    if table.depth != "0":
        raise AdapterError(f"not a depth-0 table: {table.family}")
    _ensure_related_keys(base, related, family=table.family)
    related_keys = [int(value) for value in related["case_id"].to_pylist()]
    if len(set(related_keys)) != len(related_keys):
        raise DataValidationError(f"depth-0 {table.family} case_id must be unique")
    row_for_case = {value: index for index, value in enumerate(related_keys)}
    take_indices = pa.array(
        [row_for_case.get(int(value)) for value in base["case_id"].to_pylist()],
        type=pa.int64(),
    )
    output = base
    lineage: list[FeatureLineage] = []
    base_dates = base["date_decision"].to_pylist()
    for rule in table.included_features:
        name = rule.output_prefix
        if not name or name in output.column_names:
            raise DataValidationError(f"invalid or colliding output feature: {name!r}")
        source = pc.take(related[rule.feature_name], take_indices).to_pylist()
        if rule.logical_type == "date":
            values = [
                _signed_days(value, base_dates[index], field=f"{table.family}.{rule.feature_name}")
                for index, value in enumerate(source)
            ]
            array = pa.array(values, type=pa.int64())
        else:
            array = pa.array(source, type=related[rule.feature_name].type)
        output = output.append_column(name, array)
        lineage.append(
            FeatureLineage(
                output_feature=name,
                source_family=table.family,
                source_feature=rule.feature_name,
                aggregation="identity_after_family_prefix"
                if rule.logical_type != "date"
                else "signed_days_relative_to_base_date_decision",
                logical_type=rule.logical_type,
                protocol_action=rule.action,
                protocol_output_prefix=rule.output_prefix,
            )
        )
    if output.num_rows != base.num_rows or output["case_id"].to_pylist() != base["case_id"].to_pylist():
        raise DataValidationError("depth-0 join multiplied, lost, or reordered base cases")
    return output, tuple(lineage)


def _lexical_key(value: str) -> tuple[str, bytes]:
    normalized = unicodedata.normalize("NFC", value)
    return normalized.casefold(), value.encode("utf-8")


def _sample_variance(values: Sequence[float | int]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.variance(map(float, values)))


def _numeric_values(
    values: Sequence[Any], *, family: str, feature: str
) -> list[float | int]:
    result: list[float | int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataValidationError(f"{family}.{feature} is not numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise DataValidationError(f"{family}.{feature} contains non-finite numeric data")
        result.append(value)
    return result


def _aggregate_numeric(values: Sequence[Any], row_count: int) -> dict[str, Any]:
    observed = _numeric_values(values, family="depth1", feature="value")
    count = len(observed)
    if row_count == 0:
        return {
            "count_non_missing": None,
            "missing_count": None,
            "min": None,
            "max": None,
            "mean": None,
            "sum": None,
            "sample_variance_ddof_1": None,
            "first_by_num_group1": None,
            "last_by_num_group1": None,
        }
    return {
        "count_non_missing": count,
        "missing_count": row_count - count,
        "min": min(observed) if observed else None,
        "max": max(observed) if observed else None,
        "mean": float(sum(observed) / count) if observed else None,
        "sum": sum(observed) if observed else None,
        "sample_variance_ddof_1": _sample_variance(observed),
        # Literal ordered values intentionally do not skip nulls.
        "first_by_num_group1": values[0],
        "last_by_num_group1": values[-1],
    }


def _aggregate_date(values: Sequence[Any], row_count: int) -> dict[str, Any]:
    result = _aggregate_numeric(values, row_count)
    result.pop("sum")
    return result


def _aggregate_boolean(values: Sequence[Any], row_count: int) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    if any(not isinstance(value, bool) for value in observed):
        raise DataValidationError("boolean aggregation received a non-boolean value")
    if row_count == 0:
        return {
            key: None
            for key in (
                "count_non_missing",
                "missing_count",
                "false_count",
                "true_count",
                "mean",
                "first_by_num_group1",
                "last_by_num_group1",
                "any",
                "all",
            )
        }
    true_count = sum(observed)
    return {
        "count_non_missing": len(observed),
        "missing_count": row_count - len(observed),
        "false_count": len(observed) - true_count,
        "true_count": true_count,
        "mean": float(true_count / len(observed)) if observed else None,
        "first_by_num_group1": values[0],
        "last_by_num_group1": values[-1],
        "any": any(observed) if observed else None,
        "all": all(observed) if observed else None,
    }


def _aggregate_categorical(values: Sequence[Any], row_count: int) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    if any(not isinstance(value, str) for value in observed):
        raise DataValidationError("categorical aggregation received a non-string value")
    if row_count == 0:
        return {
            key: None
            for key in (
                "count_non_missing",
                "missing_count",
                "nunique",
                "lexical_mode",
                "first_by_num_group1",
                "last_by_num_group1",
            )
        }
    counts = Counter(observed)
    lexical_mode = None
    if counts:
        maximum = max(counts.values())
        lexical_mode = min(
            (value for value, count in counts.items() if count == maximum),
            key=_lexical_key,
        )
    return {
        "count_non_missing": len(observed),
        "missing_count": row_count - len(observed),
        "nunique": len(counts),
        "lexical_mode": lexical_mode,
        "first_by_num_group1": values[0],
        "last_by_num_group1": values[-1],
    }


def _aggregation_names(contract: AdapterContract, logical_type: str) -> tuple[str, ...]:
    values = {
        "numeric": contract.numeric_aggregations,
        "date": contract.date_aggregations,
        "boolean": contract.boolean_aggregations,
        "categorical": contract.categorical_aggregations,
    }
    try:
        return values[logical_type]
    except KeyError as exc:
        raise ProtocolContractError(
            f"included feature has unsupported logical type: {logical_type}"
        ) from exc


def _aggregation_arrow_type(rule: FeatureRule, aggregation: str) -> pa.DataType:
    if aggregation in {
        "count_non_missing",
        "missing_count",
        "false_count",
        "true_count",
        "nunique",
    }:
        return pa.int64()
    if aggregation in {"mean", "sample_variance_ddof_1"}:
        return pa.float64()
    if rule.logical_type == "numeric":
        return pa.float64()
    if rule.logical_type == "date":
        return pa.int64()
    if rule.logical_type == "boolean":
        return pa.bool_()
    if rule.logical_type == "categorical":
        return pa.string()
    raise ProtocolContractError(
        f"no output dtype for {rule.logical_type}/{aggregation}"
    )


def _aggregate_values(
    values: Sequence[Any], row_count: int, logical_type: str
) -> dict[str, Any]:
    if logical_type == "numeric":
        return _aggregate_numeric(values, row_count)
    if logical_type == "date":
        return _aggregate_date(values, row_count)
    if logical_type == "boolean":
        return _aggregate_boolean(values, row_count)
    if logical_type == "categorical":
        return _aggregate_categorical(values, row_count)
    raise ProtocolContractError(f"unsupported included logical type: {logical_type}")


def aggregate_depth_1(
    base: pa.Table,
    related: pa.Table,
    table: TableSpec,
    contract: AdapterContract,
) -> tuple[pa.Table, tuple[FeatureLineage, ...]]:
    """Reduce one depth-1 family to exactly one row per base case."""

    if table.depth != "1":
        raise AdapterError(f"not a depth-1 table: {table.family}")
    _ensure_related_keys(base, related, family=table.family)
    if "num_group1" not in related.column_names:
        raise DataValidationError(f"{table.family} is missing num_group1")
    if related["num_group1"].null_count:
        raise DataValidationError(f"{table.family}.num_group1 must be non-null")
    for internal in (INTERNAL_SOURCE_PART, INTERNAL_ROW_OFFSET):
        if internal not in related.column_names:
            raise DataValidationError(f"{table.family} lacks deterministic tie key {internal}")
    order = pc.sort_indices(
        related,
        sort_keys=[
            ("case_id", "ascending"),
            ("num_group1", "ascending"),
            (INTERNAL_SOURCE_PART, "ascending"),
            (INTERNAL_ROW_OFFSET, "ascending"),
        ],
    )
    related = related.take(order)
    related_keys = [int(value) for value in related["case_id"].to_pylist()]
    if table.cardinality == "one_to_one" and len(set(related_keys)) != len(related_keys):
        raise DataValidationError(
            f"depth-1 {table.family} violates locked one-to-one cardinality"
        )
    rows_by_case: dict[int, list[int]] = {}
    for index, case_id in enumerate(related_keys):
        rows_by_case.setdefault(case_id, []).append(index)

    base_keys = [int(value) for value in base["case_id"].to_pylist()]
    base_dates = base["date_decision"].to_pylist()
    output: dict[str, pa.Array] = {"case_id": pa.array(base_keys, type=pa.int64())}
    row_count_name = f"d1__{table.family}__row_count"
    output[row_count_name] = pa.array(
        [len(rows_by_case.get(case_id, ())) for case_id in base_keys], type=pa.int64()
    )
    lineage: list[FeatureLineage] = [
        FeatureLineage(
            output_feature=row_count_name,
            source_family=table.family,
            source_feature=None,
            aggregation="row_count",
            logical_type="relational_count",
            protocol_action="generated_by_locked_depth_1_rule",
            protocol_output_prefix=f"d1__{table.family}__row_count",
        )
    ]

    for rule in table.included_features:
        source_values = related[rule.feature_name].to_pylist()
        aggregates = _aggregation_names(contract, rule.logical_type)
        per_aggregation: dict[str, list[Any]] = {name: [] for name in aggregates}
        for base_index, case_id in enumerate(base_keys):
            indices = rows_by_case.get(case_id, ())
            values = [source_values[index] for index in indices]
            if rule.logical_type == "date":
                values = [
                    _signed_days(
                        value,
                        base_dates[base_index],
                        field=f"{table.family}.{rule.feature_name}",
                    )
                    for value in values
                ]
            result = _aggregate_values(values, len(indices), rule.logical_type)
            if tuple(result) != aggregates:
                raise ProtocolContractError(
                    f"aggregation implementation order differs for {rule.logical_type}"
                )
            for aggregation in aggregates:
                per_aggregation[aggregation].append(result[aggregation])
        for aggregation in aggregates:
            output_name = f"{rule.output_prefix}__{aggregation}"
            if output_name in output:
                raise DataValidationError(f"colliding aggregate output: {output_name}")
            output[output_name] = pa.array(
                per_aggregation[aggregation],
                type=_aggregation_arrow_type(rule, aggregation),
            )
            lineage.append(
                FeatureLineage(
                    output_feature=output_name,
                    source_family=table.family,
                    source_feature=rule.feature_name,
                    aggregation=aggregation,
                    logical_type=rule.logical_type,
                    protocol_action=rule.action,
                    protocol_output_prefix=rule.output_prefix,
                )
            )
    compact = pa.table(output)
    if compact.num_rows != base.num_rows or compact["case_id"].to_pylist() != base["case_id"].to_pylist():
        raise DataValidationError("depth-1 aggregation lost or reordered base cases")
    return compact, tuple(lineage)


def join_compact(base: pa.Table, compact: pa.Table) -> pa.Table:
    """Join an already reduced table while preserving exact base identity/order."""

    keys = compact["case_id"].to_pylist()
    if len(keys) != compact.num_rows or len(set(keys)) != len(keys):
        raise DataValidationError("compact table is not one row per case_id")
    if keys != base["case_id"].to_pylist():
        raise DataValidationError("compact table identity/order differs from base shard")
    output = base
    for name in compact.column_names:
        if name == "case_id":
            continue
        if name in output.column_names:
            raise DataValidationError(f"compact join collision: {name}")
        output = output.append_column(name, compact[name])
    return output


def predictor_columns(contract: AdapterContract) -> tuple[str, ...]:
    names: list[str] = []
    for family in contract.depth_0_order:
        names.extend(rule.output_prefix for rule in contract.table(family).included_features)
    for family in contract.depth_1_order:
        table = contract.table(family)
        names.append(f"d1__{family}__row_count")
        for rule in table.included_features:
            names.extend(
                f"{rule.output_prefix}__{aggregation}"
                for aggregation in _aggregation_names(contract, rule.logical_type)
            )
    if len(names) != len(set(names)):
        raise ProtocolContractError("locked naming rules create output collisions")
    return tuple(names)


def assert_predictor_boundary(matrix: pa.Table, contract: AdapterContract) -> None:
    expected = predictor_columns(contract)
    observed = tuple(name for name in matrix.column_names if name in set(expected))
    if observed != expected:
        raise DataValidationError("predictor ordering differs from the frozen contract")
    forbidden = {"target", "case_id", "date_decision", "MONTH", "WEEK_NUM", "num_group1"}
    if forbidden & set(expected):
        raise ProtocolContractError("identifier/target/split control entered predictors")


def fit_scope_token(
    *, scope: str, case_ids: Sequence[int], memberships: Sequence[str]
) -> FitScopeToken:
    if scope not in {"dev_fold", "full_dev"}:
        raise LeakageBoundaryError("fit scope must be dev_fold or full_dev")
    if len(case_ids) != len(memberships) or not case_ids:
        raise LeakageBoundaryError("fit identity and membership must be aligned and non-empty")
    if any(value != "DEV" for value in memberships):
        raise LeakageBoundaryError("OOT membership may not influence a fitted state")
    identity = canonical_sha256([int(value) for value in case_ids])
    return FitScopeToken(scope, identity)  # type: ignore[arg-type]


def assert_transform_scope(token: FitScopeToken, *, transform_membership: str) -> None:
    if transform_membership not in {"DEV", "OOT"}:
        raise LeakageBoundaryError("transform membership must be DEV or OOT")
    if transform_membership == "OOT" and token.fitted_scope != "full_dev":
        raise LeakageBoundaryError("OOT transform requires a full-DEV-fitted state")


def expected_lineage(contract: AdapterContract) -> tuple[FeatureLineage, ...]:
    lineage: list[FeatureLineage] = []
    for family in contract.depth_0_order:
        for rule in contract.table(family).included_features:
            lineage.append(
                FeatureLineage(
                    output_feature=rule.output_prefix,
                    source_family=family,
                    source_feature=rule.feature_name,
                    aggregation=(
                        "signed_days_relative_to_base_date_decision"
                        if rule.logical_type == "date"
                        else "identity_after_family_prefix"
                    ),
                    logical_type=rule.logical_type,
                    protocol_action=rule.action,
                    protocol_output_prefix=rule.output_prefix,
                )
            )
    for family in contract.depth_1_order:
        lineage.append(
            FeatureLineage(
                output_feature=f"d1__{family}__row_count",
                source_family=family,
                source_feature=None,
                aggregation="row_count",
                logical_type="relational_count",
                protocol_action="generated_by_locked_depth_1_rule",
                protocol_output_prefix=f"d1__{family}__row_count",
            )
        )
        for rule in contract.table(family).included_features:
            for aggregation in _aggregation_names(contract, rule.logical_type):
                lineage.append(
                    FeatureLineage(
                        output_feature=f"{rule.output_prefix}__{aggregation}",
                        source_family=family,
                        source_feature=rule.feature_name,
                        aggregation=aggregation,
                        logical_type=rule.logical_type,
                        protocol_action=rule.action,
                        protocol_output_prefix=rule.output_prefix,
                    )
                )
    if tuple(item.output_feature for item in lineage) != predictor_columns(contract):
        raise ProtocolContractError("lineage and predictor order differ")
    return tuple(lineage)


def _write_json_atomic(path: Path, payload: Mapping[str, Any] | list[Any], *, overwrite: bool) -> dict[str, Any]:
    from credit_risk_fs.experiments.atomic_io import write_json_atomic

    return write_json_atomic(path, payload, overwrite=overwrite).to_dict()


def _write_text_atomic(path: Path, value: str, *, overwrite: bool) -> dict[str, Any]:
    from credit_risk_fs.experiments.atomic_io import write_text_atomic

    return write_text_atomic(path, value, overwrite=overwrite).to_dict()


def _write_parquet_atomic(path: Path, table: pa.Table, *, overwrite: bool) -> dict[str, Any]:
    from credit_risk_fs.experiments.atomic_io import atomic_publish

    metadata = atomic_publish(
        path,
        lambda partial: pq.write_table(
            table,
            partial,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        ),
        artifact_format="parquet",
        expected_columns=table.column_names,
        expected_row_count=table.num_rows,
        ordered_row_identity_column="case_id",
        overwrite=overwrite,
    )
    return metadata.to_dict()


def _checkpoint_identity(
    contract: AdapterContract,
    inventory: InputInventory,
    table: TableSpec | None,
    shard_id: int,
    base_case_identity: str,
) -> dict[str, Any]:
    return {
        "adapter_version": contract.adapter_version,
        "protocol_file_sha256": contract.lock_file_sha256,
        "protocol_internal_sha256": contract.lock_internal_sha256,
        "input_inventory_identity_sha256": inventory.identity_sha256,
        "logical_table": "final_matrix" if table is None else table.family,
        "table_schema_sha256": "final_matrix_v1" if table is None else table.schema_sha256,
        "aggregation_rules_sha256": canonical_sha256(
            {
                "numeric": contract.numeric_aggregations,
                "date": contract.date_aggregations,
                "boolean": contract.boolean_aggregations,
                "categorical": contract.categorical_aggregations,
                "empty": contract.empty_group_rule,
                "tie": contract.categorical_mode_tie_break,
            }
        ),
        "shard_id": shard_id,
        "base_case_identity_sha256": base_case_identity,
    }


def _load_reusable_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    expected_identity: Mapping[str, Any],
) -> pa.Table | None:
    if not checkpoint_path.exists() and not output_path.exists():
        return None
    if not checkpoint_path.is_file() or not output_path.is_file():
        raise CheckpointMismatchError("checkpoint and compact output must exist together")
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointMismatchError(f"checkpoint is unreadable: {checkpoint_path}") from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointMismatchError("checkpoint schema version mismatch")
    if payload.get("identity") != dict(expected_identity):
        raise CheckpointMismatchError("stale or mismatched checkpoint identity")
    if file_sha256(output_path) != payload.get("output", {}).get("sha256"):
        raise CheckpointMismatchError("checkpoint output digest mismatch")
    table = pq.read_table(output_path)
    if table.num_rows != payload.get("output", {}).get("row_count"):
        raise CheckpointMismatchError("checkpoint output row-count mismatch")
    if canonical_sha256([(field.name, str(field.type)) for field in table.schema]) != payload.get(
        "output", {}
    ).get("schema_sha256"):
        raise CheckpointMismatchError("checkpoint output schema mismatch")
    return table


def _publish_checkpointed_table(
    checkpoint_path: Path,
    output_path: Path,
    table: pa.Table,
    identity: Mapping[str, Any],
) -> None:
    metadata = _write_parquet_atomic(output_path, table, overwrite=False)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "complete",
        "identity": dict(identity),
        "output": {
            "relative_name": output_path.name,
            "size_bytes": metadata["size_bytes"],
            "sha256": metadata["sha256"],
            "row_count": table.num_rows,
            "schema_sha256": canonical_sha256(
                [(field.name, str(field.type)) for field in table.schema]
            ),
        },
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(checkpoint_path, checkpoint, overwrite=False)


def validate_output_manifest(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    marker = root / "_SUCCESS"
    manifest_path = root / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise ManifestAuthenticationError("completed output lacks marker or manifest")
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestAuthenticationError("completion metadata is unreadable") from exc
    manifest_digest = file_sha256(manifest_path)
    if marker_payload.get("manifest_sha256") != manifest_digest:
        raise ManifestAuthenticationError("completion marker does not bind manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestAuthenticationError("manifest schema version mismatch")
    for artifact in manifest.get("artifacts", []):
        relative = Path(str(artifact.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestAuthenticationError("manifest artifact path is unsafe")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestAuthenticationError(f"manifest artifact is missing: {relative}")
        if int(path.stat().st_size) != int(artifact.get("size_bytes", -1)):
            raise ManifestAuthenticationError(f"manifest size mismatch: {relative}")
        if file_sha256(path) != artifact.get("sha256"):
            raise ManifestAuthenticationError(f"manifest digest mismatch: {relative}")
    return manifest


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _existing_completed_result(
    output_root: Path,
    contract: AdapterContract,
    inventory: InputInventory,
) -> BuildResult | None:
    if not (output_root / "_SUCCESS").exists():
        return None
    manifest = validate_output_manifest(output_root)
    identity = manifest.get("identity", {})
    expected = {
        "adapter_version": contract.adapter_version,
        "protocol_file_sha256": contract.lock_file_sha256,
        "protocol_internal_sha256": contract.lock_internal_sha256,
        "input_inventory_identity_sha256": inventory.identity_sha256,
        "mode": inventory.mode,
    }
    if identity != expected:
        raise ManifestAuthenticationError(
            "valid completed artifact is incompatible and will not be overwritten"
        )
    parts = tuple(
        output_root / item["path"]
        for item in manifest["artifacts"]
        if str(item["path"]).startswith("matrix/")
    )
    return BuildResult(
        output_root=output_root,
        manifest_path=output_root / "manifest.json",
        manifest_sha256=file_sha256(output_root / "manifest.json"),
        matrix_parts=parts,
        predictor_count=int(manifest["summary"]["predictor_count"]),
        row_count=int(manifest["summary"]["row_count"]),
        reused_completed_build=True,
    )


def build_modeling_matrix(
    *,
    input_root: str | Path,
    output_root: str | Path,
    contract: AdapterContract,
    mode: str,
    shard_rows: int = 50_000,
    resource_hook: ResourceHook | None = None,
) -> BuildResult:
    """Build checkpointed matrix shards with no preprocessing or model work."""

    resolved_mode = _require_mode(mode)
    if shard_rows < 1:
        raise AdapterError("shard_rows must be positive")
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()
    if output_path == input_path or output_path.is_relative_to(input_path):
        raise AdapterError("output root must not overlap the explicit input root")
    inventory = inspect_input_inventory(input_path, contract, mode=resolved_mode)
    existing = _existing_completed_result(output_path, contract, inventory)
    if existing is not None:
        return existing
    if output_path.exists() and any(
        path.name == "_SUCCESS" for path in output_path.iterdir()
    ):
        raise ManifestAuthenticationError("unvalidated completion marker present")

    started = datetime.now(timezone.utc)
    base = validate_base(read_registered_table(input_path, contract.base_table, resource_hook=resource_hook))
    base_case_ids = [int(value) for value in base["case_id"].to_pylist()]
    shard_bounds = [
        (start, min(start + shard_rows, base.num_rows))
        for start in range(0, base.num_rows, shard_rows)
    ]
    if not shard_bounds:
        raise DataValidationError("base population must not be empty")
    output_path.mkdir(parents=True, exist_ok=True)
    intermediate = output_path / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)

    for family in (*contract.depth_0_order, *contract.depth_1_order):
        table = contract.table(family)
        family_dir = intermediate / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for shard_id, (start, end) in enumerate(shard_bounds):
            base_shard = base.slice(start, end - start)
            case_ids = base_case_ids[start:end]
            case_identity = canonical_sha256(case_ids)
            compact_path = family_dir / f"part-{shard_id:05d}.parquet"
            checkpoint_path = family_dir / f"part-{shard_id:05d}.checkpoint.json"
            checkpoint_identity = _checkpoint_identity(
                contract, inventory, table, shard_id, case_identity
            )
            reusable = _load_reusable_checkpoint(
                checkpoint_path, compact_path, checkpoint_identity
            )
            if reusable is not None:
                _emit(resource_hook, "checkpoint_reused", family=family, shard_id=shard_id)
                continue
            related = read_registered_table(
                input_path,
                table,
                case_ids=case_ids,
                resource_hook=resource_hook,
            )
            if table.depth == "0":
                joined, _ = join_depth_0(base_shard, related, table)
                compact = joined.select(
                    ["case_id", *(rule.output_prefix for rule in table.included_features)]
                )
            else:
                compact, _ = aggregate_depth_1(base_shard, related, table, contract)
            _publish_checkpointed_table(
                checkpoint_path, compact_path, compact, checkpoint_identity
            )
            _emit(
                resource_hook,
                "family_shard_completed",
                family=family,
                shard_id=shard_id,
                base_rows=base_shard.num_rows,
                related_rows=related.num_rows,
            )
            del related, compact

    matrix_dir = output_path / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_parts: list[Path] = []
    matrix_schema: pa.Schema | None = None
    for shard_id, (start, end) in enumerate(shard_bounds):
        base_shard = base.slice(start, end - start)
        case_ids = base_case_ids[start:end]
        case_identity = canonical_sha256(case_ids)
        matrix_path = matrix_dir / f"part-{shard_id:05d}.parquet"
        checkpoint_path = matrix_dir / f"part-{shard_id:05d}.checkpoint.json"
        checkpoint_identity = _checkpoint_identity(
            contract, inventory, None, shard_id, case_identity
        )
        matrix = _load_reusable_checkpoint(
            checkpoint_path, matrix_path, checkpoint_identity
        )
        if matrix is None:
            matrix = base_shard
            for family in (*contract.depth_0_order, *contract.depth_1_order):
                compact = pq.read_table(intermediate / family / f"part-{shard_id:05d}.parquet")
                matrix = join_compact(matrix, compact)
            assert_predictor_boundary(matrix, contract)
            _publish_checkpointed_table(
                checkpoint_path, matrix_path, matrix, checkpoint_identity
            )
        else:
            assert_predictor_boundary(matrix, contract)
        if matrix_schema is None:
            matrix_schema = matrix.schema
        elif matrix.schema != matrix_schema:
            raise DataValidationError("matrix shard schemas differ")
        matrix_parts.append(matrix_path)
        _emit(resource_hook, "matrix_shard_completed", shard_id=shard_id, rows=matrix.num_rows)

    predictors = predictor_columns(contract)
    lineage = expected_lineage(contract)
    assert matrix_schema is not None
    metadata_path = output_path / "metadata.json"
    lineage_path = output_path / "lineage.json"
    split_path = output_path / "split_membership.parquet"
    runtime_path = output_path / "runtime.json"
    status_path = output_path / "status.json"
    _write_json_atomic(
        metadata_path,
        {
            "schema_version": "homecredit_model_stability_2024_metadata_v1",
            "adapter_version": contract.adapter_version,
            "protocol_file_sha256": contract.lock_file_sha256,
            "mode": resolved_mode,
            "research_status": (
                "synthetic_fixture_not_research"
                if resolved_mode == "fixture"
                else "research_adapter_matrix_no_models_or_metrics"
            ),
            "row_count": base.num_rows,
            "matrix_part_count": len(matrix_parts),
            "columns": [
                {"name": field.name, "arrow_type": str(field.type)}
                for field in matrix_schema
            ],
            "predictor_columns": list(predictors),
            "non_predictor_columns": ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"],
            "split_and_fold_interface": {
                "membership": f"DEV when date_decision < {contract.split_boundary}; OOT otherwise",
                "implementation": "credit_risk_fs.models._cv_utils.GroupedTimeSeriesSplit",
                "n_splits": contract.fold_count,
                "gap_unique_time_groups": contract.fold_gap_unique_dates,
                "fit_scope": "fold training DEV only; full DEV required before OOT transform",
            },
            "fold_local_preprocessing_required": True,
            "global_preprocessing_fitted": False,
        },
        overwrite=False,
    )
    _write_json_atomic(
        lineage_path,
        {
            "schema_version": "homecredit_model_stability_2024_lineage_v1",
            "features": [asdict(item) for item in lineage],
        },
        overwrite=False,
    )
    memberships = [
        "OOT" if str(value) >= contract.split_boundary else "DEV"
        for value in base["date_decision"].to_pylist()
    ]
    split_table = pa.table(
        {
            "case_id": base["case_id"],
            "date_decision": base["date_decision"],
            "membership": pa.array(memberships, type=pa.string()),
        }
    )
    _write_parquet_atomic(split_path, split_table, overwrite=False)
    ended = datetime.now(timezone.utc)
    _write_json_atomic(
        runtime_path,
        {
            "schema_version": "homecredit_model_stability_2024_runtime_v1",
            "started_at_utc": started.isoformat(),
            "completed_at_utc": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(),
            "resource_budget": "not_frozen_in_prompt_15",
            "resource_hook_supported": True,
            "workers_started": 0,
            "selector_fits": 0,
            "model_fits": 0,
            "predictions": 0,
            "evaluations": 0,
        },
        overwrite=False,
    )
    # Status is finalized before the manifest so it is cryptographically bound.
    _write_json_atomic(
        status_path,
        {
            "schema_version": "homecredit_model_stability_2024_status_v1",
            "status": "complete",
            "mode": resolved_mode,
            "research_status": (
                "synthetic_fixture_not_research"
                if resolved_mode == "fixture"
                else "research_adapter_matrix_no_models_or_metrics"
            ),
            "completed_at_utc": ended.isoformat(),
        },
        overwrite=False,
    )

    final_artifacts = [
        *matrix_parts,
        metadata_path,
        lineage_path,
        split_path,
        runtime_path,
        status_path,
    ]
    manifest_path = output_path / "manifest.json"
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "identity": {
            "adapter_version": contract.adapter_version,
            "protocol_file_sha256": contract.lock_file_sha256,
            "protocol_internal_sha256": contract.lock_internal_sha256,
            "input_inventory_identity_sha256": inventory.identity_sha256,
            "mode": resolved_mode,
        },
        "input_inventory": inventory.to_dict(),
        "summary": {
            "row_count": base.num_rows,
            "predictor_count": len(predictors),
            "matrix_part_count": len(matrix_parts),
            "depth_2_files_opened": 0,
            "fits": 0,
            "evaluations": 0,
        },
        "artifacts": [_artifact_record(output_path, path) for path in final_artifacts],
        "completed_at_utc": ended.isoformat(),
    }
    _write_json_atomic(manifest_path, manifest, overwrite=False)
    # Validate every binding before writing the sole completion marker.
    for artifact in manifest["artifacts"]:
        path = output_path / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise ManifestAuthenticationError(f"artifact changed before sealing: {path}")
    manifest_digest = file_sha256(manifest_path)
    _write_text_atomic(
        output_path / "_SUCCESS",
        json.dumps(
            {
                "schema_version": "homecredit_model_stability_2024_completion_v1",
                "manifest_sha256": manifest_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        overwrite=False,
    )
    validate_output_manifest(output_path)
    return BuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        matrix_parts=tuple(matrix_parts),
        predictor_count=len(predictors),
        row_count=base.num_rows,
        reused_completed_build=False,
    )
