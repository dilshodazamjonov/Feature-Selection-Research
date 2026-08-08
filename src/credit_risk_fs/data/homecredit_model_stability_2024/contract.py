"""Authentication and executable contract recovery for the frozen adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ADAPTER_VERSION = "homecredit_model_stability_2024_adapter_v1"
LOCK_SCHEMA_VERSION = "third_dataset_protocol_lock_v1"
LOCK_PROTOCOL_ID = "homecredit_model_stability_2024_v1"
DATASET_ID = "homecredit_model_stability_2024"
EXPECTED_LOCK_FILE_SHA256 = (
    "e4b9f9f13286f15db0887c9dead09eb7e13f7912af786f2f2bc9c53d126b1860"
)
EXPECTED_INTERNAL_SHA256 = (
    "638e1fa2aa54bf98b771206b56ac13f6a6b77e2093deb291b794081d1a475df6"
)
EXPECTED_REVIEW_DIGEST = (
    "3f537d1b5e79faad3a2f047ec13dbe4b1797e11d4d64c4d92a06e09762a53f1e"
)
EXPECTED_INCLUDED_INPUT_DIGEST = (
    "8adb1db82c9dafb662657db08fd7d1dcf2eb4794d5ff7925e9ca4dd25f73fad2"
)
RAW_PREFIX = PurePosixPath("data/homecredit_model_stability_2024")


class ProtocolContractError(ValueError):
    """Raised before data access when the frozen contract is not exact."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    relative_path: str
    protocol_relative_path: str
    numeric_part: int | None
    expected_size_bytes: int
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class FeatureRule:
    family: str
    depth: str
    feature_name: str
    description: str
    observed_dtype: str
    logical_type: str
    intended_aggregation: str
    availability: str
    action: str
    reason: str
    output_prefix: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    family: str
    depth: str
    cardinality: str
    schema_sha256: str
    schema_fields: tuple[tuple[str, str], ...]
    partitions: tuple[PartitionSpec, ...]
    feature_rules: tuple[FeatureRule, ...]
    group_order_columns: tuple[str, ...]

    @property
    def included_features(self) -> tuple[FeatureRule, ...]:
        return tuple(rule for rule in self.feature_rules if rule.action == "include")


@dataclass(frozen=True, slots=True)
class AdapterContract:
    adapter_version: str
    lock_path: str
    lock_file_sha256: str
    lock_internal_sha256: str
    approved_review_digest: str
    protocol_id: str
    dataset_id: str
    included_raw_input_digest: str
    tables: tuple[TableSpec, ...]
    support_files: tuple[PartitionSpec, ...]
    excluded_depth_2_paths: tuple[str, ...]
    depth_0_order: tuple[str, ...]
    depth_1_order: tuple[str, ...]
    numeric_aggregations: tuple[str, ...]
    date_aggregations: tuple[str, ...]
    boolean_aggregations: tuple[str, ...]
    categorical_aggregations: tuple[str, ...]
    categorical_mode_tie_break: str
    empty_group_rule: str
    final_row_order: tuple[str, ...]
    split_boundary: str
    fold_count: int
    fold_gap_unique_dates: int
    later_stage_accounting: Mapping[str, int]
    protected_fingerprints: Mapping[str, str]

    @property
    def base_table(self) -> TableSpec:
        return self.table("base")

    def table(self, family: str) -> TableSpec:
        matches = [table for table in self.tables if table.family == family]
        if len(matches) != 1:
            raise ProtocolContractError(
                f"expected one registered table family {family!r}, found {len(matches)}"
            )
        return matches[0]

    @property
    def included_partition_paths(self) -> tuple[str, ...]:
        return tuple(
            partition.relative_path
            for table in self.tables
            for partition in table.partitions
        )

    @property
    def included_input_paths(self) -> tuple[str, ...]:
        return tuple(item.relative_path for item in self.support_files) + self.included_partition_paths

    @property
    def predictor_rules(self) -> tuple[FeatureRule, ...]:
        return tuple(
            rule
            for table in self.tables
            for rule in table.feature_rules
            if rule.action == "include"
        )

    def validate_requested_tables(self, families: Iterable[str]) -> tuple[TableSpec, ...]:
        requested = tuple(str(value) for value in families)
        registered = {table.family: table for table in self.tables}
        bad = [value for value in requested if value not in registered]
        if any("depth2" in value.lower() or value.endswith(":2") for value in requested):
            raise ProtocolContractError("depth-2 execution is forbidden by the frozen protocol")
        if bad:
            raise ProtocolContractError(f"unregistered table families requested: {sorted(bad)}")
        return tuple(registered[value] for value in requested)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prompt_15_adapter_contract_v1",
            "status": "recovered_from_authenticated_canonical_lock",
            "adapter_version": self.adapter_version,
            "protocol": {
                "path": self.lock_path,
                "file_sha256": self.lock_file_sha256,
                "internal_authentication_sha256": self.lock_internal_sha256,
                "approved_review_digest": self.approved_review_digest,
                "protocol_id": self.protocol_id,
                "dataset_id": self.dataset_id,
                "included_raw_input_digest": self.included_raw_input_digest,
                "protected_fingerprints": dict(self.protected_fingerprints),
            },
            "tables": [
                {
                    "family": table.family,
                    "depth": table.depth,
                    "cardinality": table.cardinality,
                    "schema_sha256": table.schema_sha256,
                    "schema_fields": [
                        {"name": name, "dtype": dtype}
                        for name, dtype in table.schema_fields
                    ],
                    "group_order_columns": list(table.group_order_columns),
                    "partitions": [asdict(item) for item in table.partitions],
                    "feature_rules": [asdict(rule) for rule in table.feature_rules],
                }
                for table in self.tables
            ],
            "support_files": [asdict(item) for item in self.support_files],
            "excluded_depth_2_paths": list(self.excluded_depth_2_paths),
            "ordering": {
                "depth_0_family_order": list(self.depth_0_order),
                "depth_1_family_order": list(self.depth_1_order),
                "final_row_order": list(self.final_row_order),
                "depth_1_row_order": [
                    "case_id",
                    "num_group1",
                    "numeric_source_part",
                    "physical_row_offset_within_part",
                ],
            },
            "aggregation": {
                "numeric": list(self.numeric_aggregations),
                "date": list(self.date_aggregations),
                "boolean": list(self.boolean_aggregations),
                "categorical": list(self.categorical_aggregations),
                "categorical_mode_tie_break": self.categorical_mode_tie_break,
                "empty_group_rule": self.empty_group_rule,
                "first_last_null_semantics": (
                    "literal ordered row value; an observed null at the first/last row "
                    "remains null and is distinguished by missing_count from no related row"
                ),
            },
            "roles": {
                "target": ["target"],
                "identifier": ["case_id", "num_group*"],
                "split_control": ["date_decision", "MONTH", "WEEK_NUM"],
                "predictor_rule_count": len(self.predictor_rules),
                "excluded_rule_count": sum(
                    rule.action == "exclude"
                    for table in self.tables
                    for rule in table.feature_rules
                ),
                "unresolved_rule_count": sum(
                    rule.action == "unresolved"
                    for table in self.tables
                    for rule in table.feature_rules
                ),
            },
            "split_and_fold_interface": {
                "oot_start_inclusive": self.split_boundary,
                "fold_count": self.fold_count,
                "gap_unique_dates": self.fold_gap_unique_dates,
                "fit_scope": "DEV fold training only; full DEV only for later OOT transform",
                "oot_fitting": "forbidden",
            },
            "output_invariants": {
                "rows": "exactly one per base case_id",
                "joins": "base-left only after one-row-per-case reduction",
                "predictor_order": "depth-0 family/rule order then depth-1 family/rule/aggregation order",
                "completion": "status, manifest, validation, then completion marker last",
                "checkpoint_identity": [
                    "adapter_version",
                    "protocol_digest",
                    "input_inventory_identity",
                    "logical_table",
                    "schema",
                    "aggregation_rules",
                    "output_digest",
                ],
            },
            "later_stage_accounting": dict(self.later_stage_accounting),
            "execution_boundary": {
                "prompt_15_fits": 0,
                "prompt_15_evaluations": 0,
                "depth_2_execution": "impossible",
                "real_data_access": "not authorized in Prompt 15",
            },
        }


def _strip_raw_prefix(value: str) -> str:
    path = PurePosixPath(value)
    try:
        return path.relative_to(RAW_PREFIX).as_posix()
    except ValueError as exc:
        raise ProtocolContractError(f"raw inventory path escapes dataset root: {value}") from exc


def _authenticate_lock(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    file_digest = hashlib.sha256(raw).hexdigest()
    if file_digest != EXPECTED_LOCK_FILE_SHA256:
        raise ProtocolContractError(
            f"protocol lock file digest mismatch: {file_digest}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolContractError("protocol lock is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolContractError("protocol lock must be a JSON object")
    claimed = payload.get("artifact_authentication_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_authentication_sha256", None)
    observed_internal = canonical_sha256(unsigned)
    if claimed != EXPECTED_INTERNAL_SHA256 or observed_internal != EXPECTED_INTERNAL_SHA256:
        raise ProtocolContractError("protocol internal authentication mismatch")
    return payload, file_digest, observed_internal


def load_adapter_contract(protocol_lock: str | Path) -> AdapterContract:
    """Authenticate the one allowed lock and recover its executable adapter contract.

    This function reads only the explicitly supplied protocol file. It never
    resolves or inspects a dataset path.
    """

    path = Path(protocol_lock)
    lock, file_digest, internal_digest = _authenticate_lock(path)
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ProtocolContractError("protocol schema version mismatch")
    if lock.get("protocol_id") != LOCK_PROTOCOL_ID:
        raise ProtocolContractError("protocol identity mismatch")
    if lock.get("status") != "canonical_approved_locked_no_execution":
        raise ProtocolContractError("protocol is not the approved canonical lock")
    stage_1 = lock.get("stage_1_authentication", {})
    approval = lock.get("user_approval_record", {})
    if (
        stage_1.get("approved_review_digest_sha256") != EXPECTED_REVIEW_DIGEST
        or stage_1.get("observed_review_digest_sha256") != EXPECTED_REVIEW_DIGEST
        or approval.get("approved_review_digest_sha256") != EXPECTED_REVIEW_DIGEST
    ):
        raise ProtocolContractError("approved Stage-1 review digest mismatch")

    approved = lock.get("approved_protocol", {})
    identity = approved.get("dataset_identity", {})
    if identity.get("dataset_id") != DATASET_ID:
        raise ProtocolContractError("dataset identity mismatch")
    if identity.get("included_raw_input_digest") != EXPECTED_INCLUDED_INPUT_DIGEST:
        raise ProtocolContractError("included input identity mismatch")
    if identity.get("included_parquet_file_count") != 18:
        raise ProtocolContractError("included parquet count mismatch")
    if identity.get("excluded_parquet_file_count") != 14:
        raise ProtocolContractError("excluded depth-2 count mismatch")

    raw_scope = approved.get("raw_file_scope", {})
    records = raw_scope.get("records", [])
    if not isinstance(records, list) or len(records) != 33:
        raise ProtocolContractError("raw inventory record count mismatch")
    if (
        raw_scope.get("included_count") != 19
        or raw_scope.get("included_parquet_count") != 18
        or raw_scope.get("excluded_depth_2_parquet_count") != 14
    ):
        raise ProtocolContractError("raw depth-scope accounting mismatch")
    if any(
        row.get("relational_depth") == "2"
        and row.get("inclusion_status") != "excluded"
        for row in records
    ):
        raise ProtocolContractError("depth-2 input is not fully excluded")

    leakage = approved.get("leakage_and_availability_scope", {})
    feature_records = leakage.get("records", [])
    if (
        leakage.get("candidate_rows") != 461
        or leakage.get("included") != 434
        or leakage.get("excluded") != 27
        or leakage.get("unresolved") != 0
        or len(feature_records) != 461
    ):
        raise ProtocolContractError("feature-role accounting mismatch")
    if any(row.get("action") == "unresolved" for row in feature_records):
        raise ProtocolContractError("unresolved feature rule in canonical lock")

    adapter = approved.get("adapter_protocol", {})
    if adapter.get("included_raw_input_digest") != EXPECTED_INCLUDED_INPUT_DIGEST:
        raise ProtocolContractError("adapter input identity mismatch")
    depth_0_order = tuple(adapter.get("depth_0", {}).get("merge_order", ()))
    depth_1_order = tuple(adapter.get("depth_1", {}).get("family_order", ()))
    if depth_0_order != ("static", "static_cb") or len(depth_1_order) != 10:
        raise ProtocolContractError("table family order mismatch")

    profiles = approved.get("schema_and_cardinality_profile", [])
    profile_by_family = {str(row["table_family"]): row for row in profiles}
    if len(profile_by_family) != 13:
        raise ProtocolContractError("logical-table profile count mismatch")
    features_by_family: dict[str, list[FeatureRule]] = {}
    for row in feature_records:
        family = str(row["source_table_family"])
        features_by_family.setdefault(family, []).append(
            FeatureRule(
                family=family,
                depth=str(row["relational_depth"]),
                feature_name=str(row["feature_name"]),
                description=str(row["description"]),
                observed_dtype=str(row["observed_dtype"]),
                logical_type=str(row["logical_type"]),
                intended_aggregation=str(row["intended_aggregation"]),
                availability=str(row["decision_time_availability"]),
                action=str(row["action"]),
                reason=str(row["reason"]),
                output_prefix=str(row["deterministic_output_prefix"]),
            )
        )

    table_order = ("base", *depth_0_order, *depth_1_order)
    tables: list[TableSpec] = []
    for family in table_order:
        profile = profile_by_family.get(family)
        if profile is None:
            raise ProtocolContractError(f"missing schema profile for {family}")
        depth = str(profile["relational_depth"])
        family_inventory = [
            row
            for row in records
            if row.get("table_family") == family
            and row.get("inclusion_status") == "included"
            and str(row.get("file_type")) == "parquet"
        ]
        family_inventory.sort(
            key=lambda row: (
                -1
                if str(row.get("file_part_number", "")) == ""
                else int(row["file_part_number"])
            )
        )
        partitions = tuple(
            PartitionSpec(
                relative_path=_strip_raw_prefix(str(row["relative_path"])),
                protocol_relative_path=str(row["relative_path"]),
                numeric_part=(
                    None
                    if str(row.get("file_part_number", "")) == ""
                    else int(row["file_part_number"])
                ),
                expected_size_bytes=int(row["byte_size"]),
                expected_sha256=str(row["sha256"]),
            )
            for row in family_inventory
        )
        expected_parts = int(profile["part_count"])
        if len(partitions) != expected_parts:
            raise ProtocolContractError(f"partition count mismatch for {family}")
        rules = tuple(features_by_family.get(family, ()))
        fields = tuple((rule.feature_name, rule.observed_dtype) for rule in rules)
        if canonical_sha256(
            [{"name": name, "dtype": dtype} for name, dtype in fields]
        ) != str(profile["schema_sha256"]):
            raise ProtocolContractError(f"schema registry mismatch for {family}")
        group_columns = tuple(
            rule.feature_name
            for rule in rules
            if rule.logical_type == "order_identifier"
        )
        if depth == "1" and group_columns != ("num_group1",):
            raise ProtocolContractError(f"depth-1 order contract mismatch for {family}")
        tables.append(
            TableSpec(
                family=family,
                depth=depth,
                cardinality=str(profile["cardinality"]),
                schema_sha256=str(profile["schema_sha256"]),
                schema_fields=fields,
                partitions=partitions,
                feature_rules=rules,
                group_order_columns=group_columns,
            )
        )

    matrix = approved.get("method_and_evaluation_matrix", {})
    phase = matrix.get("phase_design", {})
    accounting = {
        "resource_pilot_selector_fits": int(phase.get("pilot", {}).get("selector_fit_calls", -1)),
        "resource_pilot_evaluations": int(
            phase.get("pilot", {}).get("configuration_evaluation_cells", -1)
        ),
        "dev_selector_fits": int(phase.get("dev", {}).get("selector_fit_calls", -1)),
        "dev_fold_evaluations": int(
            phase.get("dev", {}).get("fold_evaluation_cells", -1)
        ),
        "oot_full_dev_selector_refits": int(
            phase.get("oot", {}).get("full_dev_selector_refit_calls", -1)
        ),
        "oot_evaluations": int(phase.get("oot", {}).get("evaluation_cells", -1)),
    }
    expected_accounting = {
        "resource_pilot_selector_fits": 27,
        "resource_pilot_evaluations": 30,
        "dev_selector_fits": 135,
        "dev_fold_evaluations": 150,
        "oot_full_dev_selector_refits": 27,
        "oot_evaluations": 30,
    }
    if accounting != expected_accounting:
        raise ProtocolContractError(
            f"later-stage accounting mismatch: {accounting}"
        )

    split = approved.get("split_and_fold_boundaries", {})
    fold = split.get("fold_protocol", {})
    excluded_depth_2 = tuple(
        _strip_raw_prefix(str(row["relative_path"]))
        for row in records
        if row.get("relational_depth") == "2"
    )
    support_files = tuple(
        PartitionSpec(
            relative_path=_strip_raw_prefix(str(row["relative_path"])),
            protocol_relative_path=str(row["relative_path"]),
            numeric_part=None,
            expected_size_bytes=int(row["byte_size"]),
            expected_sha256=str(row["sha256"]),
        )
        for row in records
        if row.get("inclusion_status") == "included"
        and str(row.get("file_type")) != "parquet"
    )
    if len(support_files) != 1 or support_files[0].relative_path != "feature_definitions.csv":
        raise ProtocolContractError("included support-file scope mismatch")
    depth_1 = adapter["depth_1"]
    return AdapterContract(
        adapter_version=ADAPTER_VERSION,
        lock_path=path.as_posix(),
        lock_file_sha256=file_digest,
        lock_internal_sha256=internal_digest,
        approved_review_digest=EXPECTED_REVIEW_DIGEST,
        protocol_id=LOCK_PROTOCOL_ID,
        dataset_id=DATASET_ID,
        included_raw_input_digest=EXPECTED_INCLUDED_INPUT_DIGEST,
        tables=tuple(tables),
        support_files=support_files,
        excluded_depth_2_paths=excluded_depth_2,
        depth_0_order=depth_0_order,
        depth_1_order=depth_1_order,
        numeric_aggregations=tuple(depth_1["numeric_aggregations"]),
        date_aggregations=tuple(depth_1["date_aggregations"]["statistics"]),
        boolean_aggregations=tuple(depth_1["boolean_aggregations"]),
        categorical_aggregations=tuple(depth_1["categorical_aggregations"]),
        categorical_mode_tie_break=str(depth_1["categorical_mode_tie_break"]),
        empty_group_rule=str(depth_1["empty_group_rule"]),
        final_row_order=("date_decision", "case_id"),
        split_boundary=str(split["oot_start_inclusive"]),
        fold_count=int(fold["n_splits"]),
        fold_gap_unique_dates=int(fold["gap_unique_time_groups"]),
        later_stage_accounting=accounting,
        protected_fingerprints=dict(
            lock.get("protected_contract", {}).get("fingerprints_sha256", {})
        ),
    )
