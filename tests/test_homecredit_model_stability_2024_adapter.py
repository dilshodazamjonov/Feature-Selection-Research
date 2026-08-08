from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_fs.data.homecredit_model_stability_2024.adapter import (
    AdapterError,
    CheckpointMismatchError,
    DataValidationError,
    FeatureLineage,
    InventoryError,
    LeakageBoundaryError,
    ManifestAuthenticationError,
    SchemaMismatchError,
    _checkpoint_identity,
    _load_reusable_checkpoint,
    _publish_checkpointed_table,
    _write_parquet_atomic,
    aggregate_depth_1,
    assert_predictor_boundary,
    assert_transform_scope,
    build_modeling_matrix,
    expected_lineage,
    fit_scope_token,
    inspect_input_inventory,
    join_depth_0,
    predictor_columns,
    read_registered_table,
    validate_base,
    validate_output_manifest,
    validate_partition_schema,
    validate_requested_partition,
    INTERNAL_ROW_OFFSET,
    INTERNAL_SOURCE_PART,
)
from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    FeatureRule,
    ProtocolContractError,
    TableSpec,
    canonical_sha256,
    load_adapter_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPOSITORY_ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"


@pytest.fixture(scope="module")
def contract():
    return load_adapter_contract(LOCK_PATH)


def _array_for_rule(rule: FeatureRule, values: list[object]) -> pa.Array:
    arrow_type = {
        "int64": pa.int64(),
        "double": pa.float64(),
        "string": pa.string(),
        "bool": pa.bool_(),
    }[rule.observed_dtype]
    return pa.array(values, type=arrow_type)


def _base_values(name: str) -> list[object]:
    return {
        "case_id": [3, 1, 2],
        "date_decision": ["2020-02-26", "2020-01-10", "2020-01-11"],
        "MONTH": [202002, 202001, 202001],
        "WEEK_NUM": [61, 54, 54],
        "target": [0, 1, 0],
    }[name]


def _default_values(rule: FeatureRule, rows: int, *, case_ids: list[int], groups: list[int] | None) -> list[object]:
    if rule.feature_name == "case_id":
        return case_ids
    if rule.feature_name == "num_group1":
        assert groups is not None
        return groups
    if rule.logical_type == "numeric":
        return [float(index + 1) for index in range(rows)]
    if rule.logical_type == "categorical":
        return ["b" if index % 2 == 0 else "a" for index in range(rows)]
    if rule.logical_type == "date":
        return ["2020-01-01" if index % 2 == 0 else "2020-01-03" for index in range(rows)]
    if rule.logical_type == "boolean":
        return [bool(index % 2) for index in range(rows)]
    if rule.observed_dtype == "double":
        return [0.0] * rows
    if rule.observed_dtype == "string":
        return ["excluded"] * rows
    if rule.observed_dtype == "bool":
        return [False] * rows
    return [0] * rows


def _write_full_fixture(root: Path, contract) -> None:
    (root / "feature_definitions.csv").parent.mkdir(parents=True, exist_ok=True)
    (root / "feature_definitions.csv").write_text("Variable,Description\nfixture,synthetic only\n", encoding="utf-8")
    for table in contract.tables:
        part_rows: list[list[int]] = [[] for _ in table.partitions]
        if table.family == "base":
            part_rows[0] = [0, 1, 2]
        elif table.depth == "0":
            # Partition placement is deliberately not case-id order.
            part_rows[0] = [1, 3] if len(table.partitions) > 1 else [1, 2, 3]
            if len(table.partitions) > 1:
                part_rows[1] = [2]
        else:
            # one-to-many fixtures include a duplicate order value for the
            # deterministic physical-offset tie breaker; locked one-to-one does not.
            part_rows[0] = [1] if table.cardinality == "one_to_one" else [1, 1, 2]
        for part_index, partition in enumerate(table.partitions):
            path = root / partition.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            markers = part_rows[part_index]
            if table.family == "base":
                arrays = {
                    rule.feature_name: _array_for_rule(
                        rule, [_base_values(rule.feature_name)[index] for index in markers]
                    )
                    for rule in table.feature_rules
                }
            else:
                case_ids = markers
                groups = [0, 0, 0][: len(markers)] if table.depth == "1" else None
                arrays = {
                    rule.feature_name: _array_for_rule(
                        rule,
                        _default_values(
                            rule, len(markers), case_ids=case_ids, groups=groups
                        ),
                    )
                    for rule in table.feature_rules
                }
                if table.depth == "1" and 2 in case_ids:
                    position = case_ids.index(2)
                    for rule in table.included_features:
                        values = arrays[rule.feature_name].to_pylist()
                        values[position] = None
                        arrays[rule.feature_name] = _array_for_rule(rule, values)
            pq.write_table(pa.table(arrays), path, compression=None)


@pytest.fixture()
def full_fixture(tmp_path: Path, contract) -> Path:
    root = tmp_path / "synthetic_fixture"
    _write_full_fixture(root, contract)
    return root


def _mini_rule(name: str, logical_type: str, dtype: str, prefix: str) -> FeatureRule:
    return FeatureRule(
        family="mini",
        depth="1",
        feature_name=name,
        description="synthetic",
        observed_dtype=dtype,
        logical_type=logical_type,
        intended_aggregation="locked",
        availability="available",
        action="include",
        reason="fixture",
        output_prefix=prefix,
    )


def _mini_depth1() -> TableSpec:
    rules = (
        _mini_rule("amount", "numeric", "double", "d1__mini__amount"),
        _mini_rule("category", "categorical", "string", "d1__mini__category"),
        _mini_rule("flag", "boolean", "bool", "d1__mini__flag"),
        _mini_rule("event_date", "date", "string", "d1__mini__event_date"),
    )
    return TableSpec(
        family="mini",
        depth="1",
        cardinality="one_to_many",
        schema_sha256="fixture",
        schema_fields=(),
        partitions=(),
        feature_rules=rules,
        group_order_columns=("num_group1",),
    )


def _base_table() -> pa.Table:
    return validate_base(
        pa.table(
            {
                "case_id": pa.array([2, 1, 3], type=pa.int64()),
                "date_decision": ["2020-01-10", "2020-01-10", "2020-01-10"],
                "MONTH": pa.array([1, 1, 1], type=pa.int64()),
                "WEEK_NUM": pa.array([1, 1, 1], type=pa.int64()),
                "target": pa.array([0, 1, 0], type=pa.int64()),
            }
        )
    )


def _related_table() -> pa.Table:
    return pa.table(
        {
            "case_id": pa.array([1, 1, 2], type=pa.int64()),
            "num_group1": pa.array([1, 1, 0], type=pa.int64()),
            "amount": pa.array([3.0, 1.0, None], type=pa.float64()),
            "category": pa.array(["B", "a", None], type=pa.string()),
            "flag": pa.array([True, False, None], type=pa.bool_()),
            "event_date": pa.array(["2020-01-12", "2020-01-08", None], type=pa.string()),
            INTERNAL_SOURCE_PART: pa.array([1, 0, 0], type=pa.int32()),
            INTERNAL_ROW_OFFSET: pa.array([0, 5, 0], type=pa.int64()),
        }
    )


def test_contract_authentication_and_accounting(contract):
    assert contract.lock_file_sha256 == "e4b9f9f13286f15db0887c9dead09eb7e13f7912af786f2f2bc9c53d126b1860"
    assert contract.lock_internal_sha256 == "638e1fa2aa54bf98b771206b56ac13f6a6b77e2093deb291b794081d1a475df6"
    assert contract.approved_review_digest == "3f537d1b5e79faad3a2f047ec13dbe4b1797e11d4d64c4d92a06e09762a53f1e"
    assert dict(contract.later_stage_accounting) == {
        "resource_pilot_selector_fits": 27,
        "resource_pilot_evaluations": 30,
        "dev_selector_fits": 135,
        "dev_fold_evaluations": 150,
        "oot_full_dev_selector_refits": 27,
        "oot_evaluations": 30,
    }


def test_changed_protocol_digest_is_rejected(tmp_path: Path):
    changed = tmp_path / "changed.json"
    changed.write_bytes(LOCK_PATH.read_bytes() + b" ")
    with pytest.raises(ProtocolContractError, match="digest mismatch"):
        load_adapter_contract(changed)


def test_changed_contract_instance_is_rejected_before_inventory(full_fixture: Path, contract):
    changed = replace(contract, split_boundary="1999-01-01")
    with pytest.raises(ProtocolContractError, match="differs"):
        inspect_input_inventory(full_fixture, changed, mode="fixture")


def test_contract_scope_counts_and_no_unresolved(contract):
    rules = [rule for table in contract.tables for rule in table.feature_rules]
    assert len(contract.included_input_paths) == 19
    assert len(contract.excluded_depth_2_paths) == 14
    assert sum(rule.action == "include" for rule in rules) == 434
    assert sum(rule.action == "exclude" for rule in rules) == 27
    assert sum(rule.action == "unresolved" for rule in rules) == 0


def test_base_validation_order_uniqueness_missing_and_target():
    base = _base_table()
    assert base["case_id"].to_pylist() == [1, 2, 3]
    duplicate = base.set_column(0, "case_id", pa.array([1, 1, 3], type=pa.int64()))
    with pytest.raises(DataValidationError, match="unique"):
        validate_base(duplicate)
    missing = base.set_column(0, "case_id", pa.array([1, None, 3], type=pa.int64()))
    with pytest.raises(DataValidationError, match="non-null"):
        validate_base(missing)
    bad_target = base.set_column(4, "target", pa.array([0, 2, 1], type=pa.int64()))
    with pytest.raises(DataValidationError, match="binary"):
        validate_base(bad_target)


def test_strict_date_validation():
    base = _base_table().set_column(1, "date_decision", pa.array(["2020-1-1"] * 3))
    with pytest.raises(DataValidationError, match="strict ISO|canonical"):
        validate_base(base)


def test_fixture_inventory_and_schema(full_fixture: Path, contract):
    inventory = inspect_input_inventory(full_fixture, contract, mode="fixture")
    assert len(inventory.artifacts) == 19
    assert inventory.mode == "fixture"
    for table in contract.tables:
        for partition in table.partitions:
            validate_partition_schema(full_fixture / partition.relative_path, table)


def test_unknown_and_depth2_inputs_are_rejected(full_fixture: Path, contract):
    unknown = full_fixture / "parquet_files/train/unknown.parquet"
    pq.write_table(pa.table({"x": [1]}), unknown)
    with pytest.raises(InventoryError, match="unregistered"):
        inspect_input_inventory(full_fixture, contract, mode="fixture")
    unknown.unlink()
    forbidden = contract.excluded_depth_2_paths[0]
    with pytest.raises(InventoryError, match="depth-2"):
        validate_requested_partition(contract, "applprev", forbidden)
    with pytest.raises(ProtocolContractError, match="unregistered"):
        contract.validate_requested_tables(["not_a_table"])


def test_changed_schema_is_rejected(full_fixture: Path, contract):
    table = contract.table("debitcard")
    path = full_fixture / table.partitions[0].relative_path
    changed = pq.read_table(path).append_column("extra", pa.array([1] * pq.read_table(path).num_rows))
    pq.write_table(changed, path)
    with pytest.raises(SchemaMismatchError):
        validate_partition_schema(path, table)


def test_locked_partition_order_is_independent_of_directory_enumeration(full_fixture: Path, contract):
    table = contract.table("static")
    observed = read_registered_table(full_fixture, table)
    assert observed["case_id"].to_pylist() == [1, 3, 2]
    assert [item.numeric_part for item in table.partitions] == [0, 1]


def test_depth0_join_preserves_population_and_prefixes(contract):
    base = _base_table()
    source = contract.table("static")
    rule = source.included_features[0]
    mini = replace(source, feature_rules=(source.feature_rules[0], rule))
    related = pa.table(
        {
            "case_id": pa.array([3, 1], type=pa.int64()),
            rule.feature_name: pa.array([30.0, 10.0], type=pa.float64()),
        }
    )
    joined, lineage = join_depth_0(base, related, mini)
    assert joined["case_id"].to_pylist() == [1, 2, 3]
    assert joined[rule.output_prefix].to_pylist() == [10.0, None, 30.0]
    assert lineage[0].source_feature == rule.feature_name


def test_depth0_duplicate_or_orphan_rejected(contract):
    base = _base_table()
    source = contract.table("static")
    rule = source.included_features[0]
    mini = replace(source, feature_rules=(source.feature_rules[0], rule))
    duplicate = pa.table({"case_id": pa.array([1, 1]), rule.feature_name: pa.array([1.0, 2.0])})
    with pytest.raises(DataValidationError, match="unique"):
        join_depth_0(base, duplicate, mini)
    orphan = pa.table({"case_id": pa.array([99]), rule.feature_name: pa.array([1.0])})
    with pytest.raises(DataValidationError, match="orphan"):
        join_depth_0(base, orphan, mini)


def test_depth1_typed_aggregations_ties_and_dates(contract):
    compact, lineage = aggregate_depth_1(_base_table(), _related_table(), _mini_depth1(), contract)
    # Physical source part breaks duplicate num_group1 ties: amount order is 1, 3.
    assert compact["d1__mini__amount__first_by_num_group1"].to_pylist() == [1.0, None, None]
    assert compact["d1__mini__amount__last_by_num_group1"].to_pylist() == [3.0, None, None]
    assert compact["d1__mini__amount__mean"].to_pylist() == [2.0, None, None]
    assert compact["d1__mini__amount__sample_variance_ddof_1"].to_pylist() == [2.0, None, None]
    assert compact["d1__mini__category__lexical_mode"].to_pylist() == ["a", None, None]
    assert compact["d1__mini__flag__true_count"].to_pylist() == [1, 0, None]
    # Positive scheduled dates remain signed snapshot values, never refreshed.
    assert compact["d1__mini__event_date__first_by_num_group1"].to_pylist() == [-2, None, None]
    assert compact["d1__mini__event_date__last_by_num_group1"].to_pylist() == [2, None, None]
    assert len(lineage) == compact.num_columns - 1


def test_no_row_is_distinct_from_observed_null(contract):
    compact, _ = aggregate_depth_1(_base_table(), _related_table(), _mini_depth1(), contract)
    # case 2 has a related all-null row; case 3 has no related row.
    assert compact["d1__mini__row_count"].to_pylist() == [2, 1, 0]
    assert compact["d1__mini__amount__count_non_missing"].to_pylist() == [2, 0, None]
    assert compact["d1__mini__amount__missing_count"].to_pylist() == [0, 1, None]


def test_literal_null_first_last_and_unicode_mode_tie_break(contract):
    related = _related_table()
    related = related.set_column(2, "amount", pa.array([3.0, None, None], type=pa.float64()))
    related = related.set_column(
        3,
        "category",
        pa.array(["é", "e\u0301", None], type=pa.string()),
    )
    compact, _ = aggregate_depth_1(_base_table(), related, _mini_depth1(), contract)
    assert compact["d1__mini__amount__first_by_num_group1"].to_pylist()[0] is None
    assert compact["d1__mini__amount__last_by_num_group1"].to_pylist()[0] == 3.0
    assert compact["d1__mini__amount__missing_count"].to_pylist()[0] == 1
    # NFC+casefold tie is resolved by the original UTF-8 bytes.
    assert compact["d1__mini__category__lexical_mode"].to_pylist()[0] == "e\u0301"
    assert compact["d1__mini__flag__any"].to_pylist()[0] is True
    assert compact["d1__mini__flag__all"].to_pylist()[0] is False


def test_missing_order_and_orphan_related_rejected(contract):
    related = _related_table().set_column(1, "num_group1", pa.array([1, None, 0], type=pa.int64()))
    with pytest.raises(DataValidationError, match="num_group1"):
        aggregate_depth_1(_base_table(), related, _mini_depth1(), contract)
    orphan = _related_table().set_column(0, "case_id", pa.array([99, 1, 2], type=pa.int64()))
    with pytest.raises(DataValidationError, match="orphan"):
        aggregate_depth_1(_base_table(), orphan, _mini_depth1(), contract)


def test_depth1_one_to_one_cardinality_enforced(contract):
    table = replace(_mini_depth1(), cardinality="one_to_one")
    with pytest.raises(DataValidationError, match="one-to-one"):
        aggregate_depth_1(_base_table(), _related_table(), table, contract)


def test_constants_all_missing_are_retained_and_exclusions_absent(contract):
    compact, _ = aggregate_depth_1(_base_table(), _related_table(), _mini_depth1(), contract)
    assert "d1__mini__amount__count_non_missing" in compact.column_names
    assert "num_group1" not in compact.column_names
    assert "target" not in compact.column_names
    expected = predictor_columns(contract)
    assert not {"target", "case_id", "date_decision", "MONTH", "WEEK_NUM", "num_group1"} & set(expected)


def test_predictor_and_lineage_order_are_deterministic(contract):
    first = predictor_columns(contract)
    second = predictor_columns(contract)
    lineage = expected_lineage(contract)
    assert first == second == tuple(item.output_feature for item in lineage)
    assert len(first) == len(set(first))
    assert len(first) == 1959


def test_fold_local_fit_transform_guards():
    token = fit_scope_token(scope="dev_fold", case_ids=[1, 2], memberships=["DEV", "DEV"])
    assert_transform_scope(token, transform_membership="DEV")
    with pytest.raises(LeakageBoundaryError, match="full-DEV"):
        assert_transform_scope(token, transform_membership="OOT")
    full = fit_scope_token(scope="full_dev", case_ids=[1, 2], memberships=["DEV", "DEV"])
    assert_transform_scope(full, transform_membership="OOT")
    with pytest.raises(LeakageBoundaryError, match="OOT"):
        fit_scope_token(scope="full_dev", case_ids=[1, 3], memberships=["DEV", "OOT"])


def test_atomic_failure_leaves_no_final_or_success(tmp_path: Path, monkeypatch):
    target = tmp_path / "matrix.parquet"
    original = pq.write_table

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        _write_parquet_atomic(target, pa.table({"case_id": [1]}), overwrite=False)
    monkeypatch.setattr(pq, "write_table", original)
    assert not target.exists()
    assert not (tmp_path / "_SUCCESS").exists()


def test_checkpoint_reuse_and_stale_rejection(tmp_path: Path, contract):
    output = tmp_path / "compact.parquet"
    checkpoint = tmp_path / "compact.checkpoint.json"
    table = pa.table({"case_id": pa.array([1], type=pa.int64()), "value": [2.0]})
    inventory = type("Inventory", (), {"identity_sha256": "fixture"})()
    identity = _checkpoint_identity(contract, inventory, None, 0, canonical_sha256([1]))
    _publish_checkpointed_table(checkpoint, output, table, identity)
    reused = _load_reusable_checkpoint(checkpoint, output, identity)
    assert reused is not None and reused.equals(table)
    changed = dict(identity)
    changed["adapter_version"] = "stale"
    with pytest.raises(CheckpointMismatchError, match="stale"):
        _load_reusable_checkpoint(checkpoint, output, changed)


def test_end_to_end_fixture_build_and_completed_reuse(full_fixture: Path, tmp_path: Path, contract):
    output = tmp_path / "fixture_output"
    events: list[dict[str, object]] = []
    result = build_modeling_matrix(
        input_root=full_fixture,
        output_root=output,
        contract=contract,
        mode="fixture",
        shard_rows=2,
        resource_hook=events.append,
    )
    assert result.row_count == 3
    assert result.predictor_count == 1959
    assert len(result.matrix_parts) == 2
    combined = pa.concat_tables([pq.read_table(path) for path in result.matrix_parts])
    assert combined["case_id"].to_pylist() == [1, 2, 3]
    assert combined.num_rows == 3
    assert_predictor_boundary(combined, contract)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["research_status"] == "synthetic_fixture_not_research"
    assert metadata["global_preprocessing_fitted"] is False
    manifest = validate_output_manifest(output)
    assert manifest["summary"]["fits"] == manifest["summary"]["evaluations"] == 0
    assert (output / "_SUCCESS").is_file()
    assert any(event["event"] == "input_batch" for event in events)
    reused = build_modeling_matrix(
        input_root=full_fixture,
        output_root=output,
        contract=contract,
        mode="fixture",
        shard_rows=1,
    )
    assert reused.reused_completed_build is True


def test_manifest_binds_final_status_and_detects_status_after_manifest_bug(full_fixture: Path, tmp_path: Path, contract):
    output = tmp_path / "fixture_output"
    build_modeling_matrix(
        input_root=full_fixture,
        output_root=output,
        contract=contract,
        mode="fixture",
    )
    status = output / "status.json"
    status.write_bytes(status.read_bytes() + b" ")
    with pytest.raises(ManifestAuthenticationError, match="size mismatch|digest mismatch"):
        validate_output_manifest(output)


def test_fixture_mode_cannot_be_confused_with_research(full_fixture: Path, contract):
    fixture = inspect_input_inventory(full_fixture, contract, mode="fixture")
    assert fixture.mode == "fixture"
    with pytest.raises(InventoryError, match="research input identity"):
        inspect_input_inventory(full_fixture, contract, mode="research")


def test_adapter_and_cli_imports_are_inert(tmp_path: Path):
    before = list(tmp_path.iterdir())
    cli = REPOSITORY_ROOT / "scripts/build_homecredit_model_stability_2024.py"
    spec = importlib.util.spec_from_file_location("prompt15_cli_import_test", cli)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(tmp_path.iterdir()) == before


def test_output_input_overlap_and_bad_mode_rejected(full_fixture: Path, contract):
    with pytest.raises(AdapterError, match="overlap"):
        build_modeling_matrix(
            input_root=full_fixture,
            output_root=full_fixture / "output",
            contract=contract,
            mode="fixture",
        )
    with pytest.raises(AdapterError, match="mode"):
        inspect_input_inventory(full_fixture, contract, mode="automatic")
