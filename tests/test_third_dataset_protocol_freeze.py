from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "cleanup/audits/third_dataset_protocol_freeze"
COMBINATION_ORDER = [
    "statistical_normalized_average_rank",
    "iv_then_boruta",
    "boruta_then_mrmr_mutual_information",
    "boruta_then_rfe_catboost",
]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_review_digest_authenticates_every_declared_artifact() -> None:
    digest = json.loads((AUDIT / "review_digest.json").read_text(encoding="utf-8"))
    supplied = digest.pop("review_digest_sha256")
    assert hashlib.sha256(_canonical(digest)).hexdigest() == supplied
    for artifact in digest["artifact_hashes"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["byte_size"]
        assert _sha256(path) == artifact["sha256"]
    declared = {artifact["path"] for artifact in digest["artifact_hashes"]}
    assert declared == {
        "cleanup/audits/third_dataset_protocol_freeze/dataset_identity.json",
        "cleanup/audits/third_dataset_protocol_freeze/raw_file_inventory.csv",
        "cleanup/audits/third_dataset_protocol_freeze/schema_and_cardinality_profile.csv",
        "cleanup/audits/third_dataset_protocol_freeze/feature_definition_coverage.json",
        "cleanup/audits/third_dataset_protocol_freeze/temporal_population_profile.csv",
        "cleanup/audits/third_dataset_protocol_freeze/proposed_split_and_fold_boundaries.json",
        "cleanup/audits/third_dataset_protocol_freeze/leakage_and_availability_review.csv",
        "cleanup/audits/third_dataset_protocol_freeze/proposed_adapter_protocol.json",
        "cleanup/audits/third_dataset_protocol_freeze/proposed_method_matrix.json",
        "cleanup/audits/third_dataset_protocol_freeze/preregistered_hypotheses_and_analysis.md",
        "cleanup/audits/third_dataset_protocol_freeze/protocol_review.md",
        "cleanup/tools/build_third_dataset_protocol_freeze.py",
        "tests/test_third_dataset_protocol_freeze.py",
    }


def test_inventory_depth_scope_and_input_digest_are_exact() -> None:
    with (AUDIT / "raw_file_inventory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    included = [row for row in rows if row["inclusion_status"] == "included"]
    included_parquet = [row for row in included if row["file_type"] == "parquet"]
    excluded_parquet = [
        row for row in rows
        if row["inclusion_status"] == "excluded" and row["file_type"] == "parquet"
    ]
    assert len(included) == 19
    assert len(included_parquet) == 18
    assert len(excluded_parquet) == 14
    assert {row["relational_depth"] for row in excluded_parquet} == {"2"}
    assert all(row["sha256"] for row in included)
    identity = [
        {
            "relative_path": row["relative_path"],
            "byte_size": int(row["byte_size"]),
            "sha256": row["sha256"],
        }
        for row in included
    ]
    observed = hashlib.sha256(_canonical(identity)).hexdigest()
    dataset_identity = json.loads((AUDIT / "dataset_identity.json").read_text(encoding="utf-8"))
    assert observed == dataset_identity["included_raw_input_digest"]


def test_structural_leakage_and_split_contracts_cross_validate() -> None:
    with (AUDIT / "schema_and_cardinality_profile.csv").open(encoding="utf-8", newline="") as handle:
        profiles = list(csv.DictReader(handle))
    assert len(profiles) == 13
    assert all(int(row["orphan_case_id_rows"]) == 0 for row in profiles)
    assert all(int(row["orphan_distinct_case_ids"]) == 0 for row in profiles)
    with (AUDIT / "leakage_and_availability_review.csv").open(encoding="utf-8", newline="") as handle:
        leakage = list(csv.DictReader(handle))
    assert len(leakage) == 461
    assert sum(row["action"] == "include" for row in leakage) == 434
    assert sum(row["action"] == "exclude" for row in leakage) == 27
    assert sum(row["action"] == "unresolved" for row in leakage) == 0
    split = json.loads((AUDIT / "proposed_split_and_fold_boundaries.json").read_text(encoding="utf-8"))
    assert split["oot_start_inclusive"] == "2020-02-26"
    assert split["dev"]["rows"] == 1_221_743
    assert split["oot"]["rows"] == 304_916
    assert len(split["folds"]) == 5
    assert all(fold["case_id_overlap"] == 0 for fold in split["folds"])
    assert all(fold["date_unit_overlap"] == 0 for fold in split["folds"])


def test_population_profiles_and_parquet_metadata_reconcile() -> None:
    with (AUDIT / "raw_file_inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    by_path = {row["relative_path"]: row for row in inventory}
    with (AUDIT / "schema_and_cardinality_profile.csv").open(encoding="utf-8", newline="") as handle:
        profiles = list(csv.DictReader(handle))
    for profile in profiles:
        paths = json.loads(profile["relative_paths_json"])
        assert sum(int(by_path[path]["parquet_row_count"]) for path in paths) == int(profile["row_count"])
        assert all(by_path[path]["inclusion_status"] == "included" for path in paths)
    identity = json.loads((AUDIT / "dataset_identity.json").read_text(encoding="utf-8"))
    base = identity["base_population_validation"]
    assert base["row_count"] == base["case_id_unique_count"] == 1_526_659
    assert base["case_id_missing"] == base["case_id_duplicate_count"] == 0
    assert base["target_values"] == [0, 1]
    assert base["target_0"] + base["target_1"] == base["row_count"]
    assert base["month_matches_date_decision_yyyymm"] is True
    assert base["week_num_contiguous"] is True
    with (AUDIT / "temporal_population_profile.csv").open(encoding="utf-8", newline="") as handle:
        temporal = list(csv.DictReader(handle))
    months = [row for row in temporal if row["profile_level"] == "calendar_month"]
    assert sum(int(row["rows"]) for row in months) == base["row_count"]
    assert sum(int(row["target_0"]) for row in months) == base["target_0"]
    assert sum(int(row["target_1"]) for row in months) == base["target_1"]


def test_method_matrix_and_execution_gates_remain_frozen_closed() -> None:
    matrix = json.loads((AUDIT / "proposed_method_matrix.json").read_text(encoding="utf-8"))
    assert matrix["combination_order"] == COMBINATION_ORDER
    assert len(matrix["variant_order"]) == 15
    assert len(matrix["matrix_cells"]) == 30
    assert matrix["phase_design"]["pilot"]["selector_fit_calls"] == 27
    assert matrix["phase_design"]["pilot"]["configuration_evaluation_cells"] == 30
    assert matrix["phase_design"]["dev"]["selector_fit_calls"] == 135
    assert matrix["phase_design"]["dev"]["fold_evaluation_cells"] == 150
    assert matrix["phase_design"]["oot"]["evaluation_cells"] == 30
    full_features = [cell for cell in matrix["matrix_cells"] if cell["method_id"] == "full_features"]
    iv_boruta = [cell for cell in matrix["matrix_cells"] if cell["method_id"] == "iv_then_boruta"]
    assert all(cell["requested_feature_budget"] is None for cell in full_features)
    assert all(cell["requested_feature_budget"] is None for cell in iv_boruta)
    assert matrix["gates"]["stage_1"] == "review_only_no_execution"
    identity = json.loads((AUDIT / "dataset_identity.json").read_text(encoding="utf-8"))
    assert identity["task_boundaries"] == {
        "existing_two_dataset_oot_metrics_opened": False,
        "model_or_selector_fit_run": False,
        "pilot_dev_or_oot_run": False,
        "raw_files_modified": False,
        "network_accessed": False,
        "canonical_protocol_lock_created": False,
    }
    assert not (ROOT / "configs/protocols/homecredit_model_stability_2024_v1").exists()
