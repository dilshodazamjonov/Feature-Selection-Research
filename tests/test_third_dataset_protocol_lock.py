from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cleanup/tools/create_third_dataset_protocol_lock.py"
LOCK = ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"
REPORT = ROOT / "cleanup/audits/third_dataset_protocol_approval/stage_2_validation.json"

SPEC = importlib.util.spec_from_file_location("create_third_dataset_protocol_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _errors(payload: dict[str, object]) -> list[str]:
    return MODULE.validate_lock_payload(ROOT, payload, verify_raw=False)


def test_canonical_lock_and_stage_2_report_authenticate() -> None:
    lock = _load(LOCK)
    report = _load(REPORT)
    assert MODULE.verify_self_authentication(lock)
    assert MODULE.verify_self_authentication(report)
    assert _errors(lock) == []
    assert report["canonical_lock"]["file_sha256"] == _file_sha(LOCK)
    assert (
        report["canonical_lock"]["artifact_authentication_sha256"]
        == lock["artifact_authentication_sha256"]
    )


def test_approval_identity_and_execution_gates_are_exact() -> None:
    lock = _load(LOCK)
    approval = lock["user_approval_record"]
    assert approval["approved_review_digest_sha256"] == MODULE.APPROVED_REVIEW_DIGEST
    assert approval["approval_text"] == MODULE.APPROVAL_TEXT
    assert lock["protected_contract"]["exact_approved_combination_order"] == MODULE.APPROVED_COMBINATION_ORDER
    assert lock["gates"] == {
        "prompt_14": "next_required_manual_scientific_step_not_run_in_stage_2",
        "adapter": "specification_frozen_not_implemented",
        "pilot": "closed_until_Prompt_14_implements_and_validates_adapter_then_authorizes_bounded_manual_pilot",
        "dev": "closed_until_later_authenticated_pilot_review_and_explicit_approval",
        "oot": "closed_until_later_authenticated_complete_DEV_review_and_explicit_approval",
    }
    assert not any(
        value is True
        for key, value in lock["execution_boundary"].items()
        if key not in {"raw_file_operation"}
    )


def test_dataset_and_depth_mutations_fail_closed() -> None:
    lock = _load(LOCK)
    changed_data = copy.deepcopy(lock)
    changed_data["approved_protocol"]["dataset_identity"]["included_raw_input_digest"] = "0" * 64
    assert any("dataset_identity_or_included_raw_file_hash" in error for error in _errors(changed_data))
    changed_depth = copy.deepcopy(lock)
    depth_2 = next(
        row
        for row in changed_depth["approved_protocol"]["raw_file_scope"]["records"]
        if row["relational_depth"] == "2"
    )
    depth_2["inclusion_status"] = "included"
    assert any("depth_or_raw_file_scope" in error for error in _errors(changed_depth))


def test_temporal_and_feature_scope_mutations_fail_closed() -> None:
    lock = _load(LOCK)
    changed_boundary = copy.deepcopy(lock)
    changed_boundary["approved_protocol"]["split_and_fold_boundaries"]["oot_start_inclusive"] = "2020-02-27"
    assert any("temporal_boundary_or_membership" in error for error in _errors(changed_boundary))
    changed_feature = copy.deepcopy(lock)
    changed_feature["approved_protocol"]["leakage_and_availability_scope"]["records"][0]["action"] = "include"
    assert any("adapter_or_feature_scope" in error for error in _errors(changed_feature))


def test_method_budget_seed_and_metric_mutations_fail_closed() -> None:
    lock = _load(LOCK)
    changed_order = copy.deepcopy(lock)
    order = changed_order["approved_protocol"]["method_and_evaluation_matrix"]["combination_order"]
    order[0], order[1] = order[1], order[0]
    assert any("method_or_variant_order" in error for error in _errors(changed_order))

    changed_budget = copy.deepcopy(lock)
    changed_budget["approved_protocol"]["method_and_evaluation_matrix"]["feature_budgets"]["lr"] = 21
    assert any("feature_or_pool_budget" in error for error in _errors(changed_budget))

    changed_seed = copy.deepcopy(lock)
    changed_seed["approved_protocol"]["method_and_evaluation_matrix"]["seeds"]["experiment_selector_model"] = 43
    assert any("model_or_bootstrap_seed" in error for error in _errors(changed_seed))

    changed_metric = copy.deepcopy(lock)
    changed_metric["approved_protocol"]["method_and_evaluation_matrix"]["metrics"]["primary_predictive"] = "accuracy"
    assert any("metric_or_inference_rule" in error for error in _errors(changed_metric))


def test_canonical_scope_counts_and_natural_support_are_frozen() -> None:
    protocol = _load(LOCK)["approved_protocol"]
    assert protocol["raw_file_scope"]["included_count"] == 19
    assert protocol["raw_file_scope"]["included_parquet_count"] == 18
    assert protocol["raw_file_scope"]["excluded_depth_2_parquet_count"] == 14
    leakage = protocol["leakage_and_availability_scope"]
    assert (leakage["candidate_rows"], leakage["included"], leakage["excluded"], leakage["unresolved"]) == (461, 434, 27, 0)
    matrix = protocol["method_and_evaluation_matrix"]
    assert matrix["combination_order"] == MODULE.APPROVED_COMBINATION_ORDER
    assert matrix["phase_design"]["pilot"]["configuration_evaluation_cells"] == 30
    assert matrix["phase_design"]["dev"]["fold_evaluation_cells"] == 150
    assert matrix["phase_design"]["oot"]["evaluation_cells"] == 30
    assert matrix["natural_support"]["rule"] == "report requested K and realized support; never pad"
