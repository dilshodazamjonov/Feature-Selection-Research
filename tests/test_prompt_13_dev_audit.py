from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from credit_risk_fs.experiments.prompt_13_dev_audit import (
    AUDIT_DIR,
    Prompt13AuditError,
    REVIEW_LOCK_PATH,
    authenticated_payload,
    baseline_alignment_reason,
    contamination_paths,
    pairwise_stability,
    read_authenticated_json,
    support_label,
    validate_unique_ordered_identities,
)
from credit_risk_fs.experiments.selector_combinations import (
    _validate_approval_lock,
    _validate_dev_completion_lock,
    build_phase_matrix,
    load_combination_plan,
    render_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def test_authenticated_json_rejects_corrupt_or_hash_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = authenticated_payload({"terminal_state": "completed", "identity": "cell-1"})
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_authenticated_json(path) == payload

    payload["identity"] = "cell-2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Prompt13AuditError, match="authentication_sha256_mismatch"):
        read_authenticated_json(path)


@pytest.mark.parametrize(
    "observed",
    [
        ["a", "b"],  # missing
        ["a", "b", "c", "d"],  # extra
        ["a", "b", "b"],  # duplicate and missing
        ["b", "a", "c"],  # order change
    ],
)
def test_identity_inventory_blocks_missing_extra_duplicate_or_reordered(observed: list[str]) -> None:
    with pytest.raises(Prompt13AuditError, match="identity mismatch"):
        validate_unique_ordered_identities(["a", "b", "c"], observed, "synthetic")


def test_oot_artifact_name_is_a_contamination_failure_without_opening_it(tmp_path: Path) -> None:
    path = tmp_path / "results/selector_combinations_v1/oot/evaluations/scv1-oot-001.dev_metrics.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"must not be opened")
    assert contamination_paths(tmp_path, ["scv1-oot-001"]) == [
        "results/selector_combinations_v1/oot/evaluations/scv1-oot-001.dev_metrics.json"
    ]


def test_natural_support_26_of_40_cannot_be_treated_as_matched_budget() -> None:
    label = support_label(
        "homecredit", "catboost", "boruta_then_rfe_catboost", 40, 26
    )
    assert label == "natural_support_26_of_requested_40"
    assert baseline_alignment_reason(
        combination_support_label=label,
        baseline_method="random_k",
        baseline_fold_vectors_saved=True,
        baseline_ordered_identity_hashes_saved=True,
    ) == "natural-support result is not a matched fixed-K random baseline"


def test_kuncheva_is_only_calculated_for_equal_selected_set_sizes() -> None:
    _, _, _, kuncheva, status = pairwise_stability(
        [{"a", "b"}, {"a", "c"}, {"a", "d"}], 10
    )
    assert kuncheva is not None
    assert status.startswith("applicable")
    _, _, _, kuncheva, status = pairwise_stability(
        [{"a", "b"}, {"a", "c", "d"}], 10
    )
    assert kuncheva is None
    assert status.startswith("not_applicable")


def test_baseline_alignment_is_not_coerced_without_paired_fold_identity() -> None:
    reason = baseline_alignment_reason(
        combination_support_label="matched_budget",
        baseline_method="iv_woe",
        baseline_fold_vectors_saved=False,
        baseline_ordered_identity_hashes_saved=False,
    )
    assert "cannot be authenticated" in reason


def test_safe_plan_and_review_lock_bind_the_exact_future_scope() -> None:
    plan = load_combination_plan(ROOT)
    rendered = render_plan(plan)
    assert rendered["raw_dataset_paths_resolved"] is False
    assert rendered["workers_started"] == 0
    assert rendered["oot_selection_count"] == 18
    assert rendered["oot_evaluation_count"] == 24

    approval = _validate_approval_lock(plan)
    completion = _validate_dev_completion_lock(plan, approval)
    _, expected = build_phase_matrix(
        plan, phase="oot", retained_method_ids=tuple(approval["retained_method_ids"])
    )
    lock = read_authenticated_json(ROOT / REVIEW_LOCK_PATH)
    assert lock["ordered_oot_evaluation_ids"] == [item.cell_id for item in expected]
    assert lock["dev_completion_lock_authentication_sha256"] == completion[
        "artifact_authentication_sha256"
    ]
    assert lock["expected_oot_evaluations"] == 24
    assert lock["oot_has_run"] is False
    audit_source = ROOT / lock["code_identity"]["audit_module_path"]
    assert hashlib.sha256(audit_source.read_bytes()).hexdigest() == lock["code_identity"][
        "audit_module_sha256"
    ]
    assert lock["scope_freeze_commit"] == "26158348a273876ac11956b557e8534d9edffdd2"


def test_generated_review_package_records_full_authentication_and_preservation() -> None:
    audit = ROOT / AUDIT_DIR
    authentication = read_authenticated_json(audit / "dev_authentication.json")
    decision = read_authenticated_json(audit / "review_decision.json")
    preservation = read_authenticated_json(audit / "preservation_check.json")
    scope = read_authenticated_json(audit / "oot_scope_validation.json")
    comparisons = (audit / "aligned_baseline_comparisons.csv").read_text(encoding="utf-8")

    assert authentication["dev"]["selector_fits"] == "90/90"
    assert authentication["dev"]["evaluation_cells"] == "120/120"
    assert authentication["selection_contract"]["natural_support_26_of_40_fold_rows"] == 2
    assert decision["decision"] == "ready_for_manual_oot"
    assert preservation["status"] == "preserved_byte_identical"
    assert scope["observed_oot_evaluations"] == 24
    assert "not_supported" in comparisons
    assert (audit / "report.html").is_file()


def test_audit_module_has_no_loader_or_workload_entry_point_import() -> None:
    source = (ROOT / "src/credit_risk_fs/experiments/prompt_13_dev_audit.py").read_text(
        encoding="utf-8"
    )
    assert "prepare_voting_pilot_dev_data" not in source
    assert "execute_dev" not in source
    assert "execute_oot" not in source
    assert "run_full_baseline" not in source
