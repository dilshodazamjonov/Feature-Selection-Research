"""Data-free validation for the canonical Prompt 14 v3 analysis lock."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocols/prompt_14_two_dataset_analysis_v1"
AUDIT = ROOT / "cleanup/audits/prompt_14_two_dataset_oot_review_v3"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_lock_and_registries_authenticate() -> None:
    lock_path = PROTOCOL / "analysis_protocol_lock.json"
    lock = _json(lock_path)
    registry_path = PROTOCOL / "authoritative_comparison_registry.json"
    families_path = PROTOCOL / "authoritative_holm_families.json"
    assert lock["registry_bindings"]["authoritative_comparison_registry_sha256"] == _sha(
        registry_path
    )
    assert lock["registry_bindings"]["authoritative_holm_families_sha256"] == _sha(
        families_path
    )
    digest_line = (PROTOCOL / "analysis_protocol_lock.sha256").read_text(
        encoding="utf-8"
    )
    assert digest_line == f"{_sha(lock_path)}  analysis_protocol_lock.json\n"
    assert lock["status"] == "locked_data_free_before_numeric_outcome_inspection"


def test_exact_graph_counts_ids_and_complete_family_membership() -> None:
    registry = _json(PROTOCOL / "authoritative_comparison_registry.json")
    family_registry = _json(PROTOCOL / "authoritative_holm_families.json")
    comparisons = registry["comparisons"]
    families = family_registry["families"]
    comparison_ids = [row["comparison_id"] for row in comparisons]
    family_members = [
        member for family in families for member in family["member_comparison_ids"]
    ]
    assert len(comparisons) == len(set(comparison_ids)) == 124
    assert len(families) == 36
    assert len(family_members) == len(set(family_members)) == 124
    assert set(family_members) == set(comparison_ids)
    assert sum(row["category"] == "baseline_or_full_random" for row in comparisons) == 56
    assert sum(row["category"] == "combination" for row in comparisons) == 68
    assert sum(family["complete_registered_member_count"] for family in families) == 124


def test_rejected_voting_pool_sensitivities_are_absent() -> None:
    comparisons = _json(PROTOCOL / "authoritative_comparison_registry.json")[
        "comparisons"
    ]
    voting = [
        row
        for row in comparisons
        if row["method_b"] == "cross_dataset_rank_voting_v1_primary_pool_200"
    ]
    assert len(voting) == 4
    assert all(row["method_b_configuration"] == "pool200" for row in voting)
    serialized = json.dumps(comparisons, sort_keys=True)
    assert "cross_dataset_rank_voting_v1_primary_pool_100" not in serialized
    assert "cross_dataset_rank_voting_v1_primary_pool_300" not in serialized


def test_tolerance_and_conservative_full_family_rule_are_exact() -> None:
    lock = _json(PROTOCOL / "analysis_protocol_lock.json")
    families = _json(PROTOCOL / "authoritative_holm_families.json")["families"]
    assert lock["inference"]["metric_recomputation_absolute_tolerance"] == 1e-10
    for family in families:
        rule = family["unavailable_input_rule"]
        assert rule["protocol_allowed_unavailable"] == {
            "raw_p": None,
            "holm_input_p": 1.0,
            "reject": False,
        }
        assert rule["protocol_allowed_infeasible"] == {
            "raw_p": None,
            "holm_input_p": 1.0,
            "reject": False,
        }
        assert rule["authentication_failure"] == "block_no_p1_substitution"
    assert lock["incomplete_family_rule"]["retain_original_family_denominator"] is True
    assert (
        lock["incomplete_family_rule"]["authentication_failure"]
        == "block_analysis_and_integration_never_substitute_p1"
    )


def test_two_natural_support_cases_are_unpadded_26_of_40() -> None:
    cases = _json(PROTOCOL / "analysis_protocol_lock.json")["natural_support_rules"][
        "cases"
    ]
    assert len(cases) == 2
    assert {case["method"] for case in cases} == {
        "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost",
    }
    assert all(case["dataset"] == "homecredit" for case in cases)
    assert all(case["model"] == "catboost" for case in cases)
    assert all(case["requested_k"] == 40 for case in cases)
    assert all(case["realized_k"] == 26 for case in cases)
    assert all(case["support_status"] == "infeasible_natural_support" for case in cases)
    assert all(case["padding"] is False for case in cases)


def test_corrected_provenance_and_closed_attempt_boundary() -> None:
    provenance = _json(AUDIT / "corrected_provenance_record.json")
    closure = _json(
        ROOT
        / "cleanup/audits/prompt_14a_voting_manifest_repair/blocked_prompt_14_attempt_closure.json"
    )
    assert provenance["first_plan_committed_before_numeric_outcome_inspection"] is True
    assert provenance["first_attempt_postcommit_numeric_outcomes_opened"] is True
    assert provenance["first_attempt_comparisons_analyzed"] == 0
    assert provenance["first_attempt_plan_modified_after_outcome_inspection"] is False
    assert provenance["recovery_is_not_resumption"] is True
    assert closure["state"] == "closed_blocked_attempt_not_resumable"
    assert closure["comparisons_actually_analyzed"] == 0


def test_phase_0_1_never_resolved_raw_paths_or_started_workers() -> None:
    entry = _json(AUDIT / "entry_validation.json")
    lock = _json(PROTOCOL / "analysis_protocol_lock.json")
    assert entry["phase_0_1_safety"] == {
        "comparisons_analyzed_in_this_attempt": 0,
        "numeric_outcomes_opened_in_this_attempt": False,
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
    }
    assert lock["safety"]["raw_dataset_paths_resolved"] is False
    assert lock["safety"]["workers_started"] == 0
    assert lock["safety"]["experiment_workloads_authorized"] is False


def test_pointer_selection_is_fail_closed_without_legacy_fallback() -> None:
    selection = _json(AUDIT / "manifest_selection_validation.json")
    assert selection["status"] == "pass"
    assert selection["selection_rule"] == "explicit_successor_pointer_fail_closed"
    assert selection["legacy_manifest_active"] is False
    assert selection["fallback_to_legacy_manifest_permitted"] is False
    assert selection["selected_manifest_sha256"] == (
        "45a3c3ce2773508d352d3cd9a031b0d6e35835de72ad36c877f32090c1ceabaf"
    )
