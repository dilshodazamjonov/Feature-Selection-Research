"""Outcome-blind checks for the rejected fresh Prompt 14 v2 plan stage."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "cleanup/audits/prompt_14_two_dataset_oot_review_v2"


def _json(name: str) -> dict:
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_entry_and_manifest_selection_passed_without_outcomes() -> None:
    entry = _json("entry_validation.json")
    selection = _json("manifest_selection_validation.json")
    assert entry["entry_verdict"] == "pass"
    assert entry["repository"]["starting_head"] == (
        "0824481af49f37ee1a2e1d359bd47499605dce1d"
    )
    assert entry["data_free_safety"] == {
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
        "numeric_predictive_outcomes_opened": False,
        "historical_plan_opened": False,
    }
    assert selection["selection_rule"] == "explicit_successor_pointer_fail_closed"
    assert selection["fallback_to_original_permitted"] is False
    assert selection["payload_authentication"]["authenticated_entries"] == 55
    assert selection["payload_authentication"]["unaffected_entries_byte_identical"] == 54


def test_rejected_plan_digest_and_counts_are_self_consistent() -> None:
    plan_path = AUDIT / "preinspection_analysis_plan.json"
    plan = _json("preinspection_analysis_plan.json")
    digest = _json("preinspection_analysis_plan_digest.json")
    comparisons = plan["comparison_graph"]["ordered_comparisons"]
    families = plan["multiplicity"]["families"]
    assert digest["sha256"] == _sha(plan_path)
    assert len(comparisons) == digest["comparison_count"] == 124
    assert len(families) == digest["holm_family_count"] == 36
    assert sum(family["complete_member_count"] for family in families) == 124
    assert plan["plan_status"] == (
        "rejected_material_historical_consistency_discrepancy_not_active"
    )
    assert plan["authorized_for_numeric_outcome_inspection"] is False
    assert digest["authorized_for_numeric_outcome_inspection"] is False


def test_fresh_derivation_breakdown_is_recorded_not_silently_rewritten() -> None:
    plan = _json("preinspection_analysis_plan.json")
    comparisons = plan["comparison_graph"]["ordered_comparisons"]
    roles = [item["comparison_role"] for item in comparisons]
    assert roles.count("direct_constituent_or_parent") == 64
    assert roles.count("voting_context") == 12
    assert roles.count("sanity_reference") == 48


def test_historical_consistency_discrepancy_fails_closed() -> None:
    check = _json("historical_plan_consistency_check.json")
    assert check["historical_plan_sha256"] == (
        "baa3616fec0e4a8498b018fe8d17c90c141d4fd1169bd2d66cb6252c83156901"
    )
    assert check["status"] == "material_discrepancy_fail_closed"
    assert check["comparison_identity_exact_match"] is False
    assert check["family_membership_exact_match"] is False
    assert check["thresholds_exact_match"] is False
    assert check["decision"] == (
        "stop_before_numeric_outcome_inspection_do_not_select_or_revise_either_design"
    )
    assert check["outcome_blindness"]["numeric_predictive_outcomes_opened_before_or_during_check"] is False
    assert check["raw_dataset_paths_resolved"] is False
    assert check["workers_started"] == 0


def test_natural_support_cases_remain_exact_and_unpadded() -> None:
    cases = _json("preinspection_analysis_plan.json")["budget_and_support_rules"][
        "natural_support_cases"
    ]
    assert len(cases) == 2
    assert {case["method_id"] for case in cases} == {
        "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost",
    }
    assert all(case["dataset"] == "homecredit" for case in cases)
    assert all(case["model"] == "catboost" for case in cases)
    assert all(case["requested_k"] == 40 and case["realized_k"] == 26 for case in cases)
    assert all(case["support_status"] == "infeasible_natural_support" for case in cases)
    assert all(case["padding"] is False for case in cases)
