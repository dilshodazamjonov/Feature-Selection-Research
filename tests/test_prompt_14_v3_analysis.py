"""Focused validation for the finalized Prompt 14 v3 artifact-only analysis."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "cleanup/audits/prompt_14_two_dataset_oot_review_v3"
PROTOCOL = ROOT / "configs/protocols/prompt_14_two_dataset_analysis_v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict[str, str]]:
    with (AUDIT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_authentication_counts_and_metric_reconciliation_are_exact() -> None:
    authentication = _json(AUDIT / "authentication_validation.json")
    assert authentication["status"] == "pass"
    assert authentication["pilot"]["authenticated_evaluations"] == 24
    assert authentication["pilot"]["authenticated_selector_fits"] == 18
    assert authentication["dev"]["authenticated_evaluations"] == 120
    assert authentication["dev"]["authenticated_selector_fits"] == 90
    assert authentication["combination_oot"]["authenticated_evaluations"] == 24
    assert authentication["combination_oot"]["authenticated_selector_fits"] == 18
    assert authentication["combination_oot"]["authenticated_active_files"] == 168
    assert authentication["baseline"]["authenticated_cells"] == 36
    assert authentication["voting_package"]["payload_entries_authenticated"] == 55
    assert authentication["voting_package"]["unaffected_entries_byte_identical"] == 54
    assert authentication["prediction_artifacts_authenticated"] == 64

    checks = _rows("metric_recomputation_check.csv")
    assert len(checks) == authentication["metric_reconciliation_rows"] == 448
    assert all(row["passed"] == "True" for row in checks)
    assert all(
        not row["absolute_difference"]
        or float(row["absolute_difference"]) <= 1e-10
        for row in checks
    )


def test_all_registered_comparisons_and_families_are_accounted_for_once() -> None:
    registered = _json(PROTOCOL / "authoritative_comparison_registry.json")[
        "comparisons"
    ]
    observed = _rows("paired_comparisons.csv")
    expected_ids = [row["comparison_id"] for row in registered]
    observed_ids = [row["comparison_id"] for row in observed]
    assert len(observed) == len(set(observed_ids)) == 124
    assert observed_ids == expected_ids
    assert all(row["availability"] == "evaluable" for row in observed)
    assert all(row["family_status"] == "evaluable" for row in observed)
    assert all(int(row["bootstrap_attempted"]) == 2000 for row in observed)
    assert all(int(row["bootstrap_valid"]) >= 1900 for row in observed)

    reconciliation = _json(AUDIT / "comparison_reconciliation.json")
    assert reconciliation["registered_comparisons"] == 124
    assert reconciliation["evaluable_comparisons"] == 124
    assert reconciliation["holm_families"] == 36
    assert reconciliation["protocol_allowed_unavailable"] == 0
    assert reconciliation["protocol_allowed_infeasible"] == 0
    assert reconciliation["authentication_failures"] == 0
    assert reconciliation["full_family_denominators_preserved"] is True

    families = _json(AUDIT / "multiplicity_families.json")
    assert families["family_count"] == 36
    assert families["registered_member_count"] == 124
    assert sum(row["holm_denominator"] for row in families["families"]) == 124
    assert all(row["family_status"] == "evaluable" for row in families["families"])


def test_natural_support_reference_and_oot_refit_facts_remain_distinct() -> None:
    results = _rows("two_dataset_results_long.csv")
    methods = {
        "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost",
    }
    cases = [
        row
        for row in results
        if row["dataset"] == "homecredit"
        and row["model"] == "catboost"
        and row["method"] in methods
    ]
    assert len(cases) == 2
    assert {row["method"] for row in cases} == methods
    assert all(row["requested_k"] == "40" for row in cases)
    assert all(row["realized_k"] == "40" for row in cases)
    assert all(row["reference_natural_support_k"] == "26" for row in cases)
    assert all(row["padding"] == "False" for row in cases)
    assert all(
        row["confirmatory_eligibility"] == "exploratory_natural_support_context"
        for row in cases
    )

    comparisons = _rows("paired_comparisons.csv")
    affected = [
        row
        for row in comparisons
        if row["dataset"] == "homecredit"
        and row["model"] == "catboost"
        and (row["method"] in methods or row["reference"] in methods)
    ]
    assert affected
    assert all(row["exploratory"] == "True" for row in affected)


def test_report_traceability_and_portable_delivery_are_complete() -> None:
    traceability = _json(AUDIT / "report_traceability.json")
    assert traceability["status"] == "pass"
    assert traceability["all_report_numeric_claims_traceable"] is True
    assert len(traceability["claims"]) == 6

    html = (AUDIT / "report.html").read_text(encoding="utf-8")
    assert "Two-dataset locked-OOT statistical review" in html
    assert "effect_scatter" in html
    assert "generalization_scatter" in html
    assert "paired_comparisons.csv" in html
    assert "claims_and_evidence.csv" in html


def test_results_digest_authenticates_every_required_delivery_artifact() -> None:
    digest = _json(AUDIT / "results_digest.json")
    assert digest["registered_comparisons"] == 124
    assert digest["holm_families"] == 36
    paths = [row["relative_path"] for row in digest["artifacts"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths)) == 16
    for row in digest["artifacts"]:
        path = ROOT / row["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert _sha(path) == row["sha256"]


def test_final_validation_records_no_workloads_raw_access_or_oot_adaptation() -> None:
    final = _json(AUDIT / "final_validation.json")
    assert final["status"] == "pass"
    checks = {row["check_id"]: row for row in final["checks"]}
    assert checks["raw_data_access"]["observed"] == 0
    assert checks["experiment_worker_startup"]["observed"] == 0
    assert checks["configuration_adaptation_after_oot"]["observed"] is False
    assert all(row["status"] == "pass" for row in checks.values())
