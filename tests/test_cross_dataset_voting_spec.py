from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.experiments.rank_voting import (
    aggregate_cross_dataset_rank_voting,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml"
AUDIT_ROOT = ROOT / "cleanup/audits/cross_dataset_voting_execution_spec"

EXPECTED_HASHES = {
    "configs/protocols/credit_scoring_extension_v1.yaml": (
        "f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0"
    ),
    "configs/protocols/row_alignment_contract_v1.json": (
        "fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0"
    ),
    "configs/protocols/cross_dataset_rank_voting_v1.yaml": (
        "51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49"
    ),
    "configs/execution/local_laptop_safe_v1.yaml": (
        "1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix() -> dict:
    return _parse_simple_yaml(MATRIX_PATH.read_text(encoding="utf-8"))


def test_frozen_inputs_have_expected_hashes_and_matrix_refs(matrix):
    for relative_path, expected in EXPECTED_HASHES.items():
        assert _sha256(ROOT / relative_path) == expected

    assert matrix["protocol"] == {
        "path": "configs/protocols/cross_dataset_rank_voting_v1.yaml",
        "name": "cross_dataset_rank_voting_v1",
        "version": "1.0.0",
        "sha256": EXPECTED_HASHES[
            "configs/protocols/cross_dataset_rank_voting_v1.yaml"
        ],
    }
    assert matrix["row_contract"]["version"] == "1.1.0"
    assert matrix["row_contract"]["sha256"] == EXPECTED_HASHES[
        "configs/protocols/row_alignment_contract_v1.json"
    ]
    assert matrix["execution_policy"]["sha256"] == EXPECTED_HASHES[
        "configs/execution/local_laptop_safe_v1.yaml"
    ]


def test_matrix_expands_to_exact_cartesian_design_and_reference_reruns(matrix):
    runs = matrix["runs"]
    run_order = matrix["run_order"]
    assert len(runs) == len(run_order) == 16
    assert len(set(run_order)) == 16
    assert set(runs) == set(run_order)
    assert matrix["status"] == "specification_only_not_authorized_for_execution"

    ordered = [runs[run_id] for run_id in run_order]
    for position, (run_id, run) in enumerate(zip(run_order, ordered, strict=True), 1):
        assert run["execution_order"] == position
        assert run["proposed_run_id"] == run_id
        assert run["enabled"] is True
        assert run["blocked_reason"] is None
        assert run["execution_policy"] == (
            "configs/execution/local_laptop_safe_v1.yaml"
        )
        assert run["master_seed"] == run["model_seed"] == run["selector_seed"] == 42

    voting = [run for run in ordered if run["method_id"] == "rank_voting_v1"]
    references = [run for run in ordered if run["designation"] == "reference"]
    assert len(voting) == 12
    assert len(references) == 4

    expected_voting = {
        (dataset, model, budget)
        for dataset in ("homecredit", "lendingclub_v2")
        for model in ("lr", "catboost")
        for budget in (100, 200, 300)
    }
    assert {
        (run["dataset"], run["model"], run["candidate_pool_budget"])
        for run in voting
    } == expected_voting
    assert all(run["voter_ids"] == ["rf_corr_mrmr", "boruta"] for run in voting)
    assert all(run["reference_method"] == "rf_corr_mrmr" for run in voting)

    assert {
        (run["dataset"], run["model"]) for run in references
    } == {
        (dataset, model)
        for dataset in ("homecredit", "lendingclub_v2")
        for model in ("lr", "catboost")
    }
    assert all(run["method_id"] == "rf_corr_mrmr" for run in references)


def test_matrix_budgets_designations_hashes_and_reported_counts(matrix):
    runs = matrix["runs"].values()
    voting = [run for run in runs if run["method_id"] == "rank_voting_v1"]
    assert sum(run["designation"] == "primary" for run in voting) == 4
    assert sum(run["designation"] == "sensitivity" for run in voting) == 8
    assert all(
        (run["candidate_pool_budget"] == 200) == (run["designation"] == "primary")
        for run in voting
    )
    assert all(
        run["final_feature_budget"] == (20 if run["model"] == "lr" else 40)
        for run in runs
    )

    counts = matrix["expected_counts"]
    assert counts == {
        "voting_runs": 12,
        "reference_reruns": 4,
        "total_registered_runs": 16,
        "dev_folds_per_run": 5,
        "voting_dev_fold_executions": 60,
        "reference_dev_fold_executions": 20,
        "total_dev_fold_executions": 80,
        "final_full_dev_oot_fits": 16,
        "primary_voting_runs": 4,
        "sensitivity_voting_runs": 8,
        "primary_comparisons": 4,
        "sensitivity_comparisons": 8,
    }

    expected_dev = {
        "homecredit": "722f897d531415852d00904b3c9b34f664831126b8d7afe066f2536e0a25c9b7",
        "lendingclub_v2": "4d4cd7973f00eb946fef0a6bb09e61fe6d2b9be92892786f352446660c68818e",
    }
    expected_oot = {
        "homecredit": "3e90101f56774b7e44086b3bccfa91bd5be35f3afd60a8bce0e5a53313acda7a",
        "lendingclub_v2": "86840e88a94f78f328d62e36754f14377c1765a31fb3bc73cbb3f7b2d45f8092",
    }
    for run in runs:
        assert run["expected_dev_row_hash"] == expected_dev[run["dataset"]]
        assert run["expected_oot_row_hash"] == expected_oot[run["dataset"]]


def test_hand_aggregation_orientation_missing_rank_and_ties():
    result = aggregate_cross_dataset_rank_voting(
        eligible_features=["A", "B", "C", "D"],
        rankings={
            "rf_corr_mrmr": ["A", "B", "C"],
            "boruta": ["B", "C", "D"],
        },
        fit_scopes={
            "rf_corr_mrmr": "dev_fold_training_only",
            "boruta": "dev_fold_training_only",
        },
    )
    assert result["feature"].tolist() == ["B", "C", "A", "D"]
    assert result["aggregate_score"].tolist() == pytest.approx(
        [5 / 6, 1 / 2, 1 / 2, 1 / 6]
    )
    assert result["voter_presence_count"].tolist() == [2, 2, 1, 1]


def test_audit_contracts_are_complete_and_do_not_claim_reference_reuse():
    implementation = json.loads((AUDIT_ROOT / "implementation_map.json").read_text())
    gaps = implementation["prompt_4_gaps"]
    assert len(gaps) == 9
    assert len({gap["gap_id"] for gap in gaps}) == len(gaps)
    assert all(gap["existing_owner_to_extend"] for gap in gaps)
    assert implementation["readiness"] == (
        "bounded_prompt_4_extensions_assigned_to_existing_owners"
    )

    with (AUDIT_ROOT / "reference_reuse_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        references = list(csv.DictReader(handle))
    assert len(references) == 4
    assert {row["decision"] for row in references} == {"rerun_required"}
    assert len({row["scheduled_run_id"] for row in references}) == 4

    with (AUDIT_ROOT / "forward_parity_table.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        parity = list(csv.DictReader(handle))
    required_components = {
        "target orientation",
        "row identity and chronological ordering",
        "candidate-universe construction",
        "rf_corr_mrmr voter",
        "boruta voter",
        "rank normalization",
        "aggregation and aliases",
        "candidate pools",
        "downstream selector",
        "LR model",
        "CatBoost model",
        "DEV folds",
        "full-DEV refit",
        "OOT scoring",
        "prediction schema",
        "metrics",
        "paired inference",
    }
    assert required_components <= {row["component"] for row in parity}
    assert all("mismatch" not in row["status"] for row in parity)

    validation = json.loads((AUDIT_ROOT / "validation_summary.json").read_text())
    assert validation["gate"] == validation["readiness"] == "READY_FOR_PROMPT_4"
    assert validation["matrix"]["total_future_registered_runs"] == 16
    assert validation["active_results_end_state"]["run_index_experiment_rows"] == 0
    assert validation["historical_manifest_end_comparison"]["status"] == "unchanged"
    assert validation["remaining_blockers_for_prompt_4"] == []


def test_preflight_shapes_are_explicit_and_require_a_monitored_pilot():
    preflight = json.loads((AUDIT_ROOT / "preflight_request_specs.json").read_text())
    assert preflight["evidence_policy"]["columns_none_permitted"] is False
    shapes = preflight["execution_shapes"]
    assert len(shapes) == 4
    assert {(shape["dataset"], shape["model"]) for shape in shapes} == {
        (dataset, model)
        for dataset in ("homecredit", "lendingclub_v2")
        for model in ("lr", "catboost")
    }
    assert all(shape["status"] == "pilot_required" for shape in shapes)
    assert all(shape["selector_model_peak_multiplier"] is None for shape in shapes)
    assert all(shape["projected_input_bytes"] > 0 for shape in shapes)
    assert all(shape["required_free_disk_bytes"] > 0 for shape in shapes)
    assert all(shape["expected_gpu_use"] is False for shape in shapes)

    stages = preflight["stage_projections"]
    assert {stage["stage"] for stage in stages} == {
        "row_validation",
        "voter_generation",
        "aggregation",
        "downstream_selection",
        "model_fit_predict",
        "evaluation",
    }
    assert all("columns=None" not in str(stage["required_columns"]) for stage in stages)
