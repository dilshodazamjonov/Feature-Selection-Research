"""Synthetic heavy-selector integration fixture, exercised inside the suite.

Deterministic synthetic data only. Never loads a real dataset, an OOT split, or a
saved prediction, and never executes a real fold.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.experiments.result_paths import (
    AUDITED_LEGACY_RESULTS_ROOT,
    HistoricalResultsWriteError,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/verify_heavy_selectors.py"


@pytest.fixture(scope="module")
def script():
    specification = importlib.util.spec_from_file_location(
        "verify_heavy_selectors", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_fixture_contains_every_required_shape(script) -> None:
    candidates, target, metadata = script.build_fixture()

    assert set(target.unique()) == {0, 1}
    assert 0.05 < float(target.mean()) < 0.95
    assert candidates["charlie_constant"].nunique() == 1
    assert candidates["echo_sparse_with_gaps"].isna().sum() > 0
    assert (
        candidates["mike_redundant_copy"].corr(candidates["zulu_linear_signal"]) > 0.99
    )
    # A nonlinear signal a tree can use but a linear model cannot.
    assert abs(candidates["delta_nonlinear_signal"].corr(target.astype(float))) < 0.15
    assert list(candidates.columns) != sorted(candidates.columns)
    assert not set(candidates.columns) & set(script.EXCLUDED_COLUMNS)
    assert "TARGET" in metadata.columns


@pytest.mark.parametrize(
    ("case_name", "method_id", "selection_mode", "budget"),
    [
        ("rfe_catboost_feasible_budget", "rfe_catboost", None, 3),
        ("boruta_natural_confirmed", "boruta_random_forest", "natural_confirmed", None),
        ("boruta_insufficient_confirmed", "boruta_random_forest", "confirmed_top_k", 7),
        ("catboost_shap_feasible_budget", "catboost_shap", None, 3),
    ],
)
def test_each_case_round_trips_through_the_real_artifact_path(
    script, case_name, method_id, selection_mode, budget, tmp_path: Path
) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_case(
        case_name=case_name,
        method_id=method_id,
        selection_mode=selection_mode,
        budget=budget,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )

    assert record["status"] == "PASS", record
    assert record["deterministic_on_refit"] is True
    assert record["serialization_exact"] is True
    assert record["excluded_columns_absent_from_selection"] is True
    assert record["selection_inside_candidate_universe"] is True
    assert record["cost_class"] == "heavy"
    assert record["candidate_universe_count"] == candidates.shape[1]
    assert record["estimator_config_sha256"]

    frame = pd.read_csv(tmp_path / record["artifacts"]["csv"]["path"])
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))
    assert frame["method_id"].unique().tolist() == [method_id]


def test_rfe_case_records_fits_and_elimination_history(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_case(
        case_name="rfe",
        method_id="rfe_catboost",
        selection_mode=None,
        budget=3,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )
    assert record["actual_selected_count"] == 3
    assert record["budget_status"] == "satisfied"
    assert record["estimator_fit_count"] >= 2
    assert record["elimination_history_rows"] >= 1
    assert record["natural_selected_count"] is None


def test_boruta_natural_case_reports_all_three_states(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_case(
        case_name="boruta_natural",
        method_id="boruta_random_forest",
        selection_mode="natural_confirmed",
        budget=None,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )
    counts = record["support_state_counts"]
    assert sum(counts.values()) == candidates.shape[1]
    assert record["budget_status"] == "not_applicable"
    assert record["actual_selected_count"] == counts["confirmed"]
    assert record["natural_selected_count"] == counts["confirmed"]
    # The harness imputed for the engine; the selector did not.
    assert record["harness_median_imputation_applied"] is True


def test_boruta_insufficient_case_never_pads(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_case(
        case_name="boruta_short",
        method_id="boruta_random_forest",
        selection_mode="confirmed_top_k",
        budget=7,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )
    assert record["requested_budget"] == 7
    assert record["budget_status"] == "infeasible_natural_support"
    assert record["actual_selected_count"] == record["support_state_counts"]["confirmed"]
    assert record["actual_selected_count"] < 7


def test_shap_case_records_the_explanation_sample(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_case(
        case_name="shap",
        method_id="catboost_shap",
        selection_mode=None,
        budget=3,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )
    sample = record["explanation_sample"]
    assert sample["scope"] == "selector_training_partition_only"
    assert sample["realized_size"] == 200
    assert sample["positive_count"] + sample["negative_count"] == 200
    assert len(sample["row_identity_sha256"]) == 64
    assert record["shap_calc_type"] == "Regular"
    assert record["natural_selected_count"] is None


def test_fixture_writes_nothing_into_protected_roots(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    before = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    script.run_case(
        case_name="isolation",
        method_id="rfe_catboost",
        selection_mode=None,
        budget=2,
        candidates=candidates,
        target=target,
        scratch=tmp_path,
    )
    after = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert before == after

    for path in tmp_path.rglob("*"):
        resolved = path.resolve()
        assert not resolved.is_relative_to(AUDITED_LEGACY_RESULTS_ROOT)
        assert not resolved.is_relative_to((ROOT / "results").resolve())

    with pytest.raises(HistoricalResultsWriteError):
        script._assert_scratch_is_isolated(AUDITED_LEGACY_RESULTS_ROOT / "scratch")
    with pytest.raises(SystemExit, match="active results"):
        script._assert_scratch_is_isolated(ROOT / "results/final_experiments/scratch")


def test_published_fixture_evidence_is_consistent_when_present() -> None:
    path = (
        ROOT / "cleanup/audits/prompt_08_heavy_selectors/synthetic_fixture_results.json"
    )
    if not path.is_file():
        pytest.skip("the Prompt 8 fixture evidence file has not been generated")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["failure_count"] == 0
    assert payload["fixture"]["real_dataset_loaded"] is False
    assert payload["fixture"]["oot_data_loaded"] is False
    assert payload["fixture"]["real_fold_executed"] is False
    assert payload["isolation"]["inside_active_results_root"] is False
    assert payload["isolation"]["inside_legacy_results_root"] is False
    assert payload["estimator_profiles"]["classification"] == (
        "synthetic_test_profiles_not_frozen_research_settings"
    )
    assert {entry["method_id"] for entry in payload["cases"]} == {
        "rfe_catboost",
        "boruta_random_forest",
        "catboost_shap",
    }
    for entry in payload["cases"]:
        assert entry["status"] == "PASS"
        assert entry["deterministic_on_refit"] is True
        assert entry["serialization_exact"] is True
