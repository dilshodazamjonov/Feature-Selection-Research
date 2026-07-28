"""Tiny-fixture integration pass, exercised inside the test suite.

Deterministic synthetic data only. This test must never load a real dataset, an
OOT split, or a saved prediction.
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
from credit_risk_fs.selectors.lightweight.registry import method_ids_by_cost_class

#: This fixture is the *light* integration pass. Heavy methods share the same
#: registry and contract but have their own fixture in test_heavy_integration.py
#: with per-method cases, so they are filtered out by declared cost class rather
#: than by a hard-coded name list that would go stale.
LIGHT_METHOD_IDS = [
    name for name in method_ids_by_cost_class("light")
    if name != "legacy_rf_relevance_corr"
]

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/verify_lightweight_selectors.py"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "verify_lightweight_selectors", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def test_fixture_contains_every_required_shape(script) -> None:
    candidates, target, metadata = script.build_fixture()

    assert set(target.unique()) == {0, 1}
    assert 0.05 < float(target.mean()) < 0.95
    assert candidates["constant_column"].nunique() == 1
    assert candidates["sparse_with_gaps"].isna().sum() > 0
    assert candidates["redundant_copy"].corr(candidates["strong_signal"]) > 0.99
    assert list(candidates.columns) != sorted(candidates.columns), (
        "candidate order must not be alphabetical, or a column-order leak would hide"
    )
    # Metadata and target travel outside the candidate frame.
    assert not set(candidates.columns) & set(script.EXCLUDED_COLUMNS)
    assert "TARGET" in metadata.columns


@pytest.mark.parametrize(
    "method_id", LIGHT_METHOD_IDS
)
def test_each_method_round_trips_through_the_real_artifact_path(
    script, method_id, tmp_path: Path
) -> None:
    candidates, target, _ = script.build_fixture()
    record = script.run_one(method_id, candidates, target, tmp_path)

    assert record["status"] == "PASS", record
    assert record["deterministic_on_refit"] is True
    assert record["serialization_exact"] is True
    assert record["selection_inside_candidate_universe"] is True
    assert record["excluded_columns_absent_from_selection"] is True
    assert record["fit_scope"] == "dev_fold_training_only"
    assert record["candidate_universe_count"] == candidates.shape[1]
    assert record["training_row_count"] == len(candidates)

    for artifact in record["artifacts"].values():
        written = tmp_path / artifact["path"]
        assert written.is_file()
        assert len(artifact["sha256"]) == 64

    frame = pd.read_csv(tmp_path / record["artifacts"]["csv"]["path"])
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))
    assert frame["method_id"].unique().tolist() == [method_id]


def test_supervised_methods_see_the_target_and_controls_do_not(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    records = {
        method_id: script.run_one(method_id, candidates, target, tmp_path / method_id)
        for method_id in LIGHT_METHOD_IDS
    }
    for method_id in ("iv_woe", "mrmr_mutual_information", "lasso_l1_logistic"):
        assert records[method_id]["supervised"] is True
    for method_id in ("random_k", "full_features"):
        assert records[method_id]["supervised"] is False

    # The two controls never received labels, so their fit-boundary hashes match
    # each other and differ from every supervised hash.
    control_hashes = {records[name]["training_identity_sha256"] for name in ("random_k", "full_features")}
    supervised_hashes = {
        records[name]["training_identity_sha256"]
        for name in ("iv_woe", "mrmr_mutual_information", "lasso_l1_logistic")
    }
    assert len(control_hashes) == 1
    assert len(supervised_hashes) == 1
    assert control_hashes.isdisjoint(supervised_hashes)


def test_integration_writes_nothing_into_protected_roots(script, tmp_path) -> None:
    candidates, target, _ = script.build_fixture()
    script.run_one("iv_woe", candidates, target, tmp_path)

    written = list(tmp_path.rglob("*"))
    assert written
    for path in written:
        resolved = path.resolve()
        assert not resolved.is_relative_to(AUDITED_LEGACY_RESULTS_ROOT)
        assert not resolved.is_relative_to((ROOT / "results").resolve())

    # The legacy bundle is stopped by the repository's own write barrier, which
    # fires before the script's redundant check -- a stronger guarantee, since it
    # protects every caller rather than only this script.
    with pytest.raises(HistoricalResultsWriteError):
        script._assert_scratch_is_isolated(AUDITED_LEGACY_RESULTS_ROOT / "scratch")
    with pytest.raises(SystemExit, match="active results"):
        script._assert_scratch_is_isolated(ROOT / "results/final_experiments/scratch")


def test_published_fixture_evidence_is_consistent_when_present() -> None:
    """If the audit artifact has been generated, it must report a clean pass."""

    path = ROOT / "cleanup/audits/prompt_07_lightweight_selectors/tiny_fixture_results.json"
    if not path.is_file():
        pytest.skip("the tiny-fixture evidence file has not been generated")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["failure_count"] == 0
    assert payload["fixture"]["real_dataset_loaded"] is False
    assert payload["fixture"]["oot_data_loaded"] is False
    assert payload["fixture"]["kind"] == "deterministic_synthetic_only"
    assert payload["isolation"]["inside_active_results_root"] is False
    assert payload["isolation"]["inside_legacy_results_root"] is False
    assert {entry["method_id"] for entry in payload["selectors"]} == set(LIGHT_METHOD_IDS)
    for entry in payload["selectors"]:
        assert entry["status"] == "PASS"
        assert entry["deterministic_on_refit"] is True
        assert entry["serialization_exact"] is True
