"""Prompt 6 provenance and preservation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from credit_risk_fs.analysis.voting_inference.config import (
    AuthenticationError,
    authenticate_frozen_inputs,
    load_analysis_config,
)
from credit_risk_fs.analysis.voting_inference.inventory import (
    discover_runs,
    parse_run_id,
)
from credit_risk_fs.experiments.result_paths import (
    AUDITED_LEGACY_RESULTS_ROOT,
    HistoricalResultsWriteError,
    reject_historical_write,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_CONFIG = "configs/analysis/cross_dataset_voting_inference_v1.yaml"
PACKAGE_ROOT = ROOT / "results/final_experiments/cross_dataset_voting_inference_v1"


def _package_is_published() -> bool:
    """True only for a package whose own status reports a completed PASS.

    An interrupted or blocked run leaves partial tables and a non-PASS status.
    Those must skip the published-package assertions rather than fail them: the
    package has not been published, so there is nothing to contradict.
    """

    status_path = PACKAGE_ROOT / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return str(status.get("status")) == "PASS"


@pytest.fixture(scope="module")
def config():
    return load_analysis_config(ROOT, config_path=ANALYSIS_CONFIG)


def test_frozen_inputs_authenticate(config) -> None:
    report = authenticate_frozen_inputs(config)
    assert report["status"] == "PASS", report["failures"]
    pinned = [entry for entry in report["inputs"] if entry["expected_sha256"]]
    assert len(pinned) >= 5
    assert all(entry["status"] == "hash_match" for entry in pinned)


def test_one_run_id_maps_to_one_run(config) -> None:
    runs = discover_runs(config)
    assert len(runs) == int(config.expected["total_runs"])
    assert len({run.run_id for run in runs}) == len(runs)
    assert len({run.directory for run in runs}) == len(runs)
    for run in runs:
        facets = parse_run_id(run.run_id)
        assert run.dataset == facets["dataset"]
        assert run.model == facets["model"]
        assert run.configuration == facets["configuration"]
        assert run.candidate_pool_budget == facets["candidate_pool_budget"]


def test_manifest_and_prediction_hashes_match_the_files_on_disk(config) -> None:
    from credit_risk_fs.utils.hashing import sha256_file

    for run in discover_runs(config):
        artifacts = run.manifest["artifacts"]
        for name in ("predictions_dev", "predictions_oot", "selected_features"):
            entry = artifacts[name]
            path = run.directory / str(entry["path"])
            assert path.is_file(), f"{run.run_id}: {name} missing"
            assert sha256_file(path) == entry["sha256"], f"{run.run_id}: {name} hash"


def test_analysis_output_roots_never_enter_the_legacy_bundle_or_a_run(config) -> None:
    for root in (config.package_root, config.audit_root, config.figures_root):
        assert not str(root).startswith(str(AUDITED_LEGACY_RESULTS_ROOT))
        assert reject_historical_write(root) == root.resolve()
        for run in discover_runs(config):
            assert not root.resolve().is_relative_to(run.directory.resolve())


def test_the_write_barrier_still_rejects_the_legacy_root() -> None:
    with pytest.raises(HistoricalResultsWriteError):
        reject_historical_write(AUDITED_LEGACY_RESULTS_ROOT / "anything.csv")


def test_authenticated_kuncheva_universe_matches_every_frozen_source(config) -> None:
    declared = {
        dataset: int(size)
        for dataset, size in config.metric_definitions["kuncheva"]["universe_size"].items()
    }
    protocol = yaml.safe_load(
        (ROOT / "configs/protocols/cross_dataset_rank_voting_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert declared["homecredit"] == int(protocol["candidate_universe"]["homecredit_size"])
    assert declared["lendingclub_v2"] == int(
        protocol["candidate_universe"]["lendingclub_v2_size"]
    )
    matrix = yaml.safe_load(
        (ROOT / "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    for dataset, size in declared.items():
        assert size == int(matrix["datasets"][dataset]["candidate_universe_size"])
    assert declared == {
        dataset: int(size)
        for dataset, size in config.expected["candidate_universe_size"].items()
    }


def test_saved_fold_rankings_declare_the_authenticated_universe(config) -> None:
    from credit_risk_fs.analysis.voting_inference.inventory import (
        fold_candidate_universe_counts,
    )

    for run in discover_runs(config):
        expected = config.dataset_universe_size(run.dataset)
        for fold in range(1, int(config.expected["dev_folds_per_run"]) + 1):
            declared = fold_candidate_universe_counts(run, fold)
            assert declared == {expected}, f"{run.run_id} fold {fold}: {declared}"


def test_analysis_config_rejects_an_unexpected_schema_version(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / ANALYSIS_CONFIG).read_text(encoding="utf-8"))
    payload["schema_version"] = "something_else"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(AuthenticationError, match="schema_version"):
        load_analysis_config(ROOT, config_path=path)


@pytest.mark.skipif(
    not (PACKAGE_ROOT / "prediction_inventory.csv").is_file(),
    reason="the Prompt 6 package has not been built in this checkout",
)
def test_inventoried_predictions_match_the_frozen_row_contract() -> None:
    contract = json.loads(
        (ROOT / "configs/protocols/row_alignment_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )["datasets"]
    inventory = pd.read_csv(PACKAGE_ROOT / "prediction_inventory.csv")
    oot = inventory.loc[inventory["split"] == "OOT"]
    assert len(oot) == 16
    for row in oot.itertuples(index=False):
        expected = contract[row.dataset]["oot"]
        assert row.row_count == expected["row_count"], row.run_id
        assert row.unique_identity_count == expected["unique_id_count"], row.run_id
        assert row.positive_count == expected["positive_count"], row.run_id
    dev = inventory.loc[inventory["split"] == "DEV_OOF"]
    assert len(dev) == 16
    for dataset, group in dev.groupby("dataset"):
        # DEV out-of-fold coverage is a documented subset of the DEV split, so it
        # must be smaller than the contract count yet identical across runs.
        assert group["row_count"].nunique() == 1, dataset
        assert group["positive_count"].nunique() == 1, dataset
        assert int(group["row_count"].iloc[0]) < contract[dataset]["dev"]["row_count"]
        assert int(group["positive_count"].iloc[0]) < contract[dataset]["dev"]["positive_count"]


@pytest.mark.skipif(
    not _package_is_published(),
    reason="no completed Prompt 6 package (status.json is absent or not PASS)",
)
def test_published_package_is_internally_consistent() -> None:
    status = json.loads((PACKAGE_ROOT / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "PASS"
    assert status["comparison_count"] == 12
    assert status["holm_family_count"] == 4
    assert status["bootstrap_repetitions"] == 2000

    specification = json.loads(
        (PACKAGE_ROOT / "predeclared_comparison_family.json").read_text(encoding="utf-8")
    )
    assert specification["constructed_after_viewing_oot_results"] is False
    predeclared = {
        (entry["multiplicity_family"], entry["comparison_label"])
        for entry in specification["comparisons"]
    }
    final = pd.read_csv(PACKAGE_ROOT / "paired_inference_final.csv")
    executed = set(zip(final["family"], final["comparison_label"], strict=True))
    assert predeclared == executed
    assert len(final) == len(predeclared)

    alignment = pd.read_csv(PACKAGE_ROOT / "alignment_audit.csv")
    assert (alignment["decision"] == "aligned").all()
    assert alignment["target_mismatch_count"].fillna(-1).eq(0).all()

    metrics = pd.read_csv(PACKAGE_ROOT / "run_level_metrics.csv")
    assert len(metrics) == 32
    assert ((metrics["gini"] - (2 * metrics["auc"] - 1)).abs() < 1e-12).all()

    independent = pd.read_csv(PACKAGE_ROOT / "independent_recalculation_audit.csv")
    assert independent["pass"].all()

    provenance = pd.read_csv(PACKAGE_ROOT / "provenance_audit.csv")
    assert provenance["passed"].all()

    folds = pd.read_csv(PACKAGE_ROOT / "fold_selection_inventory.csv")
    assert int(folds["present"].sum()) == 80

    manifest = json.loads(
        (PACKAGE_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["generated_file_count"] == len(manifest["generated_files"])
    for entry in manifest["generated_files"]:
        assert not entry["path"].startswith("results/runs/")
