from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cleanup.tools.validate_repository_state import main as validator_main
from cleanup.tools.validate_repository_state import validate_active_results
from credit_risk_fs.experiments.runner import main as matrix_runner_main
from credit_risk_fs.experiments.result_paths import (
    RESULT_SUBDIRECTORIES,
    RUN_INDEX_COLUMNS,
    RunDirectoryCollisionError,
    HistoricalResultsWriteError,
    LegacyEvidenceProfileError,
    LegacyResultsConfigurationError,
    append_run_index_row,
    build_run_id,
    create_run_directory,
    ensure_within_directory,
    initialize_results_layout,
    reject_historical_write,
    repository_relative_path,
    resolve_legacy_artifact,
    validate_results_root_separation,
    validate_legacy_evidence_profile,
)
from credit_risk_fs.experiments.tracking import (
    STANDARD_ARTIFACTS,
    write_resource_usage,
)


def test_initialize_results_layout_from_empty_repository(tmp_path):
    results_root = initialize_results_layout(tmp_path)

    assert results_root == (tmp_path / "results").resolve()
    assert (results_root / "README.md").is_file()
    assert all((results_root / name).is_dir() for name in RESULT_SUBDIRECTORIES)
    with (results_root / "run_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        assert next(csv.reader(file)) == list(RUN_INDEX_COLUMNS)


def test_initialization_is_idempotent_and_preserves_index_data(tmp_path):
    results_root = initialize_results_layout(tmp_path)
    append_run_index_row(
        results_root,
        {"run_id": "existing", "dataset": "homecredit", "status": "failed"},
    )
    before = (results_root / "run_index.csv").read_bytes()

    assert initialize_results_layout(tmp_path) == results_root
    assert (results_root / "run_index.csv").read_bytes() == before


def test_initializer_does_not_replace_an_existing_run_index(tmp_path):
    results_root = tmp_path / "results"
    results_root.mkdir()
    index_path = results_root / "run_index.csv"
    index_path.write_text("custom,data\nkeep,me\n", encoding="utf-8")

    initialize_results_layout(tmp_path)

    assert index_path.read_text(encoding="utf-8") == "custom,data\nkeep,me\n"


def test_run_id_sanitization_and_collision_handling(tmp_path):
    results_root = initialize_results_layout(tmp_path)
    run_id = build_run_id(
        selector="Voting / SHAP",
        model="CatBoost v2",
        run_date="2026-07-21",
    )
    assert run_id == "2026-07-21_voting_shap_catboost_v2"

    first = create_run_directory(
        results_root,
        dataset="Home Credit / 2026",
        run_id=run_id,
    )
    assert first == (
        results_root
        / "runs"
        / "home_credit_2026"
        / "2026-07-21_voting_shap_catboost_v2"
    )
    with pytest.raises(RunDirectoryCollisionError, match="already exists"):
        create_run_directory(
            results_root,
            dataset="Home Credit / 2026",
            run_id=run_id,
        )
    suffixed = create_run_directory(
        results_root,
        dataset="Home Credit / 2026",
        run_id=run_id,
        collision_policy="suffix",
    )
    assert suffixed.name == f"{run_id}_02"


def test_results_paths_do_not_depend_on_current_working_directory(
    tmp_path, monkeypatch
):
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    results_root = initialize_results_layout(
        repository_root,
        results_root="custom/results",
    )

    assert results_root == (repository_root / "custom" / "results").resolve()


def test_paths_cannot_escape_the_configured_results_root(tmp_path):
    results_root = initialize_results_layout(tmp_path)
    with pytest.raises(ValueError, match="escapes configured directory"):
        ensure_within_directory("../outside", results_root)

    run_dir = create_run_directory(
        results_root,
        dataset="../../outside",
        run_id="../../../run",
    )
    assert run_dir.is_relative_to(results_root / "runs")


def test_historical_write_barrier_rejects_direct_and_traversal_paths(
    tmp_path, monkeypatch
):
    legacy = tmp_path / "legacy" / "results"
    legacy.mkdir(parents=True)
    monkeypatch.setenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", str(legacy))

    with pytest.raises(HistoricalResultsWriteError, match="immutable historical"):
        reject_historical_write(legacy / "artifact.csv")
    with pytest.raises(HistoricalResultsWriteError, match="immutable historical"):
        reject_historical_write(tmp_path / "active" / ".." / "legacy" / "results" / "x.csv")


def test_active_and_historical_roots_must_not_overlap(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with pytest.raises(LegacyResultsConfigurationError, match="overlap"):
        validate_results_root_separation(legacy / "active", legacy)
    nested = legacy / "nested"
    nested.mkdir()
    with pytest.raises(LegacyResultsConfigurationError, match="overlap"):
        validate_results_root_separation(legacy, nested)


def test_independent_active_root_and_read_only_legacy_discovery(tmp_path):
    active = tmp_path / "active"
    legacy = tmp_path / "historical"
    active.mkdir()
    legacy.mkdir()
    artifact = legacy / "clip" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")

    assert validate_results_root_separation(active, legacy) == (
        active.resolve(),
        legacy.resolve(),
    )
    assert resolve_legacy_artifact(
        "results\\clip/manifest.json", legacy_root=legacy
    ) == artifact.resolve()


def _write_complete_evidence_profile(root: Path) -> Path:
    groups = {}
    for index, group in enumerate(
        (
            "clip_readiness",
            "clip_text_baseline",
            "clip_v2_statistical_view",
            "corrected_homecredit_clip_contrastive",
            "corrected_homecredit_clip_training",
        )
    ):
        artifact = root / f"evidence_{index}.json"
        artifact.write_text(f'{{"group":"{group}"}}\n', encoding="utf-8")
        import hashlib

        groups[group] = [
            {"path": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        ]
    profile = root / "legacy_evidence_profile.json"
    profile.write_text(
        json.dumps(
            {"profile": "clip_complete_v1", "status": "complete", "artifact_groups": groups}
        ),
        encoding="utf-8",
    )
    return profile


def test_legacy_evidence_profile_absence_is_unavailable(tmp_path):
    assert validate_legacy_evidence_profile(tmp_path) is None


def test_declared_complete_legacy_evidence_profile_is_hash_validated(tmp_path):
    _write_complete_evidence_profile(tmp_path)
    validated = validate_legacy_evidence_profile(tmp_path)
    assert validated is not None
    assert validated["verified_artifact_count"] == 5

    (tmp_path / "evidence_0.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(LegacyEvidenceProfileError, match="hash mismatch"):
        validate_legacy_evidence_profile(tmp_path)


def test_explicitly_incomplete_evidence_profile_is_unavailable(tmp_path):
    (tmp_path / "legacy_evidence_profile.json").write_text(
        json.dumps({"profile": "clip_complete_v1", "status": "incomplete"}),
        encoding="utf-8",
    )
    assert validate_legacy_evidence_profile(tmp_path) is None


def test_environment_declaration_without_manifest_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_RISK_LEGACY_EVIDENCE_PROFILE", "clip_complete_v1")
    with pytest.raises(LegacyEvidenceProfileError, match="profile.*missing"):
        validate_legacy_evidence_profile(tmp_path)


def test_resource_usage_contract_contains_required_measurements(tmp_path):
    payload = write_resource_usage(
        tmp_path,
        {
            "feature_selection_time_sec": 1.0,
            "training_time_sec": 2.0,
            "prediction_time_sec": 3.0,
            "evaluation_time_sec": 4.0,
            "total_runtime_seconds": 10.0,
        },
    )

    assert payload["timings_seconds"] == {
        "feature_selection": 1.0,
        "model_training": 2.0,
        "prediction": 3.0,
        "evaluation": 4.0,
        "total": 10.0,
    }
    assert {"peak_ram_mb", "peak_gpu_mb"}.issubset(payload)
    assert (tmp_path / "resource_usage.json").is_file()


def test_matrix_dry_run_initializes_layout_without_creating_run_directories(
    tmp_path,
):
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(
        "dataset_name: homecredit\nresults_dir: results\n",
        encoding="utf-8",
    )

    matrix_runner_main(
        [
            "--config",
            str(config_path),
            "--repository-root",
            str(tmp_path),
            "--models",
            "lr",
            "--dry-run",
        ]
    )

    results_root = tmp_path / "results"
    assert (results_root / "run_index.csv").is_file()
    assert not (results_root / "runs" / "homecredit").exists()
    assert (results_root / "comparisons" / "homecredit_matrix_runs.csv").is_file()


def _register_valid_completed_run(repository_root: Path) -> tuple[Path, Path]:
    results_root = initialize_results_layout(repository_root)
    run_dir = create_run_directory(
        results_root,
        dataset="homecredit",
        run_id="2026-07-21_mrmr_lr",
    )
    config_path = run_dir / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    artifacts = {
        name: {
            "applicable": relative in {"config.json", "manifest.json"},
            "path": relative,
            "present": relative in {"config.json", "manifest.json"},
        }
        for name, relative in STANDARD_ARTIFACTS.items()
    }
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "completed",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    append_run_index_row(
        results_root,
        {
            "run_id": run_dir.name,
            "dataset": "homecredit",
            "selector": "mrmr",
            "model": "lr",
            "split_protocol": "test",
            "seed": 42,
            "status": "completed",
            "run_directory": repository_relative_path(
                run_dir, repository_root
            ),
            "config_path": repository_relative_path(
                config_path, repository_root
            ),
            "manifest_path": repository_relative_path(
                manifest_path, repository_root
            ),
        },
    )
    return results_root, manifest_path


def test_validator_accepts_active_layout_without_historical_registry(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    _register_valid_completed_run(tmp_path)

    assert validate_active_results(tmp_path.resolve())["registered_runs"] == 1
    assert validator_main(["--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"status": "external_optional"' in output
    assert not (tmp_path / "results" / "research_summary").exists()


def test_validator_fails_when_manifest_references_missing_artifact(tmp_path):
    _, manifest_path = _register_valid_completed_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"] = {
        "applicable": True,
        "path": "metrics.csv",
        "present": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing artifact metrics"):
        validate_active_results(tmp_path.resolve())


def test_validator_rejects_index_path_escape(tmp_path):
    results_root, _ = _register_valid_completed_run(tmp_path)
    index_path = results_root / "run_index.csv"
    rows = list(csv.DictReader(index_path.read_text(encoding="utf-8").splitlines()))
    rows[0]["run_directory"] = "../outside"
    with index_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="path traversal"):
        validate_active_results(tmp_path.resolve())
