from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip_final_comparison.ablations import train_grouped_ablation_representations
from credit_risk_fs.clip_final_comparison.constants import ABLATIONS, CLIP_V2_SEEDS
from credit_risk_fs.clip_final_comparison.seeds import resolve_clip_v2_seed_artifacts
from credit_risk_fs.utils.hashing import sha256_file, sha256_text

from tests.clip_final_comparison.test_stage_orchestration_synthetic import (
    _isolate,
    _synthetic_representation_views,
)


def _write_fake_seed_training_root(root: Path) -> Path:
    training_root = root / "fake_clip_v2_training"
    for seed in CLIP_V2_SEEDS:
        seed_dir = training_root / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "best_checkpoint.pt"
        checkpoint.write_text(f"synthetic checkpoint {seed}", encoding="utf-8")
        manifest = {
            "seed": seed,
            "checkpoint_hash": sha256_file(checkpoint),
            "training_config_hash": sha256_text(f"training:{seed}"),
            "text_embedding_hash": sha256_text(f"text:{seed}"),
            "statistical_schema_hash": sha256_text(f"schema:{seed}"),
            "statistical_preprocessor_hash": sha256_text(f"preprocessor:{seed}"),
            "anchor_path": f"synthetic/anchor_{seed}.json",
            "anchor_hash": sha256_text(f"anchor:{seed}"),
            "collapse_status": "not_collapsed",
        }
        (seed_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (seed_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({"status": "complete_valid"}), encoding="utf-8")
    return training_root


def _core_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "homecredit_lr_clip_v2_5x",
                "dataset": "homecredit",
                "model": "lr",
                "screening_method": "variance",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
                "random_seed": None,
            },
            {
                "run_id": "homecredit_lr_random_5x_seed101",
                "dataset": "homecredit",
                "model": "lr",
                "screening_method": "random",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
                "random_seed": 101,
            },
        ]
    )


def _seed_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": f"{dataset}_lr_clip_v2_5x_seed{seed}",
                "dataset": dataset,
                "model": "lr",
                "screening_method": "clip_v2",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
                "random_seed": seed,
            }
            for dataset, seed in [("homecredit", 11), ("lendingclub_v2", 22)]
        ]
    )


def _ablation_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": f"homecredit_lr_{ablation}_5x",
                "dataset": "homecredit",
                "model": "lr",
                "screening_method": "clip_v2",
                "ablation": ablation,
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
            }
            for ablation in ABLATIONS
        ]
    )


def test_complete_synthetic_closure_pipeline_passes_then_missing_artifact_fails(monkeypatch, tmp_path):
    runner, root = _isolate(monkeypatch, tmp_path)
    fake_training_root = _write_fake_seed_training_root(tmp_path)
    text_view, statistical_view = _synthetic_representation_views()

    monkeypatch.setattr(runner, "build_core_experiment_plan", _core_plan)
    monkeypatch.setattr(runner, "build_seed_downstream_plan", _seed_plan)
    monkeypatch.setattr(runner, "build_ablation_plan", _ablation_plan)
    monkeypatch.setattr(runner, "EXPECTED_CORE_RUNS", 2)
    monkeypatch.setattr(runner, "EXPECTED_SEED_DOWNSTREAM_RUNS", 2)
    monkeypatch.setattr(runner, "EXPECTED_ABLATION_DOWNSTREAM_RUNS", 7)
    monkeypatch.setattr(runner, "TEMPORAL_METHODS", ("variance",))
    monkeypatch.setattr(runner, "build_source_experiment_manifest", lambda: {"status": "synthetic_valid", "source_artifact_hashes": {}})
    monkeypatch.setattr(
        runner,
        "resolve_clip_v2_seed_artifacts",
        lambda output_root: resolve_clip_v2_seed_artifacts(output_root, training_root=fake_training_root),
    )
    monkeypatch.setattr(
        runner,
        "train_grouped_ablation_representations",
        lambda output_root: train_grouped_ablation_representations(
            output_root=output_root,
            text_view=text_view,
            statistical_view=statistical_view,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_model_kwargs",
        lambda dataset, model: {"random_state": 7, "iterations": 5, "verbose": False} if model == "catboost" else {"random_state": 7},
    )

    class Completed:
        returncode = 0
        stdout = "synthetic tests passed"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())

    runner.initialize_clean_state()
    assert runner.run_selected_stages(list(runner.STAGES)) == 0

    status = runner.status_payload()
    assert all(status["stages"][stage]["status"] == "complete_valid" for stage in runner.STAGES)
    assert status["completed_scientific_runs"] == 2
    assert status["completed_seed_runs"] == 2
    assert status["completed_ablation_runs"] == 7
    assert pd.read_csv(root / "manifests/clip_v2_seed_score_caches.csv").shape[0] == 10
    assert pd.read_csv(root / "ablations/ablation_score_caches.csv").shape[0] == 14
    assert (root / "audit/final_scientific_audit.json").exists()

    (root / "ablations/training/without_location_scale/selected_checkpoint.json").unlink()
    try:
        runner.final_audit()
    except RuntimeError as exc:
        assert "ablation_training_invalid" in str(exc)
    else:
        raise AssertionError("final audit should fail when a required synthetic ablation artifact is removed")
