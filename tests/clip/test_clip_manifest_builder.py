from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.manifest_builder import build_training_manifest, load_training_manifest_config
from credit_risk_fs.utils.hashing import sha256_file


def test_source_hashes_are_deterministic():
    path = Path("results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv")

    assert sha256_file(path) == sha256_file(path)


def test_build_manifest_outputs_are_deterministic(tmp_path):
    config = load_training_manifest_config()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = build_training_manifest(config=config, output_dir=first_dir, seed=42, dry_run=True)
    second = build_training_manifest(config=config, output_dir=second_dir, seed=42, dry_run=True)

    first_manifest = json.loads(first.output_paths["training_manifest"].read_text(encoding="utf-8"))
    second_manifest = json.loads(second.output_paths["training_manifest"].read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert first_manifest["training_activity"]["model_trained"] is False
    assert first_manifest["training_activity"]["encoder_loaded"] is False
    assert first_manifest["training_activity"]["contrastive_pairs_created"] is False
    assert first_manifest["train_dataset"] == "homecredit"
    assert first_manifest["external_validation_dataset"] == "lendingclub_v2"


def test_manifest_outputs_keep_roles_and_blocked_reasons(tmp_path):
    config = load_training_manifest_config()
    result = build_training_manifest(config=config, output_dir=tmp_path, seed=42, dry_run=True)

    training = pd.read_csv(result.output_paths["training_features"])
    external = pd.read_csv(result.output_paths["external_validation_features"])
    blocked = pd.read_csv(result.output_paths["blocked_features"])
    roles = pd.read_csv(result.output_paths["field_role_manifest"])

    assert set(training["dataset"]) == {"homecredit"}
    assert set(training["dataset_role"]) == {"train"}
    assert set(training["fit_role"]) == {"trainer_fit"}
    assert set(external["dataset"]) == {"lendingclub_v2"}
    assert set(external["dataset_role"]) == {"external_validation"}
    assert set(external["fit_role"]) == {"external_validation_only"}
    assert blocked["block_reason"].fillna("").str.len().gt(0).all()
    assert not any("psi" in column.lower() or "oot" in column.lower() for column in training.columns)
    assert not any("target" in column.lower() for column in training.columns)
    train_input_roles = roles[roles["allowed_in_main_training_input"].astype(bool)]
    assert set(train_input_roles["dataset"]) == {"homecredit"}


def test_output_ordering_is_deterministic(tmp_path):
    config = load_training_manifest_config()
    result = build_training_manifest(config=config, output_dir=tmp_path, seed=42, dry_run=True)
    training = pd.read_csv(result.output_paths["training_features"])
    external = pd.read_csv(result.output_paths["external_validation_features"])

    assert training[["dataset", "feature"]].equals(
        training.sort_values(["dataset", "feature"], kind="mergesort")[["dataset", "feature"]].reset_index(drop=True)
    )
    assert external[["dataset", "feature"]].equals(
        external.sort_values(["dataset", "feature"], kind="mergesort")[["dataset", "feature"]].reset_index(drop=True)
    )
