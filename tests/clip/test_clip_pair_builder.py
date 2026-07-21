from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text

from credit_risk_fs.clip.pair_builder import build_contrastive_data, load_contrastive_data_config


def _config(tmp_path, legacy_config_paths):
    config = load_contrastive_data_config(
        "configs/corrected_homecredit_clip/contrastive_data.yaml"
    )
    return legacy_config_paths(config, output_dir=tmp_path)


def test_positive_pairs_align_same_feature_across_views(tmp_path, legacy_artifact_path):
    base = legacy_artifact_path(
        "results/corrected_homecredit_clip/contrastive_data", required=False
    )
    train = pd.read_parquet(base / "homecredit_train_positive_pairs.parquet")
    validation = pd.read_parquet(base / "homecredit_validation_positive_pairs.parquet")
    external = pd.read_parquet(base / "lendingclub_v2_external_pairs.parquet")

    assert len(train) == 349
    assert len(validation) == 87
    assert len(external) == 576
    assert set(train["split"]) == {"train"}
    assert set(validation["split"]) == {"validation"}
    assert set(external["split"]) == {"external_validation"}
    assert train["allowed_for_training"].all()
    assert validation["allowed_for_validation"].all()
    assert external["allowed_for_external_evaluation"].all()
    assert not external["allowed_for_training"].any()
    assert train["pair_id"].is_unique
    assert validation["pair_id"].is_unique
    assert external["pair_id"].is_unique
    assert train["text_embedding_dimension"].eq(384).all()
    assert train["statistical_vector_dimension"].eq(13).all()


def test_pair_building_and_pair_ids_are_deterministic(tmp_path, legacy_artifact_path):
    train = pd.read_parquet(
        legacy_artifact_path("results/corrected_homecredit_clip/contrastive_data/homecredit_train_positive_pairs.parquet")
    )
    row = train.iloc[0]
    expected = sha256_text(
        "|".join(
            [
                str(row["dataset"]),
                str(row["feature_name"]),
                str(row["split"]),
                str(row["group_key"]),
                str(row["text_hash"]),
                str(row["statistical_vector_hash"]),
            ]
        )
    )
    assert row["pair_id"] == expected
    assert train["pair_id"].tolist() == train.sort_values(["dataset", "feature_name"], kind="mergesort")["pair_id"].tolist()


def test_contrastive_dry_run_does_not_overwrite_full_pair_artifacts(
    tmp_path, legacy_config_paths
):
    config = _config(tmp_path, legacy_config_paths)
    sentinel = tmp_path / "homecredit_train_positive_pairs.parquet"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("sentinel", encoding="utf-8")

    result = build_contrastive_data(config=config, dry_run=True)

    assert sentinel.read_text(encoding="utf-8") == "sentinel"
    assert "dry_run" in str(result.output_paths["contrastive_pair_manifest"])
    assert result.summary["homecredit_train_pair_count"] == 349
    assert result.summary["model_trained"] is False
    assert result.summary["optimizer_created"] is False
    assert result.summary["checkpoint_created"] is False


def test_contrastive_dry_run_keeps_actual_full_run_hashes_unchanged(
    tmp_path, legacy_artifact_path, legacy_config_paths
):
    source_config = legacy_config_paths(load_contrastive_data_config(
        "configs/corrected_homecredit_clip/contrastive_data.yaml"
    ))
    config = source_config.__class__(**{**source_config.__dict__, "output_dir": tmp_path})
    historical_output = legacy_artifact_path(
        "results/corrected_homecredit_clip/contrastive_data", required=False
    )
    full_run_files = [
        historical_output / "homecredit_train_positive_pairs.parquet",
        historical_output / "homecredit_validation_positive_pairs.parquet",
        historical_output / "lendingclub_v2_external_pairs.parquet",
        historical_output / "negative_exclusion_pairs.parquet",
        historical_output / "near_duplicate_text_audit.csv",
        historical_output / "contrastive_tensor_schema.json",
        historical_output / "contrastive_pair_manifest.json",
        historical_output / "split_manifest.json",
        historical_output / "negative_policy_manifest.json",
        historical_output / "negative_candidate_audit.csv",
        historical_output / "near_duplicate_threshold_sensitivity.csv",
        historical_output / "pair_quality_audit.csv",
        historical_output / "pair_quality_audit.json",
    ]
    before = {path: _sha256(path) for path in full_run_files if path.exists()}

    result = build_contrastive_data(config=config, dry_run=True)

    after = {path: _sha256(path) for path in full_run_files if path.exists()}
    assert before == after
    assert all("dry_run" in str(path) for path in result.output_paths.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
