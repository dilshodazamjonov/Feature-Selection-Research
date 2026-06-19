from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_baseline import build_statistical_baseline, load_statistical_baseline_config


def _config_with_output(tmp_path: Path):
    config = load_statistical_baseline_config()
    return config.__class__(**{**config.__dict__, "output_dir": tmp_path})


def test_statistical_dry_run_does_not_overwrite_full_run_artifacts(tmp_path):
    config = _config_with_output(tmp_path)
    sentinel = tmp_path / "statistical_preprocessor.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("full-run-sentinel", encoding="utf-8")

    result = build_statistical_baseline(config=config, dry_run=True)

    assert sentinel.read_text(encoding="utf-8") == "full-run-sentinel"
    assert "dry_run" in str(result.output_paths["statistical_field_inventory_csv"])
    assert result.summary["expected"]["homecredit_vectors"] == 436
    assert result.summary["expected"]["lendingclub_v2_vectors"] == 576


def test_statistical_dry_run_keeps_actual_full_run_hashes_unchanged():
    config = load_statistical_baseline_config()
    full_run_files = [
        config.output_dir / "statistical_field_inventory.csv",
        config.output_dir / "statistical_field_inventory.json",
        config.output_dir / "statistical_preprocessor.json",
        config.output_dir / "statistical_preprocessor.joblib",
        config.output_dir / "statistical_feature_order.json",
        config.output_dir / "statistical_preprocessing_audit.json",
        config.output_dir / "homecredit_statistical_vectors.parquet",
        config.output_dir / "lendingclub_v2_statistical_vectors.parquet",
        config.output_dir / "homecredit_statistical_anchor_features.csv",
        config.output_dir / "statistical_anchor_manifest.json",
        config.output_dir / "homecredit_statistical_only_ranking.csv",
        config.output_dir / "lendingclub_v2_statistical_only_ranking.csv",
        config.output_dir / "statistical_baseline_summary.json",
    ]
    before = {path: _sha256(path) for path in full_run_files if path.exists()}

    result = build_statistical_baseline(config=config, dry_run=True)

    after = {path: _sha256(path) for path in full_run_files if path.exists()}
    assert before == after
    assert all("dry_run" in str(path) for path in result.output_paths.values())


def test_full_statistical_baseline_uses_homecredit_train_split_anchor(tmp_path):
    config = _config_with_output(tmp_path)

    result = build_statistical_baseline(config=config, dry_run=False)

    summary = result.summary
    assert summary["main_statistical_fields"] == ["missing_rate_dev"]
    assert summary["target_aware_fields_used_in_main"] == []
    assert summary["algorithm_derived_fields_in_main"] == []
    assert summary["homecredit_train_rows"] == 349
    assert summary["homecredit_validation_rows"] == 87
    assert summary["lendingclub_v2_vectors"] == 576
    assert summary["vector_dimension"] == 1
    assert summary["model_trained"] is False
    assert summary["contrastive_pairs_created"] is False

    anchors = pd.read_csv(result.output_paths["homecredit_statistical_anchor_features"])
    home_rank = pd.read_csv(result.output_paths["homecredit_statistical_only_ranking"])
    lc_rank = pd.read_csv(result.output_paths["lendingclub_v2_statistical_only_ranking"])
    home_vectors = pd.read_parquet(result.output_paths["homecredit_statistical_vectors"])
    lc_vectors = pd.read_parquet(result.output_paths["lendingclub_v2_statistical_vectors"])

    assert set(anchors["dataset"]) == {"homecredit"}
    assert set(anchors["split"]) == {"train"}
    assert int(lc_rank["is_anchor_feature"].astype(bool).sum()) == 0
    assert int(home_rank["is_anchor_feature"].astype(bool).sum()) == len(anchors)
    assert len(home_vectors) == len(home_rank) == 436
    assert len(lc_vectors) == len(lc_rank) == 576
    assert np.isfinite(home_vectors[["stat_0000"]].to_numpy()).all()
    assert np.isfinite(lc_vectors[["stat_0000"]].to_numpy()).all()

    preprocessor_state = json.loads(Path(result.output_paths["statistical_preprocessor_json"]).read_text(encoding="utf-8"))
    assert preprocessor_state["fit_dataset"] == "homecredit"
    assert preprocessor_state["fit_split"] == "train"


def test_statistical_rankings_are_deterministic(tmp_path):
    first = _config_with_output(tmp_path / "first")
    second = _config_with_output(tmp_path / "second")

    first_result = build_statistical_baseline(config=first, dry_run=False)
    second_result = build_statistical_baseline(config=second, dry_run=False)

    first_home = pd.read_csv(first_result.output_paths["homecredit_statistical_only_ranking"])
    second_home = pd.read_csv(second_result.output_paths["homecredit_statistical_only_ranking"])
    first_lc = pd.read_csv(first_result.output_paths["lendingclub_v2_statistical_only_ranking"])
    second_lc = pd.read_csv(second_result.output_paths["lendingclub_v2_statistical_only_ranking"])
    pd.testing.assert_frame_equal(first_home, second_home)
    pd.testing.assert_frame_equal(first_lc, second_lc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
