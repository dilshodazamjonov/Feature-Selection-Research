from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.text_baseline import build_text_baseline, load_text_baseline_config
from credit_risk_fs.clip.text_encoder import MockFrozenTextEncoder


def test_dry_run_performs_no_model_loading(tmp_path):
    config = load_text_baseline_config()
    config = config.__class__(**{**config.__dict__, "output_dir": tmp_path})

    result = build_text_baseline(config=config, dry_run=True)

    assert result.summary["encoder_loaded"] is False
    assert result.summary["model_trained"] is False
    assert result.summary["expected_embedding_count"] == {"homecredit": 436, "lendingclub_v2": 576}


def test_text_baseline_dry_run_does_not_modify_full_run_artifacts():
    config = load_text_baseline_config()
    full_run_files = [
        config.output_dir / "homecredit_feature_text.csv",
        config.output_dir / "lendingclub_v2_feature_text.csv",
        config.output_dir / "homecredit_group_split.csv",
        config.output_dir / "group_split_audit.json",
        config.output_dir / "feature_family_audit.csv",
        config.output_dir / "feature_family_audit.json",
        config.output_dir / "homecredit_text_embeddings.parquet",
        config.output_dir / "lendingclub_v2_text_embeddings.parquet",
        config.output_dir / "embedding_cache_manifest.json",
        config.output_dir / "text_embedding_audit.json",
        config.output_dir / "homecredit_text_only_ranking.csv",
        config.output_dir / "lendingclub_v2_text_only_ranking.csv",
        config.output_dir / "homecredit_anchor_features.csv",
        config.output_dir / "text_anchor_manifest.json",
        config.output_dir / "text_baseline_summary.json",
    ]
    before = {path: _sha256(path) for path in full_run_files if path.exists()}

    result = build_text_baseline(config=config, dry_run=True)

    after = {path: _sha256(path) for path in full_run_files if path.exists()}
    assert before == after
    assert all("dry_run" in str(path) for path in result.output_paths.values())
    assert (config.output_dir / "dry_run" / "text_baseline_dry_run_summary.json").exists()
    assert (config.output_dir / "dry_run" / "text_dry_run_audit.json").exists()


def test_text_baseline_with_mock_encoder_uses_homecredit_anchor_unchanged(tmp_path, monkeypatch):
    config = load_text_baseline_config()
    config = config.__class__(**{**config.__dict__, "output_dir": tmp_path})

    def fake_save(frame: pd.DataFrame, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out.with_suffix(".csv"), index=False)
        return out

    monkeypatch.setattr("credit_risk_fs.clip.text_baseline.save_embedding_frame", fake_save)
    result = build_text_baseline(config=config, dry_run=False, encoder=MockFrozenTextEncoder())

    home_rank = pd.read_csv(result.output_paths["homecredit_text_only_ranking"])
    lc_rank = pd.read_csv(result.output_paths["lendingclub_v2_text_only_ranking"])
    anchors = pd.read_csv(result.output_paths["homecredit_anchor_features"])

    assert set(anchors["dataset"]) == {"homecredit"}
    assert home_rank["is_anchor_feature"].any()
    assert not lc_rank["is_anchor_feature"].any()
    assert set(lc_rank["dataset"]) == {"lendingclub_v2"}
    assert result.summary["model_trained"] is False
    assert result.summary["contrastive_pairs_created"] is False


def test_legacy_lendingclub_is_rejected(tmp_path):
    config = load_text_baseline_config()
    bad = config.__class__(**{**config.__dict__, "external_validation_dataset": "lendingclub", "output_dir": tmp_path})

    try:
        build_text_baseline(config=bad, dry_run=True)
    except RuntimeError as exc:
        assert "lendingclub_v2" in str(exc) or "legacy" in str(exc)
    else:
        raise AssertionError("legacy LendingClub config should fail")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
