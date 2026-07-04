from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.pair_builder import load_contrastive_data_config
from credit_risk_fs.clip.pair_validation import validate_manifest_boundary, validate_view_frame


def test_duplicate_text_views_fail_validation():
    text = pd.read_parquet("results/clip/text_baseline/homecredit_text_embeddings.parquet")
    stat = pd.read_parquet(
        "results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet"
    )
    duplicated = pd.concat([text, text.head(1)], ignore_index=True)

    errors = validate_view_frame(
        text=duplicated,
        stat=stat,
        dataset="homecredit",
        expected_text_dim=384,
        expected_stat_dim=13,
    )

    assert any("duplicate text" in error for error in errors)


def test_missing_statistical_view_fails_validation():
    text = pd.read_parquet("results/clip/text_baseline/homecredit_text_embeddings.parquet")
    stat = pd.read_parquet(
        "results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet"
    ).iloc[1:].copy()

    errors = validate_view_frame(
        text=text,
        stat=stat,
        dataset="homecredit",
        expected_text_dim=384,
        expected_stat_dim=13,
    )

    assert any("missing view alignment" in error for error in errors)


def test_stale_source_hash_fails_manifest_boundary(tmp_path):
    config = load_contrastive_data_config(
        "configs/corrected_homecredit_clip/contrastive_data.yaml"
    )
    manifest = json.loads(Path("results/clip/dry_run/training_manifest.json").read_text(encoding="utf-8"))
    source_hashes = json.loads(Path("results/clip/dry_run/source_hashes.json").read_text(encoding="utf-8"))
    source_hashes["homecredit"]["sha256"] = "stale"

    errors = validate_manifest_boundary(manifest=manifest, source_hashes=source_hashes, config=config)

    assert any("source hash mismatch" in error for error in errors)


def test_legacy_lendingclub_config_is_rejected():
    config = load_contrastive_data_config(
        "configs/corrected_homecredit_clip/contrastive_data.yaml"
    )
    bad = config.__class__(**{**config.__dict__, "external_validation_dataset": "lendingclub"})

    from credit_risk_fs.clip.pair_validation import validate_contrastive_config

    errors = validate_contrastive_config(bad)
    assert errors
    assert any("lendingclub_v2" in error or "legacy" in error for error in errors)
