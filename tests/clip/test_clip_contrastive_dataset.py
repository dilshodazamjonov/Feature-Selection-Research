from __future__ import annotations

import numpy as np
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.contrastive_dataset import ContrastiveFeatureDataset


def test_dataset_returns_aligned_tensors_and_no_forbidden_inputs(tmp_path, legacy_artifact_path):
    dataset = ContrastiveFeatureDataset(
        pairs_path=legacy_artifact_path("results/corrected_homecredit_clip/contrastive_data/homecredit_train_positive_pairs.parquet"),
        text_embeddings_path=legacy_artifact_path("results/clip/text_baseline/homecredit_text_embeddings.parquet"),
        statistical_vectors_path=legacy_artifact_path("results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet"),
        mode="train",
    )

    item = dataset[0]

    assert item["text_embedding"].shape == (384,)
    assert item["statistical_vector"].shape == (13,)
    assert np.isfinite(item["text_embedding"]).all()
    assert np.isfinite(item["statistical_vector"]).all()
    assert "stable_core_membership" not in item
    assert "llm_best_rank" not in item
    assert "oot" not in item
    assert "psi" not in item
    assert item["metadata"]["dataset"] == "homecredit"
    assert item["metadata"]["split"] == "train"


def test_dataset_refuses_lendingclub_v2_training_mode(tmp_path, legacy_artifact_path):
    try:
        ContrastiveFeatureDataset(
            pairs_path=legacy_artifact_path("results/corrected_homecredit_clip/contrastive_data/lendingclub_v2_external_pairs.parquet"),
            text_embeddings_path=legacy_artifact_path("results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet"),
            statistical_vectors_path=legacy_artifact_path("results/clip_v2/statistical_view/lendingclub_v2_statistical_vectors.parquet"),
            mode="train",
        )
    except ValueError as exc:
        assert "HomeCredit" in str(exc) or "training" in str(exc)
    else:
        raise AssertionError("LendingClub v2 must not be accepted as train mode")


def test_dataset_hash_verification_fails_on_stale_pair_hash(tmp_path, legacy_artifact_path):
    pairs = pd.read_parquet(
        legacy_artifact_path("results/corrected_homecredit_clip/contrastive_data/homecredit_train_positive_pairs.parquet")
    )
    pairs.loc[0, "text_hash"] = "stale"
    stale_path = tmp_path / "stale_pairs.parquet"
    pairs.to_parquet(stale_path, index=False)
    dataset = ContrastiveFeatureDataset(
        pairs_path=stale_path,
        text_embeddings_path=legacy_artifact_path("results/clip/text_baseline/homecredit_text_embeddings.parquet"),
        statistical_vectors_path=legacy_artifact_path("results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet"),
        mode="train",
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "text hash" in str(exc)
    else:
        raise AssertionError("stale pair hash should fail")
