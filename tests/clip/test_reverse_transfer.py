from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from credit_risk_fs.clip.checkpointing import save_checkpoint
from credit_risk_fs.clip.exact_duplicates import find_exact_dev_duplicate_pairs
from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss
from credit_risk_fs.clip.model import ClipModelConfig, SemanticStatisticalContrastiveEncoder
from credit_risk_fs.clip.negative_policy import build_negative_policy
from credit_risk_fs.clip.reverse_transfer import (
    PAIRING_POLICY_VERSION,
    aggregate_seed_embeddings,
    build_prediction_frame,
    deterministic_feature_split,
    fixed_candidate_pool,
    frozen_project,
    reconcile_feature_universe,
    validate_checkpoint_manifest,
    validate_prediction_splits,
)
from credit_risk_fs.clip.statistical_preprocessor import StatisticalPreprocessor
from credit_risk_fs.selectors.fixed_rank_then_mrmr import FixedRankThenMRMRSelector


def _reconciled(n: int = 12) -> pd.DataFrame:
    return reconcile_feature_universe(
        pd.DataFrame(
            {
                "feature_name": [f"f{i}" for i in range(n)],
                "source_table": ["application"] * n,
                "semantic_group": ["a" if i % 2 else "b" for i in range(n)],
                "description": [f"feature {i}" for i in range(n)],
                "raw_statistical_evidence_available": [True] * n,
            }
        ),
        dataset="lendingclub_v2",
    )


def test_reconciliation_excludes_missing_views_without_fabrication() -> None:
    evidence = pd.DataFrame(
        {
            "feature_name": ["ok", "no_text", "no_stat"],
            "description": ["present", "", "present"],
            "raw_statistical_evidence_available": [True, True, False],
        }
    )
    result = reconcile_feature_universe(evidence, dataset="lendingclub_v2")
    reasons = dict(zip(result.feature_name, result.exclusion_reason))
    assert reasons["no_text"] == "missing_description"
    assert reasons["no_stat"] == "missing_approved_dev_statistical_evidence"
    assert result.feature_id.is_unique


def test_split_is_deterministic_disjoint_and_identity_group_safe() -> None:
    universe = _reconciled()
    relation = pd.DataFrame(
        {
            "feature_id_a": [universe.feature_id.iloc[0]],
            "feature_id_b": [universe.feature_id.iloc[1]],
            "reason": ["verified_alias"],
        }
    )
    first, manifest = deterministic_feature_split(
        universe,
        dataset="lendingclub_v2",
        seed=42,
        validation_fraction=0.25,
        identity_relations=relation,
    )
    second, _ = deterministic_feature_split(
        universe,
        dataset="lendingclub_v2",
        seed=42,
        validation_fraction=0.25,
        identity_relations=relation,
    )
    pd.testing.assert_frame_equal(first, second)
    aliases = first[first.feature_id.isin(relation.iloc[0, :2])]
    assert aliases.split_assignment.nunique() == 1
    assert set(manifest["train_feature_ids"]).isdisjoint(manifest["validation_feature_ids"])


def test_lendingclub_exact_duplicates_require_exact_values_and_masks() -> None:
    dev = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [1, np.nan, 3],
            "same_summary_not_duplicate": [3.0, np.nan, 1.0],
        }
    )
    pairs = find_exact_dev_duplicate_pairs(
        dev,
        feature_names=list(dev.columns),
        dataset="lendingclub_v2",
        split="train",
    )
    assert set(pairs.anchor_feature_name) == {"a", "b"}
    with pytest.raises(ValueError):
        find_exact_dev_duplicate_pairs(
            dev,
            feature_names=list(dev.columns),
            dataset="homecredit",
            split="oot",
        )


def test_negative_policy_keeps_source_table_and_summary_relations_diagnostic() -> None:
    pairs = pd.DataFrame(
        {
            "feature_name": ["a", "b", "c"],
            "feature_id": ["1", "2", "3"],
            "pair_id": ["p1", "p2", "p3"],
            "split": ["train"] * 3,
            "dataset": ["lendingclub_v2"] * 3,
            "source_table_or_formula": ["same"] * 3,
            "statistical_vector_hash": ["equal"] * 3,
            "base_feature_family": ["one", "two", "three"],
        }
    )
    result = build_negative_policy(
        train_pairs=pairs,
        all_homecredit_pairs=pairs,
        training_dataset="lendingclub_v2",
        verified_aliases=[["a", "b"]],
        min_safe_negative_count=1,
    )
    assert set(result.exclusion_pairs.exclusion_reason) == {"verified_alias"}
    assert result.manifest["diagnostic_relation_counts"]["same_source_table"] > 0
    assert result.manifest["diagnostic_relation_counts"]["diagnostic_statistical_similarity"] > 0


def test_source_preprocessor_blocks_external_refit_and_tracks_source_hash() -> None:
    fields = ["x", "y"]
    source = pd.DataFrame(
        {
            "dataset": ["lendingclub_v2"] * 3,
            "split": ["train"] * 3,
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        }
    )
    external = source.assign(dataset="homecredit")
    preprocessor = StatisticalPreprocessor(
        fields, fit_dataset="lendingclub_v2", fit_split="train"
    ).fit(source)
    before = preprocessor.preprocessor_hash_
    transformed = preprocessor.transform(external)
    assert np.isfinite(transformed.to_numpy()).all()
    assert preprocessor.preprocessor_hash_ == before
    with pytest.raises(ValueError, match="source-only"):
        preprocessor.fit(external)


def test_projection_is_frozen_and_preserves_provenance() -> None:
    config = ClipModelConfig(4, 3, shared_embedding_dim=2, dropout=0)
    model = SemanticStatisticalContrastiveEncoder(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = pd.DataFrame(
        {
            "feature_id": ["h1", "h2"],
            "feature_name": ["a", "b"],
            "dataset": ["homecredit"] * 2,
        }
    )
    embeddings, scores = frozen_project(
        model=model,
        features=features,
        text_values=np.ones((2, 4), dtype="float32"),
        statistical_values=np.ones((2, 3), dtype="float32"),
        source_dataset="lendingclub_v2",
        external_dataset="homecredit",
        checkpoint_hash="checkpoint",
        anchor=np.array([1.0, 0.0]),
        anchor_hash="anchor",
        preprocessor_hash="preprocessor",
    )
    assert embeddings.feature_id.tolist() == ["h1", "h2"]
    assert set(scores.checkpoint_hash) == {"checkpoint"}
    model.text_projection.net[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        frozen_project(
            model=model,
            features=features,
            text_values=np.ones((2, 4)),
            statistical_values=np.ones((2, 3)),
            source_dataset="lendingclub_v2",
            external_dataset="homecredit",
            checkpoint_hash="c",
            anchor=np.array([1.0, 0.0]),
            anchor_hash="a",
            preprocessor_hash="p",
        )


def test_consensus_and_candidate_pool_are_fixed() -> None:
    frames = {}
    for seed in [11, 22, 33, 44, 55]:
        frames[seed] = pd.DataFrame(
            {
                "feature_id": ["a", "b"],
                "feature_name": ["a", "b"],
                "dataset": ["homecredit"] * 2,
                "joint_0000": [1.0, 0.0],
                "joint_0001": [0.0, 1.0],
                "learned_similarity": [0.8, 0.2],
            }
        )
    consensus, manifest = aggregate_seed_embeddings(
        frames, seed_list=[11, 22, 33, 44, 55], reference_seed=11
    )
    assert manifest["reference_seed"] == 11
    ranking = consensus.rename(columns={"consensus_rank": "consensus_clip_rank"})
    pool = fixed_candidate_pool(ranking, model="lr", pool_size=2, final_budget=1)
    assert len(pool) == 2
    assert pool.candidate_pool_frozen_before_mrmr.all()


def test_prediction_provenance_and_split_overlap_boundary() -> None:
    kwargs = {
        "dataset": "homecredit",
        "run_id": "run",
        "model": "lr",
        "data_manifest_hash": "data",
        "configuration_hash": "config",
        "threshold": 0.5,
    }
    dev = build_prediction_frame(
        stable_row_ids=["d1", "d2"],
        target=[0, 1],
        probability=[0.1, 0.9],
        split="dev",
        **kwargs,
    )
    oot = build_prediction_frame(
        stable_row_ids=["o1", "o2"],
        target=[0, 1],
        probability=[0.2, 0.8],
        split="oot",
        **kwargs,
    )
    validate_prediction_splits(dev, oot)
    assert set(dev.pairing_policy_version) == {PAIRING_POLICY_VERSION}
    with pytest.raises(ValueError, match="overlap"):
        validate_prediction_splits(dev, oot.assign(stable_row_id=["d1", "o2"]))


def test_checkpoint_metadata_rejects_wrong_source_and_old_policy() -> None:
    manifest = {
        "source_dataset": "homecredit",
        "pairing_policy_version": "old",
        "seed": 11,
    }
    with pytest.raises(ValueError, match="source_dataset"):
        validate_checkpoint_manifest(
            manifest,
            expected={"source_dataset": "lendingclub_v2", "seed": 11},
        )


def test_mrmr_persists_raw_encoded_widths_and_source_lineage(
    tmp_path, monkeypatch
) -> None:
    class FakeMRMR:
        def __init__(self, **kwargs):
            self.selected_features_ = []

        def fit(self, X, y):
            self.selected_features_ = [str(X.columns[0])]
            return self

    monkeypatch.setattr(
        "credit_risk_fs.selectors.fixed_rank_then_mrmr.MRMR", FakeMRMR
    )
    ranking_path = tmp_path / "ranking.csv"
    pd.DataFrame(
        {
            "feature_name": ["raw_a", "raw_b"],
            "consensus_clip_rank": [1, 2],
        }
    ).to_csv(ranking_path, index=False)
    selector = FixedRankThenMRMRSelector(
        ranking_path=str(ranking_path),
        feature_budget=1,
        screening_pool_size=2,
        selector_label="reverse",
    )
    selector.set_artifact_dir(tmp_path / "artifacts")
    raw = pd.DataFrame({"raw_a": [1, 2], "raw_b": [3, 4]})
    selector.fit(raw, pd.Series([0, 1]))
    encoded = pd.DataFrame(
        {"raw_a_one": [1, 0], "raw_a_two": [0, 1], "raw_b": [3, 4]}
    )
    selector.fit_postprocess(encoded, pd.Series([0, 1]))
    widths = pd.read_csv(tmp_path / "artifacts" / "reverse_mrmr_widths.csv").iloc[0]
    lineage = pd.read_csv(
        tmp_path / "artifacts" / "reverse_source_to_model_lineage.csv"
    )
    assert widths["raw_source_feature_count"] == 2
    assert widths["post_preprocessing_mrmr_column_count"] == 3
    assert set(lineage["source_feature"]) == {"raw_a", "raw_b"}


def test_tiny_synthetic_source_training_and_frozen_external_smoke(tmp_path) -> None:
    torch.manual_seed(11)
    config = ClipModelConfig(
        text_input_dim=4,
        statistical_input_dim=3,
        text_hidden_dim=4,
        statistical_hidden_dim=4,
        shared_embedding_dim=2,
        dropout=0,
    )
    model = SemanticStatisticalContrastiveEncoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    source_text = torch.randn(10, 4)
    source_stat = torch.randn(10, 3)
    text_projection, stat_projection = model(source_text, source_stat)
    loss = symmetric_masked_contrastive_loss(
        text_projection,
        stat_projection,
        temperature=model.temperature(),
        false_negative_mask=torch.zeros((10, 10), dtype=torch.bool),
    ).loss
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_manifest = tmp_path / "checkpoint.json"
    saved = save_checkpoint(
        model=model,
        path=checkpoint,
        manifest_path=checkpoint_manifest,
        seed=11,
        epoch=1,
        validation_metric="synthetic_validation_loss",
        validation_value=float(loss.detach()),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        upstream_hashes={
            "configuration_hash": "config",
            "data_manifest_hash": "data",
            "statistical_preprocessor_hash": "preprocessor",
            "source_anchor_hash": "anchor",
        },
        git_commit="synthetic",
        statistical_view_scope="synthetic_target_free",
        extra={
            "source_dataset": "lendingclub_v2",
            "external_dataset": "homecredit",
            "pairing_policy_version": PAIRING_POLICY_VERSION,
        },
    )
    validate_checkpoint_manifest(
        saved,
        expected={
            "source_dataset": "lendingclub_v2",
            "pairing_policy_version": PAIRING_POLICY_VERSION,
            "seed": 11,
        },
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    external = pd.DataFrame(
        {
            "feature_id": ["h1", "h2"],
            "feature_name": ["x", "y"],
            "dataset": ["homecredit"] * 2,
        }
    )
    embeddings, _ = frozen_project(
        model=model,
        features=external,
        text_values=np.random.default_rng(1).normal(size=(2, 4)),
        statistical_values=np.random.default_rng(2).normal(size=(2, 3)),
        source_dataset="lendingclub_v2",
        external_dataset="homecredit",
        checkpoint_hash=saved["checkpoint_sha256"],
        anchor=np.array([1.0, 0.0]),
        anchor_hash="anchor",
        preprocessor_hash="preprocessor",
    )
    assert len(embeddings) == 2
