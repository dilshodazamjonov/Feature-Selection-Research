from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

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
    RegistrySchema,
    aggregate_seed_embeddings,
    align_external_feature_views,
    append_registry_rows,
    atomic_registry_transaction,
    build_prediction_frame,
    deterministic_feature_split,
    fixed_candidate_pool,
    frozen_project,
    load_identity_evidence,
    reconcile_feature_universe,
    validate_checkpoint_manifest,
    validate_prediction_splits,
)

_MINIMAL_REGISTRY_SCHEMA = RegistrySchema(
    required=frozenset({"run_id"}),
    primary_key=("run_id",),
)
from credit_risk_fs.pipelines.common import (
    ExperimentConfig,
    PredictionMetadata,
    SourceIdentityProvenance,
    _stable_row_ids,
    build_source_identity_provenance,
    export_prediction_artifact,
    prediction_metadata_from_sources,
    prediction_metrics_from_saved_files,
    prepare_modeling_data,
    validate_authenticated_split_ids,
    validate_source_identity_subset,
)
from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    canonical_raw_dev_evidence,
)
from credit_risk_fs.utils.hashing import sha256_file
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


def test_explicit_identity_evidence_is_role_and_feature_id_bound(tmp_path) -> None:
    universe = _reconciled(3)
    left, right = universe.iloc[0], universe.iloc[1]
    path = tmp_path / "identity.json"
    path.write_text(
        __import__("json").dumps(
            {
                "identity_evidence_version": "explicit_identity_evidence_v1",
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_dataset": "lendingclub_v2",
                "external_dataset": "homecredit",
                "verified_aliases": [
                    {
                        "feature_id_a": left.feature_id,
                        "feature_id_b": right.feature_id,
                        "feature_name_a": left.feature_name,
                        "feature_name_b": right.feature_name,
                    }
                ],
                "documented_identity_transforms": [],
            }
        ),
        encoding="utf-8",
    )
    relations, manifest = load_identity_evidence(
        path, reconciled=universe, source_dataset="lendingclub_v2"
    )
    assert relations.reason.tolist() == ["verified_alias"]
    assert manifest["identity_evidence_hash"] == sha256_file(path)
    bad = __import__("json").loads(path.read_text(encoding="utf-8"))
    bad["source_dataset"] = "homecredit"
    path.write_text(__import__("json").dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="source_dataset"):
        load_identity_evidence(
            path, reconciled=universe, source_dataset="lendingclub_v2"
        )


def _external_views() -> tuple[pd.DataFrame, pd.DataFrame]:
    semantic = pd.DataFrame(
        {
            "feature_id": ["b", "a"],
            "feature_name": ["B", "A"],
            "dataset": ["homecredit", "homecredit"],
            "embedding_0000": [2.0, 1.0],
        }
    )
    statistical = pd.DataFrame(
        {
            "feature_id": ["a", "b"],
            "feature_name": ["A", "B"],
            "dataset": ["homecredit", "homecredit"],
            "stat_0000": [10.0, 20.0],
        }
    )
    return semantic, statistical


def test_external_alignment_joins_by_feature_id_not_row_position() -> None:
    semantic, statistical = _external_views()
    first = align_external_feature_views(
        semantic,
        statistical,
        external_dataset="homecredit",
        semantic_hash="semantic",
        statistical_hash="statistical",
    )
    second = align_external_feature_views(
        semantic.sample(frac=1, random_state=2),
        statistical.sample(frac=1, random_state=3),
        external_dataset="homecredit",
        semantic_hash="semantic",
        statistical_hash="statistical",
    )
    assert first[0].feature_id.tolist() == first[1].feature_id.tolist() == ["a", "b"]
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_semantic", "identities differ"),
        ("missing_statistical", "identities differ"),
        ("duplicate", "duplicate feature IDs"),
        ("name_conflict", "names conflict"),
        ("dataset", "dataset metadata mismatch"),
    ],
)
def test_external_alignment_fails_closed(mutation, match) -> None:
    semantic, statistical = _external_views()
    if mutation == "missing_semantic":
        semantic = semantic.iloc[1:].copy()
    elif mutation == "missing_statistical":
        statistical = statistical.iloc[1:].copy()
    elif mutation == "duplicate":
        semantic = pd.concat([semantic, semantic.iloc[[0]]], ignore_index=True)
    elif mutation == "name_conflict":
        statistical.loc[0, "feature_name"] = "WRONG"
    else:
        statistical.loc[0, "dataset"] = "lendingclub_v2"
    with pytest.raises(ValueError, match=match):
        align_external_feature_views(
            semantic,
            statistical,
            external_dataset="homecredit",
            semantic_hash="semantic",
            statistical_hash="statistical",
        )


def test_saved_prediction_metrics_are_exactly_recomputable(tmp_path) -> None:
    common = {
        "dataset": ["homecredit"] * 4,
        "target": [0, 0, 1, 1],
        "prediction_probability": [0.1, 0.4, 0.6, 0.9],
    }
    dev = pd.DataFrame(
        {**common, "stable_row_id": ["d1", "d2", "d3", "d4"], "split": ["dev"] * 4}
    )
    oot = pd.DataFrame(
        {**common, "stable_row_id": ["o1", "o2", "o3", "o4"], "split": ["oot"] * 4}
    )
    dev_path, oot_path = tmp_path / "dev.csv", tmp_path / "oot.csv"
    dev.to_csv(dev_path, index=False)
    oot.to_csv(oot_path, index=False)
    metrics = prediction_metrics_from_saved_files(
        dev_path, oot_path, threshold=0.5
    )
    assert metrics.set_index("split").loc["dev", "auc"] == pytest.approx(1.0)
    assert metrics.set_index("split").loc["oot", "ks"] == pytest.approx(1.0)
    assert metrics["score_psi"].eq(0.0).all()
    assert metrics.set_index("split").loc["dev", "row_count"] == 4
    assert metrics.set_index("split").loc["dev", "prediction_file_hash"] == sha256_file(dev_path)


def test_registry_conflicts_and_atomic_rollback(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry.csv"
    pd.DataFrame([{"run_id": "r1", "value": "old"}]).to_csv(registry, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        append_registry_rows(
            registry_path=registry,
            rows=pd.DataFrame([{"run_id": "r1", "value": "new"}]),
            equivalence_columns=["run_id"],
            schema=_MINIMAL_REGISTRY_SCHEMA,
        )
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    one.write_text("old-one", encoding="utf-8")
    two.write_text("old-two", encoding="utf-8")
    import credit_risk_fs.clip.reverse_transfer as module

    real_replace = module.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic transaction failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        atomic_registry_transaction(
            {one: b"new-one", two: b"new-two"},
            transaction_manifest_path=tmp_path / "transaction.json",
            metadata={"test": True},
        )
    assert one.read_text(encoding="utf-8") == "old-one"
    assert two.read_text(encoding="utf-8") == "old-two"


def test_atomic_registry_success_and_idempotent_append(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    existing = pd.DataFrame([{"run_id": "r1", "value": "same"}])
    existing.to_csv(registry, index=False)
    combined = append_registry_rows(
        registry_path=registry,
        rows=existing.copy(),
        equivalence_columns=["run_id"],
        schema=_MINIMAL_REGISTRY_SCHEMA,
    )
    assert len(combined) == 1
    target = tmp_path / "target.csv"
    manifest_path = tmp_path / "transaction.json"
    transaction = atomic_registry_transaction(
        {target: b"run_id,value\nr1,same\n"},
        transaction_manifest_path=manifest_path,
        metadata={"source_dataset": "lendingclub_v2"},
    )
    assert transaction["status"] == "committed"
    assert manifest_path.exists()


def _source_identity(tmp_path, frame):
    path = tmp_path / "application_train.csv"
    frame.to_csv(path, index=False)
    return build_source_identity_provenance(
        frame,
        dataset="homecredit",
        stable_row_id_column="SK_ID_CURR",
        target_column="TARGET",
        source_artifact_path=path,
    )


def test_persistent_row_ids_survive_shuffle_and_reject_index_surrogates(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {"SK_ID_CURR": [30, 10, 20], "TARGET": [0, 1, 0], "value": [3, 1, 2]}
    )
    identity = _source_identity(tmp_path, frame)
    first = dict(
        zip(
            frame["SK_ID_CURR"],
            _stable_row_ids(
                frame,
                dataset="homecredit",
                stable_row_id_column="SK_ID_CURR",
                source_identity=identity,
            ),
        )
    )
    shuffled = frame.sample(frac=1, random_state=4).reset_index(drop=True)
    second = dict(
        zip(
            shuffled["SK_ID_CURR"],
            _stable_row_ids(
                shuffled,
                dataset="homecredit",
                stable_row_id_column="SK_ID_CURR",
                source_identity=identity,
            ),
        )
    )
    assert first == second
    with pytest.raises(ValueError, match="persistent source column"):
        _stable_row_ids(
            frame.reset_index(),
            dataset="homecredit",
            stable_row_id_column="index",
            source_identity=identity,
        )


@pytest.mark.parametrize(
    "frame,column,match",
    [
        (pd.DataFrame({"x": [1]}), "SK_ID_CURR", "lacks"),
        (pd.DataFrame({"SK_ID_CURR": [1, None]}), "SK_ID_CURR", "null"),
        (pd.DataFrame({"SK_ID_CURR": [1, 1]}), "SK_ID_CURR", "duplicated"),
    ],
)
def test_persistent_row_id_contract_fails_closed(
    frame, column, match, tmp_path
) -> None:
    raw = pd.DataFrame({"SK_ID_CURR": [10, 20], "TARGET": [0, 1]})
    identity = _source_identity(tmp_path, raw)
    if "TARGET" not in frame:
        frame = frame.assign(TARGET=0)
    with pytest.raises(ValueError, match=match):
        _stable_row_ids(
            frame,
            dataset="homecredit",
            stable_row_id_column=column,
            source_identity=identity,
        )


def test_source_identity_rejects_fabricated_and_replaced_ids(tmp_path) -> None:
    original = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100017, 100099, 100501],
            "TARGET": [0, 1, 0, 1],
            "recent_decision": [-500, -400, -200, -100],
        }
    )
    identity = _source_identity(tmp_path, original)

    no_id = original.drop(columns="SK_ID_CURR")
    no_id_path = tmp_path / "no_id.csv"
    no_id.to_csv(no_id_path, index=False)
    with pytest.raises(ValueError, match="lacks identity contract"):
        build_source_identity_provenance(
            no_id,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            source_artifact_path=no_id_path,
        )

    reset_fake = no_id.reset_index(names="SK_ID_CURR")
    with pytest.raises(ValueError, match="not authenticated"):
        validate_source_identity_subset(
            reset_fake,
            source_identity=identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            verify_source_artifact=False,
        )
    sequential = original.assign(SK_ID_CURR=range(len(original)))
    with pytest.raises(ValueError, match="not authenticated"):
        validate_source_identity_subset(
            sequential,
            source_identity=identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            verify_source_artifact=False,
        )
    overwritten_subset = original.iloc[:2].assign(SK_ID_CURR=[0, 1])
    with pytest.raises(ValueError, match="not authenticated"):
        validate_source_identity_subset(
            overwritten_subset,
            source_identity=identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            verify_source_artifact=False,
        )


def test_source_identity_manifest_and_alignment_fail_closed(tmp_path) -> None:
    original = pd.DataFrame(
        {"SK_ID_CURR": [100001, 100017, 100099], "TARGET": [0, 1, 0]}
    )
    identity = _source_identity(tmp_path, original)
    with pytest.raises(ValueError, match="manifest is missing"):
        validate_source_identity_subset(
            original,
            source_identity=None,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
        )
    incomplete = SourceIdentityProvenance(
        manifest={"dataset": "homecredit"},
        authenticated_ids=identity.authenticated_ids,
        target_by_id=identity.target_by_id,
    )
    with pytest.raises(ValueError, match="manifest is incomplete"):
        validate_source_identity_subset(
            original,
            source_identity=incomplete,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
        )
    changed_ids = SourceIdentityProvenance(
        manifest=dict(identity.manifest),
        authenticated_ids=frozenset({*identity.authenticated_ids, "999999"}),
        target_by_id={**identity.target_by_id, "999999": "0"},
    )
    with pytest.raises(ValueError, match="values hash mismatch"):
        validate_source_identity_subset(
            original,
            source_identity=changed_ids,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            verify_source_artifact=False,
        )
    wrong_target = original.copy()
    wrong_target.loc[0, "TARGET"] = 1
    with pytest.raises(ValueError, match="targets are misaligned"):
        validate_source_identity_subset(
            wrong_target,
            source_identity=identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
            verify_source_artifact=False,
        )
    source_path = Path(identity.manifest["source_artifact"])
    source_path.write_text("SK_ID_CURR,TARGET\n100001,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_source_identity_subset(
            original,
            source_identity=identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
        )


def test_authenticated_subsets_shuffle_and_feature_exclusion(tmp_path) -> None:
    original = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100017, 100099, 100501],
            "TARGET": [0, 1, 0, 1],
            "feature": [4.0, 3.0, 2.0, 1.0],
        }
    )
    identity = _source_identity(tmp_path, original)
    dev = original.iloc[[0, 2]].sample(frac=1, random_state=8)
    oot = original.iloc[[1, 3]].sample(frac=1, random_state=9)
    dev_ids = _stable_row_ids(
        dev,
        dataset="homecredit",
        stable_row_id_column="SK_ID_CURR",
        source_identity=identity,
    )
    oot_ids = _stable_row_ids(
        oot,
        dataset="homecredit",
        stable_row_id_column="SK_ID_CURR",
        source_identity=identity,
    )
    validate_authenticated_split_ids(dev_ids, oot_ids)
    assert dev_ids.tolist() == dev["SK_ID_CURR"].astype(str).tolist()
    assert oot_ids.tolist() == oot["SK_ID_CURR"].astype(str).tolist()
    assert dict(zip(dev_ids, dev["TARGET"])) == {
        str(row.SK_ID_CURR): row.TARGET for row in dev.itertuples()
    }
    model_features = dev.drop(columns=["SK_ID_CURR", "TARGET"])
    assert "SK_ID_CURR" not in model_features
    with pytest.raises(ValueError, match="overlap"):
        validate_authenticated_split_ids(dev_ids, pd.Series([dev_ids.iloc[0]]))


def test_source_identity_rejects_duplicate_null_and_unexpected_ids(tmp_path) -> None:
    original = pd.DataFrame(
        {"SK_ID_CURR": [100001, 100017, 100099], "TARGET": [0, 1, 0]}
    )
    identity = _source_identity(tmp_path, original)
    cases = [
        (original.assign(SK_ID_CURR=[100001, 100001, 100099]), "duplicated"),
        (original.assign(SK_ID_CURR=[100001, None, 100099]), "null"),
        (original.assign(SK_ID_CURR=[100001, 100017, 999999]), "not authenticated"),
    ]
    for frame, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_source_identity_subset(
                frame,
                source_identity=identity,
                dataset="homecredit",
                stable_row_id_column="SK_ID_CURR",
                target_column="TARGET",
                verify_source_artifact=False,
            )


def test_prepare_modeling_data_authenticates_raw_ids_before_split(
    tmp_path, monkeypatch
) -> None:
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": [100099, 100001, 100501, 100017],
            "TARGET": [0, 1, 1, 0],
            "recent_decision": [-300, -500, -100, -200],
            "feature": [3.0, 1.0, 4.0, 2.0],
        }
    )
    raw.to_csv(tmp_path / "application_train.csv", index=False)
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.common.resolve_dataset_mode",
        lambda **kwargs: "single_table",
    )
    config = ExperimentConfig(
        experiment_name="identity_boundary_test",
        selector_name="mrmr",
        dataset_name="homecredit",
        data_dir=str(tmp_path),
        dev_start_day=-600,
        oot_start_day=-240,
        oot_end_day=0,
        stable_row_id_column="SK_ID_CURR",
    )
    prepared = prepare_modeling_data(config)
    assert prepared.dev_stable_row_ids.tolist() == ["100099", "100001"]
    assert prepared.oot_stable_row_ids.tolist() == ["100501", "100017"]
    assert "SK_ID_CURR" not in prepared.X_train
    assert "SK_ID_CURR" not in prepared.X_oot
    assert prepared.source_identity is not None
    assert (
        prepared.source_identity.manifest["creation_stage"] == "raw_input"
    )

    fabricated = raw.drop(columns="SK_ID_CURR").reset_index(names="SK_ID_CURR")
    fabricated.to_csv(tmp_path / "application_train.csv", index=False)
    with pytest.raises(ValueError, match="not authenticated|artifact hash|lacks"):
        validate_source_identity_subset(
            fabricated,
            source_identity=prepared.source_identity,
            dataset="homecredit",
            stable_row_id_column="SK_ID_CURR",
            target_column="TARGET",
        )


def test_raw_dev_evidence_hash_changes_for_value_row_or_column_mutation() -> None:
    frame = pd.DataFrame(
        {"recent_decision": [-3, -2], "a": [1.0, 2.0], "b": ["x", None]}
    )

    def evidence(data, features=("a", "b")):
        return canonical_raw_dev_evidence(
            data,
            time_column="recent_decision",
            feature_columns=list(features),
            target_column="TARGET",
            dev_start=-4,
            dev_end=0,
        )["sha256"]

    baseline = evidence(frame)
    changed_value = frame.copy()
    changed_value.loc[0, "a"] = 9.0
    assert evidence(changed_value) != baseline
    assert evidence(frame.iloc[:1]) != baseline
    assert evidence(frame, ("a",)) != baseline
    with pytest.raises(TransferStageError, match="outside"):
        evidence(pd.concat([frame, frame.assign(recent_decision=1).iloc[[0]]]))
    with pytest.raises(TransferStageError, match="target"):
        evidence(frame.assign(TARGET=[0, 1]), ("a", "TARGET"))
    with pytest.raises(TransferStageError, match="LendingClub"):
        canonical_raw_dev_evidence(
            frame,
            time_column="recent_decision",
            feature_columns=["a", "b"],
            target_column="TARGET",
            dev_start=-4,
            dev_end=0,
            dataset="homecredit",
        )


def test_dev_oof_metrics_require_fold_exclusive_saved_rows(tmp_path) -> None:
    dev = pd.DataFrame(
        {
            "stable_row_id": ["d1", "d2", "d3", "d4"],
            "split": ["DEV_OOF"] * 4,
            "fold_id": [1, 1, 2, 2],
            "target": [0, 1, 0, 1],
            "prediction_probability": [0.1, 0.8, 0.2, 0.9],
        }
    )
    oot = dev.drop(columns="fold_id").assign(
        stable_row_id=["o1", "o2", "o3", "o4"], split="oot"
    )
    dev_path, oot_path = tmp_path / "dev_oof_predictions.csv", tmp_path / "oot.csv"
    dev.to_csv(dev_path, index=False)
    oot.to_csv(oot_path, index=False)
    expected_targets = dict(zip(dev["stable_row_id"], dev["target"]))
    metrics = prediction_metrics_from_saved_files(
        dev_path,
        oot_path,
        threshold=0.5,
        expected_dev_targets=expected_targets,
    )
    row = metrics.set_index("split").loc["DEV_OOF"]
    assert row["metric_scope"] == "dev_oof_cross_validated"
    assert row["prediction_file_hash"] == sha256_file(dev_path)
    dev.drop(columns="fold_id").to_csv(dev_path, index=False)
    with pytest.raises(ValueError, match="fold IDs"):
        prediction_metrics_from_saved_files(dev_path, oot_path, threshold=0.5)
    dev.iloc[:3].to_csv(dev_path, index=False)
    with pytest.raises(ValueError, match="coverage"):
        prediction_metrics_from_saved_files(
            dev_path,
            oot_path,
            threshold=0.5,
            expected_dev_targets=expected_targets,
        )
    misaligned = dev.copy()
    misaligned["target"] = misaligned["target"].iloc[::-1].to_numpy()
    misaligned.to_csv(dev_path, index=False)
    with pytest.raises(ValueError, match="target alignment"):
        prediction_metrics_from_saved_files(
            dev_path,
            oot_path,
            threshold=0.5,
            expected_dev_targets=expected_targets,
        )


def _prediction_metadata(model="lr"):
    return PredictionMetadata(
        dataset="homecredit",
        split="DEV_OOF",
        run_id=f"run-{model}",
        method="reverse",
        model=model,
        source_training_dataset="lendingclub_v2",
        external_dataset="homecredit",
        configuration_hash="c" * 64,
        data_manifest_hash="d" * 64,
        pairing_policy_version="identity_equivalence_v2",
        source_identity_manifest_hash="i" * 64,
        stable_row_id_column="SK_ID_CURR",
        source_stable_id_values_hash="s" * 64,
    )


def _fold_manifest():
    import hashlib, json

    rows = []
    for fold, train, val in [
        (1, ["1003", "1004"], ["1001", "1002"]),
        (2, ["1001", "1002"], ["1003", "1004"]),
    ]:
        rows.append(
            {
                "fold_id": fold,
                "training_id_hash": hashlib.sha256(
                    json.dumps(sorted(train), separators=(",", ":")).encode()
                ).hexdigest(),
                "validation_id_hash": hashlib.sha256(
                    json.dumps(sorted(val), separators=(",", ":")).encode()
                ).hexdigest(),
                "validation_row_count": len(val),
                "model_fit_scope": "fold_training_ids_only",
                "training_ids": train,
                "validation_ids": val,
            }
        )
    return rows


@pytest.mark.parametrize("model", ["lr", "catboost"])
def test_canonical_oof_export_and_saved_metric_recomputation(tmp_path, model) -> None:
    frame = pd.DataFrame(
        {
            "stable_row_id": ["1004", "1002", "1001", "1003"],
            "fold_id": [2, 1, 1, 2],
            "target": [1, 1, 0, 0],
            "prediction_probability": [0.9, 0.8, 0.1, 0.2],
            "predicted_class": [1, 1, 0, 0],
        }
    )
    path = tmp_path / model / "dev_oof_predictions.csv"
    saved, manifest = export_prediction_artifact(
        frame,
        metadata=_prediction_metadata(model),
        path=path,
        threshold=0.5,
        expected_ids={"1001", "1002", "1003", "1004"},
        expected_targets={"1001": 0, "1002": 1, "1003": 0, "1004": 1},
        fold_manifest=_fold_manifest(),
        forbidden_ids={"2001"},
    )
    assert saved["stable_row_id"].astype(str).tolist() == [
        "1001",
        "1002",
        "1003",
        "1004",
    ]
    assert manifest["row_count"] == 4
    assert manifest["fold_manifest_hash"]
    oot = saved.drop(columns="fold_id").assign(
        stable_row_id=["2001", "2002", "2003", "2004"], split="oot"
    )
    oot.to_csv(tmp_path / model / "oot_predictions.csv", index=False)
    metrics = prediction_metrics_from_saved_files(
        path, tmp_path / model / "oot_predictions.csv", threshold=0.5
    ).set_index("split")
    assert metrics.loc["DEV_OOF", "auc"] == pytest.approx(1.0)
    assert metrics.loc["DEV_OOF", "ks"] == pytest.approx(1.0)


def test_prediction_metadata_duplicate_fields_fail_closed() -> None:
    explicit = asdict(_prediction_metadata())
    with pytest.raises(ValueError, match="supplied more than once"):
        prediction_metadata_from_sources(explicit, {"dataset": "homecredit"})
    with pytest.raises(ValueError, match="supplied more than once"):
        prediction_metadata_from_sources(explicit, {"dataset": "wrong"})


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicated"),
        ("missing", "coverage"),
        ("null", "null"),
        ("fold_missing", "fold_id"),
        ("oot", "forbidden OOT"),
        ("training_overlap", "training and validation"),
        ("validation_overlap", "overlap across folds"),
        ("target", "misaligned"),
        ("probability", "within"),
    ],
)
def test_oof_export_adversarial_failures(tmp_path, mutation, match) -> None:
    frame = pd.DataFrame(
        {
            "stable_row_id": ["1001", "1002", "1003", "1004"],
            "fold_id": [1, 1, 2, 2],
            "target": [0, 1, 0, 1],
            "prediction_probability": [0.1, 0.8, 0.2, 0.9],
            "predicted_class": [0, 1, 0, 1],
        }
    )
    folds = _fold_manifest()
    forbidden = {"2001"}
    if mutation == "duplicate":
        frame.loc[3, "stable_row_id"] = "1003"
    elif mutation == "missing":
        frame = frame.iloc[:3]
    elif mutation == "null":
        frame.loc[0, "stable_row_id"] = None
    elif mutation == "fold_missing":
        frame = frame.drop(columns="fold_id")
    elif mutation == "oot":
        frame.loc[0, "stable_row_id"] = "2001"
        forbidden = {"2001"}
    elif mutation == "training_overlap":
        folds[0]["training_ids"].append("1001")
    elif mutation == "validation_overlap":
        import hashlib, json
        folds[1]["training_ids"].remove("1001")
        folds[1]["training_id_hash"] = hashlib.sha256(
            json.dumps(
                sorted(folds[1]["training_ids"]), separators=(",", ":")
            ).encode()
        ).hexdigest()
        folds[1]["validation_ids"].append("1001")
    elif mutation == "target":
        frame.loc[0, "target"] = 1
    elif mutation == "probability":
        frame.loc[0, "prediction_probability"] = 1.5
    with pytest.raises(ValueError, match=match):
        export_prediction_artifact(
            frame,
            metadata=_prediction_metadata(),
            path=tmp_path / "dev_oof_predictions.csv",
            threshold=0.5,
            expected_ids={"1001", "1002", "1003", "1004"},
            expected_targets={"1001": 0, "1002": 1, "1003": 0, "1004": 1},
            fold_manifest=folds,
            forbidden_ids=forbidden,
        )


def test_registry_canonicalization_and_atomic_second_registration_noop(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "feature_budget": 20,
                "depends_on_clip": True,
                "output_folder": "results/run",
                "source_checkpoint_hashes": '{"22":"b","11":"a"}',
            }
        ]
    ).to_csv(registry, index=False)
    equivalent = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "feature_budget": "20.0",
                "depends_on_clip": "1",
                "output_folder": ".\\results\\run\\",
                "source_checkpoint_hashes": '{"11":"a", "22":"b"}',
            }
        ]
    )
    combined = append_registry_rows(
        registry_path=registry,
        rows=equivalent,
        equivalence_columns=["run_id"],
        schema=_MINIMAL_REGISTRY_SCHEMA,
    )
    assert len(combined) == 1
    assert combined.attrs["registry_changed"] is False

    target = tmp_path / "target.csv"
    manifest = tmp_path / "transaction.json"
    payload = {target: b"run_id\nr1\n"}
    atomic_registry_transaction(
        payload,
        transaction_manifest_path=manifest,
        metadata={"source_dataset": "lendingclub_v2"},
    )
    original_target = target.read_bytes()
    original_manifest = manifest.read_bytes()
    result = atomic_registry_transaction(
        payload,
        transaction_manifest_path=manifest,
        metadata={"source_dataset": "lendingclub_v2"},
    )
    assert result["idempotent_noop"] is True
    assert target.read_bytes() == original_target
    assert manifest.read_bytes() == original_manifest
