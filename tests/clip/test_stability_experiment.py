from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import credit_risk_fs.clip.stability_experiment as stability_module
from credit_risk_fs.clip.stability_experiment import (
    DESCRIPTOR_FIELDS,
    FEATURE_COUNT,
    REQUIRED_SEEDS,
    ExperimentContractError,
    ExperimentReporter,
    FinalOOTEvaluator,
    FiveSeedConsensusBuilder,
    FrozenTransform,
    HistoricalSourceArtifactResolver,
    LegacyMRMRDownstreamRunner,
    OOTAccessToken,
    PreOOTFreezeGate,
    ProgressLogger,
    Prompt1PackageValidator,
    StageStore,
    StabilityMatrixAccess,
    TransferredRankingBuilder,
    checkpoint_epoch_from_validation_losses,
    frozen_downstream_cells,
    identity_exclusions,
    project_joint,
    sha256_file,
    stability_training_bundle,
)
from credit_risk_fs.clip.loss import symmetric_masked_contrastive_loss
from credit_risk_fs.clip.model import ClipModelConfig
from credit_risk_fs.clip.trainer import train_seed
from credit_risk_fs.clip.training_validation import false_negative_mask
from credit_risk_fs.clip.training_validation import (
    ClipTrainingConfig,
    TrainingDataBundle,
    tensors_for_pairs,
)
from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/protocols/homecredit_model_stability_2024_v2/clip_stability_experiment_v1.json"


class _SyntheticClassifier:
    def fit(self, matrix, target, eval_set=None):
        self.fit_target_ = tuple(int(value) for value in target)
        return self

    def predict_proba(self, matrix):
        score = np.linspace(0.1, 0.9, matrix.shape[0])
        return np.column_stack([1.0 - score, score])


@pytest.fixture(scope="module")
def frozen_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prompt1_package(frozen_config):
    return Prompt1PackageValidator(REPOSITORY_ROOT, frozen_config["prompt1_package"]).validate()


def test_prompt1_pass_hash_count_and_target_free_boundary(prompt1_package, frozen_config):
    assert prompt1_package.manifest_sha256 == frozen_config["prompt1_package"]["sha256_manifest_sha256"]
    assert prompt1_package.feature_universe_hash == frozen_config["prompt1_package"]["feature_universe_hash"]
    assert len(prompt1_package.feature_universe) == FEATURE_COUNT
    assert prompt1_package.feature_universe["feature_name"].is_unique
    assert {"target", "date_decision", "case_id"}.isdisjoint(
        set(prompt1_package.feature_universe["feature_name"].str.casefold())
    )
    assert len(prompt1_package.anchor_feature_ids) == 23


def test_stability_bundle_preserves_split_and_identity_mask(prompt1_package):
    bundle = stability_training_bundle(prompt1_package)
    assert len(bundle.train_pairs) == 1564
    assert len(bundle.validation_pairs) == 395
    assert bundle.text_dim == 384
    assert bundle.statistical_dim == 13
    assert bundle.statistical_fields == list(DESCRIPTOR_FIELDS)
    train_text, train_stat = tensors_for_pairs(
        bundle.train_pairs, bundle.training_text, bundle.training_stat
    )
    validation_text, validation_stat = tensors_for_pairs(
        bundle.validation_pairs, bundle.training_text, bundle.training_stat
    )
    assert tuple(train_text.shape) == (1564, 384)
    assert tuple(train_stat.shape) == (1564, 13)
    assert tuple(validation_text.shape) == (395, 384)
    assert tuple(validation_stat.shape) == (395, 13)
    assert set(bundle.negative_exclusions["policy_version"].unique()) <= {"identity_equivalence_v2"}
    train_mask = false_negative_mask(bundle.train_pairs, bundle.negative_exclusions)
    validation_mask = false_negative_mask(bundle.validation_pairs, bundle.negative_exclusions)
    assert np.array_equal(train_mask.numpy(), train_mask.numpy().T)
    assert np.array_equal(validation_mask.numpy(), validation_mask.numpy().T)


def test_identity_exclusions_are_symmetric_exact_identity_only():
    pairs = pd.DataFrame(
        {
            "feature_name": ["a", "b", "c"],
            "equivalence_group_id": ["g1", "g1", "g2"],
        }
    )
    exclusions = identity_exclusions(pairs)
    assert set(map(tuple, exclusions[["anchor_feature_name", "excluded_feature_name"]].to_numpy())) == {
        ("a", "b"), ("b", "a")
    }
    assert set(exclusions["exclusion_reason"]) == {"exact_dev_duplicate"}


def test_symmetric_loss_is_transpose_symmetric():
    import torch

    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.2, 0.9], [-0.6, 0.4]]), dim=1)
    stat = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1], [0.1, 1.0], [-0.4, 0.7]]), dim=1)
    left = symmetric_masked_contrastive_loss(text, stat, temperature=torch.tensor(0.07)).loss
    right = symmetric_masked_contrastive_loss(stat, text, temperature=torch.tensor(0.07)).loss
    assert float(left) == pytest.approx(float(right), abs=1e-7)


def test_checkpoint_selection_uses_validation_loss_not_mrr():
    assert checkpoint_epoch_from_validation_losses([1.0, 0.8, 0.80005, 0.7], 0.0001) == 4
    assert checkpoint_epoch_from_validation_losses([0.5, 0.4, 0.45], 0.0001) == 2


def test_exact_384_by_13_adapter_completes_synthetic_training_step(tmp_path):
    rng = np.random.default_rng(42)
    all_names = [f"synthetic_{index}" for index in range(6)]

    def pairs(names, split):
        order_hash = feature_order_hash(names)
        return pd.DataFrame(
            {
                "feature_id": [f"id_{name}" for name in names],
                "dataset": "synthetic",
                "feature_name": names,
                "source_manifest_hash": "m" * 64,
                "text_embedding_row_id": [f"text_{name}" for name in names],
                "statistical_vector_row_id": [f"stat_{name}" for name in names],
                "positive_pair_index": range(len(names)),
                "feature_order_hash": order_hash,
                "split": split,
                "allowed_for_training": split == "train",
                "allowed_for_validation": split == "validation",
            }
        )

    train_pairs = pairs(all_names[:4], "train")
    validation_pairs = pairs(all_names[4:], "validation")
    text = pd.DataFrame(
        rng.normal(size=(6, 384)).astype(np.float32),
        columns=[f"embedding_{index:04d}" for index in range(384)],
    )
    text.insert(0, "feature_name", all_names)
    text.insert(0, "embedding_cache_key", [f"text_{name}" for name in all_names])
    stat = pd.DataFrame(
        rng.normal(size=(6, 13)).astype(np.float32),
        columns=[f"stat_{index:04d}" for index in range(13)],
    )
    stat.insert(0, "feature_name", all_names)
    stat.insert(0, "stable_row_id", [f"stat_{name}" for name in all_names])
    bundle = TrainingDataBundle(
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        external_pairs=pd.DataFrame(),
        source_pairs=pd.concat([train_pairs, validation_pairs], ignore_index=True),
        training_text=text,
        external_text=pd.DataFrame(),
        training_stat=stat,
        external_stat=pd.DataFrame(),
        training_dataset="synthetic",
        external_dataset="none",
        negative_exclusions=pd.DataFrame(),
        upstream_hashes={"synthetic_input_hash": "i" * 64},
        text_dim=384,
        statistical_dim=13,
        statistical_fields=list(DESCRIPTOR_FIELDS),
    )
    placeholder = tmp_path / "unused"
    model_config = ClipModelConfig(text_input_dim=384, statistical_input_dim=13)
    config = ClipTrainingConfig(
        tensor_schema_path=placeholder,
        contrastive_pair_manifest_path=placeholder,
        train_pairs_path=placeholder,
        validation_pairs_path=placeholder,
        external_pairs_path=placeholder,
        negative_exclusion_pairs_path=placeholder,
        negative_policy_manifest_path=placeholder,
        homecredit_text_embeddings_path=placeholder,
        lendingclub_v2_text_embeddings_path=placeholder,
        homecredit_statistical_vectors_path=placeholder,
        lendingclub_v2_statistical_vectors_path=placeholder,
        text_embedding_manifest_path=placeholder,
        statistical_preprocessor_path=placeholder,
        source_manifest_path=placeholder,
        split_manifest_path=placeholder,
        output_dir=tmp_path,
        model=model_config,
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.01,
        batch_size=4,
        max_epochs=2,
        early_stopping_patience=15,
        minimum_improvement=0.0001,
        gradient_clipping_enabled=True,
        gradient_clip_norm=1.0,
        seeds=(11,),
        deterministic=True,
        device_policy="cpu",
        selection_metric="validation_loss",
        collapse_thresholds={},
        statistical_view_scope="compact_target_free_v2",
        smoke_test_steps=1,
        training_dataset="synthetic",
        external_dataset="none",
        configuration_hash="c" * 64,
        data_manifest_hash="d" * 64,
        statistical_preprocessor_hash="p" * 64,
        source_anchor_hash="a" * 64,
    )
    events = []
    result = train_seed(
        config=config,
        data=bundle,
        seed=11,
        output_dir=tmp_path,
        config_snapshot_text="synthetic: true\n",
        smoke_test=True,
        progress_callback=events.append,
        direction="synthetic",
    )
    assert result.checkpoint_path.is_file()
    assert result.parameter_count == 27488
    assert {event["event"] for event in events} >= {
        "seed_start", "train_batch", "new_best_checkpoint", "epoch_end", "seed_end"
    }


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q


def test_five_seed_procrustes_recovers_synthetic_rotations_and_reference():
    rng = np.random.default_rng(991)
    base = rng.normal(size=(80, 32))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    matrices = {11: base.copy()}
    for seed in REQUIRED_SEEDS[1:]:
        matrices[seed] = base @ _orthogonal(rng, 32)
    consensus, manifest = FiveSeedConsensusBuilder().build([f"f{i}" for i in range(80)], matrices)
    assert manifest["reference_seed"] == 11
    assert manifest["required_seeds"] == list(REQUIRED_SEEDS)
    assert np.abs(np.sum(consensus * base, axis=1)).mean() > 0.9999
    repeated, repeated_manifest = FiveSeedConsensusBuilder().build([f"f{i}" for i in range(80)], matrices)
    np.testing.assert_array_equal(consensus, repeated)
    assert manifest["consensus_sha256"] == repeated_manifest["consensus_sha256"]


def test_consensus_rejects_missing_seed_and_wrong_reference():
    with pytest.raises(ExperimentContractError, match="reference"):
        FiveSeedConsensusBuilder(reference_seed=22)
    matrices = {seed: np.ones((2, 32), dtype=float) for seed in REQUIRED_SEEDS[:-1]}
    with pytest.raises(ExperimentContractError, match="seed presence"):
        FiveSeedConsensusBuilder().build(["a", "b"], matrices)


def test_ranking_ties_are_deterministic():
    builder = TransferredRankingBuilder(Path("unused"), "u" * 64)
    names = [f"feature_{i:04d}" for i in reversed(range(FEATURE_COUNT))]
    ids = [hashlib.sha256(name.encode()).hexdigest() for name in names]
    ranking = builder._ranking(
        ids,
        names,
        np.ones(FEATURE_COUNT),
        direction="stability_to_stability",
        source_dataset="homecredit_model_stability_2024",
        source_anchor_identity="a" * 64,
        source_preprocessor_identity="p" * 64,
    )
    assert ranking["feature_name"].tolist() == sorted(names)
    assert ranking["rank"].tolist() == list(range(1, FEATURE_COUNT + 1))


def test_historical_hc_lc_artifacts_authenticate_and_are_frozen(frozen_config, prompt1_package):
    resolver = HistoricalSourceArtifactResolver(
        frozen_config["representation_contract"], frozen_config["training"]["seeds"]
    )
    for name in ("homecredit", "lendingclub"):
        source = resolver.authenticate(name, frozen_config["historical_sources"][name])
        assert tuple(source.checkpoint_paths) == REQUIRED_SEEDS
        assert source.authentication["status"] == "PASS"
        assert source.authentication["parameter_count"] == 27488
        assert source.authentication["pairing_policy"] == "identity_equivalence_v2"
        assert not hasattr(source.preprocessor, "fit")
        transformed = source.preprocessor.transform(prompt1_package.raw_descriptors.head(3))
        assert transformed.shape == (3, 13)
        ordered_pairs = prompt1_package.pair_frame.sort_values(
            "feature_id", kind="mergesort"
        ).head(3)
        text_columns = [f"embedding_{index:04d}" for index in range(384)]
        for seed in REQUIRED_SEEDS:
            assert sha256_file(source.checkpoint_paths[seed]) == frozen_config["historical_sources"][name]["checkpoint_hashes"][str(seed)]
            model = resolver.load_frozen_model(source, seed)
            assert not any(parameter.requires_grad for parameter in model.parameters())
            projected = project_joint(
                model,
                ordered_pairs[text_columns].to_numpy(dtype=np.float32),
                transformed,
            )
            assert projected.shape == (3, 32)
            assert np.isfinite(projected).all()
        assert all(vector.shape == (32,) for vector in source.anchor_vectors.values())


def test_frozen_transform_has_no_fit_surface():
    frozen = FrozenTransform(lambda values: np.asarray(values, dtype=np.float32), "identity")
    assert not hasattr(frozen, "fit")
    assert frozen.transform(np.zeros((2, 13))).shape == (2, 13)


def test_exact_six_cells_and_authenticated_mrmr(frozen_config):
    cells = frozen_downstream_cells(frozen_config)
    assert len(cells) == 6
    assert {(cell.classifier, cell.pool_size, cell.final_k) for cell in cells} == {
        ("lr", 60, 20), ("catboost", 100, 40)
    }
    assert RandomForestRelevanceMRMRSelector.algorithm_name == "rf_relevance_correlation_redundancy"
    assert RandomForestRelevanceMRMRSelector.canonical_mrmr is False


def test_downstream_uses_fold_train_and_full_dev_only(tmp_path, frozen_config, monkeypatch):
    artifact = tmp_path / "outputs"
    result = tmp_path / "results"
    for direction in ("stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability"):
        ranking = pd.DataFrame(
            {
                "rank": range(1, FEATURE_COUNT + 1),
                "feature_name": [f"f{i:04d}" for i in range(FEATURE_COUNT)],
                "clip_score": np.linspace(1.0, 0.0, FEATURE_COUNT),
            }
        )
        path = artifact / "rankings" / f"{direction}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(path, index=False)

    class FakeMatrix:
        split = {"folds": [{"validation": {"rows": 4}} for _ in range(5)]}

        @staticmethod
        def _frame(predictors, case_ids, target):
            frame = pd.DataFrame(
                np.arange(len(case_ids) * len(predictors), dtype=np.float32).reshape(len(case_ids), len(predictors)),
                columns=predictors,
            )
            frame.insert(0, "target", target)
            frame.insert(0, "date_decision", pd.to_datetime(["2019-01-01"] * len(case_ids)))
            frame.insert(0, "case_id", case_ids)
            return frame

        def load_fold(self, fold_id, predictors, expected_pool_size):
            assert len(predictors) == expected_pool_size
            return (
                self._frame(predictors, [fold_id * 100 + i for i in range(4)], [0, 0, 1, 1]),
                self._frame(predictors, [fold_id * 10_000 + i for i in range(4)], [1, 0, 1, 0]),
            )

        def load_full_dev(self, predictors, expected_pool_size):
            assert len(predictors) == expected_pool_size
            return self._frame(predictors, range(900_000, 900_006), [0, 1, 0, 1, 0, 1])

        def load_oot(self, predictors, token, gate):
            gate.validate(token)
            return self._frame(
                predictors,
                range(950_000, 950_006),
                [1, 0, 1, 0, 1, 0],
            )

    monkeypatch.setattr(stability_module, "_new_classifier", lambda name, config: _SyntheticClassifier())
    logger = ProgressLogger(artifact, "logs/run.log", "logs/progress.jsonl")
    runner = LegacyMRMRDownstreamRunner(frozen_config, artifact, result, FakeMatrix(), logger)
    fit_calls = []

    def fake_fit_selector(raw, target, cell, fold):
        fit_calls.append((fold, tuple(int(value) for value in target), len(raw), raw.shape[1]))
        selected = list(raw.columns[: cell.final_k])
        selector = type("SyntheticSelector", (), {})()
        selector.selected_features_ = selected
        selector.rf_importances_ = pd.Series(
            np.linspace(1.0, 0.1, raw.shape[1]), index=raw.columns
        )
        selector.selection_trace_ = pd.DataFrame(
            {
                "feature": selected,
                "mean_absolute_correlation": np.zeros(len(selected)),
                "selection_score": np.ones(len(selected)),
            }
        )
        selector.algorithm_name = RandomForestRelevanceMRMRSelector.algorithm_name
        return selector, object()

    monkeypatch.setattr(runner, "_fit_selector", fake_fit_selector)
    dev, _ = runner.run_dev()
    assert len(dev) == 6
    assert len(fit_calls) == 30
    assert all(target == (0, 0, 1, 1) and rows == 4 and columns in {60, 100} for _, target, rows, columns in fit_calls)
    fit_calls.clear()
    outputs = runner.fit_full_dev()
    assert outputs
    assert len(fit_calls) == 6
    assert all(fold == "full_dev" and target == (0, 1, 0, 1, 0, 1) and rows == 6 for fold, target, rows, _ in fit_calls)

    source_auth = artifact / "manifests/source_artifact_authentication.json"
    checkpoints = artifact / "manifests/stability_seed_checkpoints.json"
    source_auth.parent.mkdir(parents=True, exist_ok=True)
    source_auth.write_text("{}\n", encoding="utf-8")
    checkpoints.write_text("{}\n", encoding="utf-8")
    consensus_paths = []
    for name in ("stability", "homecredit", "lendingclub"):
        path = artifact / "manifests" / f"{name}_consensus.json"
        path.write_text("{}\n", encoding="utf-8")
        consensus_paths.append(path)
    gate = PreOOTFreezeGate(artifact, result, "c" * 64)
    token = gate.create(
        prompt1_hash="p" * 64,
        feature_universe_hash="u" * 64,
        source_authentication_path=source_auth,
        stability_checkpoint_path=checkpoints,
        consensus_paths=consensus_paths,
        cells=runner.cells,
        random_seeds=REQUIRED_SEEDS,
    )
    runtime_config = json.loads(json.dumps(frozen_config))
    runtime_config["downstream"]["oot_rows"] = 6
    final, final_outputs = FinalOOTEvaluator(
        runtime_config, result, FakeMatrix(), gate, logger
    ).run(token)
    assert len(final) == 6
    assert len(final_outputs) == 13
    assert final["status"].eq("COMPLETE").all()


def _gate_fixture(tmp_path: Path, frozen_config) -> tuple[PreOOTFreezeGate, dict[str, Path], list]:
    artifact = tmp_path / "outputs"
    result = tmp_path / "results"
    artifact.mkdir()
    result.mkdir()
    cells = frozen_downstream_cells(frozen_config)
    files = {
        "source": artifact / "manifests/source_artifact_authentication.json",
        "checkpoints": artifact / "manifests/stability_seed_checkpoints.json",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    consensus = []
    for name in ("stability", "hc", "lc"):
        path = artifact / f"manifests/{name}_consensus.json"
        path.write_text("{}\n", encoding="utf-8")
        consensus.append(path)
    for direction in ("stability_to_stability", "homecredit_to_stability", "lendingclub_to_stability"):
        path = artifact / "rankings" / f"{direction}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("rank,feature_name\n1,x\n", encoding="utf-8")
    for cell in cells:
        root = result / "downstream" / cell.direction / cell.classifier
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "candidate_pool.csv", "fold_selected_features.csv", "full_dev_selected_features.csv",
            "full_dev_freeze_manifest.json", "full_dev_score_reference.parquet", "frozen_full_dev_preprocessor.joblib",
            "frozen_full_dev_classifier.joblib",
        ):
            (root / name).write_bytes(b"frozen\n")
    files["consensus"] = consensus
    return PreOOTFreezeGate(artifact, result, "c" * 64), files, cells


def test_oot_gate_rejects_before_token_allows_valid_and_blocks_tamper(tmp_path, frozen_config):
    gate, files, cells = _gate_fixture(tmp_path, frozen_config)
    access = object.__new__(StabilityMatrixAccess)
    with pytest.raises(ExperimentContractError, match="token is absent"):
        access.load_oot([], None, gate)
    with pytest.raises(ExperimentContractError):
        gate.validate(OOTAccessToken(gate.path, "0" * 64, "c" * 64))
    token = gate.create(
        prompt1_hash="p" * 64,
        feature_universe_hash="u" * 64,
        source_authentication_path=files["source"],
        stability_checkpoint_path=files["checkpoints"],
        consensus_paths=files["consensus"],
        cells=cells,
        random_seeds=REQUIRED_SEEDS,
    )
    gate.validate(token)
    victim = gate.result_root / "downstream/stability_to_stability/lr/candidate_pool.csv"
    victim.write_text("tampered", encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="SHA-256"):
        gate.validate(token)


def test_progress_jsonl_schema(tmp_path):
    logger = ProgressLogger(tmp_path, "logs/run.log", "logs/progress.jsonl")
    row = logger.emit("test", stage="unit", event="tick", seed=11, metrics={"loss": 1.0})
    observed = json.loads((tmp_path / "logs/progress.jsonl").read_text(encoding="utf-8"))
    assert observed == row
    assert set(ProgressLogger.required_fields).issubset(observed)
    assert {"direction", "seed", "epoch", "batch", "fold", "metrics"}.issubset(observed)


def test_stage_store_safe_reuse_and_mismatch_failure(tmp_path):
    output = tmp_path / "value.txt"
    output.write_text("complete", encoding="utf-8")
    store = StageStore(tmp_path, "c" * 64)
    store.complete("unit", {"input": "i" * 64}, [output])
    assert store.reusable("unit", {"input": "i" * 64}) is True
    with pytest.raises(ExperimentContractError, match="input hashes"):
        store.reusable("unit", {"input": "x" * 64})
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="SHA-256"):
        store.reusable("unit", {"input": "i" * 64})


def test_reporter_has_no_optional_markdown_dependency(tmp_path, prompt1_package):
    artifact = tmp_path / "outputs"
    result = tmp_path / "results"
    for name in (
        "source_artifact_authentication.json",
        "stability_seed_checkpoints.json",
        "ranking_manifest.json",
        "downstream_manifest.json",
        "pre_oot_freeze_manifest.json",
    ):
        path = artifact / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    final = pd.DataFrame(
        {
            "direction": ["stability_to_stability"] * 2 + ["homecredit_to_stability"] * 2 + ["lendingclub_to_stability"] * 2,
            "classifier": ["lr", "catboost"] * 3,
            "oot_auc": np.linspace(0.70, 0.75, 6),
        }
    )
    metric_path = result / "analysis/final_clip_results.csv"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(metric_path, index=False)
    outputs = ExperimentReporter(REPOSITORY_ROOT, artifact, result, "c" * 64).build(final, prompt1_package)
    assert len(outputs) == 3
    assert all(path.is_file() for path in outputs)
    report = (artifact / "FINAL_CLIP_STABILITY_REPORT.md").read_text(encoding="utf-8")
    assert "| direction | classifier | oot_auc |" in report
