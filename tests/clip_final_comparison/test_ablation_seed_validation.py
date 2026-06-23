from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip_final_comparison.ablations import (
    RETRAINED_ABLATIONS,
    ablation_schema,
    train_grouped_ablation_representations,
    validate_ablation_training,
)
from credit_risk_fs.clip_final_comparison.plans import build_ablation_plan
from credit_risk_fs.clip_final_comparison.seeds import (
    generate_seed_score_cache,
    resolve_clip_v2_seed_artifacts,
    validate_seed_artifacts,
    validate_seed_score_cache,
)


def _views() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [f"f{i}" for i in range(20)]
    text = pd.DataFrame({"feature_name": features})
    for i in range(5):
        text[f"text_{i}"] = [(j + 1) * (i + 1) / 100 for j in range(len(features))]
    stat = pd.DataFrame({"feature_name": features})
    descriptors = [
        "missing_rate",
        "unique_ratio",
        "concentration_share",
        "signed_log_mean",
        "log_standard_deviation",
        "clipped_skewness",
        "normalized_entropy",
        "is_numeric",
        "is_categorical",
        "is_binary",
        "numeric_stats_valid",
        "skewness_valid",
        "entropy_valid",
    ]
    for i, name in enumerate(descriptors):
        stat[name] = [(j + i + 1) / 50 for j in range(len(features))]
    return text, stat


def test_reduced_schema_dimensions_and_deterministic_order():
    assert ablation_schema("without_location_scale")["statistical_dimension"] == 11
    assert ablation_schema("without_shape_diversity")["statistical_dimension"] == 10
    assert ablation_schema("without_type_validity")["statistical_dimension"] == 7
    assert ablation_schema("without_location_scale")["descriptor_order"] == ablation_schema("without_location_scale")["descriptor_order"]


def test_grouped_ablation_training_produces_seed_checkpoints_and_anchor(tmp_path):
    text, stat = _views()
    manifest = train_grouped_ablation_representations(output_root=tmp_path, text_view=text, statistical_view=stat)
    assert set(RETRAINED_ABLATIONS).issubset(set(manifest["ablation"]))
    validation = validate_ablation_training(tmp_path)
    assert len(validation) == 7
    for ablation in RETRAINED_ABLATIONS:
        train_dir = tmp_path / "ablations/training" / ablation
        seed_summary = pd.read_csv(train_dir / "seed_summary.csv")
        assert len(seed_summary) == 5
        selected = json.loads((train_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
        assert abs(selected["validation_loss"] - seed_summary.sort_values(["validation_loss", "seed"], kind="mergesort").iloc[0]["validation_loss"]) < 1e-12
        assert (train_dir / "selected_anchor/anchor.json").exists()


def test_fake_manifest_only_ablation_training_cannot_complete(tmp_path):
    train_dir = tmp_path / "ablations/training/without_location_scale"
    train_dir.mkdir(parents=True)
    (train_dir / "TRAINING_COMPLETE.json").write_text('{"status":"complete_valid"}', encoding="utf-8")
    try:
        validate_ablation_training(tmp_path)
    except RuntimeError as exc:
        assert "missing training artifacts" in str(exc)
    else:
        raise AssertionError("manifest-only training should not validate")


def test_reused_ablation_mapping_and_28_downstream_keys_are_unique():
    assert ablation_schema("text_only")["reference_source"] == "frozen_text_similarity_baseline"
    assert ablation_schema("statistics_only")["reference_source"] == "frozen_statistics_only_baseline_learned_in_full_clip_v2"
    assert ablation_schema("missingness_only")["reference_source"] == "frozen_clip_v1_missingness_representation"
    plan = build_ablation_plan()
    assert len(plan) == 28
    assert plan["run_id"].is_unique


def _seed_training_root(tmp_path: Path) -> Path:
    root = tmp_path / "clip_v2/training"
    for seed in [11, 22, 33, 44, 55]:
        seed_dir = root / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        ckpt = seed_dir / "best_checkpoint.pt"
        ckpt.write_text(f"checkpoint-{seed}", encoding="utf-8")
        payload = {
            "seed": seed,
            "checkpoint_hash": "",
            "training_config_hash": f"cfg-{seed}",
            "text_embedding_hash": f"text-{seed}",
            "statistical_schema_hash": f"schema-{seed}",
            "statistical_preprocessor_hash": f"prep-{seed}",
            "anchor_path": f"anchor-{seed}.json",
            "anchor_hash": f"anchor-hash-{seed}",
            "collapse_status": "not_collapsed",
        }
        import hashlib

        payload["checkpoint_hash"] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        (seed_dir / "checkpoint_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_seed_resolver_and_cache_validation(tmp_path):
    training_root = _seed_training_root(tmp_path)
    frame = resolve_clip_v2_seed_artifacts(tmp_path, training_root=training_root)
    validate_seed_artifacts(frame)
    row = frame[frame["seed"].eq(11)].iloc[0].to_dict()
    universe = ["a", "b", "c"]
    path = generate_seed_score_cache(output_root=tmp_path, dataset="synthetic", seed_row=row, candidate_universe=universe)
    validation = validate_seed_score_cache(path, seed_row=row, dataset="synthetic", candidate_universe=universe)
    assert validation["row_count"] == 3
    cache = pd.read_csv(path)
    cache["checkpoint_hash"] = "wrong"
    cache.to_csv(path, index=False)
    try:
        validate_seed_score_cache(path, seed_row=row, dataset="synthetic", candidate_universe=universe)
    except RuntimeError as exc:
        assert "checkpoint hash mismatch" in str(exc)
    else:
        raise AssertionError("stale seed cache should not validate")


def test_seed_resolver_detects_missing_checkpoint(tmp_path):
    frame = resolve_clip_v2_seed_artifacts(tmp_path, training_root=tmp_path / "missing")
    try:
        validate_seed_artifacts(frame)
    except RuntimeError as exc:
        assert "ineligible seed artifacts" in str(exc)
    else:
        raise AssertionError("missing checkpoints should be ineligible")
