from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json


FUSION_RULE = "L2-normalized average of projected text and projected statistical embeddings"


@dataclass(frozen=True)
class ClipSelectorConfig:
    experiment_version: str
    selected_checkpoint_path: Path
    selected_checkpoint_manifest_path: Path
    model_selection_manifest_path: Path
    learned_anchor_manifest_path: Path
    training_manifest_path: Path
    text_embedding_manifest_path: Path
    statistical_preprocessor_path: Path
    source_manifest_path: Path
    homecredit_scores_path: Path
    lendingclub_v2_scores_path: Path
    output_dir: Path
    cache_dir: Path
    checkpoint_hash: str
    anchor_hash: str
    statistical_view_scope: str
    screening_pool_size: int
    missing_feature_policy: str
    feature_budgets: dict[str, int]
    active_datasets: tuple[str, ...]
    legacy_datasets: tuple[str, ...]
    no_refit: bool


def load_clip_selector_config(path: str | Path = "configs/clip/selector.yaml") -> ClipSelectorConfig:
    data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    budgets = data.get("feature_budgets", {}) if isinstance(data.get("feature_budgets"), dict) else {}
    return ClipSelectorConfig(
        experiment_version=str(data.get("experiment_version", "clip_v1")),
        selected_checkpoint_path=Path(str(data.get("selected_checkpoint_path", "results/clip/training/seeds/seed_55/best_checkpoint.pt"))),
        selected_checkpoint_manifest_path=Path(
            str(data.get("selected_checkpoint_manifest_path", "results/clip/training/seeds/seed_55/checkpoint_manifest.json"))
        ),
        model_selection_manifest_path=Path(
            str(data.get("model_selection_manifest_path", "results/clip/training/model_selection_manifest.json"))
        ),
        learned_anchor_manifest_path=Path(str(data.get("learned_anchor_manifest_path", "results/clip/training/learned_anchor_manifest.json"))),
        training_manifest_path=Path(str(data.get("training_manifest_path", "results/clip/training/training_manifest.json"))),
        text_embedding_manifest_path=Path(str(data.get("text_embedding_manifest_path", "results/clip/text_baseline/embedding_cache_manifest.json"))),
        statistical_preprocessor_path=Path(
            str(data.get("statistical_preprocessor_path", "results/clip/statistical_baseline/statistical_preprocessor.json"))
        ),
        source_manifest_path=Path(str(data.get("source_manifest_path", "results/clip/dry_run/training_manifest.json"))),
        homecredit_scores_path=Path(str(data.get("homecredit_scores_path", "results/clip/training/homecredit_learned_scores.csv"))),
        lendingclub_v2_scores_path=Path(
            str(data.get("lendingclub_v2_scores_path", "results/clip/training/lendingclub_v2_learned_scores.csv"))
        ),
        output_dir=Path(str(data.get("output_dir", "results/clip/selector_integration"))),
        cache_dir=Path(str(data.get("cache_dir", "results/clip/selector_integration/cache"))),
        checkpoint_hash=str(data.get("checkpoint_hash", "")),
        anchor_hash=str(data.get("anchor_hash", "")),
        statistical_view_scope=str(data.get("statistical_view_scope", "missingness_only")),
        screening_pool_size=int(data.get("screening_pool_size", 100)),
        missing_feature_policy=str(data.get("missing_feature_policy", "error")),
        feature_budgets={
            "lr": int(budgets.get("lr", 20)),
            "catboost": int(budgets.get("catboost", 40)),
        },
        active_datasets=tuple(str(item) for item in _list(data.get("active_datasets"), ["homecredit", "lendingclub_v2"])),
        legacy_datasets=tuple(str(item) for item in _list(data.get("legacy_datasets"), ["lendingclub"])),
        no_refit=bool(data.get("no_refit", True)),
    )


def validate_clip_selector_binding(config: ClipSelectorConfig) -> dict[str, Any]:
    paths = [
        config.selected_checkpoint_path,
        config.selected_checkpoint_manifest_path,
        config.model_selection_manifest_path,
        config.learned_anchor_manifest_path,
        config.training_manifest_path,
        config.text_embedding_manifest_path,
        config.statistical_preprocessor_path,
        config.source_manifest_path,
        config.homecredit_scores_path,
        config.lendingclub_v2_scores_path,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing CLIP selector artifacts: {missing}")
    if "lendingclub" in set(config.active_datasets):
        raise RuntimeError("legacy LendingClub is forbidden for CLIP selector integration")
    if set(config.active_datasets) != {"homecredit", "lendingclub_v2"}:
        raise RuntimeError(f"active datasets must be homecredit and lendingclub_v2: {config.active_datasets}")
    if not config.no_refit:
        raise RuntimeError("CLIP selector integration requires no_refit=true")
    if config.missing_feature_policy not in {"error", "exclude_with_manifest"}:
        raise RuntimeError("missing_feature_policy must be error or exclude_with_manifest")

    checkpoint_manifest = read_json(config.selected_checkpoint_manifest_path)
    selection = read_json(config.model_selection_manifest_path)
    anchor = read_json(config.learned_anchor_manifest_path)
    training = read_json(config.training_manifest_path)
    observed_checkpoint_hash = sha256_file(config.selected_checkpoint_path)
    expected_checkpoint_hash = config.checkpoint_hash or str(selection.get("selected_checkpoint_hash", ""))
    if observed_checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError("selected checkpoint hash mismatch")
    if checkpoint_manifest.get("checkpoint_sha256") != observed_checkpoint_hash:
        raise RuntimeError("checkpoint manifest hash mismatch")
    if selection.get("selected_checkpoint_hash") != observed_checkpoint_hash:
        raise RuntimeError("model selection manifest checkpoint mismatch")
    if selection.get("lendingclub_v2_used_for_selection"):
        raise RuntimeError("selected checkpoint was influenced by LendingClub v2")
    if anchor.get("anchor_dataset") != "homecredit":
        raise RuntimeError("learned anchor is not Home Credit")
    if "training-split" not in str(anchor.get("anchor_policy", "")):
        raise RuntimeError("learned anchor is not training-split only")
    expected_anchor_hash = config.anchor_hash or str(anchor.get("anchor_hash", ""))
    if anchor.get("anchor_hash") != expected_anchor_hash:
        raise RuntimeError("anchor hash mismatch")
    if anchor.get("checkpoint_hash") != observed_checkpoint_hash:
        raise RuntimeError("anchor checkpoint hash mismatch")
    if anchor.get("statistical_view_scope") != config.statistical_view_scope:
        raise RuntimeError("anchor statistical-view scope mismatch")
    if training.get("statistical_view_scope") != config.statistical_view_scope:
        raise RuntimeError("Prompt 5 statistical-view scope is missing or mismatched")

    for dataset, path in [("homecredit", config.homecredit_scores_path), ("lendingclub_v2", config.lendingclub_v2_scores_path)]:
        frame = pd.read_csv(path)
        validate_score_frame(frame, dataset=dataset, checkpoint_hash=observed_checkpoint_hash, anchor_hash=expected_anchor_hash)

    return {
        "experiment_version": config.experiment_version,
        "checkpoint_hash": observed_checkpoint_hash,
        "anchor_hash": expected_anchor_hash,
        "statistical_view_scope": config.statistical_view_scope,
        "selection_rule": selection.get("selection_rule"),
        "anchor_count": int(anchor.get("anchor_count", 0)),
        "fusion_rule": anchor.get("fusion_rule", FUSION_RULE),
        "text_embedding_manifest_hash": sha256_file(config.text_embedding_manifest_path),
        "statistical_preprocessor_hash_file": sha256_file(config.statistical_preprocessor_path),
        "source_manifest_hash_file": sha256_file(config.source_manifest_path),
    }


def validate_score_frame(frame: pd.DataFrame, *, dataset: str, checkpoint_hash: str, anchor_hash: str) -> None:
    required = {
        "dataset",
        "feature_name",
        "learned_similarity",
        "learned_rank",
        "checkpoint_hash",
        "anchor_hash",
        "source_manifest_hash",
        "statistical_view_scope",
        "projected_text_hash",
        "projected_statistical_hash",
        "joint_embedding_hash",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{dataset}: score frame missing columns {missing}")
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise RuntimeError(f"{dataset}: score dataset mismatch")
    if frame["feature_name"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate score features")
    if not frame["checkpoint_hash"].astype(str).eq(checkpoint_hash).all():
        raise RuntimeError(f"{dataset}: score checkpoint hash mismatch")
    if not frame["anchor_hash"].astype(str).eq(anchor_hash).all():
        raise RuntimeError(f"{dataset}: score anchor hash mismatch")
    if frame["feature_name"].astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{dataset}: blank score feature")


def _list(value: Any, default: list[str]) -> list[Any]:
    if value in (None, "[]"):
        return list(default)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)
