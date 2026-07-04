from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from credit_risk_fs.clip.model import ClipModelConfig
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.negative_policy import MASK_PRODUCING_REASONS, NEGATIVE_POLICY_VERSION
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json


STATISTICAL_VIEW_SCOPE_MISSINGNESS_ONLY = "missingness_only"


@dataclass(frozen=True)
class ClipTrainingConfig:
    tensor_schema_path: Path
    contrastive_pair_manifest_path: Path
    train_pairs_path: Path
    validation_pairs_path: Path
    external_pairs_path: Path
    negative_exclusion_pairs_path: Path
    negative_policy_manifest_path: Path
    homecredit_text_embeddings_path: Path
    lendingclub_v2_text_embeddings_path: Path
    homecredit_statistical_vectors_path: Path
    lendingclub_v2_statistical_vectors_path: Path
    text_embedding_manifest_path: Path
    statistical_preprocessor_path: Path
    source_manifest_path: Path
    split_manifest_path: Path
    output_dir: Path
    model: ClipModelConfig
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    minimum_improvement: float
    gradient_clipping_enabled: bool
    gradient_clip_norm: float
    seeds: tuple[int, ...]
    deterministic: bool
    device_policy: str
    selection_metric: str
    collapse_thresholds: dict[str, float]
    statistical_view_scope: str
    smoke_test_steps: int
    training_dataset: str
    external_dataset: str
    configuration_hash: str
    data_manifest_hash: str
    statistical_preprocessor_hash: str
    source_anchor_hash: str


@dataclass(frozen=True)
class TrainingDataBundle:
    train_pairs: pd.DataFrame
    validation_pairs: pd.DataFrame
    external_pairs: pd.DataFrame
    source_pairs: pd.DataFrame
    training_text: pd.DataFrame
    external_text: pd.DataFrame
    training_stat: pd.DataFrame
    external_stat: pd.DataFrame
    training_dataset: str
    external_dataset: str
    negative_exclusions: pd.DataFrame
    upstream_hashes: dict[str, str]
    text_dim: int
    statistical_dim: int
    statistical_fields: list[str]


def load_training_config(
    path: str | Path = "configs/corrected_homecredit_clip/training.yaml",
) -> ClipTrainingConfig:
    data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    model_data = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
    return ClipTrainingConfig(
        tensor_schema_path=Path(str(data.get("tensor_schema_path", "results/corrected_homecredit_clip/contrastive_data/contrastive_tensor_schema.json"))),
        contrastive_pair_manifest_path=Path(
            str(data.get("contrastive_pair_manifest_path", "results/corrected_homecredit_clip/contrastive_data/contrastive_pair_manifest.json"))
        ),
        train_pairs_path=Path(str(data.get("train_pairs_path", "results/corrected_homecredit_clip/contrastive_data/homecredit_train_positive_pairs.parquet"))),
        validation_pairs_path=Path(
            str(data.get("validation_pairs_path", "results/corrected_homecredit_clip/contrastive_data/homecredit_validation_positive_pairs.parquet"))
        ),
        external_pairs_path=Path(str(data.get("external_pairs_path", "results/corrected_homecredit_clip/contrastive_data/lendingclub_v2_external_pairs.parquet"))),
        negative_exclusion_pairs_path=Path(
            str(data.get("negative_exclusion_pairs_path", "results/corrected_homecredit_clip/contrastive_data/negative_exclusion_pairs.parquet"))
        ),
        negative_policy_manifest_path=Path(
            str(data.get("negative_policy_manifest_path", "results/corrected_homecredit_clip/contrastive_data/negative_policy_manifest.json"))
        ),
        homecredit_text_embeddings_path=Path(
            str(data.get("homecredit_text_embeddings_path", "results/clip/text_baseline/homecredit_text_embeddings.parquet"))
        ),
        lendingclub_v2_text_embeddings_path=Path(
            str(data.get("lendingclub_v2_text_embeddings_path", "results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet"))
        ),
        homecredit_statistical_vectors_path=Path(
            str(data.get("homecredit_statistical_vectors_path", "results/clip_v2/statistical_view/homecredit_statistical_vectors.parquet"))
        ),
        lendingclub_v2_statistical_vectors_path=Path(
            str(data.get("lendingclub_v2_statistical_vectors_path", "results/clip_v2/statistical_view/lendingclub_v2_statistical_vectors.parquet"))
        ),
        text_embedding_manifest_path=Path(
            str(data.get("text_embedding_manifest_path", "results/clip/text_baseline/embedding_cache_manifest.json"))
        ),
        statistical_preprocessor_path=Path(
            str(data.get("statistical_preprocessor_path", "results/clip_v2/statistical_view/statistical_preprocessor.json"))
        ),
        source_manifest_path=Path(str(data.get("source_manifest_path", "results/clip/dry_run/training_manifest.json"))),
        split_manifest_path=Path(str(data.get("split_manifest_path", "results/corrected_homecredit_clip/contrastive_data/split_manifest.json"))),
        output_dir=Path(str(data.get("output_dir", "results/corrected_homecredit_clip/training"))),
        model=ClipModelConfig(
            text_input_dim=int(model_data.get("text_input_dim", 384)),
            statistical_input_dim=int(model_data.get("statistical_input_dim", 1)),
            text_hidden_dim=int(model_data.get("text_hidden_dim", 64)),
            statistical_hidden_dim=int(model_data.get("statistical_hidden_dim", 16)),
            shared_embedding_dim=int(model_data.get("shared_embedding_dim", 32)),
            dropout=float(model_data.get("dropout", 0.05)),
            activation=str(model_data.get("activation", "gelu")),
            initial_temperature=float(model_data.get("initial_temperature", 0.07)),
            trainable_temperature=bool(model_data.get("trainable_temperature", False)),
            min_temperature=float(model_data.get("min_temperature", 0.02)),
            max_temperature=float(model_data.get("max_temperature", 0.5)),
        ),
        optimizer=str(data.get("optimizer", "AdamW")),
        learning_rate=float(data.get("learning_rate", 0.001)),
        weight_decay=float(data.get("weight_decay", 0.01)),
        batch_size=int(data.get("batch_size", 64)),
        max_epochs=int(data.get("max_epochs", 80)),
        early_stopping_patience=int(data.get("early_stopping_patience", 15)),
        minimum_improvement=float(data.get("minimum_improvement", 0.0001)),
        gradient_clipping_enabled=bool(data.get("gradient_clipping_enabled", True)),
        gradient_clip_norm=float(data.get("gradient_clip_norm", 1.0)),
        seeds=tuple(int(seed) for seed in _list(data.get("seeds"), [11, 22, 33, 44, 55])),
        deterministic=bool(data.get("deterministic", True)),
        device_policy=str(data.get("device_policy", "cpu")),
        selection_metric=str(data.get("selection_metric", "validation_loss")),
        collapse_thresholds=data.get("collapse_thresholds", {}) if isinstance(data.get("collapse_thresholds"), dict) else {},
        statistical_view_scope=str(data.get("statistical_view_scope", STATISTICAL_VIEW_SCOPE_MISSINGNESS_ONLY)),
        smoke_test_steps=int(data.get("smoke_test_steps", 3)),
        training_dataset=str(data.get("training_dataset", "homecredit")),
        external_dataset=str(
            data.get("external_dataset", data.get("external_validation_dataset", "lendingclub_v2"))
        ),
        configuration_hash=str(data.get("configuration_hash", "")),
        data_manifest_hash=str(data.get("data_manifest_hash", "")),
        statistical_preprocessor_hash=str(data.get("statistical_preprocessor_hash", "")),
        source_anchor_hash=str(data.get("source_anchor_hash", "")),
    )


def load_and_validate_training_inputs(config: ClipTrainingConfig) -> TrainingDataBundle:
    paths = [
        config.tensor_schema_path,
        config.contrastive_pair_manifest_path,
        config.train_pairs_path,
        config.validation_pairs_path,
        config.external_pairs_path,
        config.negative_exclusion_pairs_path,
        config.negative_policy_manifest_path,
        config.homecredit_text_embeddings_path,
        config.lendingclub_v2_text_embeddings_path,
        config.homecredit_statistical_vectors_path,
        config.lendingclub_v2_statistical_vectors_path,
        config.text_embedding_manifest_path,
        config.statistical_preprocessor_path,
        config.source_manifest_path,
        config.split_manifest_path,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing CLIP training inputs: {missing}")

    tensor_schema = read_json(config.tensor_schema_path)
    pair_manifest = read_json(config.contrastive_pair_manifest_path)
    split_manifest = read_json(config.split_manifest_path)
    negative_manifest = read_json(config.negative_policy_manifest_path)
    stat_preprocessor = read_json(config.statistical_preprocessor_path)

    text_dim = int(tensor_schema["text_embedding_dimension"])
    stat_dim = int(tensor_schema["statistical_vector_dimension"])
    if text_dim != config.model.text_input_dim:
        raise RuntimeError(f"text dim mismatch: config={config.model.text_input_dim}, artifact={text_dim}")
    if stat_dim != config.model.statistical_input_dim:
        raise RuntimeError(f"statistical dim mismatch: config={config.model.statistical_input_dim}, artifact={stat_dim}")
    if stat_dim == 1 and config.statistical_view_scope != STATISTICAL_VIEW_SCOPE_MISSINGNESS_ONLY:
        raise RuntimeError("one-dimensional statistical view must be labeled missingness_only")
    if split_manifest.get("base_family_overlap_count") != 0:
        raise RuntimeError("canonical/base family overlap is nonzero")
    if negative_manifest.get("cross_dataset_negatives_enabled") or negative_manifest.get("validation_as_training_negative"):
        raise RuntimeError("negative policy violates training boundary")
    if negative_manifest.get("policy_version") != NEGATIVE_POLICY_VERSION:
        raise RuntimeError("negative policy artifact is stale and requires rebuilding")
    if config.training_dataset == config.external_dataset:
        raise RuntimeError("training and external datasets must be different")
    if (
        pair_manifest.get("training_dataset") != config.training_dataset
        or pair_manifest.get("external_validation_dataset") != config.external_dataset
    ):
        raise RuntimeError("pair manifest dataset boundary mismatch")

    train = pd.read_parquet(config.train_pairs_path)
    validation = pd.read_parquet(config.validation_pairs_path)
    external = pd.read_parquet(config.external_pairs_path)
    negative = pd.read_parquet(config.negative_exclusion_pairs_path)
    by_dataset_text = {
        "homecredit": pd.read_parquet(config.homecredit_text_embeddings_path),
        "lendingclub_v2": pd.read_parquet(config.lendingclub_v2_text_embeddings_path),
    }
    by_dataset_stat = {
        "homecredit": pd.read_parquet(config.homecredit_statistical_vectors_path),
        "lendingclub_v2": pd.read_parquet(config.lendingclub_v2_statistical_vectors_path),
    }
    if config.training_dataset not in by_dataset_text or config.external_dataset not in by_dataset_text:
        raise RuntimeError("training loader has no configured artifact path for a declared dataset")
    training_text = by_dataset_text[config.training_dataset]
    external_text = by_dataset_text[config.external_dataset]
    training_stat = by_dataset_stat[config.training_dataset]
    external_stat = by_dataset_stat[config.external_dataset]

    _validate_pair_roles(train, dataset=config.training_dataset, split="train", training=True)
    _validate_pair_roles(validation, dataset=config.training_dataset, split="validation", validation=True)
    _validate_pair_roles(
        external,
        dataset=config.external_dataset,
        split="external_validation",
        external=True,
    )
    _validate_no_forbidden_columns(set(train.columns).union(set(validation.columns)).union(set(external.columns)))
    _validate_alignment(train, training_text, training_stat, text_dim=text_dim, stat_dim=stat_dim)
    _validate_alignment(validation, training_text, training_stat, text_dim=text_dim, stat_dim=stat_dim)
    _validate_alignment(external, external_text, external_stat, text_dim=text_dim, stat_dim=stat_dim)

    return TrainingDataBundle(
        train_pairs=train.sort_values("feature_name", kind="mergesort").reset_index(drop=True),
        validation_pairs=validation.sort_values("feature_name", kind="mergesort").reset_index(drop=True),
        external_pairs=external.sort_values("feature_name", kind="mergesort").reset_index(drop=True),
        source_pairs=_reindex_combined_positive_pairs(
            pd.concat([train, validation], ignore_index=True)
        ),
        training_text=training_text,
        external_text=external_text,
        training_stat=training_stat,
        external_stat=external_stat,
        training_dataset=config.training_dataset,
        external_dataset=config.external_dataset,
        negative_exclusions=negative,
        upstream_hashes=_upstream_hashes(config),
        text_dim=text_dim,
        statistical_dim=stat_dim,
        statistical_fields=list(stat_preprocessor.get("field_order", [])),
    )


def tensors_for_pairs(
    pairs: pd.DataFrame,
    text_frame: pd.DataFrame,
    stat_frame: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_positive_identity(pairs)
    text_cols = _embedding_columns(text_frame)
    stat_cols = _statistical_columns(stat_frame)
    text_by_key = text_frame.set_index("embedding_cache_key", drop=False)
    stat_by_key = stat_frame.set_index("stable_row_id", drop=False)
    text_rows = []
    stat_rows = []
    for row in pairs.itertuples(index=False):
        text_row = text_by_key.loc[str(row.text_embedding_row_id)]
        stat_row = stat_by_key.loc[str(row.statistical_vector_row_id)]
        if str(text_row["feature_name"]) != str(row.feature_name):
            raise RuntimeError(f"text alignment failed for {row.feature_name}")
        if str(stat_row["feature_name"]) != str(row.feature_name):
            raise RuntimeError(f"statistical alignment failed for {row.feature_name}")
        text_rows.append(text_row[text_cols].to_numpy(dtype=np.float32))
        stat_rows.append(stat_row[stat_cols].to_numpy(dtype=np.float32))
    return torch.tensor(np.vstack(text_rows), dtype=torch.float32), torch.tensor(np.vstack(stat_rows), dtype=torch.float32)


def false_negative_mask(pairs: pd.DataFrame, exclusions: pd.DataFrame | None = None) -> torch.Tensor:
    features = pairs["feature_name"].astype(str).tolist()
    if len(features) != len(set(features)):
        raise ValueError("pair frame contains duplicate feature identities")
    order_hash = feature_order_hash(features)
    index = {feature: idx for idx, feature in enumerate(features)}
    mask = np.zeros((len(features), len(features)), dtype=bool)
    if exclusions is not None and len(exclusions):
        reasons = set(exclusions["exclusion_reason"].astype(str))
        unsupported = reasons - MASK_PRODUCING_REASONS
        if unsupported:
            raise ValueError(f"unsupported mask-producing reasons: {sorted(unsupported)}")
        if "policy_version" in exclusions and set(exclusions["policy_version"].astype(str)) != {NEGATIVE_POLICY_VERSION}:
            raise ValueError("negative exclusion policy version is stale")
        if "feature_order_hash" in exclusions and set(exclusions["feature_order_hash"].astype(str)) != {order_hash}:
            raise ValueError("negative exclusion feature order hash is stale")
        for row in exclusions.to_dict("records"):
            a = str(row.get("anchor_feature_name"))
            b = str(row.get("excluded_feature_name"))
            if a in index and b in index and a != b:
                mask[index[a], index[b]] = True
    np.fill_diagonal(mask, False)
    if not np.array_equal(mask, mask.T):
        raise ValueError("negative exclusion mask must be symmetric")
    valid_negatives = (~mask).sum(axis=1) - 1
    if len(features) > 1 and (valid_negatives < 1).any():
        raise ValueError("negative exclusion mask leaves a row with zero valid negatives")
    return torch.tensor(mask, dtype=torch.bool)


def resolve_device(policy: str) -> torch.device:
    if policy == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_pair_roles(frame: pd.DataFrame, *, dataset: str, split: str, training: bool = False, validation: bool = False, external: bool = False) -> None:
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise RuntimeError(f"{dataset}: pair dataset mismatch")
    if set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"{dataset}: pair split mismatch")
    if training and not frame["allowed_for_training"].astype(bool).all():
        raise RuntimeError("training pairs are not all training-eligible")
    if validation and not frame["allowed_for_validation"].astype(bool).all():
        raise RuntimeError("validation pairs are not all validation-eligible")
    if external and frame["allowed_for_training"].astype(bool).any():
        raise RuntimeError("external pairs are training-eligible")


def _validate_alignment(pairs: pd.DataFrame, text: pd.DataFrame, stat: pd.DataFrame, *, text_dim: int, stat_dim: int) -> None:
    text_cols = _embedding_columns(text)
    stat_cols = _statistical_columns(stat)
    if len(text_cols) != text_dim or len(stat_cols) != stat_dim:
        raise RuntimeError("tensor dimensions do not match schema")
    tensors_for_pairs(pairs, text, stat)


def _validate_positive_identity(pairs: pd.DataFrame) -> None:
    required = {
        "feature_id",
        "dataset",
        "feature_name",
        "source_manifest_hash",
        "text_embedding_row_id",
        "statistical_vector_row_id",
        "positive_pair_index",
        "feature_order_hash",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise RuntimeError(f"positive pair identity columns missing: {sorted(missing)}")
    for column in ["feature_id", "feature_name", "positive_pair_index"]:
        if pairs[column].duplicated().any():
            raise RuntimeError(f"duplicate positive identity values in {column}")
    expected_indexes = list(range(len(pairs)))
    if pairs["positive_pair_index"].astype(int).tolist() != expected_indexes:
        raise RuntimeError("positive-pair indices do not match row order")
    expected_hash = feature_order_hash(pairs["feature_name"].astype(str).tolist())
    if set(pairs["feature_order_hash"].astype(str)) != {expected_hash}:
        raise RuntimeError("positive-pair feature order hash is stale")


def _reindex_combined_positive_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = pairs.sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    frame["positive_pair_index"] = range(len(frame))
    frame["feature_order_hash"] = feature_order_hash(
        frame["feature_name"].astype(str).tolist()
    )
    return frame


def _validate_no_forbidden_columns(columns: set[str]) -> None:
    forbidden_tokens = ["llm", "oot", "psi", "target", "label", "prediction", "post_origination", "stable_core"]
    bad = [column for column in columns if any(token in str(column).lower() for token in forbidden_tokens)]
    if bad:
        raise RuntimeError(f"forbidden model-input columns present in pair frame: {bad}")


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([col for col in frame.columns if str(col).startswith("embedding_") and len(str(col)) == 14])


def _statistical_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([col for col in frame.columns if str(col).startswith("stat_") and len(str(col)) == 9])


def _upstream_hashes(config: ClipTrainingConfig) -> dict[str, str]:
    return {
        "tensor_schema_hash": sha256_file(config.tensor_schema_path),
        "contrastive_pair_manifest_hash": sha256_file(config.contrastive_pair_manifest_path),
        "negative_policy_manifest_hash": sha256_file(config.negative_policy_manifest_path),
        "text_embedding_manifest_hash": sha256_file(config.text_embedding_manifest_path),
        "statistical_preprocessor_hash_file": sha256_file(config.statistical_preprocessor_path),
        "source_manifest_hash": sha256_file(config.source_manifest_path),
        "split_manifest_hash": sha256_file(config.split_manifest_path),
    }


def _list(value: Any, default: list[int]) -> list[Any]:
    if value in (None, "[]"):
        return list(default)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)
