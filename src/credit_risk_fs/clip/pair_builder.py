from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip.contrastive_schema import CONTRASTIVE_DATA_VERSION, ContrastiveBuildResult, ContrastiveDataConfig
from credit_risk_fs.clip.exact_duplicates import feature_order_hash
from credit_risk_fs.clip.negative_policy import build_negative_policy
from credit_risk_fs.clip.pair_validation import (
    embedding_columns,
    statistical_columns,
    validate_contrastive_config,
    validate_group_split,
    validate_manifest_boundary,
    validate_positive_pairs,
    validate_required_contrastive_artifacts,
    validate_view_frame,
)
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.hashing import sha256_file, sha256_text
from credit_risk_fs.utils.io import read_json, write_json


def load_contrastive_data_config(
    path: str | Path = "configs/corrected_homecredit_clip/contrastive_data.yaml",
) -> ContrastiveDataConfig:
    data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    return ContrastiveDataConfig(
        manifest_path=Path(str(data.get("manifest_path", "results/clip/dry_run/training_manifest.json"))),
        source_hashes_path=Path(str(data.get("source_hashes_path", "results/clip/dry_run/source_hashes.json"))),
        group_split_path=Path(str(data.get("group_split_path", "results/clip/text_baseline/homecredit_group_split.csv"))),
        homecredit_text_embeddings_path=Path(str(data.get("homecredit_text_embeddings_path", ""))),
        lendingclub_v2_text_embeddings_path=Path(str(data.get("lendingclub_v2_text_embeddings_path", ""))),
        embedding_cache_manifest_path=Path(str(data.get("embedding_cache_manifest_path", ""))),
        text_embedding_audit_path=Path(str(data.get("text_embedding_audit_path", ""))),
        homecredit_statistical_vectors_path=Path(str(data.get("homecredit_statistical_vectors_path", ""))),
        lendingclub_v2_statistical_vectors_path=Path(str(data.get("lendingclub_v2_statistical_vectors_path", ""))),
        statistical_feature_order_path=Path(str(data.get("statistical_feature_order_path", ""))),
        statistical_preprocessor_path=Path(str(data.get("statistical_preprocessor_path", ""))),
        statistical_anchor_manifest_path=Path(str(data.get("statistical_anchor_manifest_path", ""))),
        exact_dev_duplicate_pairs_path=Path(str(data.get("exact_dev_duplicate_pairs_path", ""))),
        homecredit_feature_text_path=Path(str(data.get("homecredit_feature_text_path", ""))),
        lendingclub_v2_feature_text_path=Path(str(data.get("lendingclub_v2_feature_text_path", ""))),
        output_dir=Path(
            str(data.get("output_dir", "results/corrected_homecredit_clip/contrastive_data"))
        ),
        seed=int(data.get("seed", 42)),
        training_dataset=str(data.get("training_dataset", "homecredit")),
        external_validation_dataset=str(data.get("external_validation_dataset", "lendingclub_v2")),
        legacy_lendingclub_allowed=bool(data.get("legacy_lendingclub_allowed", False)),
        pair_key_fields=_list(data.get("pair_key_fields"), ["dataset", "feature_name", "split"]),
        negative_policy=data.get("negative_policy", {}) if isinstance(data.get("negative_policy"), dict) else {},
        external_validation_policy=data.get("external_validation_policy", {})
        if isinstance(data.get("external_validation_policy"), dict)
        else {},
        tensor_schema=data.get("tensor_schema", {}) if isinstance(data.get("tensor_schema"), dict) else {},
        training_feature_manifest=Path(
            str(data.get("training_feature_manifest", data.get("manifest_path", "")))
        ),
        external_feature_manifest=Path(
            str(data.get("external_feature_manifest", data.get("manifest_path", "")))
        ),
        training_raw_statistical_source=Path(
            str(data.get("training_raw_statistical_source", ""))
        ),
        external_raw_statistical_source=Path(
            str(data.get("external_raw_statistical_source", ""))
        ),
        training_statistical_fit_scope=str(
            data.get("training_statistical_fit_scope", "dev_training_features_only")
        ),
        external_statistical_transform_scope=str(
            data.get("external_statistical_transform_scope", "transform_only")
        ),
    )


def build_contrastive_data(*, config: ContrastiveDataConfig, dry_run: bool) -> ContrastiveBuildResult:
    errors = validate_contrastive_config(config)
    required = [
        config.manifest_path,
        config.source_hashes_path,
        config.group_split_path,
        config.homecredit_text_embeddings_path,
        config.lendingclub_v2_text_embeddings_path,
        config.embedding_cache_manifest_path,
        config.text_embedding_audit_path,
        config.homecredit_statistical_vectors_path,
        config.lendingclub_v2_statistical_vectors_path,
        config.statistical_feature_order_path,
        config.statistical_preprocessor_path,
        config.statistical_anchor_manifest_path,
        config.exact_dev_duplicate_pairs_path,
        config.homecredit_feature_text_path,
        config.lendingclub_v2_feature_text_path,
    ]
    errors.extend(validate_required_contrastive_artifacts(required))
    if errors:
        raise RuntimeError("; ".join(errors))

    manifest = read_json(config.manifest_path)
    source_hashes = read_json(config.source_hashes_path)
    errors.extend(validate_manifest_boundary(manifest=manifest, source_hashes=source_hashes, config=config))
    if errors:
        raise RuntimeError("; ".join(errors))

    embedding_manifest = read_json(config.embedding_cache_manifest_path)
    embedding_audit = read_json(config.text_embedding_audit_path)
    feature_order = read_json(config.statistical_feature_order_path)
    preprocessor = read_json(config.statistical_preprocessor_path)
    stat_anchor = read_json(config.statistical_anchor_manifest_path)
    exact_dev_duplicates = pd.read_parquet(config.exact_dev_duplicate_pairs_path)
    home_text = pd.read_parquet(config.homecredit_text_embeddings_path)
    lc_text = pd.read_parquet(config.lendingclub_v2_text_embeddings_path)
    home_stat = pd.read_parquet(config.homecredit_statistical_vectors_path)
    lc_stat = pd.read_parquet(config.lendingclub_v2_statistical_vectors_path)
    home_feature_text = pd.read_csv(config.homecredit_feature_text_path)
    lc_feature_text = pd.read_csv(config.lendingclub_v2_feature_text_path)
    split = pd.read_csv(config.group_split_path)

    text_dim = int(embedding_audit["embedding_dimension"])
    stat_dim = int(feature_order["vector_dimension"])
    errors.extend(validate_group_split(split, dataset=config.training_dataset))
    errors.extend(validate_view_frame(text=home_text, stat=home_stat, dataset="homecredit", expected_text_dim=text_dim, expected_stat_dim=stat_dim))
    errors.extend(
        validate_view_frame(text=lc_text, stat=lc_stat, dataset="lendingclub_v2", expected_text_dim=text_dim, expected_stat_dim=stat_dim)
    )
    if preprocessor.get("preprocessor_hash") != stat_anchor.get("preprocessor_hash"):
        errors.append("statistical preprocessor hash does not match statistical anchor manifest")
    if errors:
        raise RuntimeError("; ".join(errors))

    home_pairs = build_positive_pairs(
        text=home_text,
        stat=home_stat,
        feature_text=home_feature_text,
        dataset="homecredit",
        text_dim=text_dim,
        stat_dim=stat_dim,
        training_dataset=config.training_dataset,
    )
    lc_pairs = build_positive_pairs(
        text=lc_text,
        stat=lc_stat,
        feature_text=lc_feature_text,
        dataset="lendingclub_v2",
        text_dim=text_dim,
        stat_dim=stat_dim,
        training_dataset=config.training_dataset,
    )
    pairs_by_dataset = {"homecredit": home_pairs, "lendingclub_v2": lc_pairs}
    source_pairs = pairs_by_dataset[config.training_dataset]
    external_source_pairs = pairs_by_dataset[config.external_validation_dataset]
    train_pairs = source_pairs[source_pairs["split"].eq("train")].copy()
    validation_pairs = source_pairs[source_pairs["split"].eq("validation")].copy()
    external_pairs = external_source_pairs.copy()
    train_pairs = _index_positive_pairs(train_pairs)
    validation_pairs = _index_positive_pairs(validation_pairs)
    external_pairs = _index_positive_pairs(external_pairs)
    errors.extend(validate_positive_pairs(train_pairs, role="train_positive", dataset=config.training_dataset, split="train"))
    errors.extend(validate_positive_pairs(validation_pairs, role="validation_positive", dataset=config.training_dataset, split="validation"))
    errors.extend(
        validate_positive_pairs(
            external_pairs,
            role="external_validation_positive",
            dataset=config.external_validation_dataset,
            split="external_validation",
        )
    )
    base_overlap = _base_family_overlap(train_pairs, validation_pairs)
    if base_overlap:
        errors.append(f"base-family overlap between train and validation: {base_overlap[:20]}")
    if errors:
        raise RuntimeError("; ".join(errors))

    negative = build_negative_policy(
        train_pairs=train_pairs,
        all_homecredit_pairs=source_pairs,
        text_embeddings=(
            home_text if config.training_dataset == "homecredit" else lc_text
        ),
        exact_dev_duplicates=exact_dev_duplicates,
        training_dataset=config.training_dataset,
        verified_aliases=config.negative_policy.get("verified_aliases", []),
        documented_identity_transforms=config.negative_policy.get("documented_identity_transforms", []),
        near_duplicate_text_threshold=float(config.negative_policy.get("near_duplicate_text_threshold", 0.95)),
        min_safe_negative_count=int(config.negative_policy.get("min_safe_negative_count", 25)),
    )
    split_manifest = {
        "training_dataset": config.training_dataset,
        "external_validation_dataset": config.external_validation_dataset,
        "training_pair_count": int(len(train_pairs)),
        "validation_pair_count": int(len(validation_pairs)),
        "external_pair_count": int(len(external_pairs)),
        "group_overlap_count": 0,
        "base_family_overlap_count": len(base_overlap),
        "base_family_overlap": base_overlap,
        "split_hash": sha256_file(config.group_split_path),
    }
    tensor_schema = {
        "contrastive_data_version": CONTRASTIVE_DATA_VERSION,
        "text_embedding_dimension": text_dim,
        "statistical_vector_dimension": stat_dim,
        "expected_dtype": config.tensor_schema.get("expected_dtype", "float32"),
        "text_normalization_state": {"normalize_embeddings": bool(embedding_manifest.get("normalize_embeddings", True))},
        "statistical_feature_order": feature_order.get("field_order", []),
        "padding_policy": config.tensor_schema.get("padding_policy") or "none",
        "missing_value_policy": {
            "imputation_strategy": preprocessor.get("imputation_strategy"),
            "fit_dataset": preprocessor.get("fit_dataset"),
            "fit_split": preprocessor.get("fit_split"),
        },
        "text_encoder_identity": embedding_manifest.get("encoder_model"),
        "text_encoder_revision": embedding_manifest.get("encoder_revision"),
        "statistical_preprocessor_hash": preprocessor.get("preprocessor_hash"),
        "source_manifest_hash": sha256_file(config.manifest_path),
        "split_hash": sha256_file(config.group_split_path),
        "forbidden_tensor_fields": [
            "stable_core_membership",
            "llm_best_rank",
            "llm_mean_rank_if_available",
            "target",
            "oot",
            "psi",
            "prediction",
            "post_origination_outcome",
        ],
    }
    quality = _quality_audit(
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        external_pairs=external_pairs,
        negative_manifest=negative.manifest,
        split_manifest=split_manifest,
    )
    output_dir = config.output_dir / "dry_run" if dry_run else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not dry_run:
        paths["training_positive_pairs"] = _save_parquet(
            train_pairs, output_dir / f"{config.training_dataset}_train_positive_pairs.parquet"
        )
        paths["validation_positive_pairs"] = _save_parquet(
            validation_pairs,
            output_dir / f"{config.training_dataset}_validation_positive_pairs.parquet"
        )
        paths["external_positive_pairs"] = _save_parquet(
            external_pairs,
            output_dir / f"{config.external_validation_dataset}_external_pairs.parquet"
        )
        if (
            config.training_dataset == "homecredit"
            and config.external_validation_dataset == "lendingclub_v2"
        ):
            paths["homecredit_train_positive_pairs"] = paths["training_positive_pairs"]
            paths["homecredit_validation_positive_pairs"] = paths[
                "validation_positive_pairs"
            ]
            paths["lendingclub_v2_external_pairs"] = paths["external_positive_pairs"]
        paths["negative_exclusion_pairs"] = _save_parquet(negative.exclusion_pairs, output_dir / "negative_exclusion_pairs.parquet")
        paths["near_duplicate_text_audit"] = output_dir / "near_duplicate_text_audit.csv"
        negative.near_duplicate_audit.to_csv(paths["near_duplicate_text_audit"], index=False)
    paths["contrastive_tensor_schema"] = write_json(output_dir / "contrastive_tensor_schema.json", tensor_schema)
    pair_manifest = _pair_manifest(
        config=config,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        external_pairs=external_pairs,
        tensor_schema=tensor_schema,
        negative_manifest=negative.manifest,
    )
    pair_manifest["dry_run"] = bool(dry_run)
    paths["contrastive_pair_manifest"] = write_json(output_dir / "contrastive_pair_manifest.json", pair_manifest)
    paths["split_manifest"] = write_json(output_dir / "split_manifest.json", split_manifest)
    paths["negative_policy_manifest"] = write_json(output_dir / "negative_policy_manifest.json", negative.manifest)
    paths["negative_candidate_audit"] = output_dir / "negative_candidate_audit.csv"
    negative.candidate_audit.to_csv(paths["negative_candidate_audit"], index=False)
    paths["near_duplicate_threshold_sensitivity"] = output_dir / "near_duplicate_threshold_sensitivity.csv"
    negative.threshold_sensitivity.to_csv(paths["near_duplicate_threshold_sensitivity"], index=False)
    paths["pair_quality_audit_csv"] = output_dir / "pair_quality_audit.csv"
    quality.to_csv(paths["pair_quality_audit_csv"], index=False)
    paths["pair_quality_audit_json"] = write_json(
        output_dir / "pair_quality_audit.json",
        {"checks": quality.to_dict("records"), "all_passed": bool(quality["passed"].all())},
    )
    summary = {
        "dry_run": bool(dry_run),
        "model_trained": False,
        "optimizer_created": False,
        "checkpoint_created": False,
        "training_log_created": False,
        "text_embedding_dimension": text_dim,
        "statistical_vector_dimension": stat_dim,
        "training_pair_count": int(len(train_pairs)),
        "validation_pair_count": int(len(validation_pairs)),
        "external_pair_count": int(len(external_pairs)),
        "negative_excluded_counts_by_reason": negative.manifest["excluded_reason_counts"],
        "near_duplicate_text_threshold": float(config.negative_policy.get("near_duplicate_text_threshold", 0.95)),
        "remaining_safe_negative_min": int(negative.candidate_audit["remaining_safe_negative_count"].min()),
        "remaining_safe_negative_median": float(negative.candidate_audit["remaining_safe_negative_count"].median()),
        "remaining_safe_negative_max": int(negative.candidate_audit["remaining_safe_negative_count"].max()),
        "group_overlap_count": 0,
        "base_family_overlap_count": len(base_overlap),
        "tensor_schema_path": str(paths["contrastive_tensor_schema"]).replace("\\", "/"),
        "source_manifest_hash": sha256_file(config.manifest_path),
        "split_hash": sha256_file(config.group_split_path),
        "pair_manifest_hash": sha256_file(paths["contrastive_pair_manifest"]),
    }
    if (
        config.training_dataset == "homecredit"
        and config.external_validation_dataset == "lendingclub_v2"
    ):
        summary.update(
            {
                "homecredit_train_pair_count": int(len(train_pairs)),
                "homecredit_validation_pair_count": int(len(validation_pairs)),
                "lendingclub_v2_external_pair_count": int(len(external_pairs)),
            }
        )
    return ContrastiveBuildResult(output_paths=paths, summary=summary)


def build_positive_pairs(
    *,
    text: pd.DataFrame,
    stat: pd.DataFrame,
    feature_text: pd.DataFrame,
    dataset: str,
    text_dim: int,
    stat_dim: int,
    training_dataset: str = "homecredit",
) -> pd.DataFrame:
    text_meta = text[
        ["dataset", "feature_name", "feature_text_hash", "source_manifest_hash", "embedding_cache_key"]
    ].rename(columns={"feature_text_hash": "text_hash", "embedding_cache_key": "text_embedding_row_id"})
    stat_meta = stat[
        [
            "dataset",
            "feature_name",
            "split",
            "group_key",
            "semantic_group",
            "source_table_or_formula",
            "source_manifest_hash",
            "stable_row_id",
            "statistical_vector_hash",
            "vector_dimension",
            "canonical_feature_family",
            "family_resolution_source",
            "family_resolution_rule",
            "family_member_count",
        ]
    ].rename(columns={"stable_row_id": "statistical_vector_row_id"})
    text_strings = feature_text[["dataset", "feature_name", "feature_text"]].copy()
    text_strings["normalized_feature_text"] = text_strings["feature_text"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
    text_strings["normalized_text_hash"] = text_strings["normalized_feature_text"].map(sha256_text)
    merged = stat_meta.merge(text_meta, on=["dataset", "feature_name", "source_manifest_hash"], how="inner").merge(
        text_strings[["dataset", "feature_name", "normalized_text_hash"]], on=["dataset", "feature_name"], how="left"
    )
    merged["group_source"] = merged["group_key"].map(_group_source_from_key)
    merged["base_feature_family"] = [
        _base_feature_family(
            feature_name=str(row.feature_name),
            group_key=str(row.group_key),
            canonical_feature_family=str(row.canonical_feature_family),
            family_member_count=int(row.family_member_count),
        )
        for row in merged.itertuples(index=False)
    ]
    merged["text_embedding_dimension"] = int(text_dim)
    merged["statistical_vector_dimension"] = int(stat_dim)
    if dataset == training_dataset:
        merged["pair_role"] = merged["split"].map(lambda value: "train_positive" if value == "train" else "validation_positive")
        merged["allowed_for_training"] = merged["split"].eq("train")
        merged["allowed_for_validation"] = merged["split"].eq("validation")
        merged["allowed_for_external_evaluation"] = False
    else:
        merged["pair_role"] = "external_validation_positive"
        merged["allowed_for_training"] = False
        merged["allowed_for_validation"] = False
        merged["allowed_for_external_evaluation"] = True
    merged["pair_id"] = [
        sha256_text(
            "|".join(
                [
                    str(row.dataset),
                    str(row.feature_name),
                    str(row.split),
                    str(row.group_key),
                    str(row.text_hash),
                    str(row.statistical_vector_hash),
                ]
            )
        )
        for row in merged.itertuples(index=False)
    ]
    merged = merged.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
    merged["feature_id"] = [
        sha256_text(f"{row.dataset}|{row.feature_name}|{row.source_manifest_hash}")
        for row in merged.itertuples(index=False)
    ]
    merged["positive_pair_index"] = range(len(merged))
    merged["feature_order_hash"] = feature_order_hash(merged["feature_name"].astype(str).tolist())
    columns = [
        "feature_id",
        "positive_pair_index",
        "feature_order_hash",
        "pair_id",
        "dataset",
        "feature_name",
        "split",
        "group_key",
        "group_source",
        "semantic_group",
        "source_table_or_formula",
        "base_feature_family",
        "canonical_feature_family",
        "family_resolution_source",
        "family_resolution_rule",
        "family_member_count",
        "text_embedding_row_id",
        "statistical_vector_row_id",
        "text_hash",
        "statistical_vector_hash",
        "normalized_text_hash",
        "text_embedding_dimension",
        "statistical_vector_dimension",
        "source_manifest_hash",
        "pair_role",
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_external_evaluation",
    ]
    return merged[columns]


def _index_positive_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = pairs.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
    frame["positive_pair_index"] = range(len(frame))
    frame["feature_order_hash"] = feature_order_hash(frame["feature_name"].astype(str).tolist())
    return frame


def _group_source_from_key(group_key: str) -> str:
    text = str(group_key)
    if text.startswith("family:"):
        return "derived_feature_family"
    if text.startswith("source:"):
        return "source_table"
    if text.startswith("semantic:"):
        return "semantic_group"
    if text.startswith("external_validation:"):
        return "external_validation_dataset"
    return "feature_name_fallback"


def _base_feature_family(*, feature_name: str, group_key: str, canonical_feature_family: str, family_member_count: int) -> str:
    if canonical_feature_family and (canonical_feature_family != feature_name or family_member_count > 1):
        return f"family:{canonical_feature_family}"
    if group_key.startswith("family:"):
        return group_key
    return f"name:{feature_name}"


def _base_family_overlap(train_pairs: pd.DataFrame, validation_pairs: pd.DataFrame) -> list[str]:
    return sorted(set(train_pairs["base_feature_family"].astype(str)).intersection(set(validation_pairs["base_feature_family"].astype(str))))


def _quality_audit(
    *,
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    external_pairs: pd.DataFrame,
    negative_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> pd.DataFrame:
    checks = [
        ("training_pair_count", len(train_pairs) > 0, len(train_pairs)),
        ("validation_pair_count", len(validation_pairs) > 0, len(validation_pairs)),
        ("external_pair_count", len(external_pairs) > 0, len(external_pairs)),
        ("external_not_training", not external_pairs["allowed_for_training"].any(), 0),
        ("group_overlap_count", split_manifest["group_overlap_count"] == 0, split_manifest["group_overlap_count"]),
        ("base_family_overlap_count", split_manifest["base_family_overlap_count"] == 0, split_manifest["base_family_overlap_count"]),
        ("hard_negatives_disabled", not negative_manifest["explicit_hard_negatives_enabled"], 0),
        ("cross_dataset_negatives_disabled", not negative_manifest["cross_dataset_negatives_enabled"], 0),
    ]
    return pd.DataFrame([{"check": name, "passed": bool(passed), "value": value} for name, passed, value in checks])


def _pair_manifest(
    *,
    config: ContrastiveDataConfig,
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    external_pairs: pd.DataFrame,
    tensor_schema: dict[str, Any],
    negative_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contrastive_data_version": CONTRASTIVE_DATA_VERSION,
        "training_dataset": config.training_dataset,
        "external_validation_dataset": config.external_validation_dataset,
        "legacy_lendingclub_allowed": config.legacy_lendingclub_allowed,
        "pair_counts": {
            "training_positive": int(len(train_pairs)),
            "validation_positive": int(len(validation_pairs)),
            "external_positive": int(len(external_pairs)),
        },
        "pair_hashes": {
            "training_positive": sha256_text(train_pairs.to_csv(index=False)),
            "validation_positive": sha256_text(validation_pairs.to_csv(index=False)),
            "external_positive": sha256_text(external_pairs.to_csv(index=False)),
        },
        "tensor_schema": tensor_schema,
        "negative_policy": negative_manifest,
        "training_activity": {
            "model_trained": False,
            "optimizer_created": False,
            "backpropagation_performed": False,
            "projection_heads_fit": False,
            "matrix_integrated": False,
        },
    }


def _save_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _list(value: Any, default: list[str]) -> list[str]:
    if value in (None, "[]"):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(default)
