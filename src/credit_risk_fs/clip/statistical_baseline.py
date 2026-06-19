from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_fields import (
    build_statistical_field_inventory,
    main_statistical_fields,
    write_statistical_field_inventory,
)
from credit_risk_fs.clip.statistical_preprocessor import (
    StatisticalPreprocessor,
    build_vector_frame,
    input_field_hash,
    preprocessing_audit,
)
from credit_risk_fs.clip.statistical_schema import (
    STATISTICAL_BASELINE_VERSION,
    StatisticalBaselineConfig,
    StatisticalBaselineResult,
)
from credit_risk_fs.clip.statistical_validation import (
    validate_anchor_artifacts,
    validate_group_split_artifact,
    validate_main_statistical_fields,
    validate_manifest_sources_and_hashes,
    validate_required_statistical_artifacts,
    validate_statistical_config_policy,
    validate_vector_frame,
)
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.hashing import sha256_file, sha256_text
from credit_risk_fs.utils.io import read_json, write_json


def load_statistical_baseline_config(path: str | Path = "configs/clip/statistical_baseline.yaml") -> StatisticalBaselineConfig:
    data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    return StatisticalBaselineConfig(
        manifest_path=Path(str(data.get("manifest_path", "results/clip/dry_run/training_manifest.json"))),
        training_features_path=Path(str(data.get("training_features_path", "results/clip/dry_run/training_features.csv"))),
        external_validation_features_path=Path(
            str(data.get("external_validation_features_path", "results/clip/dry_run/external_validation_features.csv"))
        ),
        field_role_manifest_path=Path(str(data.get("field_role_manifest_path", "results/clip/dry_run/field_role_manifest.csv"))),
        source_hashes_path=Path(str(data.get("source_hashes_path", "results/clip/dry_run/source_hashes.json"))),
        group_split_path=Path(str(data.get("group_split_path", "results/clip/text_baseline/homecredit_group_split.csv"))),
        group_split_audit_path=Path(str(data.get("group_split_audit_path", "results/clip/text_baseline/group_split_audit.json"))),
        anchor_features_path=Path(str(data.get("anchor_features_path", "results/clip/text_baseline/homecredit_anchor_features.csv"))),
        anchor_manifest_path=Path(str(data.get("anchor_manifest_path", "results/clip/text_baseline/text_anchor_manifest.json"))),
        homecredit_feature_text_path=Path(str(data.get("homecredit_feature_text_path", "results/clip/text_baseline/homecredit_feature_text.csv"))),
        lendingclub_v2_feature_text_path=Path(
            str(data.get("lendingclub_v2_feature_text_path", "results/clip/text_baseline/lendingclub_v2_feature_text.csv"))
        ),
        train_dataset=str(data.get("train_dataset", "homecredit")),
        external_validation_dataset=str(data.get("external_validation_dataset", "lendingclub_v2")),
        legacy_lendingclub_allowed=bool(data.get("legacy_lendingclub_allowed", False)),
        approved_main_statistical_fields=_list(data.get("approved_main_statistical_fields"), ["missing_rate_dev"]),
        optional_ablation_fields=_list(
            data.get("optional_ablation_fields"),
            ["iv_score_if_available", "mrmr_selection_frequency", "boruta_selection_frequency"],
        ),
        forbidden_field_patterns=_list(data.get("forbidden_field_patterns"), []),
        field_alignment_rules=data.get("field_alignment_rules", {}) if isinstance(data.get("field_alignment_rules"), dict) else {},
        missing_value_policy=str(data.get("missing_value_policy", "median imputation fitted on Home Credit train split only")),
        imputation_strategy=str(data.get("imputation_strategy", "median")),
        scaling_strategy=str(data.get("scaling_strategy", "standard")),
        clipping_enabled=bool(data.get("clipping_enabled", False)),
        clipping_lower_quantile=float(data.get("clipping_lower_quantile", 0.01)),
        clipping_upper_quantile=float(data.get("clipping_upper_quantile", 0.99)),
        fit_preprocessing_on=str(data.get("fit_preprocessing_on", "homecredit_train_split_only")),
        external_refit_allowed=bool(data.get("external_refit_allowed", False)),
        algorithm_derived_fields_in_main_view=bool(data.get("algorithm_derived_fields_in_main_view", False)),
        llm_fields_allowed=bool(data.get("llm_fields_allowed", False)),
        stable_core_as_input=bool(data.get("stable_core_as_input", False)),
        stable_core_role=str(data.get("stable_core_role", "anchor_only")),
        oot_fields_allowed=bool(data.get("oot_fields_allowed", False)),
        psi_fields_allowed=bool(data.get("psi_fields_allowed", False)),
        target_fields_allowed=bool(data.get("target_fields_allowed", False)),
        similarity_metric=str(data.get("similarity_metric", "cosine")),
        seed=int(data.get("seed", 42)),
        output_dir=Path(str(data.get("output_dir", "results/clip/statistical_baseline"))),
        anchor_field=str(data.get("anchor_field", "stable_core_membership")),
        minimum_anchor_count=int(data.get("minimum_anchor_count", 5)),
    )


def build_statistical_baseline(*, config: StatisticalBaselineConfig, dry_run: bool) -> StatisticalBaselineResult:
    errors = validate_statistical_config_policy(config)
    required_paths = [
        config.manifest_path,
        config.training_features_path,
        config.external_validation_features_path,
        config.field_role_manifest_path,
        config.source_hashes_path,
        config.group_split_path,
        config.group_split_audit_path,
        config.anchor_features_path,
        config.anchor_manifest_path,
        config.homecredit_feature_text_path,
        config.lendingclub_v2_feature_text_path,
    ]
    errors.extend(validate_required_statistical_artifacts(required_paths))
    if errors:
        raise RuntimeError("; ".join(errors))

    manifest = read_json(config.manifest_path)
    source_hashes = read_json(config.source_hashes_path)
    errors.extend(validate_manifest_sources_and_hashes(manifest=manifest, source_hashes=source_hashes, config=config))
    if errors:
        raise RuntimeError("; ".join(errors))

    training_features = pd.read_csv(config.training_features_path)
    external_features = pd.read_csv(config.external_validation_features_path)
    home_source = pd.read_csv(manifest["source_files"][config.train_dataset])
    lc_source = pd.read_csv(manifest["source_files"][config.external_validation_dataset])
    group_split = pd.read_csv(config.group_split_path)
    group_audit = read_json(config.group_split_audit_path)
    anchors = pd.read_csv(config.anchor_features_path)
    anchor_manifest = read_json(config.anchor_manifest_path)
    home_text = pd.read_csv(config.homecredit_feature_text_path)
    lc_text = pd.read_csv(config.lendingclub_v2_feature_text_path)

    errors.extend(validate_group_split_artifact(split=group_split, audit=group_audit, training_features=training_features))
    errors.extend(
        validate_anchor_artifacts(
            anchors=anchors,
            anchor_manifest=anchor_manifest,
            split=group_split,
            minimum_anchor_count=config.minimum_anchor_count,
        )
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    inventory = build_statistical_field_inventory(
        config=config,
        homecredit_source=home_source,
        lendingclub_source=lc_source,
        training_features=training_features,
        external_validation_features=external_features,
    )
    fields = [field for field in config.approved_main_statistical_fields if field in main_statistical_fields(inventory)]
    errors.extend(
        validate_main_statistical_fields(
            fields=fields,
            training_features=training_features,
            external_validation_features=external_features,
        )
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    output_dir = config.output_dir / "dry_run" if dry_run else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_statistical_field_inventory(inventory, output_dir)
    source_manifest_hash = sha256_file(config.manifest_path)
    split_hash = sha256_file(config.group_split_path)

    home_meta, lc_meta = _metadata_frames(
        config=config,
        training_features=training_features,
        external_features=external_features,
        group_split=group_split,
        home_text=home_text,
        lc_text=lc_text,
        source_manifest_hash=source_manifest_hash,
    )
    train_mask = home_meta["split"].eq("train")
    train_anchor_count = int(
        anchors["feature_name"].astype(str).isin(set(home_meta.loc[train_mask, "feature_name"].astype(str))).sum()
    )
    expected = {
        "homecredit_vectors": int(len(home_meta)),
        "lendingclub_v2_vectors": int(len(lc_meta)),
        "homecredit_train_rows": int(train_mask.sum()),
        "homecredit_validation_rows": int((~train_mask).sum()),
        "train_split_anchor_count": train_anchor_count,
        "vector_dimension": len(fields),
    }
    if dry_run:
        summary = {
            "dry_run": True,
            "baseline_version": STATISTICAL_BASELINE_VERSION,
            "model_trained": False,
            "contrastive_pairs_created": False,
            "main_statistical_fields": fields,
            "optional_ablation_fields": config.optional_ablation_fields,
            "expected": expected,
            "group_split_hash": split_hash,
            "source_manifest_hash": source_manifest_hash,
            "dry_run_output_dir": str(output_dir).replace("\\", "/"),
        }
        paths["statistical_baseline_dry_run_summary"] = write_json(output_dir / "statistical_baseline_dry_run_summary.json", summary)
        return StatisticalBaselineResult(output_paths=paths, summary=summary)

    preprocessor = StatisticalPreprocessor(
        field_order=fields,
        imputation_strategy=config.imputation_strategy,
        scaling_strategy=config.scaling_strategy,
        clipping_enabled=config.clipping_enabled,
        clipping_lower_quantile=config.clipping_lower_quantile,
        clipping_upper_quantile=config.clipping_upper_quantile,
        fit_dataset=config.train_dataset,
        fit_split="train",
    )
    home_values = home_meta.merge(training_features[["feature", *fields]], left_on="feature_name", right_on="feature", how="left")
    lc_values = lc_meta.merge(external_features[["feature", *fields]], left_on="feature_name", right_on="feature", how="left")
    fit_values = home_values.loc[home_values["split"].eq("train"), fields]
    preprocessor.fit(fit_values)
    transformed_home = preprocessor.transform(home_values[fields])
    transformed_lc = preprocessor.transform(lc_values[fields])
    home_vectors = build_vector_frame(metadata=home_meta, transformed=transformed_home, preprocessor=preprocessor)
    lc_vectors = build_vector_frame(metadata=lc_meta, transformed=transformed_lc, preprocessor=preprocessor)
    dimension = len(fields)
    vector_errors = validate_vector_frame(home_vectors, dataset=config.train_dataset, dimension=dimension)
    vector_errors.extend(validate_vector_frame(lc_vectors, dataset=config.external_validation_dataset, dimension=dimension))
    if vector_errors:
        raise RuntimeError("; ".join(vector_errors))

    paths.update(preprocessor.save(output_dir))
    audit = preprocessing_audit(
        raw_train=fit_values,
        raw_all_homecredit=home_values[fields],
        raw_lendingclub=lc_values[fields],
        transformed_homecredit=transformed_home,
        transformed_lendingclub=transformed_lc,
        preprocessor=preprocessor,
    )
    audit.update(
        {
            "group_split_hash": split_hash,
            "prompt2_group_split_audit_has_hash": "split_hash" in group_audit,
            "source_manifest_hash": source_manifest_hash,
        }
    )
    paths["statistical_preprocessing_audit"] = write_json(output_dir / "statistical_preprocessing_audit.json", audit)
    paths["homecredit_statistical_vectors"] = _save_parquet(home_vectors, output_dir / "homecredit_statistical_vectors.parquet")
    paths["lendingclub_v2_statistical_vectors"] = _save_parquet(
        lc_vectors, output_dir / "lendingclub_v2_statistical_vectors.parquet"
    )

    anchor_payload, home_rank, lc_rank = _rank_by_statistical_anchor(
        config=config,
        home_vectors=home_vectors,
        lc_vectors=lc_vectors,
        anchors=anchors,
        preprocessor_hash=preprocessor.preprocessor_hash_,
        source_manifest_hash=source_manifest_hash,
    )
    paths["homecredit_statistical_anchor_features"] = output_dir / "homecredit_statistical_anchor_features.csv"
    paths["statistical_anchor_manifest"] = output_dir / "statistical_anchor_manifest.json"
    paths["homecredit_statistical_only_ranking"] = output_dir / "homecredit_statistical_only_ranking.csv"
    paths["lendingclub_v2_statistical_only_ranking"] = output_dir / "lendingclub_v2_statistical_only_ranking.csv"
    anchor_payload["anchor_frame"].to_csv(paths["homecredit_statistical_anchor_features"], index=False)
    home_rank.to_csv(paths["homecredit_statistical_only_ranking"], index=False)
    lc_rank.to_csv(paths["lendingclub_v2_statistical_only_ranking"], index=False)
    anchor_manifest_json = {k: v for k, v in anchor_payload.items() if k != "anchor_frame"}
    write_json(paths["statistical_anchor_manifest"], anchor_manifest_json)

    diagnostics = _diagnostics(
        home_rank=home_rank,
        lc_rank=lc_rank,
        home_text_rank_path=config.output_dir.parent / "text_baseline" / "homecredit_text_only_ranking.csv",
        lc_text_rank_path=config.output_dir.parent / "text_baseline" / "lendingclub_v2_text_only_ranking.csv",
        anchor_frame=anchor_payload["anchor_frame"],
        audit=audit,
        group_audit=group_audit,
    )
    summary = {
        "dry_run": False,
        "baseline_name": "DEV-only statistical-vector baseline with a Home Credit training-split stable-core anchor.",
        "baseline_version": STATISTICAL_BASELINE_VERSION,
        "model_trained": False,
        "neural_encoder_trained": False,
        "contrastive_pairs_created": False,
        "matrix_integrated": False,
        "main_statistical_fields": fields,
        "optional_ablation_fields": config.optional_ablation_fields,
        "target_aware_fields_used_in_main": [],
        "algorithm_derived_fields_in_main": [],
        "vector_dimension": dimension,
        "homecredit_vectors": int(len(home_vectors)),
        "lendingclub_v2_vectors": int(len(lc_vectors)),
        "homecredit_train_rows": expected["homecredit_train_rows"],
        "homecredit_validation_rows": expected["homecredit_validation_rows"],
        "anchor_count": int(anchor_manifest_json["anchor_count"]),
        "input_field_hash": input_field_hash(fields),
        "preprocessor_hash": preprocessor.preprocessor_hash_,
        "anchor_hash": anchor_manifest_json["anchor_hash"],
        "source_manifest_hash": source_manifest_hash,
        "group_split_hash": split_hash,
        "diagnostics": diagnostics,
    }
    paths["statistical_baseline_summary"] = write_json(output_dir / "statistical_baseline_summary.json", summary)
    return StatisticalBaselineResult(output_paths=paths, summary=summary)


def _metadata_frames(
    *,
    config: StatisticalBaselineConfig,
    training_features: pd.DataFrame,
    external_features: pd.DataFrame,
    group_split: pd.DataFrame,
    home_text: pd.DataFrame,
    lc_text: pd.DataFrame,
    source_manifest_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    home = training_features[["dataset", "feature", "semantic_group", "source_table"]].rename(
        columns={"feature": "feature_name", "source_table": "source_table_or_formula"}
    )
    split_columns = [
        "feature_name",
        "split",
        "group_key",
        "canonical_feature_family",
        "family_resolution_source",
        "family_resolution_rule",
        "family_member_count",
    ]
    available_split_columns = [col for col in split_columns if col in group_split.columns]
    home = home.merge(group_split[available_split_columns], on="feature_name", how="left")
    if home[["split", "group_key"]].isna().any().any():
        raise RuntimeError("HomeCredit metadata has features missing from Prompt 2 group split")
    lc = external_features[["dataset", "feature", "semantic_group", "source_table"]].rename(
        columns={"feature": "feature_name", "source_table": "source_table_or_formula"}
    )
    lc["split"] = "external_validation"
    lc["group_key"] = "external_validation:lendingclub_v2"
    lc["canonical_feature_family"] = lc["feature_name"].astype(str)
    lc["family_resolution_source"] = "external_validation_dataset"
    lc["family_resolution_rule"] = "external_validation_not_grouped"
    lc["family_member_count"] = 1
    for column, default in [
        ("canonical_feature_family", home["feature_name"].astype(str)),
        ("family_resolution_source", "feature_name_fallback"),
        ("family_resolution_rule", "exact_feature_name"),
        ("family_member_count", 1),
    ]:
        if column not in home.columns:
            home[column] = default
    home["source_manifest_hash"] = source_manifest_hash
    lc["source_manifest_hash"] = source_manifest_hash
    for name, frame, text_frame in [
        ("homecredit", home, home_text),
        ("lendingclub_v2", lc, lc_text),
    ]:
        if set(frame["feature_name"].astype(str)) != set(text_frame["feature_name"].astype(str)):
            raise RuntimeError(f"{name}: statistical metadata does not align with Prompt 2 feature text")
    return (
        home.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True),
        lc.sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True),
    )


def _rank_by_statistical_anchor(
    *,
    config: StatisticalBaselineConfig,
    home_vectors: pd.DataFrame,
    lc_vectors: pd.DataFrame,
    anchors: pd.DataFrame,
    preprocessor_hash: str,
    source_manifest_hash: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    stat_cols = [col for col in home_vectors.columns if str(col).startswith("stat_")]
    train_vectors = home_vectors[home_vectors["split"].eq("train")].copy()
    train_anchor_names = set(anchors["feature_name"].astype(str)).intersection(set(train_vectors["feature_name"].astype(str)))
    if len(train_anchor_names) < config.minimum_anchor_count:
        raise RuntimeError(f"insufficient train-split anchors for statistical baseline: {len(train_anchor_names)}")
    anchor_frame = train_vectors[train_vectors["feature_name"].isin(train_anchor_names)].copy()
    anchor_matrix = anchor_frame[stat_cols].to_numpy(dtype=float)
    centroid = anchor_matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        raise RuntimeError("statistical anchor centroid has zero norm")
    centroid = centroid / norm
    anchor_hash = sha256_text(",".join(f"{value:.12g}" for value in centroid.tolist()))
    home_rank = _ranking_frame(
        frame=home_vectors,
        stat_cols=stat_cols,
        anchor_centroid=centroid,
        anchor_names=train_anchor_names,
        preprocessor_hash=preprocessor_hash,
        anchor_hash=anchor_hash,
        source_manifest_hash=source_manifest_hash,
    )
    lc_rank = _ranking_frame(
        frame=lc_vectors,
        stat_cols=stat_cols,
        anchor_centroid=centroid,
        anchor_names=set(),
        preprocessor_hash=preprocessor_hash,
        anchor_hash=anchor_hash,
        source_manifest_hash=source_manifest_hash,
    )
    anchor_out = anchor_frame[
        ["dataset", "feature_name", "split", "semantic_group", "source_table_or_formula", "statistical_vector_hash"]
    ].copy()
    payload = {
        "anchor_dataset": config.train_dataset,
        "external_validation_dataset": config.external_validation_dataset,
        "anchor_field": config.anchor_field,
        "anchor_policy": "Home Credit training split stable-core anchors only",
        "lendingclub_v2_anchor_policy": "uses unchanged Home Credit training-split statistical anchor centroid",
        "anchor_count": int(len(anchor_out)),
        "anchor_hash": anchor_hash,
        "preprocessor_hash": preprocessor_hash,
        "source_manifest_hash": source_manifest_hash,
        "anchor_features": sorted(anchor_out["feature_name"].astype(str).tolist()),
        "anchor_frame": anchor_out.sort_values("feature_name", kind="mergesort").reset_index(drop=True),
    }
    return payload, home_rank, lc_rank


def _ranking_frame(
    *,
    frame: pd.DataFrame,
    stat_cols: list[str],
    anchor_centroid: np.ndarray,
    anchor_names: set[str],
    preprocessor_hash: str,
    anchor_hash: str,
    source_manifest_hash: str,
) -> pd.DataFrame:
    values = frame[stat_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    similarity = (values / safe_norms[:, None]) @ anchor_centroid
    out = pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str),
            "feature_name": frame["feature_name"].astype(str),
            "split": frame["split"].astype(str),
            "statistical_similarity": similarity.astype(float),
            "is_anchor_feature": frame["feature_name"].astype(str).isin(anchor_names),
            "semantic_group": frame["semantic_group"].astype(str),
            "source_table_or_formula": frame["source_table_or_formula"].astype(str),
            "statistical_vector_hash": frame["statistical_vector_hash"].astype(str),
            "preprocessor_hash": preprocessor_hash,
            "anchor_hash": anchor_hash,
            "source_manifest_hash": source_manifest_hash,
        }
    )
    out = out.sort_values(["statistical_similarity", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["statistical_rank"] = range(1, len(out) + 1)
    return out[
        [
            "dataset",
            "feature_name",
            "split",
            "statistical_similarity",
            "statistical_rank",
            "is_anchor_feature",
            "semantic_group",
            "source_table_or_formula",
            "statistical_vector_hash",
            "preprocessor_hash",
            "anchor_hash",
            "source_manifest_hash",
        ]
    ]


def _diagnostics(
    *,
    home_rank: pd.DataFrame,
    lc_rank: pd.DataFrame,
    home_text_rank_path: Path,
    lc_text_rank_path: Path,
    anchor_frame: pd.DataFrame,
    audit: dict[str, Any],
    group_audit: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = {
        "vector_dimension": len(audit["field_order"]),
        "missingness_before": audit["missingness_before"],
        "missingness_after": audit["missingness_after"],
        "finite_checks": audit["finite_checks"],
        "train_validation_group_counts": {
            "group_count": group_audit.get("group_count"),
            "train_rows": group_audit.get("train_rows"),
            "validation_rows": group_audit.get("validation_rows"),
            "group_overlap_count": group_audit.get("group_overlap_count"),
        },
        "anchor_count": int(len(anchor_frame)),
        "anchor_composition_by_semantic_group": anchor_frame.groupby("semantic_group").size().sort_index().to_dict(),
        "anchor_composition_by_source_table": anchor_frame.groupby("source_table_or_formula").size().sort_index().to_dict(),
        "rank_distribution": {
            "homecredit": _rank_distribution(home_rank),
            "lendingclub_v2": _rank_distribution(lc_rank),
        },
        "top_20": {
            "homecredit": home_rank.head(20)[["feature_name", "split", "statistical_similarity", "statistical_rank"]].to_dict("records"),
            "lendingclub_v2": lc_rank.head(20)[["feature_name", "split", "statistical_similarity", "statistical_rank"]].to_dict("records"),
        },
        "text_rank_correlation_diagnostic_only": {},
    }
    diagnostics["text_rank_correlation_diagnostic_only"]["homecredit"] = _rank_correlation(home_rank, home_text_rank_path)
    diagnostics["text_rank_correlation_diagnostic_only"]["lendingclub_v2"] = _rank_correlation(lc_rank, lc_text_rank_path)
    return diagnostics


def _rank_distribution(frame: pd.DataFrame) -> dict[str, float]:
    ranks = frame["statistical_rank"].astype(float)
    return {"min": float(ranks.min()), "median": float(ranks.median()), "max": float(ranks.max())}


def _rank_correlation(stat_rank: pd.DataFrame, text_rank_path: Path) -> float | None:
    if not text_rank_path.exists():
        return None
    text_rank = pd.read_csv(text_rank_path)
    if "text_rank" not in text_rank.columns:
        return None
    merged = stat_rank[["feature_name", "statistical_rank"]].merge(
        text_rank[["feature_name", "text_rank"]], on="feature_name", how="inner"
    )
    if len(merged) < 2:
        return None
    return float(merged["statistical_rank"].corr(merged["text_rank"], method="spearman"))


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
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)
