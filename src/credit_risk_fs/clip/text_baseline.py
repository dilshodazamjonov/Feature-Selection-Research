from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.embedding_cache import (
    EmbeddingCacheSpec,
    build_embedding_frame,
    save_embedding_frame,
    write_embedding_cache_manifest,
)
from credit_risk_fs.clip.group_split import build_group_split, save_group_split
from credit_risk_fs.clip.text_builder import TEXT_TEMPLATE_VERSION, build_feature_text_frame
from credit_risk_fs.clip.text_encoder import FrozenSentenceTransformerEncoder, TextEncoderProtocol, resolve_device
from credit_risk_fs.clip.text_validation import (
    validate_embeddings,
    validate_feature_text_frame,
    validate_no_legacy_rows,
    validate_prompt1_artifacts,
    validate_text_source_columns,
)
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json, write_json


@dataclass(frozen=True)
class TextBaselineConfig:
    manifest_path: Path
    training_features_path: Path
    external_validation_features_path: Path
    field_role_manifest_path: Path
    source_hashes_path: Path
    encoder_model_name: str
    local_model_path: str | None
    encoder_revision: str
    batch_size: int
    device_policy: str
    normalize_embeddings: bool
    text_template_version: str
    text_fields: list[str]
    forbidden_text_fields: list[str]
    grouping_priority: list[str]
    validation_fraction: float
    seed: int
    anchor_field: str
    minimum_anchor_count: int
    output_dir: Path
    cache_policy: str
    train_dataset: str
    external_validation_dataset: str
    freeze_text_encoder: bool
    use_llm_rank: bool
    use_psi: bool
    use_oot: bool
    stable_core_role: str
    legacy_lendingclub_allowed: bool


@dataclass(frozen=True)
class TextBaselineResult:
    output_paths: dict[str, Path]
    summary: dict[str, Any]


def load_text_baseline_config(path: str | Path = "configs/clip/text_baseline.yaml") -> TextBaselineConfig:
    config_path = Path(path)
    data = _parse_simple_yaml(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return TextBaselineConfig(
        manifest_path=Path(str(data.get("manifest_path", "results/clip/dry_run/training_manifest.json"))),
        training_features_path=Path(str(data.get("training_features_path", "results/clip/dry_run/training_features.csv"))),
        external_validation_features_path=Path(
            str(data.get("external_validation_features_path", "results/clip/dry_run/external_validation_features.csv"))
        ),
        field_role_manifest_path=Path(str(data.get("field_role_manifest_path", "results/clip/dry_run/field_role_manifest.csv"))),
        source_hashes_path=Path(str(data.get("source_hashes_path", "results/clip/dry_run/source_hashes.json"))),
        encoder_model_name=str(data.get("encoder_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        local_model_path=_optional_string(data.get("local_model_path")),
        encoder_revision=str(data.get("encoder_revision", "main")),
        batch_size=int(data.get("batch_size", 64)),
        device_policy=str(data.get("device_policy", "auto")),
        normalize_embeddings=bool(data.get("normalize_embeddings", True)),
        text_template_version=str(data.get("text_template_version", TEXT_TEMPLATE_VERSION)),
        text_fields=_list(data.get("text_fields"), ["feature", "description", "semantic_group", "source_table"]),
        forbidden_text_fields=_list(data.get("forbidden_text_fields"), []),
        grouping_priority=_list(data.get("grouping_priority"), ["base_feature_family", "source_table", "semantic_group", "feature_name"]),
        validation_fraction=float(data.get("validation_fraction", 0.2)),
        seed=int(data.get("seed", 42)),
        anchor_field=str(data.get("anchor_field", "stable_core_membership")),
        minimum_anchor_count=int(data.get("minimum_anchor_count", 5)),
        output_dir=Path(str(data.get("output_dir", "results/clip/text_baseline"))),
        cache_policy=str(data.get("cache_policy", "reuse_if_key_matches")),
        train_dataset=str(data.get("train_dataset", "homecredit")),
        external_validation_dataset=str(data.get("external_validation_dataset", "lendingclub_v2")),
        freeze_text_encoder=bool(data.get("freeze_text_encoder", True)),
        use_llm_rank=bool(data.get("use_llm_rank", False)),
        use_psi=bool(data.get("use_psi", False)),
        use_oot=bool(data.get("use_oot", False)),
        stable_core_role=str(data.get("stable_core_role", "anchor_only")),
        legacy_lendingclub_allowed=bool(data.get("legacy_lendingclub_allowed", False)),
    )


def build_text_baseline(
    *,
    config: TextBaselineConfig,
    dry_run: bool,
    encoder: TextEncoderProtocol | None = None,
) -> TextBaselineResult:
    _validate_config_policy(config)
    errors = validate_prompt1_artifacts(
        [
            config.manifest_path,
            config.training_features_path,
            config.external_validation_features_path,
            config.field_role_manifest_path,
            config.source_hashes_path,
        ]
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    manifest = read_json(config.manifest_path)
    source_hashes = read_json(config.source_hashes_path)
    manifest_hash = sha256_file(config.manifest_path)
    train = pd.read_csv(config.training_features_path)
    external = pd.read_csv(config.external_validation_features_path)
    source_field_errors = []
    source_field_errors.extend(validate_text_source_columns(train.columns, config.text_fields))
    source_field_errors.extend(validate_text_source_columns(external.columns, config.text_fields))
    if source_field_errors:
        raise RuntimeError("; ".join(source_field_errors))

    if validate_no_legacy_rows(train, external):
        raise RuntimeError("; ".join(validate_no_legacy_rows(train, external)))

    home_text = build_feature_text_frame(
        train,
        dataset=config.train_dataset,
        source_manifest_hash=manifest_hash,
        template_version=config.text_template_version,
    )
    lc_text = build_feature_text_frame(
        external,
        dataset=config.external_validation_dataset,
        source_manifest_hash=manifest_hash,
        template_version=config.text_template_version,
    )
    text_errors = validate_feature_text_frame(home_text, config.train_dataset) + validate_feature_text_frame(
        lc_text, config.external_validation_dataset
    )
    if text_errors:
        raise RuntimeError("; ".join(text_errors))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "homecredit_feature_text": config.output_dir / "homecredit_feature_text.csv",
        "lendingclub_v2_feature_text": config.output_dir / "lendingclub_v2_feature_text.csv",
    }
    _public_text_frame(home_text).to_csv(paths["homecredit_feature_text"], index=False)
    _public_text_frame(lc_text).to_csv(paths["lendingclub_v2_feature_text"], index=False)

    split_input = train.rename(columns={"feature": "feature_name"})
    split_result = build_group_split(
        split_input,
        dataset=config.train_dataset,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
    )
    paths.update(save_group_split(split_result, output_dir=config.output_dir))

    anchor_features = _load_anchor_features(config, manifest, source_hashes)
    if len(anchor_features) < config.minimum_anchor_count:
        raise RuntimeError(
            f"insufficient Home Credit anchors: observed={len(anchor_features)}, minimum={config.minimum_anchor_count}"
        )

    expected_embedding_count = {"homecredit": int(len(home_text)), "lendingclub_v2": int(len(lc_text))}
    if dry_run:
        summary = {
            "dry_run": True,
            "encoder_loaded": False,
            "model_trained": False,
            "expected_embedding_count": expected_embedding_count,
            "anchor_count": int(len(anchor_features)),
            "group_split": split_result.audit,
            "text_template_version": config.text_template_version,
        }
        write_json(config.output_dir / "text_baseline_summary.json", summary)
        paths["text_baseline_summary"] = config.output_dir / "text_baseline_summary.json"
        return TextBaselineResult(output_paths=paths, summary=summary)

    encoder = encoder or FrozenSentenceTransformerEncoder(
        model_name=config.encoder_model_name,
        revision=config.encoder_revision,
        local_model_path=config.local_model_path,
        device=resolve_device(config.device_policy),
    )
    spec = EmbeddingCacheSpec(
        encoder_model=encoder.model_name,
        encoder_revision=encoder.revision,
        normalize_embeddings=config.normalize_embeddings,
        text_template_version=config.text_template_version,
    )
    home_embeddings = encoder.encode(
        home_text["feature_text"].tolist(),
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
    )
    lc_embeddings = encoder.encode(
        lc_text["feature_text"].tolist(),
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
    )
    home_emb_frame = build_embedding_frame(text_frame=home_text, embeddings=home_embeddings, spec=spec)
    lc_emb_frame = build_embedding_frame(text_frame=lc_text, embeddings=lc_embeddings, spec=spec)
    embedding_dim = int(home_embeddings.shape[1])
    embedding_errors = validate_embeddings(home_emb_frame, expected_dimension=embedding_dim, normalize=config.normalize_embeddings)
    embedding_errors.extend(validate_embeddings(lc_emb_frame, expected_dimension=embedding_dim, normalize=config.normalize_embeddings))
    if embedding_errors:
        raise RuntimeError("; ".join(embedding_errors))

    paths["homecredit_text_embeddings"] = save_embedding_frame(
        home_emb_frame, config.output_dir / "homecredit_text_embeddings.parquet"
    )
    paths["lendingclub_v2_text_embeddings"] = save_embedding_frame(
        lc_emb_frame, config.output_dir / "lendingclub_v2_text_embeddings.parquet"
    )
    paths["embedding_cache_manifest"] = write_embedding_cache_manifest(
        config.output_dir / "embedding_cache_manifest.json",
        frames=[home_emb_frame, lc_emb_frame],
        spec=spec,
    )

    anchor_manifest, rankings = _rank_by_anchor(
        config=config,
        home_text=home_text,
        lc_text=lc_text,
        home_embeddings=home_embeddings,
        lc_embeddings=lc_embeddings,
        anchor_features=anchor_features,
        encoder_model=encoder.model_name,
        encoder_revision=encoder.revision,
        source_manifest_hash=manifest_hash,
    )
    paths.update(rankings)
    paths["text_anchor_manifest"] = write_json(config.output_dir / "text_anchor_manifest.json", anchor_manifest)

    audit = {
        "encoder_model": encoder.model_name,
        "encoder_revision": encoder.revision,
        "embedding_dimension": embedding_dim,
        "normalize_embeddings": config.normalize_embeddings,
        "homecredit_embeddings": int(len(home_emb_frame)),
        "lendingclub_v2_embeddings": int(len(lc_emb_frame)),
        "embedding_validation_errors": [],
    }
    paths["text_embedding_audit"] = write_json(config.output_dir / "text_embedding_audit.json", audit)
    summary = {
        "dry_run": False,
        "encoder_loaded": True,
        "model_trained": False,
        "contrastive_pairs_created": False,
        "homecredit_texts": int(len(home_text)),
        "lendingclub_v2_texts": int(len(lc_text)),
        "homecredit_embeddings": int(len(home_emb_frame)),
        "lendingclub_v2_embeddings": int(len(lc_emb_frame)),
        "anchor_count": int(len(anchor_features)),
        "encoder_model": encoder.model_name,
        "encoder_revision": encoder.revision,
        "embedding_dimension": embedding_dim,
        "group_split": split_result.audit,
    }
    paths["text_baseline_summary"] = write_json(config.output_dir / "text_baseline_summary.json", summary)
    return TextBaselineResult(output_paths=paths, summary=summary)


def _load_anchor_features(config: TextBaselineConfig, manifest: dict[str, Any], source_hashes: dict[str, Any]) -> pd.DataFrame:
    source_file = Path(manifest["source_files"][config.train_dataset])
    expected_hash = source_hashes[config.train_dataset]["sha256"]
    observed_hash = sha256_file(source_file)
    if observed_hash != expected_hash:
        raise RuntimeError(f"anchor source hash mismatch: expected={expected_hash}, observed={observed_hash}")
    source = pd.read_csv(source_file)
    if config.anchor_field not in source.columns:
        raise RuntimeError(f"anchor field missing from approved source evidence: {config.anchor_field}")
    allowed = source[source["allowed_for_clip_training"].astype(bool)].copy()
    anchors = allowed[allowed[config.anchor_field].astype(bool)].copy()
    return anchors[["dataset", "feature", config.anchor_field, "semantic_group", "source_table"]].sort_values(
        "feature", kind="mergesort"
    )


def _rank_by_anchor(
    *,
    config: TextBaselineConfig,
    home_text: pd.DataFrame,
    lc_text: pd.DataFrame,
    home_embeddings: np.ndarray,
    lc_embeddings: np.ndarray,
    anchor_features: pd.DataFrame,
    encoder_model: str,
    encoder_revision: str,
    source_manifest_hash: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    feature_to_idx = {feature: idx for idx, feature in enumerate(home_text["feature_name"].tolist())}
    anchor_names = [feature for feature in anchor_features["feature"].astype(str).tolist() if feature in feature_to_idx]
    if len(anchor_names) < config.minimum_anchor_count:
        raise RuntimeError(f"too few anchors after text alignment: {len(anchor_names)}")
    anchor_matrix = home_embeddings[[feature_to_idx[name] for name in anchor_names]]
    anchor_centroid = anchor_matrix.mean(axis=0)
    norm = float(np.linalg.norm(anchor_centroid))
    if norm == 0:
        raise RuntimeError("anchor centroid has zero norm")
    anchor_centroid = anchor_centroid / norm

    paths = {}
    home_ranking = _ranking_frame(
        text_frame=home_text,
        embeddings=home_embeddings,
        anchor_centroid=anchor_centroid,
        anchor_names=set(anchor_names),
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        source_manifest_hash=source_manifest_hash,
        config=config,
    )
    lc_ranking = _ranking_frame(
        text_frame=lc_text,
        embeddings=lc_embeddings,
        anchor_centroid=anchor_centroid,
        anchor_names=set(),
        encoder_model=encoder_model,
        encoder_revision=encoder_revision,
        source_manifest_hash=source_manifest_hash,
        config=config,
    )
    paths["homecredit_text_only_ranking"] = config.output_dir / "homecredit_text_only_ranking.csv"
    paths["lendingclub_v2_text_only_ranking"] = config.output_dir / "lendingclub_v2_text_only_ranking.csv"
    paths["homecredit_anchor_features"] = config.output_dir / "homecredit_anchor_features.csv"
    home_ranking.to_csv(paths["homecredit_text_only_ranking"], index=False)
    lc_ranking.to_csv(paths["lendingclub_v2_text_only_ranking"], index=False)
    anchor_features.rename(columns={"feature": "feature_name"}).to_csv(paths["homecredit_anchor_features"], index=False)
    anchor_manifest = {
        "anchor_dataset": config.train_dataset,
        "external_validation_dataset": config.external_validation_dataset,
        "anchor_field": config.anchor_field,
        "anchor_count": len(anchor_names),
        "anchor_features": sorted(anchor_names),
        "encoder_model": encoder_model,
        "encoder_revision": encoder_revision,
        "lendingclub_v2_anchor_policy": "uses unchanged Home Credit anchor centroid",
    }
    return anchor_manifest, paths


def _ranking_frame(
    *,
    text_frame: pd.DataFrame,
    embeddings: np.ndarray,
    anchor_centroid: np.ndarray,
    anchor_names: set[str],
    encoder_model: str,
    encoder_revision: str,
    source_manifest_hash: str,
    config: TextBaselineConfig,
) -> pd.DataFrame:
    sims = embeddings @ anchor_centroid
    frame = pd.DataFrame(
        {
            "dataset": text_frame["dataset"].astype(str),
            "feature_name": text_frame["feature_name"].astype(str),
            "cosine_similarity": sims.astype(float),
            "is_anchor_feature": text_frame["feature_name"].astype(str).isin(anchor_names),
            "semantic_group": text_frame["semantic_group"].astype(str),
            "source_formula_or_table": text_frame["source_formula_or_table"].astype(str),
            "encoder_model": encoder_model,
            "encoder_revision": encoder_revision,
            "template_version": config.text_template_version,
            "source_manifest_hash": source_manifest_hash,
        }
    )
    frame = frame.sort_values(["cosine_similarity", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    frame["text_rank"] = range(1, len(frame) + 1)
    return frame[
        [
            "dataset",
            "feature_name",
            "cosine_similarity",
            "text_rank",
            "is_anchor_feature",
            "semantic_group",
            "source_formula_or_table",
            "encoder_model",
            "encoder_revision",
            "template_version",
            "source_manifest_hash",
        ]
    ]


def _public_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "dataset",
            "feature_name",
            "feature_text",
            "description_present",
            "semantic_group_present",
            "source_formula_present",
            "text_length_chars",
            "source_manifest_hash",
            "text_template_version",
        ]
    ].copy()


def _validate_config_policy(config: TextBaselineConfig) -> None:
    if config.train_dataset != "homecredit" or config.external_validation_dataset != "lendingclub_v2":
        raise RuntimeError("text baseline requires homecredit train and lendingclub_v2 external validation")
    if not config.freeze_text_encoder:
        raise RuntimeError("text encoder must remain frozen")
    if config.use_llm_rank or config.use_psi or config.use_oot:
        raise RuntimeError("LLM rank, PSI, and OOT fields are forbidden for the text baseline")
    if config.stable_core_role != "anchor_only":
        raise RuntimeError("stable-core membership must remain anchor_only")
    if config.legacy_lendingclub_allowed:
        raise RuntimeError("legacy LendingClub is forbidden for CLIP text baseline")


def _list(value: Any, default: list[str]) -> list[str]:
    if value in (None, "[]"):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)


def _optional_string(value: Any) -> str | None:
    if value in (None, "", {}, "{}", "null", "None"):
        return None
    return str(value)
