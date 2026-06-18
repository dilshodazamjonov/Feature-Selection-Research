from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip.evidence_loader import LoadedClipEvidence, load_clip_evidence
from credit_risk_fs.clip.schemas import ClipDatasetRole, ClipFieldRole, ClipFieldSpec, ClipSourceArtifact, ClipTrainingManifest
from credit_risk_fs.clip.validation import (
    forbidden_field_matches,
    scan_forbidden_fields,
    validate_dataset_roles,
    validate_field_role_separation,
)
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.io import write_json

MANIFEST_VERSION = "clip_training_manifest_v1"
DETERMINISTIC_DRY_RUN_CREATED_AT = "1970-01-01T00:00:00Z"


@dataclass(frozen=True)
class ClipManifestConfig:
    seed: int
    output_dir: Path
    train_dataset: str
    external_validation_dataset: str
    source_files: dict[str, Path]
    text_fields: list[str]
    statistical_fields: list[str]
    supervision_only_fields: list[str]
    anchor_only_fields: list[str]
    evaluation_only_fields: list[str]
    metadata_only_fields: list[str]
    group_aware_split_fields: list[str]
    missing_value_policy: str
    numeric_scaling_policy: str
    split_policy: str
    llm_rank_policy: str
    stable_core_policy: str
    oot_policy: str
    psi_policy: str
    policy: dict[str, bool]


@dataclass(frozen=True)
class ClipManifestBuildResult:
    manifest: ClipTrainingManifest
    field_specs: list[ClipFieldSpec]
    source_artifacts: list[ClipSourceArtifact]
    evidence: dict[str, LoadedClipEvidence]
    output_paths: dict[str, Path]


def load_training_manifest_config(path: str | Path = "configs/clip/training_manifest.yaml") -> ClipManifestConfig:
    config_path = Path(path)
    data = _parse_simple_yaml(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        datasets = {}

    source_files: dict[str, Path] = {}
    for dataset, payload in datasets.items():
        if isinstance(payload, dict):
            source_files[str(dataset)] = Path(str(payload.get("source_file", "")))

    default_policy = {
        "llm_rank_as_input": False,
        "stable_core_as_input": False,
        "oot_fields_allowed": False,
        "psi_fields_allowed": False,
        "target_fields_allowed": False,
        "id_fields_allowed": False,
        "legacy_lendingclub_allowed": False,
    }
    policy = dict(default_policy)
    if isinstance(data.get("policy"), dict):
        policy.update({key: bool(value) for key, value in data["policy"].items()})

    return ClipManifestConfig(
        seed=int(data.get("seed", 42)),
        output_dir=Path(str(data.get("output_dir", "results/clip/dry_run"))),
        train_dataset=str(data.get("train_dataset", "homecredit")),
        external_validation_dataset=str(data.get("external_validation_dataset", "lendingclub_v2")),
        source_files=source_files
        or {
            "homecredit": Path("results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv"),
            "lendingclub_v2": Path("results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv"),
        },
        text_fields=_list(data.get("text_field_candidates"), ["clip_training_text", "description", "semantic_group", "source_table"]),
        statistical_fields=_list(data.get("statistical_field_candidates"), ["missing_rate_dev", "iv_score_if_available"]),
        supervision_only_fields=_list(data.get("supervision_only_candidates"), []),
        anchor_only_fields=_list(data.get("anchor_only_candidates"), ["feature"]),
        evaluation_only_fields=_list(data.get("evaluation_only_candidates"), ["evaluation_only_fields"]),
        metadata_only_fields=_list(data.get("metadata_only_candidates"), []),
        group_aware_split_fields=_list(data.get("group_aware_split_candidates"), ["semantic_group"]),
        missing_value_policy=str(data.get("missing_value_policy", "Do not silently fill missing values.")),
        numeric_scaling_policy=str(data.get("numeric_scaling_policy", "No scaling is applied in dry run.")),
        split_policy=str(data.get("split_policy", "Home Credit train, LendingClub v2 external validation.")),
        llm_rank_policy=str(data.get("llm_rank_policy", "LLM ranks are metadata-only by default.")),
        stable_core_policy=str(data.get("stable_core_policy", "Stable-core fields are metadata-only by default.")),
        oot_policy=str(data.get("oot_policy", "OOT evidence is forbidden for training.")),
        psi_policy=str(data.get("psi_policy", "PSI evidence is forbidden for training.")),
        policy=policy,
    )


def build_training_manifest(
    *,
    config: ClipManifestConfig | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    dry_run: bool = True,
) -> ClipManifestBuildResult:
    cfg = config or load_training_manifest_config()
    if output_dir is not None:
        cfg = _replace_config(cfg, output_dir=Path(output_dir))
    if seed is not None:
        cfg = _replace_config(cfg, seed=int(seed))

    validation_errors = validate_dataset_roles(cfg.train_dataset, cfg.external_validation_dataset)
    validation_warnings: list[str] = []
    if not dry_run:
        validation_errors.append("only dry-run manifest building is allowed")

    evidence: dict[str, LoadedClipEvidence] = {}
    dataset_roles = {
        cfg.train_dataset: ClipDatasetRole.TRAIN,
        cfg.external_validation_dataset: ClipDatasetRole.EXTERNAL_VALIDATION,
    }
    for dataset, role in dataset_roles.items():
        evidence[dataset] = load_clip_evidence(
            dataset=dataset,
            role=role,
            source_path=cfg.source_files[dataset],
            statistical_fields=cfg.statistical_fields,
        )
        validation_warnings.extend(evidence[dataset].audit.validation_warnings)

    field_specs = _build_field_specs(cfg, evidence)
    validation_errors.extend(validate_field_role_separation(field_specs))

    artifacts = [
        ClipSourceArtifact(
            dataset=loaded.dataset,
            role=loaded.role,
            source_file=loaded.source_path,
            source_sha256=loaded.source_hash,
            row_count=loaded.audit.row_count,
            allowed_row_count=loaded.audit.allowed_row_count,
            blocked_row_count=loaded.audit.blocked_row_count,
        )
        for loaded in evidence.values()
    ]

    forbidden_detected = {
        dataset: scan_forbidden_fields(loaded.frame.columns)
        for dataset, loaded in sorted(evidence.items())
    }
    source_files = {artifact.dataset: str(artifact.source_file).replace("\\", "/") for artifact in artifacts}
    source_hashes = {artifact.dataset: artifact.source_sha256 for artifact in artifacts}
    source_row_counts = {artifact.dataset: artifact.row_count for artifact in artifacts}
    allowed_row_counts = {artifact.dataset: artifact.allowed_row_count for artifact in artifacts}
    blocked_row_counts = {artifact.dataset: artifact.blocked_row_count for artifact in artifacts}
    allowed_feature_names = {
        dataset: sorted(loaded.allowed["feature"].dropna().astype(str).tolist())
        for dataset, loaded in evidence.items()
    }
    blocked_feature_names = {
        dataset: sorted(loaded.blocked["feature"].dropna().astype(str).tolist())
        for dataset, loaded in evidence.items()
    }
    block_reasons = {
        dataset: {
            str(row["feature"]): str(row.get("clip_training_exclusion_reason", ""))
            for row in loaded.blocked.sort_values("feature", kind="mergesort").to_dict("records")
        }
        for dataset, loaded in evidence.items()
    }

    manifest = ClipTrainingManifest(
        manifest_version=MANIFEST_VERSION,
        created_at=DETERMINISTIC_DRY_RUN_CREATED_AT,
        random_seed=cfg.seed,
        active_datasets=[cfg.train_dataset, cfg.external_validation_dataset],
        train_dataset=cfg.train_dataset,
        external_validation_dataset=cfg.external_validation_dataset,
        source_files=source_files,
        source_hashes=source_hashes,
        source_row_counts=source_row_counts,
        allowed_row_counts=allowed_row_counts,
        blocked_row_counts=blocked_row_counts,
        allowed_feature_names=allowed_feature_names,
        blocked_feature_names=blocked_feature_names,
        block_reasons=block_reasons,
        text_fields=cfg.text_fields,
        candidate_statistical_fields=cfg.statistical_fields,
        supervision_only_fields=cfg.supervision_only_fields,
        anchor_only_fields=cfg.anchor_only_fields,
        evaluation_only_fields=cfg.evaluation_only_fields,
        forbidden_fields_detected=forbidden_detected,
        missing_value_policy=cfg.missing_value_policy,
        numeric_scaling_policy=cfg.numeric_scaling_policy,
        split_policy=cfg.split_policy,
        group_aware_split_fields=cfg.group_aware_split_fields,
        llm_rank_policy=cfg.llm_rank_policy,
        stable_core_policy=cfg.stable_core_policy,
        oot_policy=cfg.oot_policy,
        psi_policy=cfg.psi_policy,
        validation_status="pass" if not validation_errors else "fail",
        validation_warnings=sorted(set(validation_warnings)),
        validation_errors=sorted(set(validation_errors)),
        training_activity={
            "dry_run": bool(dry_run),
            "model_trained": False,
            "encoder_loaded": False,
            "contrastive_pairs_created": False,
            "matrix_integrated": False,
        },
    )

    output_paths = _write_outputs(cfg.output_dir, cfg, manifest, field_specs, artifacts, evidence)
    return ClipManifestBuildResult(
        manifest=manifest,
        field_specs=field_specs,
        source_artifacts=artifacts,
        evidence=evidence,
        output_paths=output_paths,
    )


def _write_outputs(
    output_dir: Path,
    cfg: ClipManifestConfig,
    manifest: ClipTrainingManifest,
    field_specs: list[ClipFieldSpec],
    artifacts: list[ClipSourceArtifact],
    evidence: dict[str, LoadedClipEvidence],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = evidence[cfg.train_dataset]
    external = evidence[cfg.external_validation_dataset]

    training_features = _training_feature_frame(train, cfg, fit_role="trainer_fit")
    external_features = _training_feature_frame(external, cfg, fit_role="external_validation_only")
    blocked_features = pd.concat(
        [_blocked_feature_frame(loaded) for loaded in evidence.values()],
        ignore_index=True,
    ).sort_values(["dataset", "feature"], kind="mergesort")
    field_role_manifest = pd.DataFrame(
        [
            {
                "dataset": spec.dataset,
                "field_name": spec.field_name,
                "detected_dtype": spec.detected_dtype,
                "assigned_field_role": spec.field_role.value,
                "allowed_in_main_training_input": spec.allowed_in_main_training_input,
                "reason": spec.reason,
            }
            for spec in sorted(field_specs, key=lambda item: (item.dataset, item.field_name))
        ]
    )
    schema_audit = {
        dataset: {
            **loaded.audit.__dict__,
            "role": loaded.audit.role.value,
            "source_file": str(loaded.audit.source_file).replace("\\", "/"),
        }
        for dataset, loaded in sorted(evidence.items())
    }
    source_hashes = {
        artifact.dataset: {
            "role": artifact.role.value,
            "source_file": str(artifact.source_file).replace("\\", "/"),
            "sha256": artifact.source_sha256,
        }
        for artifact in sorted(artifacts, key=lambda item: item.dataset)
    }

    paths = {
        "training_manifest": output_dir / "training_manifest.json",
        "training_features": output_dir / "training_features.csv",
        "external_validation_features": output_dir / "external_validation_features.csv",
        "blocked_features": output_dir / "blocked_features.csv",
        "schema_audit": output_dir / "schema_audit.json",
        "field_role_manifest": output_dir / "field_role_manifest.csv",
        "source_hashes": output_dir / "source_hashes.json",
    }
    write_json(paths["training_manifest"], manifest.to_dict())
    training_features.to_csv(paths["training_features"], index=False)
    external_features.to_csv(paths["external_validation_features"], index=False)
    blocked_features.to_csv(paths["blocked_features"], index=False)
    write_json(paths["schema_audit"], schema_audit)
    field_role_manifest.to_csv(paths["field_role_manifest"], index=False)
    write_json(paths["source_hashes"], source_hashes)
    return paths


def _training_feature_frame(loaded: LoadedClipEvidence, cfg: ClipManifestConfig, *, fit_role: str) -> pd.DataFrame:
    columns = ["dataset", "feature", *cfg.text_fields, *cfg.statistical_fields]
    columns = list(dict.fromkeys(col for col in columns if col in loaded.allowed.columns))
    frame = loaded.allowed[columns].copy()
    frame.insert(1, "dataset_role", loaded.role.value)
    frame.insert(2, "fit_role", fit_role)
    return frame.sort_values(["dataset", "feature"], kind="mergesort").reset_index(drop=True)


def _blocked_feature_frame(loaded: LoadedClipEvidence) -> pd.DataFrame:
    frame = loaded.blocked.copy()
    return pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str),
            "dataset_role": loaded.role.value,
            "feature": frame["feature"].astype(str),
            "block_reason": frame["clip_training_exclusion_reason"].fillna("").astype(str),
            "original_allowed_for_clip_training": frame["allowed_for_clip_training"].astype(bool),
            "leakage_review_status": frame["leakage_review_status"].fillna("").astype(str),
            "leakage_review_action": frame["leakage_review_action"].fillna("").astype(str),
        }
    )


def _build_field_specs(cfg: ClipManifestConfig, evidence: dict[str, LoadedClipEvidence]) -> list[ClipFieldSpec]:
    specs: list[ClipFieldSpec] = []
    for dataset, loaded in sorted(evidence.items()):
        for field in loaded.frame.columns:
            role, reason = _assign_field_role(field, cfg)
            allowed = (
                loaded.role == ClipDatasetRole.TRAIN
                and role in {ClipFieldRole.TEXT_INPUT, ClipFieldRole.STATISTICAL_INPUT, ClipFieldRole.ANCHOR_ONLY}
                and not forbidden_field_matches(field)
            )
            specs.append(
                ClipFieldSpec(
                    dataset=dataset,
                    field_name=field,
                    detected_dtype=str(loaded.frame[field].dtype),
                    field_role=role,
                    allowed_in_main_training_input=allowed,
                    reason=reason,
                )
            )
    return specs


def _assign_field_role(field: str, cfg: ClipManifestConfig) -> tuple[ClipFieldRole, str]:
    if field in cfg.text_fields:
        return ClipFieldRole.TEXT_INPUT, "configured text input"
    if field in cfg.statistical_fields:
        return ClipFieldRole.STATISTICAL_INPUT, "configured DEV-only statistical input"
    if field in cfg.supervision_only_fields:
        return ClipFieldRole.SUPERVISION_ONLY, "configured supervision-only field"
    if field in cfg.anchor_only_fields:
        return ClipFieldRole.ANCHOR_ONLY, "configured anchor feature identifier"
    if field in cfg.evaluation_only_fields:
        return ClipFieldRole.EVALUATION_ONLY, "configured evaluation-only field"
    if forbidden_field_matches(field):
        return ClipFieldRole.FORBIDDEN, f"field-name pattern match: {forbidden_field_matches(field)}"
    return ClipFieldRole.METADATA_ONLY, "metadata or non-input audit field"


def _replace_config(cfg: ClipManifestConfig, **updates: Any) -> ClipManifestConfig:
    payload = cfg.__dict__.copy()
    payload.update(updates)
    return ClipManifestConfig(**payload)


def _list(value: Any, default: list[str]) -> list[str]:
    if value in (None, "[]"):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)
