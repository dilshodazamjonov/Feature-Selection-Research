from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch

from credit_risk_fs.clip.checkpointing import load_checkpoint
from credit_risk_fs.clip.exact_duplicates import (
    EXACT_DUPLICATE_POLICY_VERSION,
    find_exact_dev_duplicate_pairs,
)
from credit_risk_fs.clip.model import SemanticStatisticalContrastiveEncoder
from credit_risk_fs.clip.negative_policy import build_negative_policy
from credit_risk_fs.clip.reverse_transfer import (
    EXTERNAL_DATASET,
    PAIRING_POLICY_VERSION,
    REVERSE_METHOD,
    SOURCE_DATASET,
    DatasetRoles,
    align_external_feature_views,
    atomic_registry_transaction,
    aggregate_seed_embeddings,
    append_registry_rows,
    build_summary_manifest,
    build_feature_positive_pairs,
    canonical_artifact_id,
    canonical_registry_value,
    deterministic_feature_split,
    file_manifest,
    fixed_candidate_pool,
    frozen_project,
    implementation_contract,
    load_identity_evidence,
    reconcile_feature_universe,
    registry_bundle_dry_run,
    validate_checkpoint_manifest,
    validate_frozen_external_transform,
    validate_prediction_splits,
    validate_registry_bundle,
    validate_summary_manifest,
    validate_summary_manifest_payloads,
)
from credit_risk_fs.clip.source_anchor import (
    build_feature_stability_evidence,
    build_seed_anchor,
    load_source_anchor_config,
    select_anchor_members,
    validate_seed_anchor,
    validate_source_anchor_artifacts,
    write_anchor_selection_artifacts,
)
from credit_risk_fs.clip.statistical_preprocessor import (
    StatisticalPreprocessor,
    build_vector_frame,
)
from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2
from credit_risk_fs.clip.statistical_view_v2 import build_statistical_view_frame
from credit_risk_fs.clip.trainer import train_seed
from credit_risk_fs.clip.training_validation import (
    TrainingDataBundle,
    load_training_config,
    tensors_for_pairs,
)
from credit_risk_fs.experiments.config import (
    _parse_simple_yaml,
    load_named_project_config,
    resolve_model_kwargs,
)
from credit_risk_fs.experiments.tracking import build_data_version
from credit_risk_fs.pipelines.common import (
    ExperimentConfig,
    prediction_metrics_from_saved_files,
    prepare_modeling_data,
    run_experiment,
    validate_metric_provenance,
)
from credit_risk_fs.selectors.fixed_rank_then_mrmr import FixedRankThenMRMRSelector
from credit_risk_fs.utils.hashing import sha256_file, sha256_text
from credit_risk_fs.utils.io import read_json, write_json


STAGE_ORDER = ("prepare", "train", "project", "evaluate", "register")
DEFAULT_OUTPUT_ROOT = Path("results/corrected_lendingclub_to_homecredit_transfer")


class TransferStageError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone corrected LendingClub v2 to Home Credit reverse transfer"
    )
    parser.add_argument(
        "--stage",
        choices=[*STAGE_ORDER, "all"],
        required=True,
    )
    parser.add_argument(
        "--config-dir",
        default="configs/corrected_lendingclub_to_homecredit",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--models", default="lr,catboost")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config_dir(args.config_dir)
        seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
        models = tuple(value.strip() for value in args.models.split(",") if value.strip())
        stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
        plan = resolve_plan(
            config=config,
            stages=stages,
            seeds=seeds,
            models=models,
            output_dir=Path(args.output_dir),
        )
        if args.dry_run:
            if stages == ("register",):
                plan["registry_dry_run"] = _register(
                    config=config,
                    output_dir=Path(args.output_dir),
                    seeds=seeds,
                    models=models,
                    dry_run=True,
                )
            print(json.dumps(plan, indent=2, default=str))
            return 0
        execute_plan(
            config=config,
            stages=stages,
            seeds=seeds,
            models=models,
            output_dir=Path(args.output_dir),
            resume=bool(args.resume),
            skip_existing=bool(args.skip_existing),
        )
        return 0
    except (ValueError, FileNotFoundError, TransferStageError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2


def load_config_dir(config_dir: str | Path) -> dict[str, Any]:
    root = Path(config_dir)
    required = [
        root / "contrastive_data.yaml",
        root / "training.yaml",
        root / "reverse_projection.yaml",
        root / "downstream.yaml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing reverse-transfer config files: {missing}")
    merged: dict[str, Any] = {}
    for path in required:
        parsed = _parse_simple_yaml(path.read_text(encoding="utf-8"))
        merged.update(parsed)
    config_files = list(required)
    identity_path = Path(str(merged.get("identity_evidence_path", "")))
    if not identity_path.exists():
        raise FileNotFoundError(f"missing identity evidence config: {identity_path}")
    config_files.append(identity_path)
    merged["_config_files"] = [
        str(path).replace("\\", "/") for path in config_files
    ]
    merged["_config_dir"] = str(root).replace("\\", "/")
    return merged


def resolve_plan(
    *,
    config: dict[str, Any],
    stages: tuple[str, ...],
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    roles = _roles(config)
    roles.validate()
    if (
        roles.training_dataset != SOURCE_DATASET
        or roles.external_dataset != EXTERNAL_DATASET
    ):
        raise ValueError(
            "reverse-transfer execution requires training_dataset=lendingclub_v2 "
            "and external_dataset=homecredit"
        )
    if config.get("pairing_policy_version") != PAIRING_POLICY_VERSION:
        raise ValueError("active pairing policy must be identity_equivalence_v2")
    if config.get("stable_row_id_column") != "SK_ID_CURR":
        raise ValueError(
            "Home Credit prediction provenance requires stable_row_id_column=SK_ID_CURR"
        )
    if seeds != (11, 22, 33, 44, 55):
        raise ValueError("seeds must be exactly 11,22,33,44,55")
    if set(models) - {"lr", "catboost"} or not models:
        raise ValueError("models must be a non-empty subset of lr,catboost")
    budgets = _model_budgets(config)
    if budgets != {
        "lr": {"candidate_pool_size": 60, "feature_budget": 20},
        "catboost": {"candidate_pool_size": 100, "feature_budget": 40},
    }:
        raise ValueError("fixed candidate pools/budgets must be LR 60/20 and CatBoost 100/40")
    if any(stage not in STAGE_ORDER for stage in stages):
        raise ValueError("unknown stage")
    contract = implementation_contract(output_dir)
    plan = {
        "status": "dry_run_no_scientific_execution",
        "resolved_stages": list(stages),
        "roles": roles.manifest(),
        "seeds": list(seeds),
        "models": list(models),
        "fixed_budgets": budgets,
        "inputs": {
            key: file_manifest(value)
            for key, value in _input_paths(config).items()
        },
        "outputs": contract["scientific_outputs"],
        "provenance_contract": {
            "stable_row_id_column": config["stable_row_id_column"],
            "generated_index_ids_allowed": False,
            "dev_prediction_metric_artifact": "results/dev_oof_predictions.csv",
            "dev_metric_scope": "dev_oof_cross_validated",
            "raw_dev_evidence": "target-free raw LendingClub DEV values with canonical SHA-256",
            "reuse_validation": "current stage plus complete transitive upstream chain and declared inputs",
            "registry_equivalence": "canonical type/path/JSON normalization with atomic idempotent no-op",
        },
        "safeguards": {
            "external_refit_allowed": False,
            "source_oot_allowed": False,
            "external_target_allowed_before_mrmr": False,
            "baseline_execution_allowed": False,
            "llm_execution_allowed": False,
            "existing_homecredit_clip_retraining_allowed": False,
            "umap_generation_allowed": False,
            "overwrite_completed_outputs": False,
        },
        "source_anchor": {
            **load_source_anchor_config(config).to_manifest(),
            "output_paths": {
                "subwindow_config": str(
                    output_dir / "source_anchor/stability_subwindow_config.json"
                ),
                "stability_evidence": str(
                    output_dir / "source_anchor/feature_stability_evidence.csv"
                ),
                "candidate_audit": str(
                    output_dir / "source_anchor/anchor_candidate_audit.csv"
                ),
                "members": str(output_dir / "source_anchor/anchor_members.csv"),
                "seed_manifest": str(
                    output_dir / "source_anchor/seed_anchor_manifest.csv"
                ),
                "manifest": str(
                    output_dir / "source_anchor/source_anchor_manifest.json"
                ),
            },
            "fail_closed_conditions": [
                "fewer_or_more_than_23_members",
                "target_or_oot_or_external_data_used",
                "member_outside_training_split",
                "identity_equivalent_members",
                "threshold_or_hash_mismatch",
                "non_lendingclub_source_anchor",
                "downstream_performance_used_for_selection",
            ],
        },
    }
    if "register" in stages:
        required_scientific = [
            path
            for stage in STAGE_ORDER[:-1]
            for path in _stage_artifact_paths(stage, output_dir, seeds, models)
        ]
        missing_scientific = [
            str(path).replace("\\", "/")
            for path in required_scientific
            if not path.exists()
        ]
        affected = [
            "results/research_summary/run_index.csv",
            "results/research_summary/artifact_registry.csv",
            "results/research_summary/reusable_metrics.csv",
            "results/research_summary/selected_feature_registry.csv",
            "results/research_summary/results_access_guide.md",
            "results/research_summary/summary_manifest.json",
        ]
        plan["registry_dry_run"] = {
            "schema_version": "reverse_transfer_registry_v2",
            "canonicalization_version": "schema_aware_registry_v2",
            "transaction_outcome": (
                "CONFLICT" if missing_scientific else "NEW_TRANSACTION_OR_IDEMPOTENT_NO_OP"
            ),
            "missing_artifacts": missing_scientific,
            "affected_files": affected,
            "writes_performed": False,
            "success_transaction_manifest_written": False,
        }
    return plan


def execute_plan(
    *,
    config: dict[str, Any],
    stages: tuple[str, ...],
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    output_dir: Path,
    resume: bool,
    skip_existing: bool,
) -> None:
    allowed_root = DEFAULT_OUTPUT_ROOT.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != allowed_root and allowed_root not in resolved_output.parents:
        raise TransferStageError(
            "scientific output directory must be under "
            "results/corrected_lendingclub_to_homecredit_transfer"
        )
    resolve_plan(
        config=config,
        stages=stages,
        seeds=seeds,
        models=models,
        output_dir=output_dir,
    )
    _validate_unique_stage_artifact_ownership(
        stages=stages,
        output_dir=output_dir,
        seeds=seeds,
        models=models,
    )
    if "train" in stages:
        _migrate_legacy_prepare_anchor_contract(
            config=config,
            output_dir=output_dir,
            seeds=seeds,
            models=models,
        )
    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": _prepare,
        "train": _train,
        "project": _project,
        "evaluate": _evaluate,
        "register": _register,
    }
    configuration_hash = _configuration_hash(config)
    for stage in stages:
        manifest_path = output_dir / "manifests" / f"{stage}_stage_manifest.json"
        if manifest_path.exists():
            old = read_json(manifest_path)
            complete = old.get("status") == "complete"
            if complete:
                _validate_reuse_chain(
                    stage=stage,
                    output_dir=output_dir,
                    config_hash=configuration_hash,
                    seeds=seeds,
                    models=models,
                )
                if skip_existing or resume:
                    continue
                raise TransferStageError(
                    f"{stage}: completed output exists; use --skip-existing or a new output directory"
                )
            if not resume:
                raise TransferStageError(
                    f"{stage}: partial output exists; use --resume only after inspection"
                )
            _validate_stage_identity(old, configuration_hash)
            _validate_reuse_chain(
                stage=stage,
                output_dir=output_dir,
                config_hash=configuration_hash,
                seeds=seeds,
                models=models,
                include_current=False,
            )
            if stage not in {"train", "evaluate"}:
                raise TransferStageError(
                    f"{stage}: partial-stage resume is not supported safely; use a new output directory"
                )
        else:
            unmanifested = [
                path
                for path in _stage_artifact_paths(stage, output_dir, seeds, models)
                if path.exists()
            ]
            if unmanifested:
                raise TransferStageError(
                    f"{stage}: unmanifested partial outputs cannot be reused: {unmanifested}"
                )
        stage_index = STAGE_ORDER.index(stage)
        if stage_index:
            _validate_reuse_chain(
                stage=stage,
                output_dir=output_dir,
                config_hash=configuration_hash,
                seeds=seeds,
                models=models,
                include_current=False,
            )
        write_json(
            manifest_path,
            {
                "stage": stage,
                "status": "in_progress",
                "configuration_hash": configuration_hash,
                "source_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "requested_seeds": list(seeds),
                "requested_models": list(models),
            },
        )
        handler_kwargs = dict(
            config=config,
            output_dir=output_dir,
            seeds=seeds,
            models=models,
        )
        if stage in {"train", "evaluate"}:
            handler_kwargs["resume"] = resume
        payload = handlers[stage](**handler_kwargs)
        artifacts = _stage_artifact_paths(stage, output_dir, seeds, models)
        missing = [str(path) for path in artifacts if not path.exists()]
        if missing:
            raise TransferStageError(f"{stage}: required outputs are missing: {missing}")
        payload.update(
            {
                "stage": stage,
                "status": "complete",
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
                "configuration_hash": configuration_hash,
                "requested_seeds": list(seeds),
                "requested_models": list(models),
                "artifact_hashes": {
                    _relative(path): sha256_file(path) for path in artifacts
                },
            }
        )
        data_manifest_path = output_dir / "pairing" / "data_manifest.json"
        if data_manifest_path.exists():
            data_manifest = read_json(data_manifest_path)
            payload.update(
                {
                    "data_manifest_hash": sha256_file(data_manifest_path),
                    "identity_evidence_hash": data_manifest[
                        "identity_evidence_hash"
                    ],
                    "feature_universe_hash": data_manifest[
                        "source_feature_universe_hash"
                    ],
                    "feature_split_hash": data_manifest["split_hash"],
                    "raw_dev_statistical_evidence_hash": data_manifest[
                        "raw_dev_statistical_evidence_hash"
                    ],
                    "statistical_preprocessor_hash": data_manifest[
                        "statistical_preprocessor_hash"
                    ],
                }
            )
        write_json(manifest_path, payload)


def _validate_stage_identity(manifest: dict[str, Any], config_hash: str) -> None:
    expected = {
        "configuration_hash": config_hash,
        "source_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise TransferStageError(f"stage metadata mismatch: {mismatches}")


def _validate_reuse_chain(
    *,
    stage: str,
    output_dir: Path,
    config_hash: str,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    include_current: bool = True,
) -> None:
    """Fail closed unless every reused stage and its declared inputs remain valid."""
    last_index = STAGE_ORDER.index(stage) + int(include_current)
    for dependency in STAGE_ORDER[:last_index]:
        path = output_dir / "manifests" / f"{dependency}_stage_manifest.json"
        if not path.exists():
            raise TransferStageError(
                f"{stage}: required reusable stage {dependency} has no manifest"
            )
        _validate_stage_manifest(
            stage=dependency,
            manifest=read_json(path),
            output_dir=output_dir,
            config_hash=config_hash,
            seeds=seeds,
            models=models,
        )


def _validate_stage_manifest(
    *,
    stage: str,
    manifest: dict[str, Any],
    output_dir: Path,
    config_hash: str,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    artifact_paths: list[Path] | None = None,
) -> None:
    _validate_stage_identity(manifest, config_hash)
    if manifest.get("status") != "complete":
        raise TransferStageError(f"{stage}: stage is not complete")
    if manifest.get("requested_seeds") != list(seeds):
        raise TransferStageError(f"{stage}: seed set mismatch")
    if manifest.get("requested_models") != list(models):
        raise TransferStageError(f"{stage}: model set mismatch")
    expected_paths = artifact_paths or _stage_artifact_paths(
        stage, output_dir, seeds, models
    )
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise TransferStageError(f"{stage}: artifact hashes are missing")
    expected_keys = [_relative(path) for path in expected_paths]
    if len(expected_keys) != len(set(expected_keys)):
        raise TransferStageError(f"{stage}: duplicate artifact declaration")
    if set(hashes) != set(expected_keys):
        raise TransferStageError(
            f"{stage}: artifact declaration or path normalization mismatch"
        )
    for path in expected_paths:
        key = _relative(path)
        if not path.exists() or hashes.get(key) != sha256_file(path):
            raise TransferStageError(f"{stage}: artifact missing or corrupt: {path}")
    data_manifest_path = output_dir / "pairing" / "data_manifest.json"
    if data_manifest_path.exists():
        current = read_json(data_manifest_path)
        expected_provenance = {
            "data_manifest_hash": sha256_file(data_manifest_path),
            "identity_evidence_hash": current.get("identity_evidence_hash"),
            "feature_universe_hash": current.get("source_feature_universe_hash"),
            "feature_split_hash": current.get("split_hash"),
            "raw_dev_statistical_evidence_hash": current.get(
                "raw_dev_statistical_evidence_hash"
            ),
            "statistical_preprocessor_hash": current.get(
                "statistical_preprocessor_hash"
            ),
        }
        mismatches = [
            key
            for key, expected in expected_provenance.items()
            if manifest.get(key) != expected
        ]
        if mismatches:
            raise TransferStageError(
                f"{stage}: current data provenance mismatch: {mismatches}"
            )
    for raw_path, expected_hash in manifest.get("input_hashes", {}).items():
        path = Path(raw_path)
        if not path.exists() or not path.is_file() or sha256_file(path) != expected_hash:
            raise TransferStageError(f"{stage}: input changed or missing: {path}")
    for raw_path, expected_fingerprint in manifest.get(
        "input_fingerprints", {}
    ).items():
        path = Path(raw_path)
        if not path.exists() or build_data_version(path) != expected_fingerprint:
            raise TransferStageError(
                f"{stage}: input dataset changed or missing: {path}"
            )
    for raw_path, expected_hash in manifest.get(
        "registered_output_hashes", {}
    ).items():
        path = Path(raw_path)
        if not path.exists() or not path.is_file() or sha256_file(path) != expected_hash:
            raise TransferStageError(
                f"{stage}: registered output changed or missing: {path}"
            )


def _validate_unique_stage_artifact_ownership(
    *,
    stages: tuple[str, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> None:
    last_stage_index = max(STAGE_ORDER.index(stage) for stage in stages)
    owners: dict[str, str] = {}
    for stage in STAGE_ORDER[: last_stage_index + 1]:
        for path in _stage_artifact_paths(stage, output_dir, seeds, models):
            canonical = os.path.normcase(str(path.resolve()))
            previous = owners.get(canonical)
            if previous is not None:
                raise TransferStageError(
                    "duplicate artifact declaration: "
                    f"{path} is owned by both {previous} and {stage}"
                )
            owners[canonical] = stage


def _migrate_legacy_prepare_anchor_contract(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> bool:
    """Authenticate and migrate the one legacy prepare/train ownership overlap."""
    anchor_dir = output_dir / "source_anchor"
    legacy_selection = anchor_dir / "source_anchor_manifest.json"
    legacy_seed_template = anchor_dir / "seed_anchor_manifest.csv"
    selection = anchor_dir / "source_anchor_selection_manifest.json"
    seed_template = anchor_dir / "seed_anchor_manifest_template.csv"
    prepare_manifest_path = (
        output_dir / "manifests" / "prepare_stage_manifest.json"
    )
    train_manifest_path = output_dir / "manifests" / "train_stage_manifest.json"
    legacy_exists = (legacy_selection.exists(), legacy_seed_template.exists())
    migrated_exists = (selection.exists(), seed_template.exists())

    if migrated_exists == (True, True) and legacy_exists == (False, False):
        return False
    if legacy_exists == (False, False) and migrated_exists == (False, False):
        return False
    if train_manifest_path.exists():
        return False
    if legacy_exists != (True, True) or migrated_exists != (False, False):
        raise TransferStageError(
            "prepare anchor contract migration refused: mixed legacy/current paths"
        )
    if not prepare_manifest_path.exists():
        raise TransferStageError(
            "prepare anchor contract migration refused: missing prepare manifest"
        )

    config_hash = _configuration_hash(config)
    prepare_manifest = read_json(prepare_manifest_path)
    legacy_prepare_paths = [
        path
        for path in _stage_artifact_paths("prepare", output_dir, seeds, models)
        if path not in (selection, seed_template)
    ]
    legacy_prepare_paths.append(legacy_selection)
    _validate_stage_manifest(
        stage="prepare",
        manifest=prepare_manifest,
        output_dir=output_dir,
        config_hash=config_hash,
        seeds=seeds,
        models=models,
        artifact_paths=legacy_prepare_paths,
    )

    source_anchor_paths = prepare_manifest.get("source_anchor_paths")
    if not isinstance(source_anchor_paths, dict):
        raise TransferStageError(
            "prepare anchor contract migration refused: source paths are missing"
        )
    expected_declared_paths = {
        "source_anchor_manifest": legacy_selection,
        "seed_anchor_manifest": legacy_seed_template,
    }
    for key, expected in expected_declared_paths.items():
        raw = source_anchor_paths.get(key)
        if raw is None or Path(raw).resolve() != expected.resolve():
            raise TransferStageError(
                "prepare anchor contract migration refused: "
                f"path normalization mismatch for {key}"
            )

    expected_columns = [
        "seed",
        "checkpoint_hash",
        "anchor_hash",
        "member_count",
        "source_dataset",
        "pairing_policy_version",
    ]
    with legacy_seed_template.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != expected_columns or rows:
        raise TransferStageError(
            "prepare anchor contract migration refused: "
            "legacy seed manifest is not the empty prepare template"
        )

    source_manifest = read_json(legacy_selection)
    if source_manifest.get("anchor_hashes_by_seed") or source_manifest.get(
        "checkpoint_hashes_by_seed"
    ):
        raise TransferStageError(
            "prepare anchor contract migration refused: "
            "source manifest contains train outputs"
        )
    data_manifest_path = output_dir / "pairing" / "data_manifest.json"
    anchor_members_path = anchor_dir / "anchor_members.csv"
    data_manifest = read_json(data_manifest_path)
    try:
        validate_source_anchor_artifacts(
            config=load_source_anchor_config(config),
            members=pd.read_csv(anchor_members_path),
            manifest=source_manifest,
            training_feature_ids=set(
                pd.read_parquet(
                    output_dir
                    / "pairing/lendingclub_v2_train_positive_pairs.parquet",
                    columns=["feature_id"],
                )["feature_id"].astype(str)
            ),
            configuration_hash=config_hash,
            data_manifest_hash=sha256_file(data_manifest_path),
            external_dataset=EXTERNAL_DATASET,
            source_feature_universe_hash=data_manifest[
                "source_feature_universe_hash"
            ],
            feature_split_hash=data_manifest["split_hash"],
            identity_evidence_hash=data_manifest["identity_evidence_hash"],
            raw_statistical_evidence_hash=data_manifest[
                "raw_statistical_evidence_hash"
            ],
            statistical_preprocessor_hash=data_manifest[
                "statistical_preprocessor_hash"
            ],
            member_path=anchor_members_path,
            evidence_path=anchor_dir / "feature_stability_evidence.csv",
        )
    except (KeyError, OSError, ValueError) as exc:
        raise TransferStageError(
            "prepare anchor contract migration refused: "
            f"anchor provenance validation failed: {exc}"
        ) from exc

    new_manifest = dict(prepare_manifest)
    new_paths = dict(source_anchor_paths)
    del new_paths["source_anchor_manifest"]
    del new_paths["seed_anchor_manifest"]
    new_paths["source_anchor_selection_manifest"] = _relative(selection)
    new_paths["seed_anchor_manifest_template"] = _relative(seed_template)
    new_manifest["source_anchor_paths"] = new_paths
    new_manifest["artifact_hashes"] = {
        _relative(path): (
            sha256_file(legacy_selection)
            if path == selection
            else sha256_file(legacy_seed_template)
            if path == seed_template
            else sha256_file(path)
        )
        for path in _stage_artifact_paths("prepare", output_dir, seeds, models)
    }
    new_manifest["manifest_contract_migrations"] = [
        *new_manifest.get("manifest_contract_migrations", []),
        {
            "migration": "prepare_anchor_ownership_v1_to_v2",
            "reason": "separate immutable prepare intermediates from train outputs",
            "legacy_source_anchor_manifest_sha256": sha256_file(
                legacy_selection
            ),
            "legacy_seed_anchor_template_sha256": sha256_file(
                legacy_seed_template
            ),
            "original_manifest_backup": _relative(
                prepare_manifest_path.with_name(
                    "prepare_stage_manifest.pre_anchor_contract_migration.json"
                )
            ),
        },
    ]

    backup_path = prepare_manifest_path.with_name(
        "prepare_stage_manifest.pre_anchor_contract_migration.json"
    )
    if backup_path.exists():
        raise TransferStageError(
            "prepare anchor contract migration refused: backup already exists"
        )
    manifest_bytes = prepare_manifest_path.read_bytes()
    with backup_path.open("xb") as backup:
        backup.write(manifest_bytes)
        backup.flush()
        os.fsync(backup.fileno())

    moved_selection = False
    moved_template = False
    try:
        os.replace(legacy_selection, selection)
        moved_selection = True
        os.replace(legacy_seed_template, seed_template)
        moved_template = True
        _write_json_atomic(prepare_manifest_path, new_manifest)
    except Exception:
        if moved_template and seed_template.exists():
            os.replace(seed_template, legacy_seed_template)
        if moved_selection and selection.exists():
            os.replace(selection, legacy_selection)
        backup_path.unlink(missing_ok=True)
        raise
    return True


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise TransferStageError(f"atomic manifest temporary exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluate_model_artifact_paths(output_dir: Path, model: str) -> list[Path]:
    name = "logistic_regression" if model == "lr" else "catboost"
    base = output_dir / "downstream" / name
    selector_prefix = "reverse_transfer_clip_then_mrmr"
    return [
        output_dir / f"candidate_pools/{model}_candidate_pool.csv",
        base / "data/source_identity_manifest.json",
        base / "features/feature_stability_metrics.csv",
        base / "features/fold_selected_features.csv",
        base / "features/selection_frequency.csv",
        base / "features/semantic_group_stability.csv",
        base / "features/final_selected_features.csv",
        base / "models/final_model.model",
        base / "models/final_model_bundle.joblib",
        base / "models/final_model_metadata.json",
        base / "models/final_preprocessor.pkl",
        base / "results/cv_results.csv",
        base / "results/dev_predictions.csv",
        base / "results/dev_oof_predictions.csv",
        base / "results/experiment_summary.csv",
        base / "results/oof_reconciliation.csv",
        base / "results/oot_predictions.csv",
        base / "results/oot_test_results.csv",
        base / "results/prediction_metrics.csv",
        base / "results/psi_details.csv",
        base / "manifests/fold_manifest.json",
        base / "manifests/prediction_manifest.json",
        base / "manifests/metric_manifest.json",
        base
        / f"llm_responses/final_dev/{selector_prefix}_mrmr_widths.csv",
        base
        / f"llm_responses/final_dev/{selector_prefix}_selection_manifest.csv",
        base
        / f"llm_responses/final_dev/{selector_prefix}_source_to_model_lineage.csv",
    ]


def _stage_artifact_paths(
    stage: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> list[Path]:
    if stage == "prepare":
        return [
            output_dir / "feature_universe/feature_reconciliation.csv",
            output_dir / "pairing/lendingclub_v2_feature_split.csv",
            output_dir / "pairing/identity_evidence_manifest.json",
            output_dir / "pairing/identity_relations.csv",
            output_dir / "pairing/lendingclub_v2_exact_dev_duplicates.parquet",
            output_dir / "pairing/lendingclub_v2_train_positive_pairs.parquet",
            output_dir / "pairing/lendingclub_v2_validation_positive_pairs.parquet",
            output_dir / "pairing/negative_exclusion_pairs.parquet",
            output_dir / "pairing/feature_split_manifest.json",
            output_dir / "pairing/negative_policy_manifest.json",
            output_dir / "pairing/raw_dev_statistical_evidence_manifest.json",
            output_dir / "pairing/data_manifest.json",
            output_dir / "training/statistical_preprocessor/statistical_preprocessor.json",
            output_dir / "training/statistical_preprocessor/statistical_preprocessor.joblib",
            output_dir / "training/inputs/lendingclub_v2_text_embeddings.parquet",
            output_dir / "training/inputs/lendingclub_v2_statistical_vectors.parquet",
            output_dir / "source_anchor/feature_stability_evidence.csv",
            output_dir / "source_anchor/anchor_candidate_audit.csv",
            output_dir / "source_anchor/anchor_members.csv",
            output_dir / "source_anchor/source_anchor_selection_manifest.json",
            output_dir / "source_anchor/seed_anchor_manifest_template.csv",
        ]
    if stage == "train":
        paths = [output_dir / "source_anchor/source_anchor_manifest.json"]
        for seed in seeds:
            paths.extend(
                [
                    output_dir / f"training/seeds/seed_{seed}/best_checkpoint.pt",
                    output_dir / f"training/seeds/seed_{seed}/checkpoint_manifest.json",
                    output_dir / f"source_anchor/seed_{seed}_anchor.npy",
                    output_dir / f"source_anchor/seed_{seed}_anchor_manifest.json",
                    output_dir
                    / f"source_anchor/seed_{seed}_lendingclub_validation_scores.csv",
                ]
            )
        paths.extend(
            [
                output_dir / "source_anchor/seed_anchor_manifest.csv",
                output_dir
                / "source_anchor/lendingclub_validation_consensus_scores.csv",
            ]
        )
        return paths
    if stage == "project":
        return [
            *[
                output_dir
                / f"reverse_projection/seed_{seed}_homecredit_reverse_embeddings.parquet"
                for seed in seeds
            ],
            output_dir / "reverse_projection/homecredit_raw_descriptors.parquet",
            output_dir / "reverse_projection/homecredit_statistical_vectors.parquet",
            output_dir / "reverse_projection/homecredit_reverse_embeddings.parquet",
            output_dir / "reverse_projection/homecredit_reverse_scores.csv",
            output_dir / "reverse_projection/homecredit_reverse_feature_reconciliation.csv",
            output_dir / "reverse_projection/alignment_manifest.json",
            output_dir / "reverse_projection/alignment_exclusions.csv",
            output_dir / "reverse_projection/reverse_projection_manifest.json",
        ]
    if stage == "evaluate":
        paths = [
            output_dir / "comparisons/reverse_transfer_metrics.csv",
            output_dir / "manifests/registry_payload.json",
        ]
        for model in models:
            paths.extend(_evaluate_model_artifact_paths(output_dir, model))
        return paths
    if stage == "register":
        return [output_dir / "manifests/registration_transaction_manifest.json"]
    raise TransferStageError(f"unknown stage: {stage}")


def canonical_raw_dev_evidence(
    frame: pd.DataFrame,
    *,
    time_column: str,
    feature_columns: list[str],
    target_column: str,
    dev_start: int,
    dev_end: int,
    dataset: str = SOURCE_DATASET,
    stable_row_id_column: str | None = None,
) -> dict[str, Any]:
    """Cryptographically identify the exact target-free raw DEV values consumed."""
    if dataset != SOURCE_DATASET:
        raise TransferStageError("raw DEV evidence must come from LendingClub v2")
    if target_column in feature_columns:
        raise TransferStageError("target cannot be part of raw DEV statistical evidence")
    columns = [
        *([stable_row_id_column] if stable_row_id_column else []),
        time_column,
        *sorted(dict.fromkeys(feature_columns)),
    ]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise TransferStageError(f"raw DEV statistical evidence columns missing: {missing}")
    if frame.empty:
        raise TransferStageError("raw DEV statistical evidence is empty")
    if stable_row_id_column:
        identifiers = frame[stable_row_id_column]
        if identifiers.isna().any() or identifiers.astype(str).duplicated().any():
            raise TransferStageError("raw DEV stable source row IDs are invalid")
    times = pd.to_numeric(frame[time_column], errors="coerce")
    if times.isna().any() or not ((times >= dev_start) & (times < dev_end)).all():
        raise TransferStageError("raw DEV statistical evidence is outside its declared window")

    schema = {
        "version": "canonical_raw_dev_statistical_evidence_v1",
        "dataset": dataset,
        "split": "DEV",
        "dev_window": [dev_start, dev_end],
        "row_order": "source_file_order_after_dev_filter",
        "column_order": "time_column_then_lexicographically_sorted_feature_columns",
        "null_encoding": "<NA>",
        "float_format": "%.17g",
        "encoding": "utf-8",
        "target_excluded": True,
        "time_column": time_column,
        "stable_source_row_id_column": stable_row_id_column,
        "stable_source_row_ids_available": bool(stable_row_id_column),
        "feature_columns": sorted(dict.fromkeys(feature_columns)),
        "column_dtypes": {
            column: str(frame[column].dtype) for column in columns
        },
        "row_count": int(len(frame)),
        "column_count": len(columns),
    }
    digest = hashlib.sha256()
    digest.update(
        (json.dumps(schema, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )

    class _DigestWriter:
        def write(self, value: str) -> int:
            encoded = value.encode("utf-8")
            digest.update(encoded)
            return len(value)

    frame.loc[:, columns].to_csv(
        _DigestWriter(),
        index=False,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    )
    return {**schema, "sha256": digest.hexdigest()}


def load_lendingclub_required_columns(
    source_path: str | Path,
    *,
    time_column: str,
    target_column: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load an unambiguous, deterministically ordered LendingClub column subset."""

    path = Path(source_path)
    time_column = str(time_column)
    target_column = str(target_column)
    if not path.is_file():
        raise TransferStageError(f"LendingClub input is missing: {path}")
    if time_column == target_column:
        raise TransferStageError(
            "LendingClub time and target columns must be distinct; "
            f"source={path}; time_column={time_column!r}; "
            f"target_column={target_column!r}"
        )

    modelling_features: list[str] = []
    seen_features: set[str] = set()
    duplicate_requested: list[str] = []
    excluded_identity_columns = {time_column, target_column}
    for value in feature_columns:
        feature = str(value)
        if feature in excluded_identity_columns:
            if feature not in duplicate_requested:
                duplicate_requested.append(feature)
            continue
        if feature in seen_features:
            if feature not in duplicate_requested:
                duplicate_requested.append(feature)
            continue
        seen_features.add(feature)
        modelling_features.append(feature)

    required_columns = [time_column, target_column, *modelling_features]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        try:
            header = next(csv.reader(source))
        except StopIteration as exc:
            raise TransferStageError(
                f"LendingClub input has no CSV header: {path}"
            ) from exc

    header_counts = Counter(header)
    duplicate_header = sorted(
        column for column, count in header_counts.items() if count > 1
    )
    ambiguous_required = sorted(set(required_columns) & set(duplicate_header))
    if ambiguous_required:
        raise TransferStageError(
            "LendingClub input has ambiguous duplicate required header columns: "
            f"{ambiguous_required}; source={path}; "
            f"required_column_count={len(required_columns)}; "
            f"duplicate_requested_columns={duplicate_requested}; "
            f"time_column={time_column!r}; target_column={target_column!r}"
        )

    missing = [column for column in required_columns if column not in header_counts]
    if missing:
        raise TransferStageError(
            f"LendingClub input is missing required columns: {missing}; "
            f"source={path}; required_column_count={len(required_columns)}; "
            f"duplicate_requested_columns={duplicate_requested}; "
            f"time_column={time_column!r}; target_column={target_column!r}"
        )

    # Reproduced with pandas 3.0.1: the low-memory C parser infers mixed
    # float/string chunks for total_bal_to_income_band (source index 643), then
    # indexes the selected-name list with that source index while constructing
    # a DtypeWarning, raising IndexError inside _concatenate_chunks.
    loaded = pd.read_csv(
        path,
        usecols=required_columns,
        low_memory=False,
    )
    loaded_row_count = len(loaded)
    loaded_counts = Counter(str(column) for column in loaded.columns)
    missing_after_load = [
        column for column in required_columns if loaded_counts[column] != 1
    ]
    unexpected = [
        str(column)
        for column in loaded.columns
        if str(column) not in set(required_columns)
    ]
    if missing_after_load or unexpected:
        raise TransferStageError(
            "LendingClub loaded-column validation failed; "
            f"source={path}; required_column_count={len(required_columns)}; "
            f"missing_or_ambiguous_columns={missing_after_load}; "
            f"unexpected_columns={unexpected}; "
            f"duplicate_requested_columns={duplicate_requested}; "
            f"time_column={time_column!r}; target_column={target_column!r}"
        )

    loaded = loaded.loc[:, required_columns]
    if len(loaded) != loaded_row_count:
        raise TransferStageError(
            "LendingClub row count changed while ordering loaded columns; "
            f"source={path}; before={loaded_row_count}; after={len(loaded)}"
        )
    if loaded.columns.tolist() != required_columns:
        raise TransferStageError(
            "LendingClub loaded-column order is not deterministic; "
            f"source={path}; required_column_count={len(required_columns)}"
        )
    if time_column in modelling_features or target_column in modelling_features:
        raise TransferStageError(
            "LendingClub time or target leakage entered modelling features; "
            f"source={path}; time_column={time_column!r}; "
            f"target_column={target_column!r}"
        )
    return loaded, modelling_features


def _prepare(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> dict[str, Any]:
    evidence_path = Path(config["training_feature_manifest"])
    data_path = Path(config["training_raw_statistical_source"])
    text_path = Path(config["training_text_embeddings_path"])
    for path in (evidence_path, data_path, text_path):
        if not path.exists():
            raise TransferStageError(f"prepare prerequisite missing: {path}")
    evidence = pd.read_csv(evidence_path)
    if "feature" in evidence and "feature_name" not in evidence:
        evidence = evidence.rename(columns={"feature": "feature_name"})
    evidence["text_available"] = evidence.get("description", "").fillna("").astype(str).str.strip().ne("")
    evidence["raw_statistical_evidence_available"] = evidence.get(
        "allowed_for_clip_training", False
    ).fillna(False).astype(bool)
    reconciliation = reconcile_feature_universe(evidence, dataset=SOURCE_DATASET)
    identity_relations, identity_manifest = load_identity_evidence(
        config["identity_evidence_path"],
        reconciled=reconciliation,
        source_dataset=SOURCE_DATASET,
    )

    time_col = str(config["training_time_column"])
    target_col = str(config["training_target_column"])
    eligible_names = reconciliation.loc[
        reconciliation["eligible_for_pairing"], "feature_name"
    ].astype(str).tolist()
    raw, feature_names = load_lendingclub_required_columns(
        data_path,
        time_column=time_col,
        target_column=target_col,
        feature_columns=eligible_names,
    )
    dev = raw[
        (raw[time_col] >= int(config["training_dev_start_day"]))
        & (raw[time_col] < int(config["training_oot_start_day"]))
    ].copy()
    if dev.empty:
        raise TransferStageError("approved LendingClub DEV window is empty")
    raw_dev_evidence = canonical_raw_dev_evidence(
        dev,
        time_column=time_col,
        feature_columns=feature_names,
        target_column=target_col,
        dev_start=int(config["training_dev_start_day"]),
        dev_end=int(config["training_oot_start_day"]),
    )
    reconciliation.loc[
        reconciliation["eligible_for_pairing"]
        & ~reconciliation["feature_name"].isin(feature_names),
        ["eligible_for_pairing", "eligible_for_training", "eligible_for_validation", "excluded"],
    ] = [False, False, False, True]
    reconciliation.loc[
        reconciliation["excluded"]
        & reconciliation["exclusion_reason"].eq(""),
        "exclusion_reason",
    ] = "missing_approved_dev_matrix_column"

    duplicates = find_exact_dev_duplicate_pairs(
        dev[feature_names],
        feature_names=feature_names,
        dataset=SOURCE_DATASET,
        split="train",
    )
    ids_by_name = dict(zip(reconciliation["feature_name"], reconciliation["feature_id"]))
    identity_rows = [
        {
            "feature_id_a": ids_by_name[str(row.anchor_feature_name)],
            "feature_id_b": ids_by_name[str(row.excluded_feature_name)],
            "reason": "exact_dev_duplicate",
        }
        for row in duplicates.itertuples(index=False)
    ]
    identity_rows.extend(
        identity_relations[["feature_id_a", "feature_id_b", "reason"]].to_dict(
            "records"
        )
    )
    split, split_manifest = deterministic_feature_split(
        reconciliation,
        dataset=SOURCE_DATASET,
        seed=int(config["feature_split_seed"]),
        validation_fraction=float(config["validation_fraction"]),
        identity_relations=pd.DataFrame(
            identity_rows, columns=["feature_id_a", "feature_id_b", "reason"]
        ),
    )
    reconciliation = reconciliation.merge(
        split[["feature_id", "split_assignment"]],
        on="feature_id",
        how="left",
        suffixes=("", "_resolved"),
    )
    reconciliation["split_assignment"] = reconciliation[
        "split_assignment_resolved"
    ].fillna("excluded")
    reconciliation = reconciliation.drop(columns=["split_assignment_resolved"])
    source_anchor_config = load_source_anchor_config(config)
    stability_evidence, _ = build_feature_stability_evidence(
        dev,
        reconciliation,
        time_column=time_col,
        config=source_anchor_config,
        exact_duplicates=duplicates,
        verified_aliases=identity_relations.loc[
            identity_relations["reason"].eq("verified_alias"),
            ["feature_name_a", "feature_name_b"],
        ].itertuples(index=False, name=None),
        documented_identity_transforms=identity_relations.loc[
            identity_relations["reason"].eq("documented_identity_transform"),
            ["feature_name_a", "feature_name_b"],
        ].itertuples(index=False, name=None),
    )
    anchor_members, anchor_audit = select_anchor_members(
        stability_evidence,
        config=source_anchor_config,
    )

    descriptors = build_statistical_view_frame(dev[feature_names])
    descriptors = descriptors.merge(
        reconciliation[
            ["feature_id", "feature_name", "semantic_group", "source_table"]
        ],
        on="feature_name",
        how="inner",
    ).merge(
        split[["feature_id", "split_assignment", "identity_group"]],
        on="feature_id",
        how="inner",
    )
    descriptors["dataset"] = SOURCE_DATASET
    descriptors["split"] = descriptors["split_assignment"]
    descriptors["group_key"] = "identity:" + descriptors["identity_group"].astype(str)
    descriptors["source_table_or_formula"] = descriptors["source_table"]
    descriptors["source_manifest_hash"] = sha256_file(evidence_path)
    preprocessor = StatisticalPreprocessor(
        field_order=list(DESCRIPTOR_COLUMNS_V2),
        fit_dataset=SOURCE_DATASET,
        fit_split="train",
        raw_dev_statistical_evidence_hash=raw_dev_evidence["sha256"],
    )
    fit_rows = descriptors[descriptors["split"].eq("train")].copy()
    preprocessor.fit(fit_rows, dataset=SOURCE_DATASET, split="train")
    transformed = preprocessor.transform(descriptors)
    stat_vectors = build_vector_frame(
        metadata=descriptors,
        transformed=transformed,
        preprocessor=preprocessor,
    )
    stat_vectors = stat_vectors.merge(
        descriptors[["feature_name", "feature_id"]], on="feature_name", how="left"
    )
    text = pd.read_parquet(text_path)
    text = text.merge(
        reconciliation[["feature_id", "feature_name"]],
        on="feature_name",
        how="inner",
    )
    pairs = build_feature_positive_pairs(
        split=split,
        text_view=text,
        statistical_view=stat_vectors,
        dataset=SOURCE_DATASET,
        source_manifest_hash=sha256_file(evidence_path),
    )
    train_pairs = _reindex_pairs(pairs[pairs["split"].eq("train")])
    validation_pairs = _reindex_pairs(pairs[pairs["split"].eq("validation")])
    negative = build_negative_policy(
        train_pairs=train_pairs,
        all_homecredit_pairs=pairs,
        text_embeddings=text,
        exact_dev_duplicates=duplicates,
        verified_aliases=identity_relations.loc[
            identity_relations["reason"].eq("verified_alias"),
            ["feature_name_a", "feature_name_b"],
        ].itertuples(index=False, name=None),
        documented_identity_transforms=identity_relations.loc[
            identity_relations["reason"].eq("documented_identity_transform"),
            ["feature_name_a", "feature_name_b"],
        ].itertuples(index=False, name=None),
        training_dataset=SOURCE_DATASET,
        min_safe_negative_count=int(config["min_safe_negative_count"]),
    )

    feature_dir = output_dir / "feature_universe"
    pairing_dir = output_dir / "pairing"
    training_input_dir = output_dir / "training" / "inputs"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pairing_dir.mkdir(parents=True, exist_ok=True)
    training_input_dir.mkdir(parents=True, exist_ok=True)
    reconciliation.to_csv(feature_dir / "feature_reconciliation.csv", index=False)
    split.to_csv(pairing_dir / "lendingclub_v2_feature_split.csv", index=False)
    identity_relations.to_csv(pairing_dir / "identity_relations.csv", index=False)
    write_json(pairing_dir / "identity_evidence_manifest.json", identity_manifest)
    duplicates.to_parquet(pairing_dir / "lendingclub_v2_exact_dev_duplicates.parquet", index=False)
    write_json(
        pairing_dir / "lendingclub_v2_exact_duplicate_manifest.json",
        {
            "policy_version": EXACT_DUPLICATE_POLICY_VERSION,
            "dataset": SOURCE_DATASET,
            "split": "dev",
            "aligned_row_count": len(dev),
            "feature_count": len(feature_names),
            "directed_duplicate_pair_count": len(duplicates),
            "target_excluded": True,
            "oot_excluded": True,
            "evidence_hash": sha256_text(duplicates.to_csv(index=False)),
        },
    )
    train_pairs.to_parquet(pairing_dir / "lendingclub_v2_train_positive_pairs.parquet", index=False)
    validation_pairs.to_parquet(
        pairing_dir / "lendingclub_v2_validation_positive_pairs.parquet", index=False
    )
    negative.exclusion_pairs.to_parquet(
        pairing_dir / "negative_exclusion_pairs.parquet", index=False
    )
    text.to_parquet(training_input_dir / "lendingclub_v2_text_embeddings.parquet", index=False)
    stat_vectors.to_parquet(
        training_input_dir / "lendingclub_v2_statistical_vectors.parquet", index=False
    )
    preprocessor_paths = preprocessor.save(output_dir / "training" / "statistical_preprocessor")
    write_json(pairing_dir / "feature_split_manifest.json", split_manifest)
    write_json(pairing_dir / "negative_policy_manifest.json", negative.manifest)
    write_json(
        pairing_dir / "raw_dev_statistical_evidence_manifest.json",
        raw_dev_evidence,
    )
    data_manifest = {
        "eligible_feature_count": int(len(pairs)),
        "train_feature_count": int(len(train_pairs)),
        "validation_feature_count": int(len(validation_pairs)),
        "source_dataset": SOURCE_DATASET,
        "dev_window": [
            int(config["training_dev_start_day"]),
            int(config["training_oot_start_day"]),
        ],
        "target_used": False,
        "oot_used": False,
        "external_used": False,
        "statistical_preprocessor_hash": preprocessor.preprocessor_hash_,
        "split_hash": split_manifest["split_hash"],
        "source_feature_universe_hash": sha256_file(
            feature_dir / "feature_reconciliation.csv"
        ),
        "identity_evidence_hash": identity_manifest["identity_evidence_hash"],
        "raw_dev_statistical_evidence_hash": raw_dev_evidence["sha256"],
        "raw_statistical_evidence_hash": raw_dev_evidence["sha256"],
        "raw_dev_statistical_evidence_manifest_hash": sha256_file(
            pairing_dir / "raw_dev_statistical_evidence_manifest.json"
        ),
        "configuration_hash": _configuration_hash(config),
        "pairing_policy_version": PAIRING_POLICY_VERSION,
    }
    data_manifest_path = write_json(pairing_dir / "data_manifest.json", data_manifest)
    anchor_paths = write_anchor_selection_artifacts(
        output_dir=output_dir / "source_anchor",
        config=source_anchor_config,
        evidence=stability_evidence,
        audit=anchor_audit,
        members=anchor_members,
        configuration_hash=_configuration_hash(config),
        data_manifest_hash=sha256_file(data_manifest_path),
        external_dataset=EXTERNAL_DATASET,
        source_feature_universe_hash=data_manifest["source_feature_universe_hash"],
        feature_split_hash=split_manifest["split_hash"],
        identity_evidence_hash=identity_manifest["identity_evidence_hash"],
        raw_statistical_evidence_hash=data_manifest["raw_statistical_evidence_hash"],
        statistical_preprocessor_hash=preprocessor.preprocessor_hash_,
    )
    return {
        "eligible_feature_count": len(pairs),
        "data_manifest_hash": sha256_file(data_manifest_path),
        "preprocessor_hash": preprocessor.preprocessor_hash_,
        "preprocessor_paths": {key: str(value) for key, value in preprocessor_paths.items()},
        "source_anchor_member_count": len(anchor_members),
        "source_anchor_paths": {
            key: str(value).replace("\\", "/") for key, value in anchor_paths.items()
        },
        "input_hashes": {
            str(path).replace("\\", "/"): sha256_file(path)
            for path in (
                evidence_path,
                data_path,
                text_path,
                Path(config["identity_evidence_path"]),
            )
        },
    }


def _train(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    resume: bool = False,
) -> dict[str, Any]:
    pairing_dir = output_dir / "pairing"
    train_pairs = pd.read_parquet(pairing_dir / "lendingclub_v2_train_positive_pairs.parquet")
    validation_pairs = pd.read_parquet(
        pairing_dir / "lendingclub_v2_validation_positive_pairs.parquet"
    )
    negative = pd.read_parquet(pairing_dir / "negative_exclusion_pairs.parquet")
    input_dir = output_dir / "training" / "inputs"
    text = pd.read_parquet(input_dir / "lendingclub_v2_text_embeddings.parquet")
    stat = pd.read_parquet(input_dir / "lendingclub_v2_statistical_vectors.parquet")
    data_manifest_path = pairing_dir / "data_manifest.json"
    preprocessor_path = (
        output_dir / "training" / "statistical_preprocessor" / "statistical_preprocessor.json"
    )
    training_config_path = Path(config["_config_dir"]) / "training.yaml"
    training_config = load_training_config(training_config_path)
    config_hash = _configuration_hash(config)
    anchor_dir = output_dir / "source_anchor"
    anchor_members_path = anchor_dir / "anchor_members.csv"
    source_anchor_selection_manifest_path = (
        anchor_dir / "source_anchor_selection_manifest.json"
    )
    source_anchor_manifest_path = anchor_dir / "source_anchor_manifest.json"
    if (
        not anchor_members_path.exists()
        or not source_anchor_selection_manifest_path.exists()
    ):
        raise TransferStageError(
            "train requires completed source-anchor stability selection from prepare"
        )
    source_anchor_config = load_source_anchor_config(config)
    anchor_members = pd.read_csv(anchor_members_path)
    source_anchor_manifest = read_json(source_anchor_selection_manifest_path)
    data_manifest = read_json(data_manifest_path)
    validate_source_anchor_artifacts(
        config=source_anchor_config,
        members=anchor_members,
        manifest=source_anchor_manifest,
        training_feature_ids=set(train_pairs["feature_id"].astype(str)),
        configuration_hash=config_hash,
        data_manifest_hash=sha256_file(data_manifest_path),
        external_dataset=EXTERNAL_DATASET,
        source_feature_universe_hash=data_manifest["source_feature_universe_hash"],
        feature_split_hash=data_manifest["split_hash"],
        identity_evidence_hash=data_manifest["identity_evidence_hash"],
        raw_statistical_evidence_hash=data_manifest["raw_statistical_evidence_hash"],
        statistical_preprocessor_hash=data_manifest["statistical_preprocessor_hash"],
        member_path=anchor_members_path,
        evidence_path=anchor_dir / "feature_stability_evidence.csv",
    )
    training_config = replace(
        training_config,
        output_dir=output_dir / "training",
        training_dataset=SOURCE_DATASET,
        external_dataset=EXTERNAL_DATASET,
        configuration_hash=config_hash,
        data_manifest_hash=sha256_file(data_manifest_path),
        statistical_preprocessor_hash=read_json(preprocessor_path)["preprocessor_hash"],
        source_anchor_hash=sha256_file(source_anchor_selection_manifest_path),
    )
    upstream = {
        "configuration_hash": config_hash,
        "data_manifest_hash": sha256_file(data_manifest_path),
        "statistical_preprocessor_hash": read_json(preprocessor_path)["preprocessor_hash"],
        "source_anchor_selection_manifest_hash": sha256_file(
            source_anchor_selection_manifest_path
        ),
    }
    data = TrainingDataBundle(
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        external_pairs=pd.DataFrame(),
        source_pairs=_reindex_pairs(pd.concat([train_pairs, validation_pairs])),
        training_text=text,
        external_text=pd.DataFrame(),
        training_stat=stat,
        external_stat=pd.DataFrame(),
        training_dataset=SOURCE_DATASET,
        external_dataset=EXTERNAL_DATASET,
        negative_exclusions=negative,
        upstream_hashes=upstream,
        text_dim=training_config.model.text_input_dim,
        statistical_dim=training_config.model.statistical_input_dim,
        statistical_fields=list(DESCRIPTOR_COLUMNS_V2),
    )
    results = []
    anchor_dir.mkdir(parents=True, exist_ok=True)
    seed_anchor_rows = []
    anchor_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    validation_score_frames = []
    for seed in seeds:
        seed_dir = output_dir / "training" / "seeds" / f"seed_{seed}"
        existing_checkpoint = seed_dir / "best_checkpoint.pt"
        existing_checkpoint_manifest = seed_dir / "checkpoint_manifest.json"
        existing_anchor = anchor_dir / f"seed_{seed}_anchor.npy"
        existing_anchor_manifest = anchor_dir / f"seed_{seed}_anchor_manifest.json"
        if resume and all(
            path.exists()
            for path in (
                existing_checkpoint,
                existing_checkpoint_manifest,
                existing_anchor,
                existing_anchor_manifest,
            )
        ):
            checkpoint_manifest = read_json(existing_checkpoint_manifest)
            seed_manifest = read_json(existing_anchor_manifest)
            vector = np.load(existing_anchor)
            if sha256_file(existing_checkpoint) != checkpoint_manifest.get(
                "checkpoint_sha256"
            ):
                raise TransferStageError(f"seed {seed}: existing checkpoint is corrupt")
            validate_checkpoint_manifest(
                checkpoint_manifest,
                expected={
                    "source_dataset": SOURCE_DATASET,
                    "external_dataset": EXTERNAL_DATASET,
                    "pairing_policy_version": PAIRING_POLICY_VERSION,
                    "configuration_hash": config_hash,
                    "data_manifest_hash": sha256_file(data_manifest_path),
                    "statistical_preprocessor_hash": upstream[
                        "statistical_preprocessor_hash"
                    ],
                    "source_anchor_hash": seed_manifest["anchor_hash"],
                },
            )
            temporary_source_manifest = {
                **source_anchor_manifest,
                "anchor_hashes_by_seed": {
                    **source_anchor_manifest.get("anchor_hashes_by_seed", {}),
                    str(seed): seed_manifest.get("anchor_hash"),
                },
            }
            validate_seed_anchor(
                vector=vector,
                seed_manifest=seed_manifest,
                source_manifest=temporary_source_manifest,
                expected_seed=seed,
                expected_checkpoint_hash=checkpoint_manifest["checkpoint_sha256"],
            )
            validation_scores = pd.read_csv(
                anchor_dir / f"seed_{seed}_lendingclub_validation_scores.csv"
            )
            checkpoint_hashes[str(seed)] = checkpoint_manifest["checkpoint_sha256"]
            anchor_hashes[str(seed)] = seed_manifest["anchor_hash"]
            seed_anchor_rows.append(seed_manifest)
            validation_score_frames.append(validation_scores)
            results.append(
                {
                    "seed": seed,
                    "checkpoint_hash": checkpoint_manifest["checkpoint_sha256"],
                    "source_anchor_hash": seed_manifest["anchor_hash"],
                    "resume_status": "reused_valid_completed_seed",
                }
            )
            continue
        if not resume and any(
            path.exists()
            for path in (
                existing_checkpoint,
                existing_checkpoint_manifest,
                existing_anchor,
                existing_anchor_manifest,
            )
        ):
            raise TransferStageError(
                f"seed {seed}: existing output would be overwritten"
            )
        result = train_seed(
            config=training_config,
            data=data,
            seed=seed,
            output_dir=output_dir / "training",
            config_snapshot_text=training_config_path.read_text(encoding="utf-8"),
        )
        model = load_checkpoint(
            checkpoint_path=result.checkpoint_path,
            manifest_path=result.checkpoint_manifest_path,
            config=training_config,
            upstream_hashes=upstream,
            expected_metadata={
                "source_dataset": SOURCE_DATASET,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
            },
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        source_pairs = data.source_pairs
        source_text, source_stat = tensors_for_pairs(source_pairs, text, stat)
        with torch.no_grad():
            text_projection, stat_projection = model(source_text, source_stat)
            joint = torch.nn.functional.normalize(
                (text_projection + stat_projection) / 2.0, p=2, dim=-1
            ).numpy()
        source_embeddings = source_pairs[
            ["feature_id", "feature_name", "dataset", "split"]
        ].copy()
        for index in range(joint.shape[1]):
            source_embeddings[f"joint_{index:04d}"] = joint[:, index]
        anchor, anchor_manifest = build_seed_anchor(
            source_embeddings[source_embeddings["split"].eq("train")],
            anchor_members,
            seed=seed,
            checkpoint_hash=result.checkpoint_hash,
        )
        anchor_manifest["raw_dev_statistical_evidence_hash"] = data_manifest[
            "raw_dev_statistical_evidence_hash"
        ]
        np.save(anchor_dir / f"seed_{seed}_anchor.npy", anchor)
        joint_columns = sorted(
            column
            for column in source_embeddings.columns
            if str(column).startswith("joint_")
        )
        validation_embeddings = source_embeddings[
            source_embeddings["split"].eq("validation")
        ].copy()
        validation_values = validation_embeddings[joint_columns].to_numpy(
            dtype=float
        )
        validation_norms = np.linalg.norm(validation_values, axis=1, keepdims=True)
        if bool((validation_norms <= 0).any()):
            raise TransferStageError(
                "LendingClub validation embedding contains a zero-norm row"
            )
        validation_scores = validation_embeddings[
            ["feature_id", "feature_name", "dataset", "split"]
        ].copy()
        validation_scores["seed"] = seed
        validation_scores["anchor_similarity"] = (
            validation_values / validation_norms
        ) @ anchor
        validation_scores["anchor_hash"] = anchor_manifest["anchor_hash"]
        validation_scores["checkpoint_hash"] = result.checkpoint_hash
        validation_score_path = (
            anchor_dir / f"seed_{seed}_lendingclub_validation_scores.csv"
        )
        validation_scores.to_csv(validation_score_path, index=False)
        anchor_manifest["lendingclub_validation_score_path"] = str(
            validation_score_path
        ).replace("\\", "/")
        anchor_manifest["lendingclub_validation_score_hash"] = sha256_file(
            validation_score_path
        )
        write_json(anchor_dir / f"seed_{seed}_anchor_manifest.json", anchor_manifest)
        checkpoint_manifest = read_json(result.checkpoint_manifest_path)
        checkpoint_manifest["source_anchor_hash"] = anchor_manifest["anchor_hash"]
        write_json(result.checkpoint_manifest_path, checkpoint_manifest)
        anchor_hashes[str(seed)] = anchor_manifest["anchor_hash"]
        checkpoint_hashes[str(seed)] = result.checkpoint_hash
        seed_anchor_rows.append(anchor_manifest)
        validation_score_frames.append(validation_scores)
        results.append(
            {
                "seed": seed,
                "checkpoint_hash": result.checkpoint_hash,
                "source_anchor_hash": anchor_manifest["anchor_hash"],
            }
        )
    source_anchor_manifest["anchor_hashes_by_seed"] = anchor_hashes
    source_anchor_manifest["checkpoint_hashes_by_seed"] = checkpoint_hashes
    pd.DataFrame(seed_anchor_rows).to_csv(
        anchor_dir / "seed_anchor_manifest.csv", index=False
    )
    validation_consensus = (
        pd.concat(validation_score_frames, ignore_index=True)
        .groupby(["feature_id", "feature_name", "dataset", "split"], as_index=False)
        .agg(
            consensus_anchor_similarity=("anchor_similarity", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(
            ["consensus_anchor_similarity", "feature_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if not validation_consensus["seed_count"].eq(len(seeds)).all():
        raise TransferStageError(
            "LendingClub validation consensus does not contain every fixed seed"
        )
    validation_consensus["consensus_rank"] = range(
        1, len(validation_consensus) + 1
    )
    validation_consensus_path = (
        anchor_dir / "lendingclub_validation_consensus_scores.csv"
    )
    validation_consensus.to_csv(validation_consensus_path, index=False)
    source_anchor_manifest["lendingclub_validation_consensus_score_path"] = str(
        validation_consensus_path
    ).replace("\\", "/")
    source_anchor_manifest["lendingclub_validation_consensus_score_hash"] = (
        sha256_file(validation_consensus_path)
    )
    write_json(source_anchor_manifest_path, source_anchor_manifest)
    validate_source_anchor_artifacts(
        config=source_anchor_config,
        members=anchor_members,
        manifest=source_anchor_manifest,
        training_feature_ids=set(train_pairs["feature_id"].astype(str)),
        configuration_hash=config_hash,
        data_manifest_hash=sha256_file(data_manifest_path),
        external_dataset=EXTERNAL_DATASET,
        source_feature_universe_hash=data_manifest["source_feature_universe_hash"],
        feature_split_hash=data_manifest["split_hash"],
        identity_evidence_hash=data_manifest["identity_evidence_hash"],
        raw_statistical_evidence_hash=data_manifest["raw_statistical_evidence_hash"],
        statistical_preprocessor_hash=data_manifest["statistical_preprocessor_hash"],
        require_seed_hashes=True,
        member_path=anchor_members_path,
        evidence_path=anchor_dir / "feature_stability_evidence.csv",
    )
    return {
        "seeds": results,
        "selection_used_external_results": False,
        "source_anchor_member_count": len(anchor_members),
        "anchor_hashes_by_seed": anchor_hashes,
    }


def _project(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> dict[str, Any]:
    pairing_dir = output_dir / "pairing"
    data_manifest_path = pairing_dir / "data_manifest.json"
    train_pairs = pd.read_parquet(
        pairing_dir / "lendingclub_v2_train_positive_pairs.parquet"
    )
    anchor_dir = output_dir / "source_anchor"
    anchor_members_path = anchor_dir / "anchor_members.csv"
    source_anchor_manifest_path = anchor_dir / "source_anchor_manifest.json"
    anchor_members = pd.read_csv(anchor_members_path)
    source_anchor_manifest = read_json(source_anchor_manifest_path)
    data_manifest = read_json(data_manifest_path)
    source_anchor_config = load_source_anchor_config(config)
    validate_source_anchor_artifacts(
        config=source_anchor_config,
        members=anchor_members,
        manifest=source_anchor_manifest,
        training_feature_ids=set(train_pairs["feature_id"].astype(str)),
        configuration_hash=_configuration_hash(config),
        data_manifest_hash=sha256_file(data_manifest_path),
        external_dataset=EXTERNAL_DATASET,
        source_feature_universe_hash=data_manifest["source_feature_universe_hash"],
        feature_split_hash=data_manifest["split_hash"],
        identity_evidence_hash=data_manifest["identity_evidence_hash"],
        raw_statistical_evidence_hash=data_manifest["raw_statistical_evidence_hash"],
        statistical_preprocessor_hash=data_manifest["statistical_preprocessor_hash"],
        require_seed_hashes=True,
        member_path=anchor_members_path,
        evidence_path=anchor_dir / "feature_stability_evidence.csv",
    )
    validated_seed_anchors: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for seed in seeds:
        checkpoint_manifest = read_json(
            output_dir
            / "training"
            / "seeds"
            / f"seed_{seed}"
            / "checkpoint_manifest.json"
        )
        seed_manifest = read_json(
            anchor_dir / f"seed_{seed}_anchor_manifest.json"
        )
        vector = np.load(anchor_dir / f"seed_{seed}_anchor.npy")
        validate_seed_anchor(
            vector=vector,
            seed_manifest=seed_manifest,
            source_manifest=source_anchor_manifest,
            expected_seed=seed,
            expected_checkpoint_hash=checkpoint_manifest["checkpoint_sha256"],
        )
        validated_seed_anchors[seed] = (vector, seed_manifest)

    # Home Credit is loaded only after every source anchor has been validated
    # and frozen in its own seed space.
    project = load_named_project_config("homecredit")
    experiment = ExperimentConfig(
        experiment_name="reverse_projection_descriptor_build",
        selector_name="mrmr",
        dataset_name="homecredit",
        data_dir=str(project["data_dir"]),
        description_path=str(project["description_path"]),
        dev_start_day=int(project["dev_start_day"]),
        oot_start_day=int(project["oot_start_day"]),
        oot_end_day=int(project["oot_end_day"]),
        excluded_feature_columns=tuple(project["excluded_feature_columns"]),
    )
    prepared = prepare_modeling_data(experiment)
    source_target_free_dev = prepared.X_train.drop(
        columns=[prepared.time_col], errors="ignore"
    )
    text = pd.read_parquet(config["external_text_embeddings_path"])
    text_names = set(text["feature_name"].astype(str))
    external_manifest = pd.read_csv(config["external_feature_manifest"])
    if "feature_id" not in external_manifest or "feature_name" not in external_manifest:
        raise TransferStageError("external feature manifest lacks feature_id/feature_name")
    manifest_ids = dict(
        zip(
            external_manifest["feature_name"].astype(str),
            external_manifest["feature_id"].astype(str),
        )
    )
    all_external_features = [
        str(name) for name in source_target_free_dev.columns
    ]
    available = [
        name
        for name in all_external_features
        if name in text_names and name in manifest_ids
    ]
    descriptors = build_statistical_view_frame(source_target_free_dev[available])
    descriptors["feature_id"] = descriptors["feature_name"].map(manifest_ids)
    descriptors["dataset"] = EXTERNAL_DATASET
    descriptors["split"] = "external_validation"
    external_metadata = external_manifest.copy()
    external_metadata["feature_id"] = external_metadata["feature_id"].astype(str)
    for column, default in [
        ("semantic_group", "unknown"),
        ("source_table", "unknown"),
    ]:
        if column not in external_metadata:
            external_metadata[column] = default
    descriptors = descriptors.merge(
        external_metadata[["feature_id", "semantic_group", "source_table"]],
        on="feature_id",
        how="left",
    )
    descriptors["semantic_group"] = descriptors["semantic_group"].fillna("unknown")
    descriptors["group_key"] = "external:" + descriptors["feature_id"].astype(str)
    descriptors["source_table_or_formula"] = descriptors["source_table"].fillna("unknown")
    descriptors["source_manifest_hash"] = sha256_file(
        config["external_feature_manifest"]
    )
    projection_dir = output_dir / "reverse_projection"
    projection_dir.mkdir(parents=True, exist_ok=True)
    raw_descriptor_path = projection_dir / "homecredit_raw_descriptors.parquet"
    descriptors.to_parquet(raw_descriptor_path, index=False)
    preprocessor_path = (
        output_dir / "training" / "statistical_preprocessor" / "statistical_preprocessor.joblib"
    )
    preprocessor: StatisticalPreprocessor = joblib.load(preprocessor_path)
    if (
        preprocessor.fit_dataset != SOURCE_DATASET
        or preprocessor.preprocessor_hash_
        != data_manifest["statistical_preprocessor_hash"]
        or preprocessor.raw_dev_statistical_evidence_hash
        != data_manifest["raw_dev_statistical_evidence_hash"]
    ):
        raise TransferStageError(
            "source statistical preprocessor role or hash mismatch"
        )
    transformed = preprocessor.transform(descriptors)
    external_stat_vectors = build_vector_frame(
        metadata=descriptors,
        transformed=transformed,
        preprocessor=preprocessor,
    ).merge(
        descriptors[["feature_name", "feature_id"]],
        on="feature_name",
        how="left",
    )
    validate_frozen_external_transform(
        external_stat_vectors,
        source_dataset=SOURCE_DATASET,
        external_dataset=EXTERNAL_DATASET,
        preprocessor_hash=preprocessor.preprocessor_hash_,
    )
    text = text[text["feature_name"].isin(available)].copy()
    if set(text["dataset"].astype(str)) != {EXTERNAL_DATASET}:
        raise TransferStageError("external text embedding dataset metadata mismatch")
    text["feature_id"] = text["feature_name"].map(manifest_ids)
    stat_path = projection_dir / "homecredit_statistical_vectors.parquet"
    external_stat_vectors.to_parquet(stat_path, index=False)
    text, external_stat_vectors, alignment_exclusions, alignment_manifest = (
        align_external_feature_views(
            text,
            external_stat_vectors,
            external_dataset=EXTERNAL_DATASET,
            semantic_hash=sha256_file(config["external_text_embeddings_path"]),
            statistical_hash=sha256_file(stat_path),
        )
    )
    alignment_manifest.update(
        {
            "external_feature_universe_hash": sha256_file(
                config["external_feature_manifest"]
            ),
            "external_raw_statistical_evidence_hash": sha256_file(
                raw_descriptor_path
            ),
            "configuration_hash": _configuration_hash(config),
            "data_manifest_hash": sha256_file(data_manifest_path),
        }
    )
    alignment_manifest["alignment_manifest_hash"] = sha256_text(
        json.dumps(alignment_manifest, sort_keys=True)
    )
    alignment_manifest_path = write_json(
        projection_dir / "alignment_manifest.json", alignment_manifest
    )
    alignment_exclusions.to_csv(
        projection_dir / "alignment_exclusions.csv", index=False
    )
    features = text[["feature_id", "feature_name", "dataset"]].copy()
    text_columns = sorted(
        column
        for column in text.columns
        if str(column).startswith("embedding_") and len(str(column)) == 14
    )
    seed_frames: dict[int, pd.DataFrame] = {}
    stat_columns = sorted(
        column
        for column in external_stat_vectors.columns
        if str(column).startswith("stat_")
    )
    training_config = load_training_config(Path(config["_config_dir"]) / "training.yaml")
    for seed in seeds:
        seed_dir = output_dir / "training" / "seeds" / f"seed_{seed}"
        checkpoint_path = seed_dir / "best_checkpoint.pt"
        manifest_path = seed_dir / "checkpoint_manifest.json"
        checkpoint_manifest = read_json(manifest_path)
        anchor, anchor_manifest = validated_seed_anchors[seed]
        if sha256_file(checkpoint_path) != checkpoint_manifest.get("checkpoint_sha256"):
            raise TransferStageError("checkpoint hash does not match its manifest")
        config_hash = sha256_text(
            "".join(
                Path(path).read_text(encoding="utf-8")
                for path in config["_config_files"]
            )
        )
        validate_checkpoint_manifest(
            checkpoint_manifest,
            expected={
                "source_dataset": SOURCE_DATASET,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "configuration_hash": config_hash,
                "data_manifest_hash": sha256_file(
                    output_dir / "pairing" / "data_manifest.json"
                ),
                "statistical_preprocessor_hash": preprocessor.preprocessor_hash_,
                "source_anchor_hash": anchor_manifest["anchor_hash"],
            },
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = SemanticStatisticalContrastiveEncoder(training_config.model)
        model.load_state_dict(payload["model_state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embeddings, scores = frozen_project(
            model=model,
            features=features,
            text_values=text[text_columns].to_numpy(dtype=np.float32),
            statistical_values=external_stat_vectors[stat_columns].to_numpy(
                dtype=np.float32
            ),
            source_dataset=SOURCE_DATASET,
            external_dataset=EXTERNAL_DATASET,
            checkpoint_hash=checkpoint_manifest["checkpoint_sha256"],
            anchor=anchor,
            anchor_hash=anchor_manifest["anchor_hash"],
            preprocessor_hash=preprocessor.preprocessor_hash_,
        )
        embeddings["seed"] = seed
        embeddings["configuration_hash"] = _configuration_hash(config)
        embeddings["data_manifest_hash"] = sha256_file(data_manifest_path)
        embeddings["alignment_manifest_hash"] = sha256_file(
            alignment_manifest_path
        )
        embeddings["source_feature_universe_hash"] = data_manifest[
            "source_feature_universe_hash"
        ]
        embeddings["source_split_hash"] = data_manifest["split_hash"]
        embeddings = embeddings.merge(
            scores[["feature_id", "learned_similarity"]], on="feature_id"
        )
        seed_frames[seed] = embeddings
        embeddings.to_parquet(
            projection_dir / f"seed_{seed}_homecredit_reverse_embeddings.parquet",
            index=False,
        )
    consensus, consensus_manifest = aggregate_seed_embeddings(
        seed_frames, seed_list=seeds, reference_seed=11
    )
    consensus = consensus.rename(
        columns={
            "consensus_score": "learned_similarity",
            "consensus_rank": "consensus_clip_rank",
        }
    )
    consensus["source_dataset"] = SOURCE_DATASET
    consensus["external_dataset"] = EXTERNAL_DATASET
    consensus["pairing_policy_version"] = PAIRING_POLICY_VERSION
    consensus["configuration_hash"] = _configuration_hash(config)
    consensus["data_manifest_hash"] = sha256_file(data_manifest_path)
    consensus["statistical_preprocessor_hash"] = preprocessor.preprocessor_hash_
    consensus["alignment_manifest_hash"] = sha256_file(alignment_manifest_path)
    consensus.to_parquet(
        projection_dir / "homecredit_reverse_embeddings.parquet", index=False
    )
    score_columns = [
        "feature_id",
        "feature_name",
        "dataset",
        "learned_similarity",
        "consensus_clip_rank",
        "source_dataset",
        "external_dataset",
        "pairing_policy_version",
        "configuration_hash",
        "data_manifest_hash",
        "statistical_preprocessor_hash",
        "alignment_manifest_hash",
    ]
    consensus[score_columns].to_csv(
        projection_dir / "homecredit_reverse_scores.csv", index=False
    )
    reconciliation = pd.DataFrame({"feature_name": all_external_features})
    reconciliation["feature_id"] = reconciliation["feature_name"].map(manifest_ids)
    reconciliation["text_available"] = reconciliation["feature_name"].isin(text_names)
    reconciliation["raw_statistical_evidence_available"] = True
    reconciliation["compatible"] = (
        reconciliation["text_available"] & reconciliation["feature_id"].notna()
    )
    reconciliation["exclusion_reason"] = np.where(
        reconciliation["compatible"],
        "",
        np.where(
            ~reconciliation["text_available"],
            "missing_semantic_text_embedding",
            "missing_deterministic_feature_identity",
        ),
    )
    reconciliation.to_csv(
        projection_dir / "homecredit_reverse_feature_reconciliation.csv", index=False
    )
    manifest = {
        **consensus_manifest,
        "source_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "external_refit": False,
        "fit_scope": "transform_only_with_frozen_lendingclub_objects",
        "external_target_used": False,
        "external_oot_used": False,
        "projected_feature_count": len(consensus),
        "source_feature_universe_hash": data_manifest[
            "source_feature_universe_hash"
        ],
        "source_split_hash": data_manifest["split_hash"],
        "source_preprocessor_hash": preprocessor.preprocessor_hash_,
        "source_raw_dev_statistical_evidence_hash": data_manifest[
            "raw_dev_statistical_evidence_hash"
        ],
        "source_checkpoint_hashes_by_seed": source_anchor_manifest[
            "checkpoint_hashes_by_seed"
        ],
        "source_anchor_hashes_by_seed": source_anchor_manifest[
            "anchor_hashes_by_seed"
        ],
        "external_feature_universe_hash": alignment_manifest[
            "external_feature_universe_hash"
        ],
        "external_raw_statistical_evidence_hash": alignment_manifest[
            "external_raw_statistical_evidence_hash"
        ],
        "external_text_embedding_hash": alignment_manifest[
            "semantic_input_hash"
        ],
        "alignment_manifest_hash": sha256_file(alignment_manifest_path),
        "configuration_hash": _configuration_hash(config),
        "data_manifest_hash": sha256_file(data_manifest_path),
    }
    write_json(projection_dir / "reverse_projection_manifest.json", manifest)
    return {
        **manifest,
        "input_hashes": {
            str(Path(path)).replace("\\", "/"): sha256_file(path)
            for path in (
                config["external_feature_manifest"],
                config["external_text_embeddings_path"],
            )
        },
        "input_fingerprints": {
            str(project["data_dir"]).replace("\\", "/"): build_data_version(
                project["data_dir"]
            )
        },
    }


def _validate_candidate_pool(
    *,
    actual_path: Path,
    expected: pd.DataFrame,
    model: str,
    config_hash: str,
) -> None:
    actual = pd.read_csv(actual_path)
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise TransferStageError(
            f"evaluate {model}: candidate pool mismatch"
        ) from exc
    if (
        actual["feature_id"].astype(str).duplicated().any()
        or actual["feature_name"].astype(str).duplicated().any()
        or not actual["configuration_hash"].astype(str).eq(config_hash).all()
        or not actual["model"].astype(str).eq(model).all()
    ):
        raise TransferStageError(
            f"evaluate {model}: candidate pool provenance is invalid"
        )


def _evaluate_model_resume_state(
    *,
    config: dict[str, Any],
    output_dir: Path,
    model: str,
    project: dict[str, Any],
    config_hash: str,
    expected_pool: pd.DataFrame,
    budget: dict[str, int],
) -> str:
    paths = _evaluate_model_artifact_paths(output_dir, model)
    run_paths = paths[1:]
    existing = [path for path in run_paths if path.exists()]
    if not existing:
        if paths[0].exists():
            _validate_candidate_pool(
                actual_path=paths[0],
                expected=expected_pool,
                model=model,
                config_hash=config_hash,
            )
        return "absent"
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise TransferStageError(
            f"evaluate {model}: incomplete existing model artifacts: {missing}"
        )
    _validate_completed_evaluate_model(
        config=config,
        output_dir=output_dir,
        model=model,
        project=project,
        config_hash=config_hash,
        expected_pool=expected_pool,
        budget=budget,
    )
    return "complete"


def _validate_completed_evaluate_model(
    *,
    config: dict[str, Any],
    output_dir: Path,
    model: str,
    project: dict[str, Any],
    config_hash: str,
    expected_pool: pd.DataFrame,
    budget: dict[str, int],
) -> None:
    paths = _evaluate_model_artifact_paths(output_dir, model)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise TransferStageError(
            f"evaluate {model}: required model artifacts are missing: {missing}"
        )
    _validate_candidate_pool(
        actual_path=paths[0],
        expected=expected_pool,
        model=model,
        config_hash=config_hash,
    )
    run_dir = output_dir / "downstream" / (
        "logistic_regression" if model == "lr" else "catboost"
    )
    selected = pd.read_csv(run_dir / "features/final_selected_features.csv")
    required_selected = {
        "feature_name",
        "feature",
        "semantic_group",
        "rank",
        "selector",
    }
    if (
        not required_selected.issubset(selected.columns)
        or len(selected) != int(budget["feature_budget"])
        or selected["feature_name"].astype(str).duplicated().any()
    ):
        raise TransferStageError(
            f"evaluate {model}: selected-feature schema or count mismatch"
        )
    selected_names = selected.sort_values("rank", kind="mergesort")[
        "feature_name"
    ].astype(str).tolist()
    candidate_names = set(expected_pool["feature_name"].astype(str))
    if not set(selected_names).issubset(candidate_names):
        raise TransferStageError(
            f"evaluate {model}: selected features are outside the candidate pool"
        )
    metadata = read_json(run_dir / "models/final_model_metadata.json")
    expected_model_params = resolve_model_kwargs(project, model)
    if (
        metadata.get("model") != model
        or metadata.get("config_hash") != config_hash
        or int(metadata.get("feature_budget", -1)) != int(budget["feature_budget"])
        or list(map(str, metadata.get("selected_features", []))) != selected_names
        or metadata.get("model_params") != expected_model_params
    ):
        raise TransferStageError(
            f"evaluate {model}: model configuration or selected-feature mismatch"
        )
    selector_manifest = pd.read_csv(
        run_dir
        / "llm_responses/final_dev/"
        "reverse_transfer_clip_then_mrmr_selection_manifest.csv"
    )
    required_selector = {
        "feature_name",
        "configuration_hash",
        "screening_pool_member",
        "final_selected",
        "final_rank",
    }
    if not required_selector.issubset(selector_manifest.columns):
        raise TransferStageError(
            f"evaluate {model}: selector manifest schema mismatch"
        )
    final_mask = (
        selector_manifest["final_selected"].astype(str).str.lower().eq("true")
    )
    selector_selected = (
        selector_manifest.loc[final_mask]
        .sort_values("final_rank", kind="mergesort")["feature_name"]
        .astype(str)
        .tolist()
    )
    if (
        selector_selected != selected_names
        or not selector_manifest["configuration_hash"]
        .astype(str)
        .eq(config_hash)
        .all()
    ):
        raise TransferStageError(
            f"evaluate {model}: selector lineage does not match final features"
        )

    identity_manifest = read_json(
        run_dir / "data/source_identity_manifest.json"
    )
    if (
        identity_manifest.get("stable_row_id_column") != "SK_ID_CURR"
        or identity_manifest.get("dataset") != EXTERNAL_DATASET
    ):
        raise TransferStageError(
            f"evaluate {model}: source identity provenance mismatch"
        )
    prediction_manifest = read_json(
        run_dir / "manifests/prediction_manifest.json"
    )
    prediction_entries = {
        str(entry.get("split")): entry
        for entry in prediction_manifest.get("predictions", [])
    }
    expected_prediction_paths = {
        "dev": run_dir / "results/dev_predictions.csv",
        "DEV_OOF": run_dir / "results/dev_oof_predictions.csv",
        "oot": run_dir / "results/oot_predictions.csv",
    }
    if set(prediction_entries) != set(expected_prediction_paths):
        raise TransferStageError(
            f"evaluate {model}: prediction manifest split mismatch"
        )
    prediction_frames: dict[str, pd.DataFrame] = {}
    expected_metadata = {
        "dataset": EXTERNAL_DATASET,
        "method": REVERSE_METHOD,
        "model": model,
        "source_training_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "configuration_hash": config_hash,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "stable_row_id_column": "SK_ID_CURR",
        "source_identity_manifest_hash": identity_manifest.get(
            "source_identity_manifest_hash"
        ),
    }
    for split, path in expected_prediction_paths.items():
        entry = prediction_entries[split]
        if Path(str(entry.get("prediction_path"))).resolve() != path.resolve():
            raise TransferStageError(
                f"evaluate {model}: prediction path mismatch for {split}"
            )
        if entry.get("prediction_hash") != sha256_file(path):
            raise TransferStageError(
                f"evaluate {model}: modified prediction artifact for {split}"
            )
        if any(entry.get(key) != value for key, value in expected_metadata.items()):
            raise TransferStageError(
                f"evaluate {model}: prediction provenance mismatch for {split}"
            )
        frame = pd.read_csv(path)
        required_prediction = {
            "stable_row_id",
            "target",
            "prediction_probability",
            "predicted_class",
        }
        probabilities = pd.to_numeric(
            frame.get("prediction_probability"), errors="coerce"
        )
        if (
            not required_prediction.issubset(frame.columns)
            or len(frame) != int(entry.get("row_count", -1))
            or frame["stable_row_id"].isna().any()
            or frame["stable_row_id"].astype(str).duplicated().any()
            or probabilities.isna().any()
            or not np.isfinite(probabilities).all()
            or not probabilities.between(0, 1).all()
        ):
            raise TransferStageError(
                f"evaluate {model}: invalid prediction rows for {split}"
            )
        prediction_frames[split] = frame
    dev_ids = set(prediction_frames["dev"]["stable_row_id"].astype(str))
    oot_ids = set(prediction_frames["oot"]["stable_row_id"].astype(str))
    if dev_ids & oot_ids:
        raise TransferStageError(
            f"evaluate {model}: DEV/OOT stable-ID overlap"
        )

    fold_payload = read_json(run_dir / "manifests/fold_manifest.json")
    folds = fold_payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise TransferStageError(
            f"evaluate {model}: fold manifest is incomplete"
        )
    fold_hash = hashlib.sha256(
        json.dumps(folds, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        fold_payload.get("fold_manifest_hash") != fold_hash
        or prediction_entries["DEV_OOF"].get("fold_manifest_hash") != fold_hash
    ):
        raise TransferStageError(f"evaluate {model}: fold manifest hash mismatch")
    oof = prediction_frames["DEV_OOF"].copy()
    validation_union: set[str] = set()
    reconciliation_rows = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        validation_values = list(map(str, fold["validation_ids"]))
        validation_ids = set(validation_values)
        training_ids = set(map(str, fold["training_ids"]))
        observed = set(
            oof.loc[
                oof["fold_id"].astype(int).eq(fold_id), "stable_row_id"
            ].astype(str)
        )
        validation_hash = hashlib.sha256(
            json.dumps(
                sorted(validation_values), separators=(",", ":")
            ).encode()
        ).hexdigest()
        if (
            len(validation_values) != len(validation_ids)
            or len(validation_values) != int(fold["validation_row_count"])
            or validation_hash != fold["validation_id_hash"]
            or validation_union & validation_ids
            or training_ids & validation_ids
            or observed != validation_ids
        ):
            raise TransferStageError(
                f"evaluate {model}: fold {fold_id} OOF identity mismatch"
            )
        reconciliation_rows.append(
            {
                "fold": fold_id,
                "validation_rows": len(validation_values),
                "prediction_rows": int(
                    oof["fold_id"].astype(int).eq(fold_id).sum()
                ),
                "validation_unique_ids": len(validation_ids),
                "prediction_unique_ids": len(observed),
                "missing_prediction_ids": len(validation_ids - observed),
                "extra_prediction_ids": len(observed - validation_ids),
                "duplicate_validation_ids": 0,
                "duplicate_prediction_ids": 0,
                "non_finite_predictions": 0,
            }
        )
        validation_union.update(validation_ids)
    if validation_union != set(oof["stable_row_id"].astype(str)):
        raise TransferStageError(
            f"evaluate {model}: OOF union does not match fold validation IDs"
        )
    reconciliation = pd.read_csv(
        run_dir / "results/oof_reconciliation.csv"
    )
    expected_reconciliation = pd.DataFrame(reconciliation_rows)
    try:
        pd.testing.assert_frame_equal(
            reconciliation.reset_index(drop=True),
            expected_reconciliation.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise TransferStageError(
            f"evaluate {model}: incomplete OOF reconciliation"
        ) from exc

    metric_manifest_path = run_dir / "manifests/metric_manifest.json"
    metric_manifest = read_json(metric_manifest_path)
    try:
        validate_metric_provenance(metric_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferStageError(
            f"evaluate {model}: metric provenance is invalid"
        ) from exc
    metrics_path = run_dir / "results/prediction_metrics.csv"
    if (
        metric_manifest.get("model") != model
        or metric_manifest.get("configuration_hash") != config_hash
        or metric_manifest.get("metrics_hash") != sha256_file(metrics_path)
    ):
        raise TransferStageError(
            f"evaluate {model}: metric manifest hash or configuration mismatch"
        )
    saved_metrics = pd.read_csv(metrics_path).set_index("split")
    recomputed = prediction_metrics_from_saved_files(
        run_dir / "results/dev_oof_predictions.csv",
        run_dir / "results/oot_predictions.csv",
        threshold=float(metric_manifest["threshold"]),
    ).set_index("split")
    for split in ("DEV_OOF", "oot"):
        if (
            int(saved_metrics.loc[split, "row_count"])
            != int(recomputed.loc[split, "row_count"])
            or saved_metrics.loc[split, "prediction_file_hash"]
            != recomputed.loc[split, "prediction_file_hash"]
            or not np.isclose(
                float(saved_metrics.loc[split, "auc"]),
                float(recomputed.loc[split, "auc"]),
                rtol=0,
                atol=1e-12,
            )
            or not np.isclose(
                float(saved_metrics.loc[split, "ks"]),
                float(recomputed.loc[split, "ks"]),
                rtol=0,
                atol=1e-12,
            )
        ):
            raise TransferStageError(
                f"evaluate {model}: saved metrics do not match predictions"
            )


def _evaluate_comparison_row(
    *,
    run_dir: Path,
    model: str,
    budget: dict[str, int],
) -> dict[str, Any]:
    prediction_metrics = pd.read_csv(
        run_dir / "results/prediction_metrics.csv"
    )
    dev_metric = prediction_metrics[
        prediction_metrics["split"].astype(str).eq("DEV_OOF")
    ].iloc[0]
    oot_metric = prediction_metrics[
        prediction_metrics["split"].astype(str).eq("oot")
    ].iloc[0]
    selected = pd.read_csv(run_dir / "features/final_selected_features.csv")
    return {
        "dataset": EXTERNAL_DATASET,
        "source_training_dataset": SOURCE_DATASET,
        "model": model,
        "method": REVERSE_METHOD,
        "dev_auc": dev_metric["auc"],
        "oot_auc": oot_metric["auc"],
        "auc_drop": float(dev_metric["auc"]) - float(oot_metric["auc"]),
        "dev_ks": dev_metric["ks"],
        "oot_ks": oot_metric["ks"],
        "model_score_psi": oot_metric["score_psi"],
        "dev_metric_scope": dev_metric["metric_scope"],
        "oot_metric_scope": oot_metric["metric_scope"],
        "dev_prediction_hash": dev_metric["prediction_file_hash"],
        "oot_prediction_hash": oot_metric["prediction_file_hash"],
        "selected_feature_count": len(selected),
        "candidate_pool_count": budget["candidate_pool_size"],
        "semantic_group_coverage": selected["semantic_group"].nunique(),
        **_comparator_overlap(
            selected_features=set(selected["feature_name"].astype(str)),
            model=model,
        ),
        "pairing_policy_version": PAIRING_POLICY_VERSION,
    }


def _evaluate(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    resume: bool = False,
) -> dict[str, Any]:
    ranking_path = output_dir / "reverse_projection" / "homecredit_reverse_scores.csv"
    if not ranking_path.exists():
        raise TransferStageError("evaluate requires completed reverse projection")
    ranking = pd.read_csv(ranking_path)
    project = load_named_project_config("homecredit")
    prepared = None
    runs = []
    comparison_rows = []
    pool_dir = output_dir / "candidate_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_text(
        "".join(Path(path).read_text(encoding="utf-8") for path in config["_config_files"])
    )
    for model in models:
        budget = _model_budgets(config)[model]
        expected_pool = fixed_candidate_pool(
            ranking,
            model=model,
            pool_size=budget["candidate_pool_size"],
            final_budget=budget["feature_budget"],
        )
        pool_path = pool_dir / f"{model}_candidate_pool.csv"
        run_dir = output_dir / "downstream" / (
            "logistic_regression" if model == "lr" else "catboost"
        )
        if resume:
            state = _evaluate_model_resume_state(
                config=config,
                output_dir=output_dir,
                model=model,
                project=project,
                config_hash=config_hash,
                expected_pool=expected_pool,
                budget=budget,
            )
            if state == "complete":
                comparison_rows.append(
                    _evaluate_comparison_row(
                        run_dir=run_dir,
                        model=model,
                        budget=budget,
                    )
                )
                runs.append(
                    {
                        "model": model,
                        "run_dir": str(run_dir).replace("\\", "/"),
                        "resume_status": "reused_authenticated_complete",
                    }
                )
                continue
        if pool_path.exists():
            _validate_candidate_pool(
                actual_path=pool_path,
                expected=expected_pool,
                model=model,
                config_hash=config_hash,
            )
        else:
            expected_pool.to_csv(pool_path, index=False)
        experiment = ExperimentConfig(
            experiment_name=f"homecredit_{model}_reverse_transfer",
            selector_name="reverse_transfer_clip_then_mrmr",
            dataset_name="homecredit",
            model_name=model,
            model_kwargs=resolve_model_kwargs(project, model),
            data_dir=str(project["data_dir"]),
            description_path=str(project["description_path"]),
            base_output_dir=str(run_dir.parent),
            experiment_output_dir=str(run_dir),
            dev_start_day=int(project["dev_start_day"]),
            oot_start_day=int(project["oot_start_day"]),
            oot_end_day=int(project["oot_end_day"]),
            n_splits=int(project["n_splits"]),
            cv_gap_groups=int(project["cv_gap_groups"]),
            random_state=int(project["random_seed"]),
            feature_budget=budget["feature_budget"],
            excluded_feature_columns=tuple(project["excluded_feature_columns"]),
            preprocessor_kwargs=dict(project.get("preprocessor_kwargs", {})),
            selector_cls=FixedRankThenMRMRSelector,
            selector_kwargs={
                "ranking_path": str(ranking_path),
                "rank_column": "consensus_clip_rank",
                "feature_budget": budget["feature_budget"],
                "screening_pool_size": budget["candidate_pool_size"],
                "random_state": int(project["random_seed"]),
                "selector_label": "reverse_transfer_clip_then_mrmr",
            },
            experiment_type="corrected_reverse_transfer",
            config_hash=config_hash,
            data_fingerprint=build_data_version(project["data_dir"]),
            method=REVERSE_METHOD,
            source_training_dataset=SOURCE_DATASET,
            external_dataset=EXTERNAL_DATASET,
            pairing_policy_version=PAIRING_POLICY_VERSION,
            stable_row_id_column=str(config["stable_row_id_column"]),
            stability_candidate_pool_path=str(pool_path),
        )
        if prepared is None:
            prepared = prepare_modeling_data(experiment)
        run = run_experiment(experiment, prepared_data=prepared)
        _validate_completed_evaluate_model(
            config=config,
            output_dir=output_dir,
            model=model,
            project=project,
            config_hash=config_hash,
            expected_pool=expected_pool,
            budget=budget,
        )
        comparison_rows.append(
            _evaluate_comparison_row(
                run_dir=Path(run.exp_dir),
                model=model,
                budget=budget,
            )
        )
        runs.append(
            {
                "model": model,
                "run_dir": str(run.exp_dir).replace("\\", "/"),
                "resume_status": "executed_and_authenticated",
            }
        )
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(
        comparisons_dir / "reverse_transfer_metrics.csv", index=False
    )
    payload = {
        "runs": runs,
        "llm_invoked": False,
        "valid_baselines_invoked": False,
        "input_fingerprints": {
            str(project["data_dir"]).replace("\\", "/"): build_data_version(
                project["data_dir"]
            )
        },
        "raw_dev_statistical_evidence_hash": read_json(
            output_dir / "pairing" / "data_manifest.json"
        )["raw_dev_statistical_evidence_hash"],
    }
    write_json(output_dir / "manifests" / "registry_payload.json", payload)
    return payload


def _register(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    dry_run: bool = False,
) -> dict[str, Any]:
    for prerequisite in ("prepare", "train", "project", "evaluate"):
        path = output_dir / "manifests" / f"{prerequisite}_stage_manifest.json"
        if not path.exists():
            raise TransferStageError(
                f"register requires completed {prerequisite} stage"
            )
        _validate_stage_manifest(
            stage=prerequisite,
            manifest=read_json(path),
            output_dir=output_dir,
            config_hash=_configuration_hash(config),
            seeds=seeds,
            models=models,
        )
    payload_path = output_dir / "manifests" / "registry_payload.json"
    if not payload_path.exists():
        raise TransferStageError("register requires a completed evaluate registry payload")
    payload = read_json(payload_path)
    if len(payload.get("runs", [])) != len(models):
        raise TransferStageError("registry payload does not contain every requested model")
    registry_root = Path("results/research_summary")
    required = [
        registry_root / "run_index.csv",
        registry_root / "artifact_registry.csv",
        registry_root / "reusable_metrics.csv",
        registry_root / "selected_feature_registry.csv",
    ]
    if any(not path.exists() for path in required):
        raise TransferStageError("central registry files are missing")
    run_index_rows = []
    metric_rows = []
    selected_rows = []
    artifact_rows = []
    projection_manifest = read_json(
        output_dir / "reverse_projection" / "reverse_projection_manifest.json"
    )
    for run in payload["runs"]:
        run_dir = Path(run["run_dir"])
        summary_path = run_dir / "results" / "experiment_summary.csv"
        oot_metric_path = run_dir / "results" / "oot_test_results.csv"
        selected_path = run_dir / "features" / "final_selected_features.csv"
        split_manifest_path = run_dir / "data_split_manifest.json"
        if not summary_path.exists() or not oot_metric_path.exists() or not selected_path.exists():
            raise TransferStageError(f"incomplete run cannot be registered: {run_dir}")
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        oot = pd.read_csv(oot_metric_path).iloc[0].to_dict()
        selected = pd.read_csv(selected_path)
        model = str(run["model"])
        run_id = run_dir.name
        dev_prediction_path = run_dir / "results" / "dev_predictions.csv"
        dev_oof_prediction_path = run_dir / "results" / "dev_oof_predictions.csv"
        oot_prediction_path = run_dir / "results" / "oot_predictions.csv"
        for split in ("dev", "oot"):
            prediction = pd.read_csv(run_dir / "results" / f"{split}_predictions.csv")
            if prediction["stable_row_id"].duplicated().any():
                raise TransferStageError("registry refused non-unique prediction row IDs")
        prediction_metrics_path = run_dir / "results" / "prediction_metrics.csv"
        if not prediction_metrics_path.exists():
            raise TransferStageError("registry requires saved prediction metrics")
        metric_manifest_path = run_dir / "manifests" / "metric_manifest.json"
        if not metric_manifest_path.exists():
            raise TransferStageError("registry requires a metric manifest")
        try:
            validate_metric_provenance(read_json(metric_manifest_path))
        except (KeyError, TypeError, ValueError) as exc:
            raise TransferStageError(
                f"registry refused invalid metric provenance: {exc}"
            ) from exc
        prediction_metrics = pd.read_csv(prediction_metrics_path)
        recomputed_metrics = prediction_metrics_from_saved_files(
            dev_oof_prediction_path,
            oot_prediction_path,
            threshold=0.5,
        ).set_index("split")
        for row in prediction_metrics.itertuples(index=False):
            prediction_path = (
                dev_oof_prediction_path
                if str(row.split) == "DEV_OOF"
                else run_dir / "results" / f"{str(row.split).lower()}_predictions.csv"
            )
            if sha256_file(prediction_path) != str(row.prediction_file_hash):
                raise TransferStageError("prediction metric hash mismatch")
            recomputed = recomputed_metrics.loc[str(row.split)]
            if (
                int(row.row_count) != int(recomputed["row_count"])
                or not np.isclose(float(row.auc), float(recomputed["auc"]))
                or not np.isclose(float(row.ks), float(recomputed["ks"]))
                or not np.isclose(float(row.score_psi), float(recomputed["score_psi"]))
            ):
                raise TransferStageError(
                    "prediction metric values do not match saved predictions"
                )
        first_prediction = pd.read_csv(dev_prediction_path, nrows=1).iloc[0]
        budget = _model_budgets(config)[model]
        run_index_rows.append(
            {
                "run_id": run_id,
                "dataset": EXTERNAL_DATASET,
                "method": REVERSE_METHOD,
                "model": model,
                "seed": int(load_named_project_config("homecredit")["random_seed"]),
                "split": "DEV[-600,-240);OOT[-240,0]",
                "feature_budget": budget["feature_budget"],
                "configuration_hash": first_prediction["configuration_hash"],
                "data_manifest_hash": first_prediction["data_manifest_hash"],
                "metric_artifact_path": _relative(summary_path),
                "prediction_artifact_path": _relative(oot_prediction_path),
                "selected_feature_path": _relative(selected_path),
                "checkpoint_path": _relative(output_dir / "training" / "seeds"),
                "manifest_path": _relative(split_manifest_path),
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_training_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
                "source_preprocessor_hash": projection_manifest[
                    "source_preprocessor_hash"
                ],
                "source_raw_dev_statistical_evidence_hash": projection_manifest[
                    "source_raw_dev_statistical_evidence_hash"
                ],
                "source_checkpoint_hashes": json.dumps(
                    projection_manifest["source_checkpoint_hashes_by_seed"],
                    sort_keys=True,
                ),
                "source_anchor_hashes": json.dumps(
                    projection_manifest["source_anchor_hashes_by_seed"],
                    sort_keys=True,
                ),
                "depends_on_clip": True,
                "reuse_status": "newly_executed",
                "reason": "LendingClub v2-trained corrected CLIP frozen on Home Credit; stable DEV/OOT row provenance.",
            }
        )
        metric_rows.append(
            {
                "dataset_name": EXTERNAL_DATASET,
                "model": model,
                "selector": REVERSE_METHOD,
                "experiment_type": "corrected_reverse_transfer",
                "feature_budget": budget["feature_budget"],
                "llm_shared_ranking_enabled": False,
                "llm_ranking_budget": 0,
                "oot_auc": oot.get("auc"),
                "oot_gini": oot.get("gini"),
                "oot_ks": oot.get("ks"),
                "oot_log_loss": oot.get("log_loss"),
                "oot_brier": oot.get("brier"),
                "model_score_psi": oot.get("model_score_psi"),
                "selected_feature_count": len(selected),
                "total_candidate_feature_count": budget["candidate_pool_size"],
                "config_hash": first_prediction["configuration_hash"],
                "data_manifest_hash": first_prediction["data_manifest_hash"],
                "source_identity_manifest_hash": first_prediction[
                    "source_identity_manifest_hash"
                ],
                "run_id": run_id,
                "output_folder": _relative(run_dir),
                "runtime_seconds": summary.get("runtime_seconds"),
                "result_origin": "newly_executed",
                "reuse_status": "newly_executed",
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_training_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
                "source_raw_dev_statistical_evidence_hash": projection_manifest[
                    "source_raw_dev_statistical_evidence_hash"
                ],
                "metric_artifact_path": _relative(summary_path),
                "dev_prediction_hash": str(
                    prediction_metrics.set_index("split").loc[
                        "DEV_OOF", "prediction_file_hash"
                    ]
                ),
                "oot_prediction_hash": str(
                    prediction_metrics.set_index("split").loc[
                        "oot", "prediction_file_hash"
                    ]
                ),
            }
        )
        selected_rows.append(
            {
                "run_id": run_id,
                "dataset": EXTERNAL_DATASET,
                "model": model,
                "selector": REVERSE_METHOD,
                "experiment_type": "corrected_reverse_transfer",
                "feature_budget": budget["feature_budget"],
                "selected_feature_count": len(selected),
                "selected_feature_path": _relative(selected_path),
                "selected_feature_hash": sha256_file(selected_path),
                "configuration_hash": first_prediction["configuration_hash"],
                "data_manifest_hash": first_prediction["data_manifest_hash"],
                "source_identity_manifest_hash": first_prediction[
                    "source_identity_manifest_hash"
                ],
                "depends_on_clip": True,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_training_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
                "reuse_status": "newly_executed",
                "reason": "Home Credit DEV-only mRMR after frozen reverse-transfer screening.",
            }
        )
        for artifact_type, path in [
            ("metric", summary_path),
            ("metric", oot_metric_path),
            ("prediction_metric", prediction_metrics_path),
            ("prediction_dev", dev_prediction_path),
            ("prediction_dev_oof", dev_oof_prediction_path),
            ("prediction_oot", oot_prediction_path),
            ("selected_features", selected_path),
            ("manifest", split_manifest_path),
        ]:
            artifact_rows.append(
                {
                    "artifact_id": canonical_artifact_id(
                        run_id=run_id,
                        artifact_type=artifact_type,
                        relative_path=_relative(path),
                        content_hash=sha256_file(path),
                    ),
                    "artifact_type": artifact_type,
                    "relative_path": _relative(path),
                    "file_exists": path.exists(),
                    "file_hash": sha256_file(path),
                    "configuration_hash": first_prediction["configuration_hash"],
                    "data_manifest_hash": first_prediction["data_manifest_hash"],
                    "created_by_run_id": run_id,
                    "dataset": EXTERNAL_DATASET,
                    "model": model,
                    "method": REVERSE_METHOD,
                    "scientific_stage": "evaluate",
                    "depends_on_clip": True,
                    "depends_on_old_pairing": False,
                    "pairing_policy_version": PAIRING_POLICY_VERSION,
                    "source_training_dataset": SOURCE_DATASET,
                    "external_dataset": EXTERNAL_DATASET,
                    "reuse_status": "newly_executed",
                    "human_description": (
                        f"Corrected reverse-transfer {artifact_type}; source={SOURCE_DATASET}, "
                        f"external={EXTERNAL_DATASET}."
                    ),
                }
            )

    updates = [
        (registry_root / "run_index.csv", pd.DataFrame(run_index_rows), ["run_id"]),
        (
            registry_root / "artifact_registry.csv",
            pd.DataFrame(artifact_rows),
            ["artifact_id"],
        ),
        (
            registry_root / "reusable_metrics.csv",
            pd.DataFrame(metric_rows),
            ["run_id"],
        ),
        (
            registry_root / "selected_feature_registry.csv",
            pd.DataFrame(selected_rows),
            ["run_id"],
        ),
    ]
    registry_payloads: dict[Path, bytes] = {}
    proposed_frames: dict[str, pd.DataFrame] = {}
    for path, rows, keys in updates:
        current_columns = pd.read_csv(path, nrows=0).columns.tolist()
        output_columns = current_columns + [
            column for column in rows.columns if column not in current_columns
        ]
        combined = append_registry_rows(
            registry_path=path,
            rows=rows.reindex(columns=output_columns),
            equivalence_columns=keys,
        )
        proposed_frames[path.name] = combined.reindex(columns=output_columns)
        registry_payloads[path] = (
            combined.reindex(columns=output_columns).to_csv(index=False).encode("utf-8")
            if combined.attrs.get("registry_changed", True)
            else path.read_bytes()
        )
    validate_registry_bundle(
        proposed_frames,
        verify_artifacts=True,
        repository_root=Path.cwd(),
        enforced_run_ids={str(row["run_id"]) for row in run_index_rows},
    )

    guide_path = registry_root / "results_access_guide.md"
    guide = guide_path.read_text(encoding="utf-8")
    marker = "## Corrected LendingClub v2 to Home Credit Reverse Transfer"
    if marker not in guide:
        guide += (
            f"\n\n{marker}\n\n"
            "New reverse-transfer rows are source-trained on LendingClub v2 and applied "
            "frozen to Home Credit under `identity_equivalence_v2`. DEV and OOT "
            "predictions include stable row IDs, and DEV metrics are computed from "
            "persisted fold-exclusive OOF predictions. Existing baselines remain reused.\n"
        )
    registry_payloads[guide_path] = guide.encode("utf-8")
    guide_hash = hashlib.sha256(registry_payloads[guide_path]).hexdigest()
    artifact_frame = proposed_frames["artifact_registry.csv"].copy()
    guide_key = _relative(guide_path)
    guide_mask = artifact_frame["relative_path"].map(
        lambda value: canonical_registry_value(
            "relative_path", value, expected_type="path"
        )
    ).eq(guide_key)
    if guide_mask.any():
        for index in artifact_frame.index[guide_mask]:
            artifact_type = str(
                artifact_frame.at[index, "artifact_type"]
            )
            owner = str(
                artifact_frame.at[index, "created_by_run_id"]
                if "created_by_run_id" in artifact_frame
                else ""
            )
            artifact_frame.at[index, "file_hash"] = guide_hash
            artifact_frame.at[index, "artifact_id"] = canonical_artifact_id(
                run_id="" if owner.lower() == "nan" else owner,
                artifact_type=artifact_type,
                relative_path=guide_key,
                content_hash=guide_hash,
            )
        proposed_frames["artifact_registry.csv"] = artifact_frame
        artifact_path = registry_root / "artifact_registry.csv"
        registry_payloads[artifact_path] = artifact_frame.to_csv(
            index=False
        ).encode("utf-8")
        validate_registry_bundle(
            proposed_frames,
            verify_artifacts=True,
            repository_root=Path.cwd(),
            enforced_run_ids={
                str(row["run_id"]) for row in run_index_rows
            },
        )

    summary_manifest_path = registry_root / "summary_manifest.json"
    summary_manifest = read_json(summary_manifest_path)
    validate_summary_manifest(summary_manifest)
    summary_manifest["corrected_lendingclub_to_homecredit_transfer"] = {
        "status": "executed",
        "source_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "new_runs": len(run_index_rows),
        "stable_prediction_row_ids": True,
        "dev_metric_scope": "dev_oof_cross_validated",
        "raw_dev_statistical_evidence_hash": projection_manifest[
            "source_raw_dev_statistical_evidence_hash"
        ],
        "output_root": _relative(output_dir),
    }
    summary_manifest = build_summary_manifest(
        summary_manifest,
        registry_root=registry_root,
        payloads=registry_payloads,
    )
    validate_summary_manifest_payloads(
        summary_manifest,
        registry_root=registry_root,
        payloads=registry_payloads,
    )
    registry_payloads[summary_manifest_path] = json.dumps(
        summary_manifest, indent=2, ensure_ascii=False
    ).encode("utf-8")
    transaction_manifest_path = (
        output_dir / "manifests" / "registration_transaction_manifest.json"
    )
    if dry_run:
        validation = registry_bundle_dry_run(
            proposed_frames,
            verify_artifacts=True,
            repository_root=Path.cwd(),
            enforced_run_ids={
                str(row["run_id"]) for row in run_index_rows
            },
        )
        if validation["transaction_outcome"] == "CONFLICT":
            return validation
        changed = any(
            not path.exists() or path.read_bytes() != content
            for path, content in registry_payloads.items()
        )
        return {
            **validation,
            "transaction_outcome": (
                "NEW_TRANSACTION" if changed else "IDEMPOTENT_NO_OP"
            ),
            "canonical_method": REVERSE_METHOD,
            "payload_hashes": {
                _relative(path): hashlib.sha256(content).hexdigest()
                for path, content in registry_payloads.items()
            },
        }
    transaction = atomic_registry_transaction(
        registry_payloads,
        transaction_manifest_path=transaction_manifest_path,
        metadata={
            "source_dataset": SOURCE_DATASET,
            "external_dataset": EXTERNAL_DATASET,
            "pairing_policy_version": PAIRING_POLICY_VERSION,
            "configuration_hash": _configuration_hash(config),
            "data_manifest_hash": sha256_file(
                output_dir / "pairing" / "data_manifest.json"
            ),
            "raw_dev_statistical_evidence_hash": projection_manifest[
                "source_raw_dev_statistical_evidence_hash"
            ],
            "run_ids": sorted(row["run_id"] for row in run_index_rows),
            "old_invalid_rows_preserved": True,
        },
    )
    return {
        "status_detail": "schema-preserving registry append complete",
        "registry_files_updated": [
            str(path).replace("\\", "/")
            for path in [*required, guide_path, summary_manifest_path]
        ],
        "result_origin": "newly_executed",
        "registration_transaction_hash": sha256_file(transaction_manifest_path),
        "registration_transaction": transaction,
        "registered_output_hashes": transaction["updated_files"],
    }


def _roles(config: dict[str, Any]) -> DatasetRoles:
    return DatasetRoles(
        training_dataset=str(config["training_dataset"]),
        external_dataset=str(config["external_dataset"]),
        training_feature_manifest=str(config["training_feature_manifest"]),
        external_feature_manifest=str(config["external_feature_manifest"]),
        training_raw_statistical_source=str(config["training_raw_statistical_source"]),
        external_raw_statistical_source=str(config["external_raw_statistical_source"]),
        training_statistical_fit_scope=str(config["training_statistical_fit_scope"]),
        external_statistical_transform_scope=str(
            config["external_statistical_transform_scope"]
        ),
    )


def _input_paths(config: dict[str, Any]) -> dict[str, str]:
    keys = [
        "training_feature_manifest",
        "external_feature_manifest",
        "training_raw_statistical_source",
        "external_raw_statistical_source",
        "training_text_embeddings_path",
        "external_text_embeddings_path",
        "identity_evidence_path",
    ]
    return {key: str(config[key]) for key in keys}


def _model_budgets(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        "lr": {
            "candidate_pool_size": int(config["lr_candidate_pool_size"]),
            "feature_budget": int(config["lr_feature_budget"]),
        },
        "catboost": {
            "candidate_pool_size": int(config["catboost_candidate_pool_size"]),
            "feature_budget": int(config["catboost_feature_budget"]),
        },
    }


def _reindex_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    output["positive_pair_index"] = range(len(output))
    from credit_risk_fs.clip.exact_duplicates import feature_order_hash

    output["feature_order_hash"] = feature_order_hash(
        output["feature_name"].astype(str).tolist()
    )
    return output


def _relative(path: str | Path) -> str:
    value = Path(path)
    try:
        value = value.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return str(value).replace("\\", "/")


def _configuration_hash(config: dict[str, Any]) -> str:
    return sha256_text(
        "".join(
            Path(path).read_text(encoding="utf-8")
            for path in config["_config_files"]
        )
    )


def _comparator_overlap(
    *,
    selected_features: set[str],
    model: str,
) -> dict[str, int]:
    registry = Path("results/research_summary/selected_feature_registry.csv")
    if not registry.exists():
        return {"valid_comparator_selection_overlap": 0}
    rows = pd.read_csv(registry)
    rows = rows[
        rows["dataset"].astype(str).eq(EXTERNAL_DATASET)
        & rows["model"].astype(str).eq(model)
        & rows["reuse_status"].astype(str).isin({"reusable_existing", "newly_executed"})
    ]
    overlap = 0
    for path in rows["selected_feature_path"].dropna().astype(str):
        artifact = Path(path)
        if not artifact.exists():
            continue
        frame = pd.read_csv(artifact)
        column = "feature_name" if "feature_name" in frame else "feature"
        overlap = max(overlap, len(selected_features & set(frame[column].astype(str))))
    return {"valid_comparator_selection_overlap": overlap}
