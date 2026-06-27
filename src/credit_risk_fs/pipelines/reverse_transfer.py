from __future__ import annotations

import argparse
from dataclasses import replace
import json
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
    SOURCE_DATASET,
    DatasetRoles,
    aggregate_seed_embeddings,
    append_registry_rows,
    build_feature_positive_pairs,
    deterministic_feature_split,
    file_manifest,
    fixed_candidate_pool,
    frozen_project,
    implementation_contract,
    reconcile_feature_universe,
    validate_checkpoint_manifest,
    validate_frozen_external_transform,
    validate_prediction_splits,
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
    prepare_modeling_data,
    run_experiment,
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
    merged["_config_files"] = [str(path).replace("\\", "/") for path in required]
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
    if config.get("pairing_policy_version") != PAIRING_POLICY_VERSION:
        raise ValueError("active pairing policy must be identity_equivalence_v2")
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
    return {
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
    resolve_plan(
        config=config,
        stages=stages,
        seeds=seeds,
        models=models,
        output_dir=output_dir,
    )
    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": _prepare,
        "train": _train,
        "project": _project,
        "evaluate": _evaluate,
        "register": _register,
    }
    for stage in stages:
        manifest_path = output_dir / "manifests" / f"{stage}_stage_manifest.json"
        if manifest_path.exists():
            old = read_json(manifest_path)
            complete = old.get("status") == "complete"
            if complete and skip_existing:
                continue
            if complete and not resume:
                raise TransferStageError(
                    f"{stage}: completed output exists; use --skip-existing or a new output directory"
                )
        payload = handlers[stage](
            config=config,
            output_dir=output_dir,
            seeds=seeds,
            models=models,
        )
        payload.update(
            {
                "stage": stage,
                "status": "complete",
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "source_dataset": SOURCE_DATASET,
                "external_dataset": EXTERNAL_DATASET,
            }
        )
        write_json(manifest_path, payload)


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

    time_col = str(config["training_time_column"])
    target_col = str(config["training_target_column"])
    available_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    if time_col not in available_columns or target_col not in available_columns:
        raise TransferStageError("source matrix lacks declared time or target column")
    eligible_names = reconciliation.loc[
        reconciliation["eligible_for_pairing"], "feature_name"
    ].astype(str)
    feature_names = [name for name in eligible_names if name in available_columns]
    raw = pd.read_csv(data_path, usecols=[time_col, target_col, *feature_names])
    dev = raw[
        (raw[time_col] >= int(config["training_dev_start_day"]))
        & (raw[time_col] < int(config["training_oot_start_day"]))
    ].copy()
    if dev.empty:
        raise TransferStageError("approved LendingClub DEV window is empty")
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
    }


def _train(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
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
    source_anchor_manifest_path = anchor_dir / "source_anchor_manifest.json"
    if not anchor_members_path.exists() or not source_anchor_manifest_path.exists():
        raise TransferStageError(
            "train requires completed source-anchor stability selection from prepare"
        )
    source_anchor_config = load_source_anchor_config(config)
    anchor_members = pd.read_csv(anchor_members_path)
    source_anchor_manifest = read_json(source_anchor_manifest_path)
    validate_source_anchor_artifacts(
        config=source_anchor_config,
        members=anchor_members,
        manifest=source_anchor_manifest,
        training_feature_ids=set(train_pairs["feature_id"].astype(str)),
        configuration_hash=config_hash,
        data_manifest_hash=sha256_file(data_manifest_path),
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
        source_anchor_hash=sha256_file(source_anchor_manifest_path),
    )
    upstream = {
        "configuration_hash": config_hash,
        "data_manifest_hash": sha256_file(data_manifest_path),
        "statistical_preprocessor_hash": read_json(preprocessor_path)["preprocessor_hash"],
        "source_anchor_selection_manifest_hash": sha256_file(
            source_anchor_manifest_path
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
    validation_score_frames = []
    for seed in seeds:
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
    source_anchor_config = load_source_anchor_config(config)
    validate_source_anchor_artifacts(
        config=source_anchor_config,
        members=anchor_members,
        manifest=source_anchor_manifest,
        training_feature_ids=set(train_pairs["feature_id"].astype(str)),
        configuration_hash=_configuration_hash(config),
        data_manifest_hash=sha256_file(data_manifest_path),
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
    preprocessor_path = (
        output_dir / "training" / "statistical_preprocessor" / "statistical_preprocessor.joblib"
    )
    preprocessor: StatisticalPreprocessor = joblib.load(preprocessor_path)
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
    text["feature_id"] = text["feature_name"].map(manifest_ids)
    text = text.sort_values("feature_id").reset_index(drop=True)
    descriptors = descriptors.sort_values("feature_id").reset_index(drop=True)
    features = descriptors[["feature_id", "feature_name", "dataset"]]
    text_columns = sorted(
        column
        for column in text.columns
        if str(column).startswith("embedding_") and len(str(column)) == 14
    )
    seed_frames: dict[int, pd.DataFrame] = {}
    projection_dir = output_dir / "reverse_projection"
    projection_dir.mkdir(parents=True, exist_ok=True)
    external_stat_vectors.to_parquet(
        projection_dir / "homecredit_statistical_vectors.parquet", index=False
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
            statistical_values=transformed[list(DESCRIPTOR_COLUMNS_V2)].to_numpy(
                dtype=np.float32
            ),
            source_dataset=SOURCE_DATASET,
            external_dataset=EXTERNAL_DATASET,
            checkpoint_hash=checkpoint_manifest["checkpoint_sha256"],
            anchor=anchor,
            anchor_hash=anchor_manifest["anchor_hash"],
            preprocessor_hash=preprocessor.preprocessor_hash_,
        )
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
    consensus.to_parquet(
        projection_dir / "homecredit_reverse_embeddings.parquet", index=False
    )
    score_columns = [
        "feature_id",
        "feature_name",
        "dataset",
        "learned_similarity",
        "consensus_clip_rank",
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
        "external_target_used": False,
        "external_oot_used": False,
        "projected_feature_count": len(consensus),
    }
    write_json(projection_dir / "reverse_projection_manifest.json", manifest)
    return manifest


def _evaluate(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
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
        pool = fixed_candidate_pool(
            ranking,
            model=model,
            pool_size=budget["candidate_pool_size"],
            final_budget=budget["feature_budget"],
        )
        pool.to_csv(pool_dir / f"{model}_candidate_pool.csv", index=False)
        run_dir = output_dir / "downstream" / (
            "logistic_regression" if model == "lr" else "catboost"
        )
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
            method="lendingclub_clip_to_homecredit_mrmr",
            source_training_dataset=SOURCE_DATASET,
            external_dataset=EXTERNAL_DATASET,
            pairing_policy_version=PAIRING_POLICY_VERSION,
        )
        if prepared is None:
            prepared = prepare_modeling_data(experiment)
        run = run_experiment(experiment, prepared_data=prepared)
        dev_predictions = pd.read_csv(run.exp_dir / "results" / "dev_predictions.csv")
        oot_predictions = pd.read_csv(run.exp_dir / "results" / "oot_predictions.csv")
        validate_prediction_splits(dev_predictions, oot_predictions)
        summary = pd.read_csv(run.exp_dir / "results" / "experiment_summary.csv").iloc[0]
        oot = pd.read_csv(run.exp_dir / "results" / "oot_test_results.csv").iloc[0]
        selected = pd.read_csv(run.exp_dir / "features" / "final_selected_features.csv")
        comparator_overlap = _comparator_overlap(
            selected_features=set(selected["feature_name"].astype(str)),
            model=model,
        )
        comparison_rows.append(
            {
                "dataset": EXTERNAL_DATASET,
                "source_training_dataset": SOURCE_DATASET,
                "model": model,
                "method": "lendingclub_clip_to_homecredit_mrmr",
                "dev_auc": summary.get("cv_auc_mean"),
                "oot_auc": oot.get("auc"),
                "auc_drop": (
                    float(summary.get("cv_auc_mean")) - float(oot.get("auc"))
                    if pd.notna(summary.get("cv_auc_mean")) and pd.notna(oot.get("auc"))
                    else np.nan
                ),
                "dev_ks": summary.get("cv_ks_mean"),
                "oot_ks": oot.get("ks"),
                "model_score_psi": oot.get("model_score_psi"),
                "selected_feature_count": len(selected),
                "candidate_pool_count": budget["candidate_pool_size"],
                "semantic_group_coverage": selected["semantic_group"].nunique(),
                **comparator_overlap,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
            }
        )
        runs.append({"model": model, "run_dir": str(run.exp_dir).replace("\\", "/")})
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(
        comparisons_dir / "reverse_transfer_metrics.csv", index=False
    )
    payload = {"runs": runs, "llm_invoked": False, "valid_baselines_invoked": False}
    write_json(output_dir / "manifests" / "registry_payload.json", payload)
    return payload


def _register(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
) -> dict[str, Any]:
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
        oot_prediction_path = run_dir / "results" / "oot_predictions.csv"
        for split in ("dev", "oot"):
            prediction = pd.read_csv(run_dir / "results" / f"{split}_predictions.csv")
            if prediction["stable_row_id"].duplicated().any():
                raise TransferStageError("registry refused non-unique prediction row IDs")
        first_prediction = pd.read_csv(dev_prediction_path, nrows=1).iloc[0]
        budget = _model_budgets(config)[model]
        run_index_rows.append(
            {
                "run_id": run_id,
                "dataset": EXTERNAL_DATASET,
                "method": "lendingclub_clip_to_homecredit_mrmr",
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
                "depends_on_clip": True,
                "reuse_status": "newly_executed",
                "reason": "LendingClub v2-trained corrected CLIP frozen on Home Credit; stable DEV/OOT row provenance.",
            }
        )
        metric_rows.append(
            {
                "dataset_name": EXTERNAL_DATASET,
                "model": model,
                "selector": "lendingclub_clip_to_homecredit_mrmr",
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
                "run_id": run_id,
                "output_folder": _relative(run_dir),
                "runtime_seconds": summary.get("runtime_seconds"),
                "result_origin": "newly_executed",
                "reuse_status": "newly_executed",
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "metric_artifact_path": _relative(summary_path),
            }
        )
        selected_rows.append(
            {
                "run_id": run_id,
                "dataset": EXTERNAL_DATASET,
                "model": model,
                "selector": "lendingclub_clip_to_homecredit_mrmr",
                "experiment_type": "corrected_reverse_transfer",
                "feature_budget": budget["feature_budget"],
                "selected_feature_count": len(selected),
                "selected_feature_path": _relative(selected_path),
                "selected_feature_hash": sha256_file(selected_path),
                "depends_on_clip": True,
                "pairing_policy_version": PAIRING_POLICY_VERSION,
                "reuse_status": "newly_executed",
                "reason": "Home Credit DEV-only mRMR after frozen reverse-transfer screening.",
            }
        )
        for artifact_type, path in [
            ("metric", summary_path),
            ("metric", oot_metric_path),
            ("prediction_dev", dev_prediction_path),
            ("prediction_oot", oot_prediction_path),
            ("selected_features", selected_path),
            ("manifest", split_manifest_path),
        ]:
            artifact_rows.append(
                {
                    "artifact_id": sha256_text(f"{run_id}|{artifact_type}|{_relative(path)}"),
                    "artifact_type": artifact_type,
                    "relative_path": _relative(path),
                    "file_exists": path.exists(),
                    "file_hash": sha256_file(path),
                    "created_by_run_id": run_id,
                    "depends_on_clip": True,
                    "depends_on_old_pairing": False,
                    "pairing_policy_version": PAIRING_POLICY_VERSION,
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
    for path, rows, keys in updates:
        current_columns = pd.read_csv(path, nrows=0).columns.tolist()
        combined = append_registry_rows(
            registry_path=path,
            rows=rows.reindex(columns=current_columns),
            equivalence_columns=keys,
        )
        combined.reindex(columns=current_columns).to_csv(path, index=False)

    guide_path = registry_root / "results_access_guide.md"
    guide = guide_path.read_text(encoding="utf-8")
    marker = "## Corrected LendingClub v2 to Home Credit Reverse Transfer"
    if marker not in guide:
        guide += (
            f"\n\n{marker}\n\n"
            "New reverse-transfer rows are source-trained on LendingClub v2 and applied "
            "frozen to Home Credit under `identity_equivalence_v2`. DEV and OOT "
            "predictions include stable row IDs. Existing baselines remain reused.\n"
        )
        guide_path.write_text(guide, encoding="utf-8")

    summary_manifest_path = registry_root / "summary_manifest.json"
    summary_manifest = read_json(summary_manifest_path)
    summary_manifest["corrected_lendingclub_to_homecredit_transfer"] = {
        "status": "executed",
        "source_dataset": SOURCE_DATASET,
        "external_dataset": EXTERNAL_DATASET,
        "pairing_policy_version": PAIRING_POLICY_VERSION,
        "new_runs": len(run_index_rows),
        "stable_prediction_row_ids": True,
        "output_root": _relative(output_dir),
    }
    summary_manifest["registry_file_hashes"] = {
        _relative(path): sha256_file(path)
        for path in [*required, guide_path]
    }
    write_json(summary_manifest_path, summary_manifest)
    return {
        "status_detail": "schema-preserving registry append complete",
        "registry_files_updated": [
            str(path).replace("\\", "/")
            for path in [*required, guide_path, summary_manifest_path]
        ],
        "result_origin": "newly_executed",
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
