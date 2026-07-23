from __future__ import annotations

import argparse
import copy
import json
import logging
import tempfile
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from credit_risk_fs.experiments._common import build_experiment_config
from credit_risk_fs.experiments.atomic_io import sha256_file, write_csv_atomic, write_json_atomic
from credit_risk_fs.experiments.config import (
    DEFAULT_CONFIG_PATH,
    apply_feature_budget_to_selector_kwargs,
    compute_config_hash,
    load_project_config,
    normalize_llm_ranking_budget,
    resolve_feature_budget,
    resolve_llm_candidate_pool_budget,
    resolve_llm_shared_pool_size,
)
from credit_risk_fs.experiments.matrix import (
    MODELS,
    MatrixRunSpec,
    cross_dataset_matrix_expansion_summary,
    expand_lendingclub_memory_capacity_scenarios,
    expand_cross_dataset_voting_matrix,
    expand_cross_dataset_voting_pilot,
    iter_matrix,
    validate_matrix,
)
from credit_risk_fs.experiments.execution import RegisteredRunRequest, execute_registered_run
from credit_risk_fs.experiments.checkpointing import resolve_resume_target
from credit_risk_fs.experiments.resource_policy import (
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.experiments.resource_monitor import wait_for_inter_run_readiness
from credit_risk_fs.experiments.result_paths import (
    build_run_id,
    create_run_directory,
    initialize_results_layout,
    planned_run_directory,
    sanitize_component,
)
from credit_risk_fs.experiments.tracking import write_run_manifest
from credit_risk_fs.pipelines.common import ExperimentConfig
from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector
from credit_risk_fs.selectors.registry import get_selector
from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector
from credit_risk_fs.utils.logging import setup_logging


logger = setup_logging("experiment_matrix", level=logging.INFO)


RESEARCH_PROTOCOL_HASH = "51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49"
RESEARCH_ROW_CONTRACT_HASH = "fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0"
RESEARCH_SCIENTIFIC_PROTOCOL_HASH = "f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0"
RESEARCH_POLICY_HASH = "1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012"
RESEARCH_REFINEMENT_HASH = "4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7"


LLM_SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "selector",
    "experiment_type",
    "status",
    "llm_shared_ranking_enabled",
    "llm_ranking_budget",
    "llm_calls_actually_made",
    "llm_cache_hits",
    "llm_cache_key",
    "llm_metadata_signatures",
    "llm_cache_key_hashes",
    "llm_prompt_versions",
    "llm_prompt_hashes",
    "llm_request_models",
    "llm_response_models",
    "llm_response_ids",
    "llm_cache_file_names",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "runs_sharing_metadata_signatures",
    "runs_sharing_cache_key_hashes",
    "output_folder",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full research matrix: LR/CatBoost x statistical/LLM/hybrid selectors."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory. Defaults to results_dir from config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help="Optional subset of matrix models. Default runs the full model matrix.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the config random_seed for every matrix entry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compatibility flag; every execution creates a new isolated run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix entries and output folders without training.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Explicit repository root used to resolve configured result paths.",
    )
    parser.add_argument(
        "--execution-policy",
        default="configs/execution/local_laptop_safe_v1.yaml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--allow-gpu-without-telemetry", action="store_true")
    parser.add_argument(
        "--cross-dataset-voting-matrix-dry-expand",
        type=Path,
        default=None,
        help="Purely validate and print the frozen 16-run voting matrix; performs no writes.",
    )
    parser.add_argument(
        "--voting-pilot-config",
        type=Path,
        default=None,
        help="Execute the four authorized sequential voting integration pilots.",
    )
    parser.add_argument(
        "--lendingclub-memory-refinement-config",
        type=Path,
        default=None,
        help="Execute one authorized Prompt-5 memory-capacity scenario.",
    )
    parser.add_argument(
        "--capacity-scenario-id",
        default=None,
        help="Exact authorized capacity scenario ID; required with the refinement config.",
    )
    return parser


def _matrix_config_for_spec(project_config: dict[str, Any], spec: MatrixRunSpec) -> dict[str, Any]:
    config = copy.deepcopy(project_config)
    config["model_selector"] = spec.model
    config["matrix_run"] = {
        "model": spec.model,
        "selector": spec.selector,
        "experiment_type": spec.experiment_type,
        "experiment_name": spec.experiment_name,
    }
    return config


def _args_for_config(config: dict[str, Any], model: str) -> SimpleNamespace:
    llm_config = config.get("llm", {})
    llm_ranking_budget = normalize_llm_ranking_budget(llm_config.get("ranking_budget"))
    return SimpleNamespace(
        project_config=config,
        model=model,
        data_dir=config["data_dir"],
        description_path=config["description_path"],
        n_splits=int(config["n_splits"]),
        dev_start_day=int(config["dev_start_day"]),
        oot_start_day=int(config["oot_start_day"]),
        oot_end_day=int(config["oot_end_day"]),
        cv_gap_groups=int(config["cv_gap_groups"]),
        random_seed=int(config["random_seed"]),
        llm_model=llm_config.get("model", "gpt-4.1-mini"),
        llm_max_features=resolve_llm_shared_pool_size(llm_config),
        llm_shared_pool_size=resolve_llm_shared_pool_size(llm_config),
        llm_ranking_budget=resolve_llm_shared_pool_size(llm_config),
        llm_ranking_budget_config=llm_ranking_budget,
        llm_prompt_version=llm_config.get("prompt_version", "stability_expert_v3"),
        llm_shared_ranking_enabled=bool(llm_config.get("shared_ranking_enabled", True)),
        llm_cache_dir=llm_config.get("cache_dir", "results/_llm_rankings_cache"),
    )


def _run_dir_for_spec(
    *,
    output_root: Path,
    dataset: str,
    spec: MatrixRunSpec,
) -> tuple[str, Path]:
    run_id = build_run_id(
        model=spec.model,
        selector=spec.experiment_name,
    )
    return run_id, planned_run_directory(
        output_root,
        dataset=dataset,
        run_id=run_id,
    )


def _hybrid_selector_kwargs(
    *,
    spec: MatrixRunSpec,
    args: SimpleNamespace,
) -> dict[str, Any]:
    llm_config = args.project_config.get("llm", {})
    stat_selector_cls, stat_selector_kwargs = get_selector(spec.selector)
    if stat_selector_cls is None:
        raise ValueError(f"Unsupported hybrid downstream selector: {spec.selector}")

    feature_budget = resolve_feature_budget(args.project_config, spec.model)
    stat_selector_kwargs = apply_feature_budget_to_selector_kwargs(
        spec.selector,
        stat_selector_kwargs,
        feature_budget,
    )
    llm_cache_dir = Path(args.llm_cache_dir)
    return {
        "description_csv_path": args.description_path,
        "stat_selector_cls": stat_selector_cls,
        "stat_selector_kwargs": stat_selector_kwargs,
        "cache_dir": str(llm_cache_dir),
        "llm_model": args.llm_model,
        "llm_max_features": args.llm_shared_pool_size,
        "llm_candidate_pool_budget": resolve_llm_candidate_pool_budget(llm_config, spec.model),
        "llm_shared_ranking_enabled": args.llm_shared_ranking_enabled,
        "llm_config_hash": args.project_config.get("llm_ranking_config_hash"),
        "llm_prompt_version": args.llm_prompt_version,
        "llm_ranking_budget_config": args.llm_ranking_budget_config,
        "llm_shared_pool_size": args.llm_shared_pool_size,
        "final_feature_budget": feature_budget,
        "llm_selector_kwargs": {
            "max_missing_rate": 0.95,
            "lr_feature_budget": int(args.project_config.get("feature_budgets", {}).get("lr", 20)),
            "catboost_feature_budget": int(args.project_config.get("feature_budgets", {}).get("catboost", 40)),
            "lr_candidate_pool_budget": int(llm_config.get("ranking_budget", {}).get("lr_candidate_pool", 60))
            if isinstance(llm_config.get("ranking_budget"), dict)
            else 60,
            "catboost_candidate_pool_budget": int(llm_config.get("ranking_budget", {}).get("catboost_candidate_pool", 100))
            if isinstance(llm_config.get("ranking_budget"), dict)
            else int(args.llm_shared_pool_size),
        },
        "iv_filter_kwargs": {
            "min_iv": 0.01,
            "max_iv_for_leakage": 0.5,
            "encode": True,
            "n_jobs": 1,
            "verbose": False,
        },
    }


def _experiment_config_for_spec(
    *,
    spec: MatrixRunSpec,
    run_config: dict[str, Any],
    run_dir: Path,
) -> ExperimentConfig:
    args = _args_for_config(run_config, spec.model)

    selector_cls = None
    selector_kwargs: dict[str, Any] = {}
    if spec.experiment_type == "llm":
        llm_cache_dir = Path(args.llm_cache_dir)
        feature_budget = resolve_feature_budget(run_config, spec.model)
        selector_kwargs = {
            "model": args.llm_model,
            "max_features": args.llm_shared_pool_size,
            "ranking_budget": args.llm_shared_pool_size,
            "feature_budget": feature_budget,
            "shared_ranking_enabled": args.llm_shared_ranking_enabled,
            "config_hash": run_config.get("llm_ranking_config_hash"),
            "prompt_version": args.llm_prompt_version,
            "ranking_budget_config": args.llm_ranking_budget_config,
            "shared_pool_size": args.llm_shared_pool_size,
            "lr_feature_budget": int(run_config.get("feature_budgets", {}).get("lr", 20)),
            "catboost_feature_budget": int(run_config.get("feature_budgets", {}).get("catboost", 40)),
            "lr_candidate_pool_budget": int(
                resolve_llm_candidate_pool_budget(run_config.get("llm", {}), "lr")
            ),
            "catboost_candidate_pool_budget": int(
                resolve_llm_candidate_pool_budget(run_config.get("llm", {}), "catboost")
            ),
            "cache_dir": str(llm_cache_dir),
        }
    elif spec.experiment_type == "hybrid":
        if spec.selector == "stable_core_llm_fill":
            selector_cls = StableCoreLLMFillSelector
            selector_kwargs = {
                "description_csv_path": args.description_path,
                "cache_dir": str(Path(args.llm_cache_dir)),
                "llm_model": args.llm_model,
                "llm_max_features": args.llm_shared_pool_size,
                "llm_shared_ranking_enabled": args.llm_shared_ranking_enabled,
                "llm_config_hash": args.project_config.get("llm_ranking_config_hash"),
                "llm_prompt_version": args.llm_prompt_version,
                "llm_ranking_budget_config": args.llm_ranking_budget_config,
                "llm_shared_pool_size": args.llm_shared_pool_size,
                "final_feature_budget": resolve_feature_budget(run_config, spec.model),
                "random_state": int(run_config.get("random_seed", 42)),
                "llm_selector_kwargs": {
                    "max_missing_rate": 0.95,
                    "lr_feature_budget": int(run_config.get("feature_budgets", {}).get("lr", 20)),
                    "catboost_feature_budget": int(run_config.get("feature_budgets", {}).get("catboost", 40)),
                    "lr_candidate_pool_budget": int(
                        resolve_llm_candidate_pool_budget(run_config.get("llm", {}), "lr")
                    ),
                    "catboost_candidate_pool_budget": int(
                        resolve_llm_candidate_pool_budget(run_config.get("llm", {}), "catboost")
                    ),
                },
                "iv_filter_kwargs": {
                    "min_iv": 0.01,
                    "max_iv_for_leakage": 0.5,
                    "encode": True,
                    "n_jobs": 1,
                    "verbose": False,
                },
            }
        else:
            selector_cls = LLMThenStatSelector
            selector_kwargs = _hybrid_selector_kwargs(spec=spec, args=args)

    return build_experiment_config(
        args=args,
        experiments_dir=run_dir,
        experiment_name=spec.experiment_name,
        selector_name=spec.selector_name,
        selector_cls=selector_cls,
        selector_kwargs=selector_kwargs,
        experiment_output_dir=run_dir,
    )


def _prepare_data_config(project_config: dict[str, Any], model: str, output_root: Path) -> ExperimentConfig:
    args = _args_for_config(project_config, model)
    return build_experiment_config(
        args=args,
        experiments_dir=output_root / "_data_prep",
        experiment_name="data_prep",
        selector_name="none",
    )


def _write_matrix_status(
    output_root: Path,
    dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if rows:
        write_csv_atomic(
            output_root / "comparisons" / f"{dataset}_matrix_runs.csv",
            pd.DataFrame(rows),
        )


def _unique_sorted_strings(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = values.split(";")
    elif isinstance(values, (list, tuple, set)):
        parts = list(values)
    else:
        parts = [values]
    normalized = {
        str(value).strip()
        for value in parts
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return sorted(normalized)


def _join_unique_strings(values: object) -> str:
    return ";".join(_unique_sorted_strings(values))


def _llm_ranking_stats(run_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    summary_path = run_dir / "features" / "llm_rankings_summary.csv"
    defaults = {
        "llm_cache_key": None,
        "llm_metadata_signatures": [],
        "llm_cache_key_hashes": [],
        "llm_prompt_versions": [],
        "llm_prompt_hashes": [],
        "llm_request_models": [],
        "llm_response_models": [],
        "llm_response_ids": [],
        "llm_cache_file_names": [],
        "llm_calls_actually_made": 0,
        "llm_cache_hits": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
    }
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            stats = defaults.copy()
            list_mappings = {
                "llm_metadata_signatures": "metadata_signature",
                "llm_cache_key_hashes": "cache_key_hash",
                "llm_prompt_versions": "prompt_version",
                "llm_prompt_hashes": "prompt_hash",
                "llm_request_models": "request_model",
                "llm_response_models": "response_model",
                "llm_response_ids": "response_id",
                "llm_cache_file_names": "cache_file_name",
            }
            for target_key, column in list_mappings.items():
                if column in df.columns:
                    stats[target_key] = _unique_sorted_strings(df[column].dropna().tolist())
            metadata_signatures = stats["llm_metadata_signatures"]
            stats["llm_cache_key"] = metadata_signatures[0] if metadata_signatures else None
            if "cache_hit" in df.columns:
                scope_keys = [
                    "scope",
                    "fold_id",
                    "metadata_signature",
                    "cache_key_hash",
                    "cache_file_name",
                    "response_id",
                ]
                available_keys = [key for key in scope_keys if key in df.columns]
                call_df = df.drop_duplicates(subset=available_keys) if available_keys else df
                cache_flags = call_df["cache_hit"].astype(str).str.lower().isin(["true", "1"])
                stats["llm_cache_hits"] = int(cache_flags.sum())
                stats["llm_calls_actually_made"] = int((~cache_flags).sum())
                actual_call_df = call_df.loc[~cache_flags]
            else:
                actual_call_df = df
            for column, target in [
                ("prompt_tokens", "prompt"),
                ("completion_tokens", "completion"),
                ("total_tokens", "total"),
            ]:
                if column not in actual_call_df.columns:
                    continue
                value = pd.to_numeric(actual_call_df[column], errors="coerce").fillna(0).sum()
                if target == "prompt":
                    stats["llm_prompt_tokens"] = int(value)
                elif target == "completion":
                    stats["llm_completion_tokens"] = int(value)
                else:
                    stats["llm_total_tokens"] = int(value)
            return stats
    manifest = manifest or {}
    return {
        **defaults,
        "llm_cache_key": manifest.get("llm_cache_key"),
        "llm_metadata_signatures": _unique_sorted_strings(manifest.get("llm_metadata_signatures")),
        "llm_cache_key_hashes": _unique_sorted_strings(manifest.get("llm_cache_key_hashes")),
        "llm_prompt_versions": _unique_sorted_strings(manifest.get("llm_prompt_versions")),
        "llm_prompt_hashes": _unique_sorted_strings(manifest.get("llm_prompt_hashes")),
        "llm_request_models": _unique_sorted_strings(manifest.get("llm_request_models")),
        "llm_response_models": _unique_sorted_strings(manifest.get("llm_response_models")),
        "llm_response_ids": _unique_sorted_strings(manifest.get("llm_response_ids")),
        "llm_cache_file_names": _unique_sorted_strings(manifest.get("llm_cache_file_names")),
        "llm_calls_actually_made": int(manifest.get("llm_calls_actually_made", 0) or 0),
        "llm_cache_hits": int(manifest.get("llm_cache_hits", 0) or 0),
        "llm_prompt_tokens": int(manifest.get("llm_prompt_tokens", 0) or 0),
        "llm_completion_tokens": int(manifest.get("llm_completion_tokens", 0) or 0),
        "llm_total_tokens": int(manifest.get("llm_total_tokens", 0) or 0),
    }


def _allowed_run_dirs_from_rows(output_root: Path, matrix_rows: list[dict[str, Any]] | None) -> set[Path] | None:
    if matrix_rows is None:
        return None
    return {
        (Path.cwd() / str(row["output_folder"])).resolve()
        if not Path(str(row["output_folder"])).is_absolute()
        else Path(str(row["output_folder"])).resolve()
        for row in matrix_rows
        if row.get("output_folder")
    }


def _write_llm_call_summary(
    output_root: Path,
    dataset: str,
    matrix_rows: list[dict[str, Any]] | None = None,
) -> None:
    records = []
    allowed_dirs = _allowed_run_dirs_from_rows(output_root, matrix_rows)
    for manifest_path in sorted(output_root.rglob("run_manifest.json")):
        if manifest_path.parent == output_root:
            continue
        if allowed_dirs is not None and manifest_path.parent.resolve() not in allowed_dirs:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats = _llm_ranking_stats(manifest_path.parent, manifest)
        records.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "llm_shared_ranking_enabled": manifest.get("llm_shared_ranking_enabled"),
                "llm_ranking_budget": manifest.get("llm_ranking_budget"),
                "llm_calls_actually_made": stats["llm_calls_actually_made"],
                "llm_cache_hits": stats["llm_cache_hits"],
                "llm_cache_key": stats["llm_cache_key"],
                "llm_metadata_signatures": _join_unique_strings(stats["llm_metadata_signatures"]),
                "llm_cache_key_hashes": _join_unique_strings(stats["llm_cache_key_hashes"]),
                "llm_prompt_versions": _join_unique_strings(stats["llm_prompt_versions"]),
                "llm_prompt_hashes": _join_unique_strings(stats["llm_prompt_hashes"]),
                "llm_request_models": _join_unique_strings(stats["llm_request_models"]),
                "llm_response_models": _join_unique_strings(stats["llm_response_models"]),
                "llm_response_ids": _join_unique_strings(stats["llm_response_ids"]),
                "llm_cache_file_names": _join_unique_strings(stats["llm_cache_file_names"]),
                "llm_prompt_tokens": stats["llm_prompt_tokens"],
                "llm_completion_tokens": stats["llm_completion_tokens"],
                "llm_total_tokens": stats["llm_total_tokens"],
                "output_folder": str(manifest_path.parent),
            }
        )
    signature_to_runs: dict[str, list[str]] = {}
    cache_key_hash_to_runs: dict[str, list[str]] = {}
    for row in records:
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            if signature:
                signature_to_runs.setdefault(signature, []).append(str(row["run_id"]))
        for cache_key_hash in str(row.get("llm_cache_key_hashes") or "").split(";"):
            if cache_key_hash:
                cache_key_hash_to_runs.setdefault(cache_key_hash, []).append(str(row["run_id"]))
    for row in records:
        sharing = set()
        sharing_by_hash = set()
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            sharing.update(signature_to_runs.get(signature, []))
        for cache_key_hash in str(row.get("llm_cache_key_hashes") or "").split(";"):
            sharing_by_hash.update(cache_key_hash_to_runs.get(cache_key_hash, []))
        row["runs_sharing_metadata_signatures"] = ";".join(sorted(sharing))
        row["runs_sharing_cache_key_hashes"] = ";".join(sorted(sharing_by_hash))
    write_csv_atomic(
        output_root / "comparisons" / f"{dataset}_llm_call_summary.csv",
        pd.DataFrame(records, columns=LLM_SUMMARY_COLUMNS),
    )


def _write_failed_runs(
    output_root: Path,
    dataset: str,
    matrix_rows: list[dict[str, Any]] | None = None,
) -> None:
    columns = [
        "run_id",
        "model",
        "selector",
        "experiment_type",
        "status",
        "error",
        "failed_at",
        "output_folder",
    ]
    rows = []
    allowed_dirs = _allowed_run_dirs_from_rows(output_root, matrix_rows)
    for manifest_path in sorted(output_root.rglob("run_manifest.json")):
        if manifest_path.parent == output_root:
            continue
        if allowed_dirs is not None and manifest_path.parent.resolve() not in allowed_dirs:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "failed":
            continue
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "error": manifest.get("error"),
                "failed_at": manifest.get("failed_at"),
                "output_folder": str(manifest_path.parent),
            }
        )
    write_csv_atomic(
        output_root / "comparisons" / f"{dataset}_failed_runs.csv",
        pd.DataFrame(rows, columns=columns),
    )


def dry_expand_cross_dataset_voting_matrix(path: str | Path) -> dict[str, Any]:
    """Return the validated frozen expansion without initializing result paths."""

    specs = expand_cross_dataset_voting_matrix(path)
    return cross_dataset_matrix_expansion_summary(specs)


def _verify_pilot_frozen_inputs(repository_root: Path, pilot_payload: dict[str, Any]) -> None:
    for section in ("protocol", "row_contract", "scientific_protocol", "execution_policy"):
        values = pilot_payload.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"pilot configuration section is missing: {section}")
        path = repository_root / str(values.get("path", ""))
        expected = str(values.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen pilot input hash mismatch: {section}")


def run_cross_dataset_voting_pilots(
    *,
    repository_root: str | Path,
    pilot_config_path: str | Path,
    resume_run_id: str | None = None,
) -> list[Any]:
    """Execute the four authorized jobs sequentially through the common lifecycle."""

    import yaml

    root = Path(repository_root).resolve()
    pilot_path = Path(pilot_config_path)
    if not pilot_path.is_absolute():
        pilot_path = (root / pilot_path).resolve()
    specs = expand_cross_dataset_voting_pilot(pilot_path)
    if resume_run_id is not None:
        matching = [index for index, item in enumerate(specs) if item.run_id == resume_run_id]
        if len(matching) != 1:
            raise ValueError("pilot resume must name one exact authorized pilot ID")
        specs = specs[matching[0] :]
    payload = yaml.safe_load(pilot_path.read_text(encoding="utf-8"))
    _verify_pilot_frozen_inputs(root, payload)
    output_root = initialize_results_layout(root, results_root="results")
    policy_path = str(payload["execution_policy"]["path"])
    preflight_specs = json.loads(
        (
            root
            / "cleanup/audits/cross_dataset_voting_execution_spec/preflight_request_specs.json"
        ).read_text(encoding="utf-8")
    )
    disk_requirements = {
        (item["dataset"], item["model"]): int(item["required_free_disk_bytes"])
        for item in preflight_specs["execution_shapes"]
    }
    artifact_applicability = {
        "config": True,
        "manifest": True,
        "selected_features": True,
        "fold_selections": True,
        "metrics": False,
        "predictions_dev": True,
        "predictions_oot": False,
        "stability": False,
        "resource_usage": True,
        "preflight": True,
        "checkpoint": True,
        "run_log": True,
    }
    outcomes = []
    for spec in specs:
        resume_this_run = resume_run_id == spec.run_id
        active_locks = sorted(output_root.glob("runs/*/*/.execution.lock"))
        if active_locks:
            raise RuntimeError(
                f"another registered experiment is active before {spec.run_id}: {active_locks}"
            )
        temp_root = Path(tempfile.gettempdir()).resolve()
        configured_policy = load_execution_policy(root, policy_path)
        capacity = detect_hardware(output_root, temp_root)
        resolved_policy = resolve_execution_policy(configured_policy, capacity)
        parallel = resolved_policy.to_dict()["parallelism"]
        if parallel != {
            "concurrent_experiment_runs": 1,
            "concurrent_folds": 1,
            "data_loader_workers": 0,
            "estimator_threads": 4,
            "allow_nested_parallelism": False,
        }:
            raise ValueError(f"resolved pilot parallelism widened or drifted: {parallel}")
        preflight = run_preflight(
            repository_root=root,
            config_path=policy_path,
            results_root="results",
            temp_root=temp_root,
            requested_accelerator="cpu",
            allow_gpu_without_telemetry=False,
            requested_run_directory=(
                resolve_resume_target(output_root, spec.run_id)
                if resume_this_run
                else None
            ),
            capacity=capacity,
        )
        required_bytes = disk_requirements[(spec.dataset, spec.model)]
        available_bytes = int(capacity.results_free_disk_gb * 1024**3)
        temp_available_bytes = int(capacity.temp_free_disk_gb * 1024**3)
        disk_check = {
            "name": "prompt3_atomic_write_safety_accounting",
            "passed": available_bytes >= required_bytes and temp_available_bytes >= required_bytes,
            "blocking": True,
            "detail": (
                f"required={required_bytes}, results_available={available_bytes}, "
                f"temp_available={temp_available_bytes}, safety_factor=2.5"
            ),
        }
        preflight["checks"].append(disk_check)
        if not disk_check["passed"]:
            preflight["blocking_reasons"].append(disk_check["name"])
            preflight["status"] = "fail"
        preflight["pilot_job"] = {
            "run_id": spec.run_id,
            "dataset": spec.dataset,
            "model": spec.model,
            "candidate_pool_budget": 200,
            "fold_id": 1,
            "seed": 42,
            "load_oot": False,
            "requested_accelerator": "cpu",
        }
        if preflight["status"] != "pass":
            raise RuntimeError(
                f"preflight rejected pilot {spec.run_id}: {preflight['blocking_reasons']}"
            )
        run_dir = (
            resolve_resume_target(output_root, spec.run_id)
            if resume_this_run
            else create_run_directory(
                output_root,
                dataset=spec.dataset,
                run_id=spec.run_id,
                collision_policy="error",
            )
        )
        dataset_config_path = root / f"configs/experiments/{spec.dataset}_matrix.yaml"
        dataset_config = load_project_config(dataset_config_path)
        data_dir = Path(str(dataset_config["data_dir"]))
        if not data_dir.is_absolute():
            data_dir = (root / data_dir).resolve()
        experiment_config = ExperimentConfig(
            experiment_name=spec.run_id,
            selector_name="rank_voting_v1",
            dataset_name=spec.dataset,
            model_name=spec.model,
            data_dir=str(data_dir),
            target="TARGET",
            time_col="recent_decision",
            random_state=spec.seed,
            feature_budget=spec.final_feature_budget,
            estimator_threads=resolved_policy.parallelism.estimator_threads,
            stable_row_id_column=("SK_ID_CURR" if spec.dataset == "homecredit" else "loan_id"),
        )
        effective_config = {
            "schema_version": "cross_dataset_rank_voting_pilot_run_v1",
            "pilot_config_path": str(pilot_path.relative_to(root)).replace("\\", "/"),
            "pilot_config_sha256": sha256_file(pilot_path),
            "run_id": spec.run_id,
            "purpose": "integration_resource_pilot",
            "research_eligible": False,
            "comparison_eligible": False,
            "dataset": spec.dataset,
            "model": spec.model,
            "method": "rank_voting_v1",
            "candidate_pool_budget": 200,
            "feature_budgets": {spec.model: spec.final_feature_budget},
            "seed": 42,
            "fold_count": 1,
            "canonical_fold_id": 1,
            "pilot_id_fold_suffix": "f0",
            "fold_protocol": "grouped_time_series_cv_5_splits_gap_1_expanding",
            "load_oot": False,
            "final_refit": False,
            "reference_execution": False,
            "sensitivity_execution": False,
            "accelerator": "cpu",
            "resolved_execution_policy": resolved_policy.to_dict(),
            "frozen_hashes": {
                section: payload[section]["sha256"]
                for section in (
                    "protocol",
                    "row_contract",
                    "scientific_protocol",
                    "execution_policy",
                )
            },
        }
        outcome = execute_registered_run(
            RegisteredRunRequest(
                repository_root=root,
                results_root=output_root,
                run_directory=run_dir,
                dataset=spec.dataset,
                selector="rank_voting_v1",
                model=spec.model,
                experiment_type="integration_resource_pilot",
                split_protocol="grouped_time_series_cv_5_splits_gap_1_first_fold_only",
                seed=spec.seed,
                effective_config=effective_config,
                experiment_config=experiment_config,
                preflight_report=preflight,
                resolved_policy=resolved_policy,
                resume=resume_this_run,
                worker_target="credit_risk_fs.experiments.rank_voting:voting_pilot_worker",
                worker_kwargs={
                    "repository_root": str(root),
                    "dataset": spec.dataset,
                    "model_name": spec.model,
                    "candidate_pool_budget": spec.candidate_pool_budget,
                    "seed": spec.seed,
                    "estimator_threads": resolved_policy.parallelism.estimator_threads,
                    "protocol_sha256": payload["protocol"]["sha256"],
                },
                merge_default_worker_kwargs=True,
                manifest_metadata={
                    "purpose": "integration_resource_pilot",
                    "research_eligible": False,
                    "comparison_eligible": False,
                    "coverage_type": "single_dev_fold_pilot",
                    "candidate_pool_budget": 200,
                    "final_feature_budget": spec.final_feature_budget,
                    "canonical_fold_id": 1,
                    "pilot_id_fold_suffix": "f0",
                    "load_oot": False,
                    "final_refit": False,
                },
                artifact_applicability=artifact_applicability,
                protocol_path=payload["protocol"]["path"],
                row_contract_path=payload["row_contract"]["path"],
            )
        )
        outcomes.append(outcome)
        resume_run_id = None
        if outcome.status != "completed":
            break
    return outcomes


def _capacity_projection_from_parent(
    *,
    parent_run: Path,
    parent_training_rows: int,
    parent_boundary_rows: int,
    parent_k: int,
    target_training_rows: int,
    target_boundary_rows: int,
    target_k: int,
    full_dev_rows: int,
    candidate_universe: int,
    safety_factor: float,
    fixed_uncertainty_gib: float,
) -> dict[str, Any]:
    resource_paths = sorted(
        (parent_run / "incomplete/attempt_history").glob("*resource_usage.json")
    ) + [parent_run / "resource_usage.json"]
    resources = [
        json.loads(path.read_text(encoding="utf-8")) for path in resource_paths
    ]
    ownership = pd.read_csv(parent_run / "memory_ownership_trace.csv")
    source_rows = ownership.loc[
        ownership["object_name"].eq("candidate_source_frame"), "logical_bytes"
    ]
    if source_rows.empty:
        data_access = json.loads(
            (parent_run / "data_access_log.json").read_text(encoding="utf-8")
        )
        source_bytes = int(
            data_access["load_report"]["application_train"]["dtype_bytes"]
        )
        source_logical_evidence = "data_access_log.load_report.application_train.dtype_bytes"
    else:
        source_bytes = int(source_rows.max())
        source_logical_evidence = "memory_ownership_trace.candidate_source_frame"
    raw_bytes_per_cell = source_bytes / (full_dev_rows * candidate_universe)
    numeric_bytes_per_cell = 4.0
    delta_training_rows = max(0, target_training_rows - parent_training_rows)
    projection_delta = delta_training_rows * candidate_universe * raw_bytes_per_cell
    voter_delta = delta_training_rows * candidate_universe * numeric_bytes_per_cell * 3.0
    parent_top_cells = parent_boundary_rows * parent_k
    target_top_cells = target_boundary_rows * target_k
    top_boundary_delta = max(0, target_top_cells - parent_top_cells) * (
        raw_bytes_per_cell + numeric_bytes_per_cell
    )
    top_source_delta = full_dev_rows * max(0, target_k - parent_k) * raw_bytes_per_cell
    top_k_delta = top_boundary_delta + top_source_delta
    observed_peak = max(int(item["peak_process_tree_rss_bytes"]) for item in resources)
    uncertainty = fixed_uncertainty_gib * 1024**3
    raw_estimate = observed_peak + max(projection_delta, voter_delta, top_k_delta) + uncertainty
    safety_estimate = raw_estimate * safety_factor
    return {
        "schema_version": "lendingclub_refined_capacity_projection_v1",
        "parent_run": str(parent_run),
        "parent_attempt_count": len(resources),
        "parent_cumulative_wall_seconds": sum(
            float(
                item.get("total_runtime_seconds")
                or item.get("timings_seconds", {}).get("total", 0)
            )
            for item in resources
        ),
        "parent_observed_peak_process_tree_rss_bytes": observed_peak,
        "parent_minimum_system_available_ram_bytes": int(
            min(item["minimum_system_available_ram_bytes"] for item in resources)
        ),
        "parent_training_rows": parent_training_rows,
        "target_training_rows": target_training_rows,
        "parent_boundary_rows": parent_boundary_rows,
        "target_boundary_rows": target_boundary_rows,
        "parent_candidate_pool": parent_k,
        "target_candidate_pool": target_k,
        "candidate_universe": candidate_universe,
        "source_logical_bytes": source_bytes,
        "source_logical_evidence": source_logical_evidence,
        "raw_bytes_per_candidate_cell": raw_bytes_per_cell,
        "projection_delta_bytes": int(projection_delta),
        "voter_delta_bytes": int(voter_delta),
        "top_k_delta_bytes": int(top_k_delta),
        "fixed_uncertainty_bytes": int(uncertainty),
        "raw_estimated_peak_process_tree_rss_bytes": int(raw_estimate),
        "safety_factor": safety_factor,
        "safety_factored_peak_process_tree_rss_bytes": int(safety_estimate),
        "assumptions": [
            "source representation bytes per candidate cell remain bounded by the measured parent source frame",
            "Boruta growth is bounded by three additional float32 numeric-equivalent cells per added training cell",
            "top-K reload/RFE growth includes both measured raw-cell and float32 numeric-cell volume",
            "the maximum stage delta is added to the global observed parent peak plus a fixed uncertainty reserve",
        ],
    }


def run_lendingclub_memory_capacity_scenario(
    *,
    repository_root: str | Path,
    refinement_config_path: str | Path,
    scenario_id: str,
    resume_run_id: str | None = None,
) -> Any:
    """Execute one exact authorized capacity scenario under an alternate result root."""

    import yaml

    root = Path(repository_root).resolve()
    refinement_path = Path(refinement_config_path)
    if not refinement_path.is_absolute():
        refinement_path = (root / refinement_path).resolve()
    specs = expand_lendingclub_memory_capacity_scenarios(refinement_path)
    matching = [item for item in specs if item.scenario_id == scenario_id]
    if len(matching) != 1:
        raise ValueError("capacity execution requires one exact authorized scenario ID")
    spec = matching[0]
    payload = yaml.safe_load(refinement_path.read_text(encoding="utf-8"))
    output_root = initialize_results_layout(
        root,
        results_root=str(payload["publication"]["capacity_results_root"]),
    )
    canonical_root = (root / "results").resolve()
    if output_root == canonical_root or output_root.is_relative_to(canonical_root):
        raise ValueError("capacity validation cannot register under canonical results")
    active_locks = sorted(
        [*canonical_root.glob("runs/*/*/.execution.lock"), *output_root.glob("runs/*/*/.execution.lock")]
    )
    if active_locks:
        raise RuntimeError(f"another registered execution is active: {active_locks}")
    resume = resume_run_id is not None
    if resume and resume_run_id != scenario_id:
        raise ValueError("capacity resume ID must equal the authorized scenario ID")

    equivalence_id = specs[0].scenario_id
    largest_id = specs[1].scenario_id
    if spec.execution_order >= 2:
        equivalence_gate = (
            root
            / "cleanup/audits/lendingclub_memory_refinement_capacity_gate/equivalence_validation.json"
        )
        if not equivalence_gate.is_file() or json.loads(
            equivalence_gate.read_text(encoding="utf-8")
        ).get("status") != "passed":
            raise RuntimeError("scientific equivalence gate must pass before a large scenario")
    if spec.execution_order == 3:
        largest_run = output_root / "runs" / "lendingclub_v2" / largest_id
        if not largest_run.is_dir() or json.loads(
            (largest_run / "manifest.json").read_text(encoding="utf-8")
        ).get("status") != "completed":
            raise RuntimeError("largest-fold capacity scenario must complete before full DEV")

    policy_path = str(payload["parents"]["execution_policy"]["path"])
    configured_policy = load_execution_policy(root, policy_path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    capacity = detect_hardware(output_root, temp_root)
    resolved_policy = resolve_execution_policy(configured_policy, capacity)
    if resolved_policy.to_dict()["parallelism"] != {
        "concurrent_experiment_runs": 1,
        "concurrent_folds": 1,
        "data_loader_workers": 0,
        "estimator_threads": 4,
        "allow_nested_parallelism": False,
    }:
        raise ValueError("capacity scenario widened frozen parallelism")
    run_dir = resolve_resume_target(output_root, scenario_id) if resume else None
    preflight = run_preflight(
        repository_root=root,
        config_path=policy_path,
        results_root=str(output_root),
        temp_root=temp_root,
        requested_accelerator="cpu",
        allow_gpu_without_telemetry=False,
        requested_run_directory=run_dir,
        capacity=capacity,
    )
    projection = None
    if spec.execution_order == 2:
        parent = output_root / "runs" / "lendingclub_v2" / equivalence_id
        projection = _capacity_projection_from_parent(
            parent_run=parent,
            parent_training_rows=83_283,
            parent_boundary_rows=197_550,
            parent_k=200,
            target_training_rows=458_602,
            target_boundary_rows=530_513,
            target_k=300,
            full_dev_rows=598_649,
            candidate_universe=675,
            safety_factor=float(payload["capacity_estimation"]["safety_factor"]),
            fixed_uncertainty_gib=float(
                payload["capacity_estimation"]["fixed_uncertainty_gib"]
            ),
        )
    elif spec.execution_order == 3:
        parent = output_root / "runs" / "lendingclub_v2" / largest_id
        projection = _capacity_projection_from_parent(
            parent_run=parent,
            parent_training_rows=458_602,
            parent_boundary_rows=530_513,
            parent_k=300,
            target_training_rows=598_649,
            target_boundary_rows=598_649,
            target_k=300,
            full_dev_rows=598_649,
            candidate_universe=675,
            safety_factor=float(payload["capacity_estimation"]["safety_factor"]),
            fixed_uncertainty_gib=float(
                payload["capacity_estimation"]["fixed_uncertainty_gib"]
            ),
        )
    if projection is not None:
        safety_gib = projection["safety_factored_peak_process_tree_rss_bytes"] / 1024**3
        available_for_worker_gib = capacity.available_ram_gb - float(
            payload["capacity_estimation"]["system_available_ram_floor_gib"]
        )
        passed = (
            safety_gib < resolved_policy.memory.abort_process_tree_rss_gb
            and safety_gib <= available_for_worker_gib
        )
        check = {
            "name": "refined_conservative_capacity_projection",
            "passed": passed,
            "blocking": True,
            "detail": (
                f"safety_peak={safety_gib:.3f} GiB, "
                f"current_available_minus_floor={available_for_worker_gib:.3f} GiB, "
                f"process_abort={resolved_policy.memory.abort_process_tree_rss_gb:.3f} GiB"
            ),
        }
        preflight["checks"].append(check)
        preflight["scenario_capacity_projection"] = projection
        if not passed:
            preflight["blocking_reasons"].append(check["name"])
            preflight["status"] = "fail"
    else:
        preflight["scenario_capacity_projection"] = {
            "status": "not_required_for_bounded_equivalence_replay",
            "bounding_observation": "completed Prompt-4 first-fold K200 execution under identical policy",
        }
    preflight["capacity_scenario"] = {
        "scenario_id": spec.scenario_id,
        "mode": spec.mode,
        "fold_id": spec.fold_id,
        "candidate_universe": 675,
        "candidate_pool": spec.candidate_pool,
        "branches": list(spec.branches),
        "seed": 42,
        "load_oot": False,
        "research_eligible": False,
        "comparison_eligible": False,
    }
    if preflight["status"] != "pass":
        # Preserve the rejected estimate outside a run directory; no workload starts.
        rejected = (
            output_root / f"{spec.scenario_id}.preflight_rejected.json"
        )
        from credit_risk_fs.experiments.atomic_io import write_json_atomic

        write_json_atomic(rejected, preflight, overwrite=False)
        raise RuntimeError(
            "BLOCKED_RESOURCE_CAPACITY: " + ", ".join(preflight["blocking_reasons"])
        )
    if run_dir is None:
        run_dir = create_run_directory(
            output_root,
            dataset="lendingclub_v2",
            run_id=scenario_id,
            collision_policy="error",
        )

    dataset_config = load_project_config(root / "configs/experiments/lendingclub_v2_matrix.yaml")
    data_dir = Path(str(dataset_config["data_dir"]))
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    experiment_config = ExperimentConfig(
        experiment_name=spec.scenario_id,
        selector_name="rank_voting_v1",
        dataset_name="lendingclub_v2",
        model_name="catboost",
        data_dir=str(data_dir),
        target="TARGET",
        time_col="recent_decision",
        random_state=42,
        feature_budget=40,
        estimator_threads=4,
        stable_row_id_column="loan_id",
    )
    scenario_payload = {
        "scenario_id": spec.scenario_id,
        "execution_order": spec.execution_order,
        "dataset": spec.dataset,
        "mode": spec.mode,
        "fold_id": spec.fold_id,
        "candidate_pool": spec.candidate_pool,
        "seed": spec.seed,
        "branches": list(spec.branches),
        "load_oot": False,
        "research_eligible": False,
        "comparison_eligible": False,
    }
    effective_config = {
        "schema_version": "lendingclub_memory_capacity_run_v1",
        "refinement_config_path": refinement_path.relative_to(root).as_posix(),
        "refinement_config_sha256": sha256_file(refinement_path),
        "scenario": scenario_payload,
        "purpose": "memory_capacity_validation",
        "research_eligible": False,
        "comparison_eligible": False,
        "load_oot": False,
        "oot_scored": False,
        "method": "rank_voting_v1",
        "candidate_universe": 675,
        "feature_budgets": {"lr": 20, "catboost": 40},
        "seed": 42,
        "accelerator": "cpu",
        "resolved_execution_policy": resolved_policy.to_dict(),
        "frozen_hashes": {
            name: values["sha256"] for name, values in payload["parents"].items()
        },
    }
    return execute_registered_run(
        RegisteredRunRequest(
            repository_root=root,
            results_root=output_root,
            run_directory=run_dir,
            dataset="lendingclub_v2",
            selector="rank_voting_v1_memory_safe_v1",
            model="lr_catboost_sequential",
            experiment_type="memory_capacity_validation",
            split_protocol=(
                f"grouped_time_series_cv_5_splits_gap_1_fold_{spec.fold_id}"
                if spec.mode == "fold"
                else "full_dev_capacity_fit_no_oot"
            ),
            seed=42,
            effective_config=effective_config,
            experiment_config=experiment_config,
            preflight_report=preflight,
            resolved_policy=resolved_policy,
            resume=resume,
            worker_target=(
                "credit_risk_fs.experiments.rank_voting:"
                "lendingclub_memory_capacity_worker"
            ),
            worker_kwargs={
                "repository_root": str(root),
                "scenario": scenario_payload,
                "estimator_threads": 4,
                "protocol_sha256": payload["parents"]["voting_protocol"]["sha256"],
            },
            merge_default_worker_kwargs=True,
            manifest_metadata={
                "purpose": "memory_capacity_validation",
                "research_eligible": False,
                "comparison_eligible": False,
                "load_oot": False,
                "oot_scored": False,
                "capacity_scenario_id": spec.scenario_id,
                "candidate_universe": 675,
                "candidate_pool_budget": spec.candidate_pool,
                "branches": list(spec.branches),
            },
            artifact_applicability={
                "config": True,
                "manifest": True,
                "selected_features": True,
                "fold_selections": True,
                "metrics": False,
                "predictions_dev": spec.mode == "fold",
                "predictions_oot": False,
                "stability": False,
                "resource_usage": True,
                "preflight": True,
                "checkpoint": True,
                "run_log": True,
            },
            protocol_path=payload["parents"]["voting_protocol"]["path"],
            row_contract_path=payload["parents"]["row_alignment_contract"]["path"],
        )
    )


def _research_run_directory(root: Path, spec: Any) -> Path:
    return planned_run_directory(
        root / "results", dataset=spec.dataset, run_id=spec.run_id
    )


def _research_effective_config(root: Path, spec: Any, provenance: Any) -> dict[str, Any]:
    matrix_path = root / "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml"
    return {
        "schema_version": "cross_dataset_voting_research_run_v1",
        "run_id": spec.run_id,
        "execution_order": spec.execution_order,
        "dataset": spec.dataset,
        "model": spec.model,
        "method_id": spec.method_id,
        "candidate_pool_budget": spec.candidate_pool_budget,
        "final_feature_budget": spec.final_feature_budget,
        "feature_budgets": {spec.model: spec.final_feature_budget},
        "designation": spec.designation,
        "comparison_family": spec.comparison_family,
        "reference_method": spec.reference_method,
        "fold_count": 5,
        "fold_protocol": "grouped_time_series_cv_5_splits_gap_1_expanding",
        "full_dev_fit_count": 1,
        "oot_policy": "locked_single_final_evaluation_after_global_dev_gate",
        "seed": 42,
        "accelerator": "cpu",
        "matrix_path": matrix_path.relative_to(root).as_posix(),
        "matrix_sha256": sha256_file(matrix_path),
        "frozen_hashes": {
            "scientific_protocol": RESEARCH_SCIENTIFIC_PROTOCOL_HASH,
            "row_alignment_contract": RESEARCH_ROW_CONTRACT_HASH,
            "voting_protocol": RESEARCH_PROTOCOL_HASH,
            "execution_policy": RESEARCH_POLICY_HASH,
            "memory_refinement": RESEARCH_REFINEMENT_HASH,
        },
        "release_provenance": {
            "git_commit": provenance.git_commit,
            "git_tag": provenance.git_tag,
            "pyproject_sha256": provenance.pyproject_sha256,
            "dependency_lock_path": provenance.dependency_lock_path,
            "dependency_lock_sha256": provenance.dependency_lock_sha256,
        },
    }


def preflight_cross_dataset_research(root: str | Path, plan: Any, provenance: Any) -> None:
    """Run a live global preflight without loading a research dataset."""

    repository = Path(root).resolve()
    output_root = initialize_results_layout(repository, results_root="results")
    active_locks = sorted(output_root.glob("runs/*/*/.execution.lock"))
    if active_locks:
        raise RuntimeError(f"registered experiment lock is active: {active_locks}")
    expected_ids = {item.run_id for item in plan.run_specs}
    unexpected = [
        path.name
        for path in output_root.glob("runs/*/cdv1-0*")
        if path.name not in expected_ids
        and not path.name.startswith("cdv1-pilot-")
        and not path.name.startswith("cdv1-capacity-")
        and not path.name.startswith("cdv1-equivalence-")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected cross-dataset research directories: {unexpected}")
    policy = load_execution_policy(repository, "configs/execution/local_laptop_safe_v1.yaml")
    capacity = detect_hardware(output_root, Path(tempfile.gettempdir()).resolve())
    resolved = resolve_execution_policy(policy, capacity)
    parallel = resolved.to_dict()["parallelism"]
    if (
        parallel["concurrent_experiment_runs"] != 1
        or parallel["concurrent_folds"] != 1
        or parallel["data_loader_workers"] != 0
        or parallel["estimator_threads"] > 4
        or parallel["allow_nested_parallelism"] is not False
    ):
        raise ValueError(f"research parallelism widened frozen limits: {parallel}")
    report = run_preflight(
        repository_root=repository,
        config_path="configs/execution/local_laptop_safe_v1.yaml",
        results_root="results",
        temp_root=Path(tempfile.gettempdir()).resolve(),
        requested_accelerator="cpu",
        allow_gpu_without_telemetry=False,
        capacity=capacity,
    )
    if report["status"] != "pass":
        raise RuntimeError(f"global research preflight failed: {report['blocking_reasons']}")
    if report["git_commit"] != provenance.git_commit or report["git_dirty"] is not False:
        raise RuntimeError("live preflight Git provenance differs from the release tag")


def ensure_cross_dataset_inter_run_readiness(root: str | Path) -> Any:
    """Confirm cleanup and unchanged resource floors before another research run."""

    repository = Path(root).resolve()
    results_root = initialize_results_layout(repository, results_root="results")
    temp_root = Path(tempfile.gettempdir()).resolve()
    configured = load_execution_policy(
        repository, "configs/execution/local_laptop_safe_v1.yaml"
    )
    resolved = resolve_execution_policy(
        configured, detect_hardware(results_root, temp_root)
    )
    return wait_for_inter_run_readiness(
        policy=resolved,
        results_root=results_root,
        temp_root=temp_root,
    )


def cross_dataset_research_run_state(root: str | Path, spec: Any) -> str:
    run_dir = _research_run_directory(Path(root).resolve(), spec)
    if not run_dir.is_dir():
        return "missing"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return "invalid"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "invalid"))
    if status in {"failed", "interrupted", "aborted_resource_limit"}:
        checkpoint_path = run_dir / "checkpoint.json"
        if checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if set(checkpoint.get("completed_fold_ids", [])) == {
                "1",
                "2",
                "3",
                "4",
                "5",
            } and (run_dir / "results/dev_predictions.csv").is_file():
                # A stopped OOT attempt resumes OOT; validated DEV is never repeated.
                return "dev_complete"
    return status


def execute_cross_dataset_research_phase(
    root: str | Path,
    spec: Any,
    *,
    phase: str,
    provenance: Any,
    frozen_set_sha256: str | None = None,
) -> str:
    """Submit one phase through ``execute_registered_run`` and return its state."""

    if phase not in {"dev", "oot"}:
        raise ValueError("research phase must be dev or oot")
    repository = Path(root).resolve()
    output_root = initialize_results_layout(repository, results_root="results")
    existing = _research_run_directory(repository, spec)
    resume = existing.is_dir()
    if phase == "oot" and not resume:
        raise RuntimeError("OOT phase cannot create a run before DEV")
    run_dir = (
        resolve_resume_target(output_root, spec.run_id)
        if resume
        else create_run_directory(
            output_root,
            dataset=spec.dataset,
            run_id=spec.run_id,
            collision_policy="error",
        )
    )
    configured_policy = load_execution_policy(
        repository, "configs/execution/local_laptop_safe_v1.yaml"
    )
    temp_root = Path(tempfile.gettempdir()).resolve()
    capacity = detect_hardware(output_root, temp_root)
    resolved_policy = resolve_execution_policy(configured_policy, capacity)
    parallel = resolved_policy.to_dict()["parallelism"]
    if (
        parallel["concurrent_experiment_runs"] != 1
        or parallel["concurrent_folds"] != 1
        or parallel["data_loader_workers"] != 0
        or parallel["estimator_threads"] > 4
        or parallel["allow_nested_parallelism"] is not False
    ):
        raise ValueError(f"research phase widened resource limits: {parallel}")
    preflight = run_preflight(
        repository_root=repository,
        config_path="configs/execution/local_laptop_safe_v1.yaml",
        results_root="results",
        temp_root=temp_root,
        requested_accelerator="cpu",
        allow_gpu_without_telemetry=False,
        requested_run_directory=run_dir if resume else None,
        capacity=capacity,
    )
    if preflight["status"] != "pass":
        raise RuntimeError(
            f"research phase preflight failed for {spec.run_id}: {preflight['blocking_reasons']}"
        )
    if preflight["git_commit"] != provenance.git_commit or preflight["git_dirty"] is not False:
        raise RuntimeError("research phase Git provenance drifted")
    dataset_config = load_project_config(
        repository / f"configs/experiments/{spec.dataset}_matrix.yaml"
    )
    data_dir = Path(str(dataset_config["data_dir"]))
    if not data_dir.is_absolute():
        data_dir = (repository / data_dir).resolve()
    experiment_config = ExperimentConfig(
        experiment_name=spec.run_id,
        selector_name=spec.method_id,
        dataset_name=spec.dataset,
        model_name=spec.model,
        data_dir=str(data_dir),
        target="TARGET",
        time_col="recent_decision",
        random_state=42,
        feature_budget=spec.final_feature_budget,
        estimator_threads=resolved_policy.parallelism.estimator_threads,
        stable_row_id_column=("SK_ID_CURR" if spec.dataset == "homecredit" else "loan_id"),
    )
    effective_config = _research_effective_config(repository, spec, provenance)
    checkpoint_identity_override = None
    resume_metadata = None
    if resume:
        from credit_risk_fs.experiments.provenance_bridge import (
            compatible_resume_identity,
        )

        compatible = compatible_resume_identity(
            repository,
            run_dir,
            current_commit=provenance.git_commit,
            current_tag=provenance.git_tag,
        )
        if compatible is not None:
            (
                checkpoint_identity_override,
                effective_config,
                resume_metadata,
            ) = compatible
    outcome = execute_registered_run(
        RegisteredRunRequest(
            repository_root=repository,
            results_root=output_root,
            run_directory=run_dir,
            dataset=spec.dataset,
            selector=spec.method_id,
            model=spec.model,
            experiment_type="cross_dataset_voting_research",
            split_protocol="grouped_time_series_cv_5_splits_gap_1_with_locked_oot",
            seed=42,
            effective_config=effective_config,
            experiment_config=experiment_config,
            preflight_report=preflight,
            resolved_policy=resolved_policy,
            resume=resume,
            worker_target=(
                "credit_risk_fs.experiments.cross_dataset_research:"
                "cross_dataset_research_phase_worker"
            ),
            worker_kwargs={
                "repository_root": str(repository),
                "phase": phase,
                "spec": asdict(spec),
                "protocol_sha256": RESEARCH_PROTOCOL_HASH,
                "estimator_threads": resolved_policy.parallelism.estimator_threads,
                "frozen_set_sha256": frozen_set_sha256,
            },
            merge_default_worker_kwargs=True,
            manifest_metadata={
                "purpose": "cross_dataset_voting_research",
                "research_eligible": True,
                "comparison_eligible": True,
                "candidate_pool_budget": spec.candidate_pool_budget,
                "final_feature_budget": spec.final_feature_budget,
                "dev_fold_count": 5,
                "global_dev_barrier_required": True,
                "oot_embargo": True,
                "release_tag": provenance.git_tag,
            },
            artifact_applicability={
                "config": True,
                "manifest": True,
                "selected_features": True,
                "fold_selections": True,
                "metrics": True,
                "predictions_dev": True,
                "predictions_oot": True,
                "stability": True,
                "resource_usage": True,
                "preflight": True,
                "checkpoint": True,
                "run_log": True,
            },
            protocol_path="configs/protocols/cross_dataset_rank_voting_v1.yaml",
            row_contract_path="configs/protocols/row_alignment_contract_v1.json",
            defer_terminal_success=phase == "dev",
            deferred_success_status="dev_complete",
            checkpoint_identity_override=checkpoint_identity_override,
            resume_metadata=resume_metadata,
        )
    )
    return outcome.status


def validate_cross_dataset_research_run(
    root: str | Path, spec: Any, *, phase: str
) -> None:
    from credit_risk_fs.experiments.prediction_contract import (
        COMPLETE_OOF_COVERAGE,
        COMPLETE_OOT_COVERAGE,
        validate_prediction_frame,
    )

    repository = Path(root).resolve()
    run_dir = _research_run_directory(repository, spec)
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_config_hash = compute_config_hash(
        _research_effective_config(
            repository,
            spec,
            SimpleNamespace(
                git_commit=checkpoint["identity"]["git_commit"],
                git_tag=manifest.get("release_tag"),
                pyproject_sha256=manifest["config"]["release_provenance"]["pyproject_sha256"],
                dependency_lock_path=manifest["config"]["release_provenance"]["dependency_lock_path"],
                dependency_lock_sha256=manifest["config"]["release_provenance"]["dependency_lock_sha256"],
            ),
        )
    )
    if checkpoint["identity"]["resolved_config_hash"] != expected_config_hash:
        raise ValueError(f"configuration drift detected for {spec.run_id}")
    if set(checkpoint.get("completed_fold_ids", [])) != {"1", "2", "3", "4", "5"}:
        raise ValueError(f"DEV fold coverage is incomplete for {spec.run_id}")
    dev = pd.read_csv(run_dir / "results" / "dev_predictions.csv")
    validate_prediction_frame(
        dev,
        expected_identities=dev["stable_row_id"],
        expected_targets=dev["target"],
        coverage_type=COMPLETE_OOF_COVERAGE,
        expected_split="DEV",
        research_eligible=True,
        comparison_eligible=True,
    )
    fold_selected = pd.read_csv(run_dir / "features" / "fold_selected_features.csv")
    observed = fold_selected.groupby("fold_id")["feature"].nunique().to_dict()
    if observed != {fold: spec.final_feature_budget for fold in range(1, 6)}:
        raise ValueError(f"selected-feature budgets are invalid for {spec.run_id}: {observed}")
    if phase == "dev":
        dev_valid_statuses = {
            "dev_complete",
            "completed",
            "failed",
            "interrupted",
            "aborted_resource_limit",
        }
        if manifest.get("status") not in dev_valid_statuses:
            raise ValueError(f"DEV terminal status is invalid for {spec.run_id}")
        if manifest.get("status") == "dev_complete" and (
            (run_dir / "results" / "oot_predictions.csv").exists()
            or (run_dir / "data_access_oot.json").exists()
        ):
            raise ValueError(f"OOT artifact exists before the global barrier for {spec.run_id}")
        return
    if manifest.get("status") != "completed" or not (run_dir / "_SUCCESS").is_file():
        raise ValueError(f"completed terminal status is invalid for {spec.run_id}")
    oot = pd.read_csv(run_dir / "results" / "oot_predictions.csv")
    validate_prediction_frame(
        oot,
        expected_identities=oot["stable_row_id"],
        expected_targets=oot["target"],
        coverage_type=COMPLETE_OOT_COVERAGE,
        expected_split="OOT",
        research_eligible=True,
        comparison_eligible=True,
    )
    final_selected = pd.read_csv(run_dir / "features" / "final_selected_features.csv")
    if len(final_selected) != spec.final_feature_budget:
        raise ValueError(f"final selected-feature budget is invalid for {spec.run_id}")


def freeze_cross_dataset_configuration_set(
    root: str | Path, plan: Any, provenance: Any
) -> str:
    repository = Path(root).resolve()
    for spec in plan.run_specs:
        validate_cross_dataset_research_run(repository, spec, phase="dev")
    payload = {
        "schema_version": "cross_dataset_voting_configuration_lock_v1",
        "configuration_set_sha256": plan.configuration_set_sha256,
        "matrix_sha256": plan.matrix_sha256,
        "run_ids": [item.run_id for item in plan.run_specs],
        "git_commit": provenance.git_commit,
        "git_tag": provenance.git_tag,
        "status": "all_dev_validated_oot_configuration_locked",
    }
    path = repository / "results/comparisons/cross_dataset_voting_configuration_lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("existing configuration lock differs from the validated DEV set")
    else:
        write_json_atomic(path, payload, overwrite=False)
    return plan.configuration_set_sha256


def finalize_cross_dataset_research(
    root: str | Path, plan: Any, frozen_set_sha256: str
) -> None:
    """Publish consolidated comparisons only after all OOT runs validate."""

    from credit_risk_fs.evaluation.paired_inference import (
        apply_holm_to_family,
        paired_delong_test,
        paired_stratified_bootstrap,
        validate_paired_comparison_contract,
    )
    from credit_risk_fs.experiments.compare import build_cross_dataset_voting_comparison_plan

    repository = Path(root).resolve()
    for spec in plan.run_specs:
        validate_cross_dataset_research_run(repository, spec, phase="oot")
    comparison_plan = build_cross_dataset_voting_comparison_plan(plan.run_specs)
    by_id = {item.run_id: item for item in plan.run_specs}
    results = []
    raw_by_family: dict[str, dict[str, float]] = {}
    for comparison in comparison_plan:
        voting_id = str(comparison["voting_run_id"])
        reference_id = str(comparison["reference_run_id"])
        voting_spec = by_id[voting_id]
        reference_spec = by_id[reference_id]
        voting_dir = _research_run_directory(repository, voting_spec)
        reference_dir = _research_run_directory(repository, reference_spec)
        voting = pd.read_csv(voting_dir / "results/oot_predictions.csv")
        reference = pd.read_csv(reference_dir / "results/oot_predictions.csv")
        voting_meta = json.loads(
            (voting_dir / "results/oot_prediction_metadata.json").read_text(encoding="utf-8")
        )
        reference_meta = json.loads(
            (reference_dir / "results/oot_prediction_metadata.json").read_text(encoding="utf-8")
        )
        metadata_keys = {
            "dataset": voting_spec.dataset,
            "model": voting_spec.model,
            "split": "OOT",
            "fold_definition": "locked_single_final_evaluation",
            "probability_orientation": "class_1_higher_default_risk",
            "research_eligible": True,
            "comparison_eligible": True,
        }
        aligned = validate_paired_comparison_contract(
            voting,
            {**metadata_keys, "identity_target_sha256": voting_meta["identity_target_sha256"]},
            reference,
            {**metadata_keys, "identity_target_sha256": reference_meta["identity_target_sha256"]},
        )
        delong = paired_delong_test(aligned.target, aligned.score_a, aligned.score_b)
        bootstrap = paired_stratified_bootstrap(aligned)
        family = voting_spec.comparison_family
        raw_by_family.setdefault(family, {})[voting_id] = delong["two_sided_p_value"]
        results.append(
            {
                **comparison,
                "paired_rows": len(aligned),
                "delong": delong,
                "bootstrap": bootstrap,
            }
        )
    holm = {family: apply_holm_to_family(values) for family, values in raw_by_family.items()}
    payload = {
        "schema_version": "cross_dataset_voting_consolidated_comparison_v1",
        "configuration_set_sha256": frozen_set_sha256,
        "all_oot_runs_validated_before_publication": True,
        "comparison_count": len(results),
        "comparisons": results,
        "holm_by_dataset_model_family": holm,
    }
    output = repository / "results/comparisons/cross_dataset_voting_consolidated.json"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != payload:
            raise ValueError("consolidated comparison artifact already exists with different content")
    else:
        write_json_atomic(output, payload, overwrite=False)
    write_json_atomic(
        repository / "results/comparisons/cross_dataset_voting_completeness_summary.json",
        {
            "schema_version": "cross_dataset_voting_completeness_v1",
            "status": "complete",
            "configuration_set_sha256": frozen_set_sha256,
            "registered_runs": 16,
            "dev_fold_executions": 80,
            "full_dev_oot_fits": 16,
            "comparison_count": 12,
            "oot_embargo_preserved": True,
        },
        overwrite=False,
    )


def validate_completed_cross_dataset_research(root: str | Path, plan: Any) -> None:
    repository = Path(root).resolve()
    for spec in plan.run_specs:
        validate_cross_dataset_research_run(repository, spec, phase="oot")
    result = subprocess.run(
        [sys.executable, "cleanup/tools/validate_repository_state.py", "--root", "."],
        cwd=repository,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"repository/result validator failed with exit {result.returncode}")


def main(argv: list[str] | None = None) -> None:
    validate_matrix()
    cli_args = build_parser().parse_args(argv)
    repository_root = Path(cli_args.repository_root).resolve()

    if cli_args.cross_dataset_voting_matrix_dry_expand is not None:
        path = cli_args.cross_dataset_voting_matrix_dry_expand
        if not path.is_absolute():
            path = repository_root / path
        print(json.dumps(dry_expand_cross_dataset_voting_matrix(path), indent=2))
        return
    if cli_args.voting_pilot_config is not None:
        pilot_path = cli_args.voting_pilot_config
        if not pilot_path.is_absolute():
            pilot_path = repository_root / pilot_path
        authorized_specs = expand_cross_dataset_voting_pilot(pilot_path)
        expected_outcome_count = len(authorized_specs)
        if cli_args.resume is not None:
            matching = [
                index
                for index, item in enumerate(authorized_specs)
                if item.run_id == cli_args.resume
            ]
            if len(matching) != 1:
                raise ValueError("pilot resume must name one exact authorized pilot ID")
            expected_outcome_count -= matching[0]
        outcomes = run_cross_dataset_voting_pilots(
            repository_root=repository_root,
            pilot_config_path=cli_args.voting_pilot_config,
            resume_run_id=cli_args.resume,
        )
        if len(outcomes) != expected_outcome_count or any(
            item.status != "completed" for item in outcomes
        ):
            last = outcomes[-1] if outcomes else None
            raise RuntimeError(
                "voting pilot sequence stopped: "
                f"completed={sum(item.status == 'completed' for item in outcomes)}, "
                f"last_status={getattr(last, 'status', None)}, "
                f"stop_code={getattr(last, 'stop_code', None)}"
            )
        return
    if cli_args.lendingclub_memory_refinement_config is not None:
        if cli_args.capacity_scenario_id is None:
            raise ValueError("--capacity-scenario-id is required for capacity validation")
        outcome = run_lendingclub_memory_capacity_scenario(
            repository_root=repository_root,
            refinement_config_path=cli_args.lendingclub_memory_refinement_config,
            scenario_id=cli_args.capacity_scenario_id,
            resume_run_id=cli_args.resume,
        )
        if outcome.status != "completed":
            raise RuntimeError(
                "capacity scenario stopped: "
                f"status={outcome.status}, stop_code={outcome.stop_code}"
            )
        return
    if cli_args.capacity_scenario_id is not None:
        raise ValueError(
            "--capacity-scenario-id requires --lendingclub-memory-refinement-config"
        )

    project_config = load_project_config(cli_args.config)
    if cli_args.random_seed is not None:
        project_config["random_seed"] = int(cli_args.random_seed)
    if cli_args.output_dir is not None:
        project_config["results_dir"] = cli_args.output_dir

    output_root = initialize_results_layout(
        repository_root,
        results_root=project_config.get("results_dir", "results"),
    )
    dataset = sanitize_component(
        project_config.get("dataset_name", "homecredit"),
        field_name="dataset",
    )
    shared_llm_config = copy.deepcopy(project_config)
    shared_llm_config.pop("model_selector", None)
    shared_llm_config.pop("matrix_run", None)
    shared_llm_config["llm_ranking_scope"] = {
        "shared_ranking_enabled": True,
        "ranking_budget": normalize_llm_ranking_budget(
            shared_llm_config.get("llm", {}).get("ranking_budget")
        ),
        "prompt_version": shared_llm_config.get("llm", {}).get("prompt_version", "stability_expert_v3"),
    }
    project_config["llm_ranking_config_hash"] = compute_config_hash(shared_llm_config)
    llm_cache_dir = Path(project_config.get("llm", {}).get("cache_dir", "artifacts/llm_cache"))
    project_config.setdefault("llm", {})["cache_dir"] = str(llm_cache_dir)

    resolved_policy = None
    preflight_report = None
    resume_directory = None
    if not cli_args.dry_run:
        temp_root = Path(tempfile.gettempdir()).resolve()
        configured_policy = load_execution_policy(
            repository_root, cli_args.execution_policy
        )
        capacity = detect_hardware(output_root, temp_root)
        resolved_policy = resolve_execution_policy(configured_policy, capacity)
        project_config["_resolved_execution_policy"] = resolved_policy.to_dict()
        if cli_args.resume:
            resume_directory = resolve_resume_target(output_root, cli_args.resume)
        preflight_report = run_preflight(
            repository_root=repository_root,
            config_path=cli_args.execution_policy,
            results_root=project_config.get("results_dir", "results"),
            temp_root=temp_root,
            requested_accelerator=cli_args.accelerator,
            allow_gpu_without_telemetry=cli_args.allow_gpu_without_telemetry,
            requested_run_directory=resume_directory,
            capacity=capacity,
        )
        if preflight_report["status"] != "pass":
            raise RuntimeError(
                f"preflight_rejected: {preflight_report['blocking_reasons']}"
            )

    selected_models = set(cli_args.models)
    specs = [spec for spec in iter_matrix() if spec.model in selected_models]
    if resume_directory is not None:
        resume_manifest = json.loads(
            (resume_directory / "manifest.json").read_text(encoding="utf-8")
        )
        specs = [
            spec
            for spec in specs
            if spec.model == resume_manifest.get("model")
            and spec.experiment_name == resume_manifest.get("selector")
        ]
        if len(specs) != 1:
            raise ValueError(
                "--resume must identify exactly one matrix entry matching the current config"
            )

    matrix_rows: list[dict[str, Any]] = []
    pending: list[
        tuple[MatrixRunSpec, dict[str, Any], str, dict[str, Any]]
    ] = []

    for spec in specs:
        run_config = _matrix_config_for_spec(project_config, spec)
        config_hash = compute_config_hash(run_config)
        proposed_run_id, proposed_run_dir = _run_dir_for_spec(
            output_root=output_root,
            dataset=dataset,
            spec=spec,
        )
        row = {
            "run_id": proposed_run_id,
            "model": spec.model,
            "selector": spec.experiment_name,
            "experiment_type": spec.experiment_type,
            "status": "scheduled",
            "config_hash": config_hash,
            "output_folder": str(proposed_run_dir),
        }
        matrix_rows.append(row)
        pending.append((spec, run_config, config_hash, row))

    _write_matrix_status(output_root, dataset, matrix_rows)
    _write_llm_call_summary(output_root, dataset, matrix_rows)
    _write_failed_runs(output_root, dataset, matrix_rows)

    if cli_args.dry_run:
        for row in matrix_rows:
            print(
                f"{row['status']}: {row['model']} | {row['experiment_type']} | "
                f"{row['selector']} -> {row['output_folder']}"
            )
        return

    for spec, run_config, config_hash, matrix_row in pending:
        run_dir = resume_directory or create_run_directory(
                output_root,
                dataset=dataset,
                run_id=build_run_id(
                    selector=spec.experiment_name,
                    model=spec.model,
                ),
                collision_policy="suffix",
            )
        run_id = run_dir.name
        matrix_row.update(
            {
                "run_id": run_id,
                "status": "running",
                "output_folder": str(run_dir),
            }
        )
        logger.info(
            "Starting matrix run %s | model=%s | type=%s | selector=%s",
            run_id,
            spec.model,
            spec.experiment_type,
            spec.experiment_name,
        )

        experiment_config = _experiment_config_for_spec(
            spec=spec,
            run_config=run_config,
            run_dir=run_dir,
        )
        if resolved_policy is None or preflight_report is None:
            raise RuntimeError("execution policy/preflight was not resolved")
        outcome = execute_registered_run(
            RegisteredRunRequest(
                repository_root=repository_root,
                results_root=output_root,
                run_directory=run_dir,
                dataset=dataset,
                selector=spec.experiment_name,
                model=spec.model,
                experiment_type=spec.experiment_type,
                split_protocol="grouped_time_series_cv_with_oot",
                seed=int(run_config["random_seed"]),
                effective_config=run_config,
                experiment_config=experiment_config,
                preflight_report=preflight_report,
                resolved_policy=resolved_policy,
                resume=resume_directory is not None,
            )
        )
        matrix_row["status"] = outcome.status
        if outcome.status == "completed":
            outcome.manifest.update(_llm_ranking_stats(run_dir))
            write_run_manifest(run_dir, outcome.manifest)
        else:
            _write_llm_call_summary(output_root, dataset, matrix_rows)
            _write_failed_runs(output_root, dataset, matrix_rows)
            raise RuntimeError(
                f"matrix run {run_id} ended with status={outcome.status}, "
                f"stop_code={outcome.stop_code}"
            )

    _write_matrix_status(output_root, dataset, matrix_rows)
    _write_llm_call_summary(output_root, dataset, matrix_rows)
    _write_failed_runs(output_root, dataset, matrix_rows)
    logger.info("Full experiment matrix completed. Output root: %s", output_root)


if __name__ == "__main__":
    main()
