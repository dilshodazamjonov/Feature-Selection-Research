from __future__ import annotations

import argparse
import copy
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from credit_risk_fs.experiments._common import build_experiment_config
from credit_risk_fs.experiments.atomic_io import write_csv_atomic
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
from credit_risk_fs.experiments.matrix import MODELS, MatrixRunSpec, iter_matrix, validate_matrix
from credit_risk_fs.experiments.execution import RegisteredRunRequest, execute_registered_run
from credit_risk_fs.experiments.checkpointing import resolve_resume_target
from credit_risk_fs.experiments.resource_policy import (
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
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


def main(argv: list[str] | None = None) -> None:
    validate_matrix()
    cli_args = build_parser().parse_args(argv)
    repository_root = Path(cli_args.repository_root).resolve()

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
