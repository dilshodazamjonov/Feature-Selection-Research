from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from Models.utils import get_selector
from experiments.common import build_experiment_config
from experiments.config import (
    DEFAULT_CONFIG_PATH,
    apply_feature_budget_to_selector_kwargs,
    compute_config_hash,
    load_project_config,
    normalize_llm_ranking_budget,
    resolve_feature_budget,
    resolve_llm_candidate_pool_budget,
    resolve_llm_shared_pool_size,
)
from experiments.matrix import MODELS, MatrixRunSpec, iter_matrix, validate_matrix
from experiments.tracking import (
    build_run_manifest,
    is_completed_run,
    mark_completed,
    run_id_for_config,
    utc_timestamp,
    write_run_manifest,
)
from feature_selection.hybrid import LLMThenStatSelector
from feature_selection.hybrid import StableCoreLLMFillSelector
from pipelines.common import ExperimentConfig, prepare_modeling_data, run_experiment
from utils.logging_config import run_log_context, setup_logging


logger = setup_logging("experiment_matrix", level=logging.INFO)


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
        help="Rerun completed matrix entries instead of reusing their outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix entries and output folders without training.",
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
    spec: MatrixRunSpec,
    config_hash: str,
) -> tuple[str, Path]:
    run_id = run_id_for_config(
        model=spec.model,
        experiment_type=spec.experiment_type,
        selector=spec.experiment_name,
        config_hash=config_hash,
    )
    return run_id, output_root / spec.model / spec.output_bucket / run_id


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


def _write_matrix_status(output_root: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(output_root / "matrix_runs.csv", index=False)


def _llm_ranking_stats(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "features" / "llm_rankings_summary.csv"
    cache_hits = 0
    calls_made = 0
    signatures: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            if "metadata_signature" in df.columns:
                signatures = [
                    str(value)
                    for value in df["metadata_signature"].dropna().drop_duplicates().tolist()
                ]
            if "cache_hit" in df.columns:
                scope_keys = ["scope", "fold_id", "metadata_signature"]
                available_keys = [key for key in scope_keys if key in df.columns]
                call_df = df.drop_duplicates(subset=available_keys) if available_keys else df
                cache_flags = call_df["cache_hit"].astype(str).str.lower().isin(["true", "1"])
                cache_hits = int(cache_flags.sum())
                calls_made = int((~cache_flags).sum())
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
                    prompt_tokens = int(value)
                elif target == "completion":
                    completion_tokens = int(value)
                else:
                    total_tokens = int(value)
    return {
        "llm_cache_key": signatures[0] if signatures else None,
        "llm_metadata_signatures": signatures,
        "llm_calls_actually_made": calls_made,
        "llm_cache_hits": cache_hits,
        "llm_prompt_tokens": prompt_tokens,
        "llm_completion_tokens": completion_tokens,
        "llm_total_tokens": total_tokens,
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


def _write_llm_call_summary(output_root: Path, matrix_rows: list[dict[str, Any]] | None = None) -> None:
    columns = [
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
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "runs_sharing_metadata_signatures",
        "output_folder",
    ]
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
        records.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "llm_shared_ranking_enabled": manifest.get("llm_shared_ranking_enabled"),
                "llm_ranking_budget": manifest.get("llm_ranking_budget"),
                "llm_calls_actually_made": manifest.get("llm_calls_actually_made", 0),
                "llm_cache_hits": manifest.get("llm_cache_hits", 0),
                "llm_cache_key": manifest.get("llm_cache_key"),
                "llm_metadata_signatures": ";".join(manifest.get("llm_metadata_signatures", []) or []),
                "llm_prompt_tokens": manifest.get("llm_prompt_tokens", 0),
                "llm_completion_tokens": manifest.get("llm_completion_tokens", 0),
                "llm_total_tokens": manifest.get("llm_total_tokens", 0),
                "output_folder": str(manifest_path.parent),
            }
        )
    signature_to_runs: dict[str, list[str]] = {}
    for row in records:
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            if signature:
                signature_to_runs.setdefault(signature, []).append(str(row["run_id"]))
    for row in records:
        sharing = set()
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            sharing.update(signature_to_runs.get(signature, []))
        row["runs_sharing_metadata_signatures"] = ";".join(sorted(sharing))
    pd.DataFrame(records, columns=columns).to_csv(output_root / "llm_call_summary.csv", index=False)


def _write_failed_runs(output_root: Path, matrix_rows: list[dict[str, Any]] | None = None) -> None:
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
    pd.DataFrame(rows, columns=columns).to_csv(output_root / "failed_runs.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    validate_matrix()
    cli_args = build_parser().parse_args(argv)

    project_config = load_project_config(cli_args.config)
    if cli_args.random_seed is not None:
        project_config["random_seed"] = int(cli_args.random_seed)
    if cli_args.output_dir is not None:
        project_config["results_dir"] = cli_args.output_dir

    output_root = Path(project_config.get("results_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
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
    llm_cache_dir = output_root / "_llm_rankings_cache"
    project_config.setdefault("llm", {})["cache_dir"] = str(llm_cache_dir)

    selected_models = set(cli_args.models)
    specs = [spec for spec in iter_matrix() if spec.model in selected_models]

    matrix_rows: list[dict[str, Any]] = []
    pending: list[tuple[MatrixRunSpec, dict[str, Any], str, Path]] = []

    for spec in specs:
        run_config = _matrix_config_for_spec(project_config, spec)
        config_hash = compute_config_hash(run_config)
        run_id, run_dir = _run_dir_for_spec(
            output_root=output_root,
            spec=spec,
            config_hash=config_hash,
        )

        status = "completed" if is_completed_run(run_dir) else "pending"
        if status == "completed" and not cli_args.force:
            logger.info("Skipping completed run: %s", run_dir)
        else:
            pending.append((spec, run_config, run_id, run_dir))
            status = "scheduled"

        matrix_rows.append(
            {
                "run_id": run_id,
                "model": spec.model,
                "selector": spec.experiment_name,
                "experiment_type": spec.experiment_type,
                "status": status,
                "config_hash": config_hash,
                "output_folder": str(run_dir),
            }
        )

    _write_matrix_status(output_root, matrix_rows)
    _write_llm_call_summary(output_root, matrix_rows)
    _write_failed_runs(output_root, matrix_rows)

    if cli_args.dry_run:
        for row in matrix_rows:
            print(
                f"{row['status']}: {row['model']} | {row['experiment_type']} | "
                f"{row['selector']} -> {row['output_folder']}"
            )
        return

    if not pending:
        _write_llm_call_summary(output_root, matrix_rows)
        _write_failed_runs(output_root, matrix_rows)
        logger.info("All matrix entries already completed. Nothing to rerun.")
        return

    first_model = pending[0][0].model
    logger.info("Preparing shared modeling data once for %s pending matrix runs.", len(pending))
    prepared_data = prepare_modeling_data(
        _prepare_data_config(project_config, model=first_model, output_root=output_root)
    )

    completed_rows: list[dict[str, Any]] = []
    for spec, run_config, run_id, run_dir in pending:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_run_manifest(
            run_id=run_id,
            model=spec.model,
            selector=spec.experiment_name,
            experiment_type=spec.experiment_type,
            config=run_config,
            data_dir=run_config["data_dir"],
            random_seed=int(run_config["random_seed"]),
            output_folder=run_dir,
            project_root=Path.cwd(),
            status="running",
        )
        llm_config = run_config.get("llm", {})
        manifest.update(
            {
                "llm_shared_ranking_enabled": bool(llm_config.get("shared_ranking_enabled", True)),
                "llm_ranking_budget": int(resolve_llm_shared_pool_size(llm_config)),
                "llm_ranking_budget_config": normalize_llm_ranking_budget(llm_config.get("ranking_budget")),
                "llm_shared_pool_size": int(resolve_llm_shared_pool_size(llm_config)),
                "llm_candidate_pool_budget": int(
                    resolve_llm_candidate_pool_budget(llm_config, spec.model)
                ),
                "llm_prompt_version": llm_config.get("prompt_version", "stability_expert_v3"),
                "lr_feature_budget": int(run_config.get("feature_budgets", {}).get("lr", 20)),
                "catboost_feature_budget": int(run_config.get("feature_budgets", {}).get("catboost", 40)),
                "feature_budget": int(run_config.get("feature_budgets", {}).get(spec.model, 40)),
            }
        )
        write_run_manifest(run_dir, manifest)

        logger.info(
            "Starting matrix run %s | model=%s | type=%s | selector=%s",
            run_id,
            spec.model,
            spec.experiment_type,
            spec.experiment_name,
        )

        try:
            with run_log_context(run_dir / "run.log"):
                experiment_config = _experiment_config_for_spec(
                    spec=spec,
                    run_config=run_config,
                    run_dir=run_dir,
                )
                run = run_experiment(experiment_config, prepared_data=prepared_data)
                manifest["status"] = "completed"
                manifest["completed_at"] = utc_timestamp()
                manifest["summary"] = run.summary
                manifest.update(_llm_ranking_stats(run_dir))
                write_run_manifest(run_dir, manifest)
                mark_completed(run_dir)
                completed_rows.append(
                    {
                        "run_id": run_id,
                        "model": spec.model,
                        "selector": spec.experiment_name,
                        "experiment_type": spec.experiment_type,
                        "status": "completed",
                        "config_hash": manifest["config_hash"],
                        "output_folder": str(run_dir),
                    }
                )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failed_at"] = utc_timestamp()
            manifest["error"] = repr(exc)
            write_run_manifest(run_dir, manifest)
            _write_llm_call_summary(output_root, matrix_rows)
            _write_failed_runs(output_root, matrix_rows)
            logger.exception("Matrix run failed: %s", run_id)
            raise

    for row in matrix_rows:
        for completed in completed_rows:
            if row["run_id"] == completed["run_id"]:
                row.update(completed)
                break

    _write_matrix_status(output_root, matrix_rows)
    _write_llm_call_summary(output_root, matrix_rows)
    _write_failed_runs(output_root, matrix_rows)
    logger.info("Full experiment matrix completed. Output root: %s", output_root)


if __name__ == "__main__":
    main()
