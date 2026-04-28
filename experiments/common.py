from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.config import (
    apply_feature_budget_to_selector_kwargs,
    compute_config_hash,
    resolve_feature_budget,
    resolve_model_kwargs,
)
from experiments.tracking import build_data_version
from pipelines.common import (
    ExperimentConfig,
    create_run_output_dir,
    prepare_modeling_data,
    write_run_manifest,
)


@dataclass(slots=True)
class RunLayout:
    run_dir: Path
    experiments_dir: Path
    feature_overlap_dir: Path | None = None


def add_common_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--model-selector",
        "--model",
        dest="model",
        default="lr",
        help="Model name: lr, rf, or catboost.",
    )
    parser.add_argument("--data-dir", default="data/inputs")
    parser.add_argument("--description-path", default="data/HomeCredit_columns_description.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--dev-start-day", type=int, default=-600)
    parser.add_argument("--oot-start-day", type=int, default=-240)
    parser.add_argument("--oot-end-day", type=int, default=0)
    parser.add_argument("--cv-gap-groups", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-max-features", type=int, default=100)
    parser.add_argument("--llm-ranking-budget", type=int, default=100)
    parser.add_argument("--llm-shared-ranking-enabled", action="store_true", default=True)
    parser.add_argument("--llm-cache-dir", default="results/_llm_rankings_cache")


def create_run_layout(
    *,
    output_dir: str | Path,
    run_label: str,
    manifest_payload: dict[str, Any],
    include_feature_overlap_dir: bool = False,
) -> RunLayout:
    run_dir = create_run_output_dir(output_dir, run_label)
    experiments_dir = run_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    feature_overlap_dir: Path | None = None
    if include_feature_overlap_dir:
        feature_overlap_dir = run_dir / "feature_overlap"
        feature_overlap_dir.mkdir(parents=True, exist_ok=True)

    write_run_manifest(run_dir, manifest_payload)
    return RunLayout(
        run_dir=run_dir,
        experiments_dir=experiments_dir,
        feature_overlap_dir=feature_overlap_dir,
    )


def build_experiment_config(
    *,
    args: argparse.Namespace,
    experiments_dir: str | Path,
    experiment_name: str,
    selector_name: str,
    selector_kwargs: dict[str, Any] | None = None,
    selector_cls: type | None = None,
    experiment_output_dir: str | Path | None = None,
) -> ExperimentConfig:
    excluded_feature_columns = args.project_config.get("excluded_feature_columns", ())
    project_config = dict(args.project_config)
    project_config["random_seed"] = args.random_seed
    matrix_run = project_config.get("matrix_run", {})
    feature_budget = resolve_feature_budget(project_config, args.model)
    selector_kwargs = apply_feature_budget_to_selector_kwargs(
        selector_name,
        dict(selector_kwargs or {}),
        feature_budget,
    )
    return ExperimentConfig(
        experiment_name=experiment_name,
        selector_name=selector_name,
        selector_cls=selector_cls,
        selector_kwargs=selector_kwargs,
        experiment_type=matrix_run.get("experiment_type", "single"),
        config_hash=compute_config_hash(project_config),
        data_fingerprint=build_data_version(args.data_dir),
        model_name=args.model,
        model_kwargs=resolve_model_kwargs(project_config, args.model),
        data_dir=args.data_dir,
        description_path=args.description_path,
        base_output_dir=str(experiments_dir),
        experiment_output_dir=str(experiment_output_dir) if experiment_output_dir is not None else None,
        dev_start_day=args.dev_start_day,
        oot_start_day=args.oot_start_day,
        oot_end_day=args.oot_end_day,
        n_splits=args.n_splits,
        cv_gap_groups=args.cv_gap_groups,
        random_state=args.random_seed,
        feature_budget=feature_budget,
        excluded_feature_columns=tuple(excluded_feature_columns),
    )


def prepare_shared_data(args: argparse.Namespace, experiments_dir: str | Path):
    base_config = build_experiment_config(
        args=args,
        experiments_dir=experiments_dir,
        experiment_name="data_prep",
        selector_name="none",
    )
    return prepare_modeling_data(base_config)


def resolve_llm_cache_dir(run_dir: str | Path, configured_cache_dir: str) -> str:
    cache_name = Path(configured_cache_dir).name or "_llm_rankings_cache"
    return str(Path(run_dir) / cache_name)
