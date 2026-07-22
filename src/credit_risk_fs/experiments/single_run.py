from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from credit_risk_fs.experiments._common import (
    add_common_experiment_args,
    build_experiment_config,
)
from credit_risk_fs.experiments.config import (
    build_parser_defaults,
    extract_config_path,
    load_project_config,
)
from credit_risk_fs.experiments.result_paths import (
    build_run_id,
    create_run_directory,
    initialize_results_layout,
)
from credit_risk_fs.experiments.checkpointing import resolve_resume_target
from credit_risk_fs.experiments.execution import (
    RegisteredRunRequest,
    execute_registered_run,
)
from credit_risk_fs.experiments.resource_policy import (
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one experiment configuration as a lightweight compatibility entrypoint.",
    )
    parser.add_argument("--selector", default=defaults["selector"], help="Selector name to run.")
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    add_common_experiment_args(parser)
    parser.set_defaults(
        config=defaults["config_path"],
        model=defaults["model"],
        data_dir=defaults["data_dir"],
        description_path=defaults["description_path"],
        n_splits=defaults["n_splits"],
        dev_start_day=defaults["dev_start_day"],
        oot_start_day=defaults["oot_start_day"],
        oot_end_day=defaults["oot_end_day"],
        cv_gap_groups=defaults["cv_gap_groups"],
        random_seed=defaults["random_seed"],
    )
    return parser


def run(args: argparse.Namespace) -> None:
    repository_root = Path(args.repository_root).resolve()
    dataset = str(args.project_config.get("dataset_name", "homecredit"))
    results_root = initialize_results_layout(
        repository_root,
        results_root=args.output_dir,
    )
    temp_root = Path(tempfile.gettempdir()).resolve()
    configured_policy = load_execution_policy(repository_root, args.execution_policy)
    capacity = detect_hardware(results_root, temp_root)
    resolved_policy = resolve_execution_policy(configured_policy, capacity)
    effective_config = {
        **args.project_config,
        "model_selector": args.model,
        "_resolved_execution_policy": resolved_policy.to_dict(),
        "single_run": {
            "selector": args.selector,
            "n_splits": args.n_splits,
            "dev_start_day": args.dev_start_day,
            "oot_start_day": args.oot_start_day,
            "oot_end_day": args.oot_end_day,
            "cv_gap_groups": args.cv_gap_groups,
        },
    }
    args.project_config = effective_config
    resume_directory = (
        resolve_resume_target(results_root, args.resume) if args.resume else None
    )
    preflight = run_preflight(
        repository_root=repository_root,
        config_path=args.execution_policy,
        results_root=args.output_dir,
        temp_root=temp_root,
        requested_accelerator=args.accelerator,
        allow_gpu_without_telemetry=args.allow_gpu_without_telemetry,
        requested_run_directory=resume_directory,
        capacity=capacity,
    )
    if preflight["status"] != "pass":
        raise RuntimeError(f"preflight_rejected: {preflight['blocking_reasons']}")
    run_dir = (
        resume_directory
        if resume_directory is not None
        else create_run_directory(
            results_root,
            dataset=dataset,
            run_id=build_run_id(selector=args.selector, model=args.model),
            collision_policy="suffix",
        )
    )
    config = build_experiment_config(
        args=args,
        experiments_dir=run_dir,
        experiment_name=args.selector.lower(),
        selector_name=args.selector.lower(),
        experiment_output_dir=run_dir,
    )
    outcome = execute_registered_run(
        RegisteredRunRequest(
            repository_root=repository_root,
            results_root=results_root,
            run_directory=run_dir,
            dataset=dataset,
            selector=args.selector,
            model=args.model,
            experiment_type="single",
            split_protocol="grouped_time_series_cv_with_oot",
            seed=int(args.random_seed),
            effective_config=effective_config,
            experiment_config=config,
            preflight_report=preflight,
            resolved_policy=resolved_policy,
            resume=resume_directory is not None,
        )
    )
    if outcome.status != "completed":
        raise RuntimeError(
            f"run ended with status={outcome.status}, stop_code={outcome.stop_code}"
        )
    print(f"Run directory: {outcome.run_directory}")


def main(argv: list[str] | None = None) -> None:
    config_path = extract_config_path(argv)
    project_config = load_project_config(config_path)
    defaults = build_parser_defaults(project_config, "single_experiment")
    defaults["config_path"] = config_path
    defaults["output_dir"] = project_config.get("results_dir", "results")
    args = build_parser(defaults).parse_args(argv)
    args.project_config = project_config
    run(args)


if __name__ == "__main__":
    main()
