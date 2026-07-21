from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_risk_fs.experiments._common import (
    add_common_experiment_args,
    build_experiment_config,
    create_run_layout,
)
from credit_risk_fs.experiments.config import (
    build_parser_defaults,
    extract_config_path,
    load_project_config,
)
from credit_risk_fs.experiments.result_paths import (
    append_run_index_row,
    repository_relative_path,
    update_run_index_row,
)
from credit_risk_fs.experiments.tracking import (
    build_run_manifest,
    mark_completed,
    materialize_standard_artifacts,
    utc_timestamp,
    write_json,
    write_run_manifest,
)
from credit_risk_fs.pipelines.common import prepare_modeling_data, run_experiment
from credit_risk_fs.utils.logging import run_log_context


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
    layout = create_run_layout(
        repository_root=repository_root,
        results_root=args.output_dir,
        dataset=dataset,
        selector=args.selector,
        model=args.model,
    )
    run_id = layout.run_dir.name
    effective_config = {
        **args.project_config,
        "model_selector": args.model,
        "single_run": {
            "selector": args.selector,
            "n_splits": args.n_splits,
            "dev_start_day": args.dev_start_day,
            "oot_start_day": args.oot_start_day,
            "oot_end_day": args.oot_end_day,
            "cv_gap_groups": args.cv_gap_groups,
        },
    }
    config_path = write_json(layout.run_dir / "config.json", effective_config)
    manifest = build_run_manifest(
        run_id=run_id,
        model=args.model,
        selector=args.selector,
        experiment_type="single",
        config=effective_config,
        data_dir=args.data_dir,
        random_seed=args.random_seed,
        output_folder=layout.run_dir,
        project_root=repository_root,
        status="running",
    )
    manifest["split_protocol"] = "grouped_time_series_cv_with_oot"
    manifest_path = write_run_manifest(layout.run_dir, manifest)
    append_run_index_row(
        layout.results_root,
        {
            "run_id": run_id,
            "dataset": dataset,
            "selector": args.selector,
            "model": args.model,
            "split_protocol": manifest["split_protocol"],
            "seed": args.random_seed,
            "status": "running",
            "started_at_utc": manifest["started_at_utc"],
            "run_directory": repository_relative_path(
                layout.run_dir, repository_root
            ),
            "config_path": repository_relative_path(config_path, repository_root),
            "manifest_path": repository_relative_path(
                manifest_path, repository_root
            ),
        },
    )

    config = build_experiment_config(
        args=args,
        experiments_dir=layout.experiments_dir,
        experiment_name=args.selector.lower(),
        selector_name=args.selector.lower(),
        experiment_output_dir=layout.run_dir,
    )
    try:
        with run_log_context(layout.run_dir / "run.log"):
            prepared_data = prepare_modeling_data(config)
            completed_run = run_experiment(config, prepared_data=prepared_data)
        materialize_standard_artifacts(layout.run_dir)
        resource_usage = json.loads(
            (layout.run_dir / "resource_usage.json").read_text(encoding="utf-8")
        )
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = utc_timestamp()
        manifest["summary"] = completed_run.summary
        write_run_manifest(layout.run_dir, manifest)
        mark_completed(layout.run_dir)
        update_run_index_row(
            layout.results_root,
            run_id,
            {
                "status": "completed",
                "completed_at_utc": manifest["completed_at_utc"],
                "runtime_seconds": resource_usage["timings_seconds"]["total"],
                "peak_ram_mb": resource_usage["peak_ram_mb"],
                "peak_gpu_mb": resource_usage["peak_gpu_mb"],
            },
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_timestamp()
        manifest["error"] = repr(exc)
        write_run_manifest(layout.run_dir, manifest)
        update_run_index_row(
            layout.results_root,
            run_id,
            {"status": "failed", "notes": repr(exc)},
        )
        raise
    print(f"Run directory: {layout.run_dir}")


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
