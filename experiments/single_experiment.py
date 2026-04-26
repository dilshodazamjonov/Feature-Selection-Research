from __future__ import annotations

import argparse

from experiments.common import add_common_experiment_args, build_experiment_config, create_run_layout
from experiments.config import build_parser_defaults, extract_config_path, load_project_config
from pipelines.common import prepare_modeling_data, run_experiment


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
    layout = create_run_layout(
        output_dir=args.output_dir,
        run_label="run_single_experiment",
        manifest_payload={
            "script": "main.py",
            "model": args.model,
            "selector": args.selector,
            "data_dir": args.data_dir,
            "description_path": args.description_path,
            "n_splits": args.n_splits,
            "dev_start_day": args.dev_start_day,
            "oot_start_day": args.oot_start_day,
            "oot_end_day": args.oot_end_day,
            "cv_gap_groups": args.cv_gap_groups,
        },
    )

    config = build_experiment_config(
        args=args,
        experiments_dir=layout.experiments_dir,
        experiment_name=args.selector.lower(),
        selector_name=args.selector.lower(),
    )
    prepared_data = prepare_modeling_data(config)
    run_experiment(config, prepared_data=prepared_data)
    print(f"Run directory: {layout.run_dir}")


def main(argv: list[str] | None = None) -> None:
    config_path = extract_config_path(argv)
    project_config = load_project_config(config_path)
    defaults = build_parser_defaults(project_config, "single_experiment")
    defaults["config_path"] = config_path
    args = build_parser(defaults).parse_args(argv)
    args.project_config = project_config
    run(args)


if __name__ == "__main__":
    main()
