from __future__ import annotations

import argparse

import pandas as pd

from experiments.common import (
    add_common_experiment_args,
    build_experiment_config,
    create_run_layout,
    prepare_shared_data,
)
from experiments.config import build_parser_defaults, extract_config_path, load_project_config
from pipelines.common import run_experiment


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run statistical feature-selection experiments and summarize them.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=defaults["selectors"],
        help="Statistical selectors to compare.",
    )
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
    )
    return parser


def run(args: argparse.Namespace) -> None:
    layout = create_run_layout(
        output_dir=args.output_dir,
        run_label="run_statistical_comparison",
        manifest_payload={
            "script": "run_statistical_comparison.py",
            "model": args.model,
            "selectors": args.selectors,
            "data_dir": args.data_dir,
            "description_path": args.description_path,
            "n_splits": args.n_splits,
            "dev_start_day": args.dev_start_day,
            "oot_start_day": args.oot_start_day,
            "oot_end_day": args.oot_end_day,
            "cv_gap_groups": args.cv_gap_groups,
        },
    )

    prepared_data = prepare_shared_data(args, layout.experiments_dir)
    runs = [
        run_experiment(
            build_experiment_config(
                args=args,
                experiments_dir=layout.experiments_dir,
                experiment_name=selector_name.lower(),
                selector_name=selector_name.lower(),
            ),
            prepared_data=prepared_data,
        )
        for selector_name in args.selectors
    ]

    summary_df = pd.DataFrame([run.summary for run in runs]).sort_values(
        ["oot_auc", "cv_auc_mean"],
        ascending=[False, False],
        na_position="last",
    )
    summary_df.to_csv(layout.run_dir / "statistical_methods_summary.csv", index=False)
    print(f"Run directory: {layout.run_dir}")


def main(argv: list[str] | None = None) -> None:
    config_path = extract_config_path(argv)
    project_config = load_project_config(config_path)
    defaults = build_parser_defaults(project_config, "statistical_comparison")
    defaults["config_path"] = config_path
    args = build_parser(defaults).parse_args(argv)
    args.project_config = project_config
    run(args)


if __name__ == "__main__":
    main()
