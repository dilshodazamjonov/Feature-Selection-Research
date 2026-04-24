from __future__ import annotations

import argparse

import pandas as pd

from experiments.common import (
    add_common_experiment_args,
    add_llm_args,
    build_experiment_config,
    create_run_layout,
    prepare_shared_data,
    resolve_llm_cache_dir,
)
from experiments.config import build_parser_defaults, extract_config_path, load_project_config
from pipelines.common import run_experiment
from pipelines.comparison import compare_experiment_pair


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM vs statistical selector comparisons.",
    )
    parser.add_argument(
        "--stat-selectors",
        nargs="+",
        default=defaults["stat_selectors"],
        help="Statistical selectors to compare against the LLM selector.",
    )
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    add_common_experiment_args(parser)
    add_llm_args(parser)
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
        llm_model=defaults["llm_model"],
        llm_max_features=defaults["llm_max_features"],
        llm_cache_dir=defaults["llm_cache_dir"],
    )
    return parser


def run(args: argparse.Namespace) -> None:
    layout = create_run_layout(
        output_dir=args.output_dir,
        run_label="run_llm_vs_statistical",
        manifest_payload={
            "script": "run_llm_vs_statistical.py",
            "model": args.model,
            "stat_selectors": args.stat_selectors,
            "llm_model": args.llm_model,
            "llm_max_features": args.llm_max_features,
            "llm_cache_dir": args.llm_cache_dir,
            "data_dir": args.data_dir,
            "description_path": args.description_path,
            "n_splits": args.n_splits,
            "dev_start_day": args.dev_start_day,
            "oot_start_day": args.oot_start_day,
            "oot_end_day": args.oot_end_day,
            "cv_gap_groups": args.cv_gap_groups,
        },
        include_feature_overlap_dir=True,
    )

    llm_cache_dir = resolve_llm_cache_dir(layout.run_dir, args.llm_cache_dir)
    prepared_data = prepare_shared_data(args, layout.experiments_dir)

    llm_run = run_experiment(
        build_experiment_config(
            args=args,
            experiments_dir=layout.experiments_dir,
            experiment_name="llm",
            selector_name="llm",
            selector_kwargs={
                "model": args.llm_model,
                "max_features": args.llm_max_features,
                "cache_dir": llm_cache_dir,
            },
        ),
        prepared_data=prepared_data,
    )

    stat_runs = [
        run_experiment(
            build_experiment_config(
                args=args,
                experiments_dir=layout.experiments_dir,
                experiment_name=selector_name.lower(),
                selector_name=selector_name.lower(),
            ),
            prepared_data=prepared_data,
        )
        for selector_name in args.stat_selectors
    ]

    experiment_summary_df = pd.DataFrame([llm_run.summary, *[run.summary for run in stat_runs]])
    experiment_summary_df.to_csv(layout.run_dir / "experiment_summaries.csv", index=False)

    comparison_rows = []
    for stat_run in stat_runs:
        comparison_row, overlap_df = compare_experiment_pair(
            left_label=stat_run.config.experiment_name,
            left_exp_dir=stat_run.exp_dir,
            left_model_name=stat_run.config.model_name,
            left_selector_name=stat_run.config.selector_name,
            right_label=llm_run.config.experiment_name,
            right_exp_dir=llm_run.exp_dir,
            right_model_name=llm_run.config.model_name,
            right_selector_name=llm_run.config.selector_name,
        )
        if layout.feature_overlap_dir is not None:
            overlap_df.to_csv(
                layout.feature_overlap_dir
                / f"llm_vs_{stat_run.config.experiment_name}_feature_overlap.csv",
                index=False,
            )
        comparison_rows.append(comparison_row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "delta_oot_auc_right_minus_left",
        ascending=False,
        na_position="last",
    )
    comparison_df.to_csv(layout.run_dir / "llm_vs_statistical_summary.csv", index=False)
    print(f"Run directory: {layout.run_dir}")


def main(argv: list[str] | None = None) -> None:
    config_path = extract_config_path(argv)
    project_config = load_project_config(config_path)
    defaults = build_parser_defaults(project_config, "llm_vs_statistical")
    defaults["config_path"] = config_path
    args = build_parser(defaults).parse_args(argv)
    args.project_config = project_config
    run(args)


if __name__ == "__main__":
    main()
