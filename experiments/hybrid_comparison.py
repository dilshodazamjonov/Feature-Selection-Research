from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from Models.utils import get_selector
from experiments.common import (
    add_common_experiment_args,
    add_llm_args,
    build_experiment_config,
    create_run_layout,
    prepare_shared_data,
    resolve_llm_cache_dir,
)
from experiments.config import (
    apply_feature_budget_to_selector_kwargs,
    build_parser_defaults,
    extract_config_path,
    load_project_config,
    resolve_feature_budget,
)
from feature_selection.hybrid import LLMThenStatSelector
from pipelines.common import run_experiment
from pipelines.comparison import compare_experiment_pair


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a hybrid LLM -> statistical selector experiment and compare it.",
    )
    parser.add_argument(
        "--stat-selector",
        default=defaults["stat_selector"],
        help="Downstream statistical selector to use after the LLM stage.",
    )
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument(
        "--improvement-tolerance",
        type=float,
        default=defaults["improvement_tolerance"],
    )
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
        random_seed=defaults["random_seed"],
        llm_model=defaults["llm_model"],
        llm_max_features=defaults["llm_max_features"],
        llm_ranking_budget=defaults["llm_ranking_budget"],
        llm_shared_ranking_enabled=defaults["llm_shared_ranking_enabled"],
        llm_cache_dir=defaults["llm_cache_dir"],
    )
    return parser


def classify_help(delta_cv_auc: float, delta_oot_auc: float, tolerance: float) -> str:
    if pd.notna(delta_cv_auc) and pd.notna(delta_oot_auc):
        if delta_cv_auc > tolerance and delta_oot_auc > tolerance:
            return "yes"
        if delta_cv_auc < -tolerance and delta_oot_auc < -tolerance:
            return "no"
    return "mixed"


def run(args: argparse.Namespace) -> None:
    stat_selector_name = args.stat_selector.lower()
    if stat_selector_name == "llm":
        raise ValueError("Use a statistical selector for --stat-selector, not 'llm'.")

    stat_selector_cls, stat_selector_kwargs = get_selector(stat_selector_name)
    if stat_selector_cls is None:
        raise ValueError(f"Unsupported hybrid downstream selector: {args.stat_selector}")
    feature_budget = resolve_feature_budget(args.project_config, args.model)
    stat_selector_kwargs = apply_feature_budget_to_selector_kwargs(
        stat_selector_name,
        stat_selector_kwargs,
        feature_budget,
    )

    layout = create_run_layout(
        output_dir=args.output_dir,
        run_label="run_hybrid_comparison",
        manifest_payload={
            "script": "run_hybrid_comparison.py",
            "model": args.model,
            "stat_selector": stat_selector_name,
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
            "improvement_tolerance": args.improvement_tolerance,
        },
        include_feature_overlap_dir=True,
    )

    llm_cache_dir = resolve_llm_cache_dir(layout.run_dir, args.llm_cache_dir)
    prepared_data = prepare_shared_data(args, layout.experiments_dir)

    statistical_run = run_experiment(
        build_experiment_config(
            args=args,
            experiments_dir=layout.experiments_dir,
            experiment_name=stat_selector_name,
            selector_name=stat_selector_name,
        ),
        prepared_data=prepared_data,
    )

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

    hybrid_run = run_experiment(
        build_experiment_config(
            args=args,
            experiments_dir=layout.experiments_dir,
            experiment_name=f"llm_then_{stat_selector_name}",
            selector_name=f"llm_then_{stat_selector_name}",
            selector_cls=LLMThenStatSelector,
            selector_kwargs={
                "description_csv_path": args.description_path,
                "stat_selector_cls": stat_selector_cls,
                "stat_selector_kwargs": stat_selector_kwargs,
                "cache_dir": llm_cache_dir,
                "llm_model": args.llm_model,
                "llm_max_features": args.llm_ranking_budget,
                "llm_feature_budget": args.project_config.get("feature_budgets", {}).get(args.model, args.llm_max_features),
                "llm_shared_ranking_enabled": args.llm_shared_ranking_enabled,
                "llm_selector_kwargs": {
                    "max_missing_rate": 0.95,
                },
                "iv_filter_kwargs": {
                    "min_iv": 0.01,
                    "max_iv_for_leakage": 0.5,
                    "encode": True,
                    "n_jobs": 1,
                    "verbose": False,
                },
            },
        ),
        prepared_data=prepared_data,
    )

    experiment_summary_df = pd.DataFrame(
        [statistical_run.summary, llm_run.summary, hybrid_run.summary]
    )
    experiment_summary_df.to_csv(layout.run_dir / "experiment_summaries.csv", index=False)

    stat_vs_hybrid_row, stat_vs_hybrid_overlap = compare_experiment_pair(
        left_label=statistical_run.config.experiment_name,
        left_exp_dir=statistical_run.exp_dir,
        left_model_name=statistical_run.config.model_name,
        left_selector_name=statistical_run.config.selector_name,
        right_label=hybrid_run.config.experiment_name,
        right_exp_dir=hybrid_run.exp_dir,
        right_model_name=hybrid_run.config.model_name,
        right_selector_name=hybrid_run.config.selector_name,
    )
    llm_vs_hybrid_row, llm_vs_hybrid_overlap = compare_experiment_pair(
        left_label=llm_run.config.experiment_name,
        left_exp_dir=llm_run.exp_dir,
        left_model_name=llm_run.config.model_name,
        left_selector_name=llm_run.config.selector_name,
        right_label=hybrid_run.config.experiment_name,
        right_exp_dir=hybrid_run.exp_dir,
        right_model_name=hybrid_run.config.model_name,
        right_selector_name=hybrid_run.config.selector_name,
    )

    if layout.feature_overlap_dir is not None:
        stat_vs_hybrid_overlap.to_csv(
            layout.feature_overlap_dir / f"hybrid_vs_{stat_selector_name}_feature_overlap.csv",
            index=False,
        )
        llm_vs_hybrid_overlap.to_csv(
            layout.feature_overlap_dir / "hybrid_vs_llm_feature_overlap.csv",
            index=False,
        )

    comparison_df = pd.DataFrame([stat_vs_hybrid_row, llm_vs_hybrid_row])
    comparison_df.to_csv(layout.run_dir / "hybrid_comparison_summary.csv", index=False)

    verdict_rows = []
    for row in [stat_vs_hybrid_row, llm_vs_hybrid_row]:
        delta_cv_auc = row.get("delta_cv_auc_mean_right_minus_left", np.nan)
        delta_oot_auc = row.get("delta_oot_auc_right_minus_left", np.nan)
        verdict_rows.append(
            {
                "baseline_method": row["left_method"],
                "hybrid_method": row["right_method"],
                "delta_cv_auc_mean": delta_cv_auc,
                "delta_oot_auc": delta_oot_auc,
                "delta_cv_jaccard_similarity_mean": row.get(
                    "delta_cv_jaccard_similarity_mean_right_minus_left",
                    np.nan,
                ),
                "overall_verdict": classify_help(
                    delta_cv_auc=delta_cv_auc,
                    delta_oot_auc=delta_oot_auc,
                    tolerance=args.improvement_tolerance,
                ),
            }
        )

    pd.DataFrame(verdict_rows).to_csv(layout.run_dir / "hybrid_help_verdict.csv", index=False)
    print(f"Run directory: {layout.run_dir}")


def main(argv: list[str] | None = None) -> None:
    config_path = extract_config_path(argv)
    project_config = load_project_config(config_path)
    defaults = build_parser_defaults(project_config, "hybrid_comparison")
    defaults["config_path"] = config_path
    args = build_parser(defaults).parse_args(argv)
    args.project_config = project_config
    run(args)


if __name__ == "__main__":
    main()
