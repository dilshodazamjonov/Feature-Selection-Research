import pandas as pd

from evaluation.plotting import (
    generate_experiment_plots,
    generate_matrix_comparison_plots,
    load_plot_data,
)


def test_plotting_creates_required_tables_and_pngs(tmp_path):
    run_dir = tmp_path / "results" / "lr" / "statistical" / "run_a"
    (run_dir / "results").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "fold": 1,
                "val_time_start": -300,
                "val_time_end": -270,
                "gini": 0.30,
                "psi_model": 0.05,
                "lift_at_10": 2.0,
            },
            {
                "fold": 2,
                "val_time_start": -270,
                "val_time_end": -240,
                "gini": 0.34,
                "psi_model": 0.07,
                "lift_at_10": 2.2,
            },
        ]
    ).to_csv(run_dir / "results" / "cv_results.csv", index=False)
    pd.DataFrame([{"gini": 0.35, "auc": 0.675, "ks": 0.28, "lift_at_10": 2.1}]).to_csv(
        run_dir / "results" / "oot_test_results.csv",
        index=False,
    )

    plot_data = [load_plot_data(run_dir, label="lr_mrmr")]
    out_dir = tmp_path / "plots"
    generate_experiment_plots(experiments=plot_data, output_dir=out_dir)
    comparison_df = pd.DataFrame(
        [
            {
                "model": "lr",
                "selector": "mrmr",
                "experiment_type": "statistical",
                "oot_gini": 0.35,
                "nogueira_stability": 0.8,
                "selected_feature_count": 20,
                "selected_feature_psi_mean": 0.04,
                "model_score_psi": 0.05,
                "lift_at_10": 2.1,
            }
        ]
    )
    generate_matrix_comparison_plots(
        comparison_df=comparison_df,
        experiments=plot_data,
        output_dir=out_dir,
    )

    for name in [
        "oot_metric_table.csv",
        "stability_metric_table.csv",
        "monthly_metric_table.csv",
        "oot_performance_comparison.png",
        "stability_comparison.png",
        "performance_vs_stability.png",
        "feature_count_vs_gini.png",
        "selected_feature_psi_comparison.png",
        "model_score_psi_comparison.png",
        "lift_at_10_comparison.png",
        "monthly_gini_trend.png",
        "monthly_psi_trend.png",
        "monthly_lift_trend.png",
    ]:
        assert (out_dir / name).exists()
