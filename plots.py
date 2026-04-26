from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from evaluation.plotting import (
    generate_experiment_plots,
    generate_matrix_comparison_plots,
    load_plot_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _experiment_arg(value: str) -> tuple[str, str | None]:
    if "=" in value:
        label, path = value.split("=", 1)
        return path, label
    return value, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparison and stability plots from finished experiment folders.",
    )
    parser.add_argument(
        "--all",
        dest="all_results_dir",
        default=None,
        metavar="RESULTS_DIR",
        help="Discover and plot all completed experiment runs under a results directory.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "Run directory that contains an 'experiments/' folder. "
            "All experiment subdirectories inside it will be plotted."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Experiment directory to plot. Use either a raw path or "
            "'label=path' to control the legend label. Repeat this argument for multiple runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the generated plots and summary tables will be saved.",
    )
    return parser


def _manifest_label(exp_dir: Path) -> str | None:
    manifest_path = exp_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    model = manifest.get("model")
    selector = manifest.get("selector")
    experiment_type = manifest.get("experiment_type")
    if model and selector and experiment_type:
        return f"{model}_{experiment_type}_{selector}"
    return manifest.get("run_id")


def _discover_experiments(results_dir: str | Path) -> list[tuple[Path, str | None]]:
    root = Path(results_dir)
    discovered: list[tuple[Path, str | None]] = []
    for cv_results_path in sorted(root.rglob("cv_results.csv")):
        if "plot_reports" in cv_results_path.parts:
            continue
        exp_dir = cv_results_path.parents[1]
        if not (exp_dir / "results" / "oot_test_results.csv").exists():
            continue
        discovered.append((exp_dir, _manifest_label(exp_dir)))
    return discovered


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.experiment and not args.run_dir and not args.all_results_dir:
        raise ValueError("Provide at least one --experiment, --run-dir, or --all.")

    plot_inputs = []
    if args.all_results_dir:
        for exp_dir, label in _discover_experiments(args.all_results_dir):
            plot_inputs.append(load_plot_data(exp_dir, label=label))

    for run_dir in args.run_dir:
        experiments_root = Path(run_dir) / "experiments"
        if experiments_root.exists():
            for exp_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
                plot_inputs.append(load_plot_data(exp_dir))
        elif (Path(run_dir) / "results" / "cv_results.csv").exists():
            plot_inputs.append(load_plot_data(run_dir, label=_manifest_label(Path(run_dir))))
        else:
            raise ValueError(f"No experiments directory found under: {run_dir}")

    for item in args.experiment:
        exp_path, label = _experiment_arg(item)
        plot_inputs.append(load_plot_data(exp_path, label=label))

    if not plot_inputs:
        raise ValueError("No experiment directories were resolved for plotting.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            str(Path(args.all_results_dir) / "plot_reports" / "all")
            if args.all_results_dir
            else "outputs/plot_reports"
        )

    results = generate_experiment_plots(
        experiments=plot_inputs,
        output_dir=output_dir,
    )
    if args.all_results_dir:
        comparison_path = Path(args.all_results_dir) / "final_comparison_table.csv"
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            results.update(
                generate_matrix_comparison_plots(
                    comparison_df=comparison_df,
                    experiments=plot_inputs,
                    output_dir=output_dir,
                )
            )

    generated = [name for name, saved in results.items() if saved]
    print(f"Plot output directory: {Path(output_dir).resolve()}")
    print(f"Generated plots: {', '.join(generated) if generated else 'none'}")


if __name__ == "__main__":
    main()
