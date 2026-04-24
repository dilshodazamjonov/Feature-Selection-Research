from __future__ import annotations

import argparse
from pathlib import Path
import sys
from evaluation.plotting import generate_experiment_plots, load_plot_data

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
        default="outputs/plot_reports",
        help="Directory where the generated plots and summary tables will be saved.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.experiment and not args.run_dir:
        raise ValueError("Provide at least one --experiment or --run-dir.")

    plot_inputs = []
    for run_dir in args.run_dir:
        experiments_root = Path(run_dir) / "experiments"
        if not experiments_root.exists():
            raise ValueError(f"No experiments directory found under: {run_dir}")
        for exp_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
            plot_inputs.append(load_plot_data(exp_dir))

    for item in args.experiment:
        exp_path, label = _experiment_arg(item)
        plot_inputs.append(load_plot_data(exp_path, label=label))

    if not plot_inputs:
        raise ValueError("No experiment directories were resolved for plotting.")

    results = generate_experiment_plots(
        experiments=plot_inputs,
        output_dir=args.output_dir,
    )

    generated = [name for name, saved in results.items() if saved]
    print(f"Plot output directory: {Path(args.output_dir).resolve()}")
    print(f"Generated plots: {', '.join(generated) if generated else 'none'}")


if __name__ == "__main__":
    main()
