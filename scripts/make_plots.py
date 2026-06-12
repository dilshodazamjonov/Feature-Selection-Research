from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.evaluation.plotting import generate_experiment_plots, generate_matrix_comparison_plots, load_plot_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plot bundles from completed experiment runs.")
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub", "lendingclub_v2"], required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def _manifest_label(exp_dir: Path) -> str | None:
    manifest_path = exp_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model")
    selector = manifest.get("selector")
    experiment_type = manifest.get("experiment_type")
    if model and selector and experiment_type:
        return f"{model}_{experiment_type}_{selector}"
    return manifest.get("run_id")


def _discover_experiments(results_dir: Path) -> list[tuple[Path, str | None]]:
    discovered: list[tuple[Path, str | None]] = []
    for cv_results_path in sorted(results_dir.rglob("cv_results.csv")):
        if "plot_reports" in cv_results_path.parts:
            continue
        exp_dir = cv_results_path.parents[1]
        if (exp_dir / "results" / "oot_test_results.csv").exists():
            discovered.append((exp_dir, _manifest_label(exp_dir)))
    return discovered


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results" / args.dataset
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plot_reports" / "all"

    plot_inputs = [load_plot_data(exp_dir, label=label) for exp_dir, label in _discover_experiments(results_dir)]
    if not plot_inputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"No completed experiments found under {results_dir}")
        print(f"Plot output directory: {output_dir.resolve()}")
        return 0

    results = generate_experiment_plots(experiments=plot_inputs, output_dir=output_dir)
    comparison_path = results_dir / "final_comparison_table.csv"
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
    print(f"Plot output directory: {output_dir.resolve()}")
    print(f"Generated plots: {', '.join(generated) if generated else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
