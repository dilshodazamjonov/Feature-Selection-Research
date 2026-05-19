from __future__ import annotations
from scripts.aggregate_results import main as aggregate_results_main
from scripts.check_setup import main as check_setup_main
from scripts.make_plots import main as make_plots_main
from scripts.prepare_homecredit import main as prepare_homecredit_main
from scripts.prepare_lendingclub import main as prepare_lendingclub_main
from scripts.run_matrix import main as run_matrix_main
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


DATASETS = ("homecredit", "lendingclub")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single entrypoint for the end-to-end research workflow.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Run one dataset or both datasets in sequence.",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Optional model subset for matrix runs.")
    parser.add_argument("--force", action="store_true", help="Force rerun completed experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Only schedule the matrix; skip aggregation and plots.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation.")
    parser.add_argument("--skip-check", action="store_true", help="Skip setup validation.")
    parser.add_argument("--skip-matrix", action="store_true", help="Skip matrix execution.")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip result aggregation.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--lendingclub-raw-file",
        default=None,
        help="Optional raw LendingClub accepted-loans CSV path for preparation.",
    )
    return parser


def _selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DATASETS)
    return [dataset_arg]


def _run_step(label: str, func, argv: list[str] | None = None) -> None:
    print(f"[STEP] {label}")
    exit_code = func(argv) if argv is not None else func()
    if exit_code not in (None, 0):
        raise SystemExit(exit_code)


def _prepare_dataset(dataset_name: str, lendingclub_raw_file: str | None) -> None:
    if dataset_name == "homecredit":
        _run_step("prepare homecredit", prepare_homecredit_main)
        return
    args: list[str] = []
    if lendingclub_raw_file:
        args.extend(["--raw-file", lendingclub_raw_file])
    _run_step("prepare lendingclub", prepare_lendingclub_main, args)


def _check_dataset(dataset_name: str) -> None:
    _run_step(f"check setup {dataset_name}", check_setup_main, ["--dataset", dataset_name])


def _run_matrix(dataset_name: str, *, models: list[str] | None, force: bool, dry_run: bool) -> None:
    args = ["--dataset", dataset_name]
    if models:
        args.extend(["--models", *models])
    if force:
        args.append("--force")
    if dry_run:
        args.append("--dry-run")
    _run_step(f"run matrix {dataset_name}", run_matrix_main, args)


def _aggregate_dataset(dataset_name: str) -> None:
    _run_step(f"aggregate results {dataset_name}", aggregate_results_main, ["--dataset", dataset_name])


def _plot_dataset(dataset_name: str) -> None:
    _run_step(f"make plots {dataset_name}", make_plots_main, ["--dataset", dataset_name])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = _selected_datasets(args.dataset)

    for dataset_name in datasets:
        if not args.skip_prepare:
            _prepare_dataset(dataset_name, args.lendingclub_raw_file)

        if not args.skip_check:
            _check_dataset(dataset_name)

        if not args.skip_matrix:
            _run_matrix(
                dataset_name,
                models=args.models,
                force=args.force,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            continue

        if not args.skip_aggregate:
            _aggregate_dataset(dataset_name)

        if not args.skip_plots:
            _plot_dataset(dataset_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
