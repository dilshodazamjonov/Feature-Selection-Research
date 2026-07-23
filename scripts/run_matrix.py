from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.experiments.config import experiment_config_path  # noqa: E402
from credit_risk_fs.experiments.runner import main as runner_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full research matrix for one dataset.")
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub", "lendingclub_v2"], required=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--execution-policy",
        default="configs/execution/local_laptop_safe_v1.yaml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--allow-gpu-without-telemetry", action="store_true")
    parser.add_argument("--cross-dataset-voting-matrix-dry-expand", default=None)
    parser.add_argument("--voting-pilot-config", default=None)
    parser.add_argument("--lendingclub-memory-refinement-config", default=None)
    parser.add_argument("--capacity-scenario-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cross_dataset_voting_matrix_dry_expand:
        runner_main(
            [
                "--repository-root",
                str(PROJECT_ROOT),
                "--cross-dataset-voting-matrix-dry-expand",
                args.cross_dataset_voting_matrix_dry_expand,
            ]
        )
        return 0
    if args.voting_pilot_config:
        pilot_args = [
            "--repository-root",
            str(PROJECT_ROOT),
            "--voting-pilot-config",
            args.voting_pilot_config,
        ]
        if args.resume:
            pilot_args.extend(["--resume", args.resume])
        runner_main(pilot_args)
        return 0
    if args.lendingclub_memory_refinement_config:
        if not args.capacity_scenario_id:
            raise SystemExit(
                "--capacity-scenario-id is required with "
                "--lendingclub-memory-refinement-config"
            )
        capacity_args = [
            "--repository-root",
            str(PROJECT_ROOT),
            "--lendingclub-memory-refinement-config",
            args.lendingclub_memory_refinement_config,
            "--capacity-scenario-id",
            args.capacity_scenario_id,
        ]
        if args.resume:
            capacity_args.extend(["--resume", args.resume])
        runner_main(capacity_args)
        return 0
    if args.dataset is None:
        raise SystemExit("--dataset is required unless a cross-dataset voting mode is selected")
    config_path = Path(args.config) if args.config else experiment_config_path(args.dataset, "matrix")
    cli_args = [
        "--config",
        str(config_path),
        "--repository-root",
        str(PROJECT_ROOT),
    ]
    if args.models:
        cli_args.extend(["--models", *args.models])
    if args.force:
        cli_args.append("--force")
    if args.dry_run:
        cli_args.append("--dry-run")
    if args.output_dir:
        cli_args.extend(["--output-dir", args.output_dir])
    cli_args.extend(["--execution-policy", args.execution_policy, "--accelerator", args.accelerator])
    if args.resume:
        cli_args.extend(["--resume", args.resume])
    if args.allow_gpu_without_telemetry:
        cli_args.append("--allow-gpu-without-telemetry")
    runner_main(cli_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
