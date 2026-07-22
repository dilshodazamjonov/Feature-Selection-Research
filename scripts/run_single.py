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
from credit_risk_fs.experiments.single_run import main as single_run_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one selector/model configuration.")
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub", "lendingclub_v2"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--execution-policy",
        default="configs/execution/local_laptop_safe_v1.yaml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--allow-gpu-without-telemetry", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config) if args.config else experiment_config_path(args.dataset, "matrix")
    cli_args = ["--config", str(config_path), "--model", args.model, "--selector", args.selector]
    output_dir = args.output_dir or str(PROJECT_ROOT / "results")
    cli_args.extend(
        ["--output-dir", output_dir, "--repository-root", str(PROJECT_ROOT)]
    )
    cli_args.extend(["--execution-policy", args.execution_policy, "--accelerator", args.accelerator])
    if args.resume:
        cli_args.extend(["--resume", args.resume])
    if args.allow_gpu_without_telemetry:
        cli_args.append("--allow-gpu-without-telemetry")
    single_run_main(cli_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
