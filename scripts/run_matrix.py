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
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub", "lendingclub_v2"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    runner_main(cli_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
