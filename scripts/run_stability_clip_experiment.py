"""Manual entry point for the frozen Stability CLIP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from credit_risk_fs.clip.stability_experiment import run_stability_clip_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Home Credit Model Stability CLIP experiment.")
    parser.add_argument("--config", required=True, type=Path, help="Frozen experiment JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stability_clip_experiment(args.config)


if __name__ == "__main__":
    main()
