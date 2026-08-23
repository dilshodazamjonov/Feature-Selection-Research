from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.stability_preparation import (  # noqa: E402
    StabilityPreparationError,
    run_preparation,
)


DEFAULT_CONFIG = (
    "configs/protocols/homecredit_model_stability_2024_v2/"
    "clip_stability_preparation_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the target-free, DEV-only Home Credit Model Stability 2024 "
            "CLIP preparation package. This command does not train CLIP, selectors, or classifiers."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Deliberately rebuild an existing package after moving it to a timestamped backup. "
            "The default is verify-and-reuse or fail closed."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_preparation(
            args.config,
            repo_root=ROOT,
            rebuild=bool(args.rebuild),
            progress=lambda message: print(message, flush=True),
        )
    except StabilityPreparationError as exc:
        print(f"BLOCKED — {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("BLOCKED — preparation interrupted; no partial package was published", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
