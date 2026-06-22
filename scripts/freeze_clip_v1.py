from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.v1_freeze import verify_freeze_package, write_freeze_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify the CLIP-v1 missingness-only freeze package.")
    parser.add_argument("--write", action="store_true", help="Write the hash-based freeze package. Does not train or evaluate.")
    parser.add_argument("--verify", action="store_true", help="Verify the existing freeze package.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.write and not args.verify:
        print(json.dumps({"status": "plan", "write": False, "verify": False, "model_training": False}, indent=2))
        return 0
    payload = {}
    if args.write:
        payload["write"] = write_freeze_package()
    if args.verify:
        payload["verify"] = verify_freeze_package()
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("verify", {"status": "passed"}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
