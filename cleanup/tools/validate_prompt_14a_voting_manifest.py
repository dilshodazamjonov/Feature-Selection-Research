"""Run the Prompt 14A voting-package validator without loading any dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    ROOT
    / "src"
    / "credit_risk_fs"
    / "analysis"
    / "voting_inference"
    / "manifest_authentication.py"
)
SPEC = importlib.util.spec_from_file_location("prompt14a_manifest_authentication", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load data-free validator: {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)
reproduce_legacy_blocker = VALIDATOR.reproduce_legacy_blocker
validate_voting_package = VALIDATOR.validate_voting_package


def fixed_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_fixed_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".prompt14a.partial")
    temporary.write_bytes(fixed_json_bytes(payload))
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write-blocker-reproduction", type=Path)
    parser.add_argument("--validate-successor", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    if arguments.validate_successor:
        result = validate_voting_package(root)
    else:
        result = reproduce_legacy_blocker(root)

    if arguments.write_blocker_reproduction is not None:
        output = arguments.write_blocker_reproduction
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_fixed_json(output, result)

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
