"""Explicit CLI for inspecting or building the frozen third-dataset adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or build the authenticated Home Credit 2024 adapter"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--stage", choices=("inspect", "build"), required=True)
    parser.add_argument("--mode", choices=("fixture", "research"), required=True)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from credit_risk_fs.data.homecredit_model_stability_2024.adapter import (
        build_modeling_matrix,
        inspect_input_inventory,
    )
    from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
        load_adapter_contract,
    )

    contract = load_adapter_contract(args.protocol_lock)
    if args.stage == "inspect":
        inventory = inspect_input_inventory(
            args.input_root, contract, mode=args.mode
        )
        # Inspection writes only to the explicitly supplied output root.
        args.output_root.mkdir(parents=True, exist_ok=False)
        path = args.output_root / "inspection.json"
        from credit_risk_fs.experiments.atomic_io import write_json_atomic

        write_json_atomic(
            path,
            {
                "schema_version": "homecredit_model_stability_2024_inspection_v1",
                "protocol_file_sha256": contract.lock_file_sha256,
                "dataset_id": contract.dataset_id,
                "mode": args.mode,
                "research_status": (
                    "synthetic_fixture_not_research"
                    if args.mode == "fixture"
                    else "research_input_inspection_only"
                ),
                "inventory": inventory.to_dict(),
                "fits": 0,
                "evaluations": 0,
            },
            overwrite=False,
        )
        print(path)
        return 0
    result = build_modeling_matrix(
        input_root=args.input_root,
        output_root=args.output_root,
        contract=contract,
        mode=args.mode,
        shard_rows=args.shard_rows,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
