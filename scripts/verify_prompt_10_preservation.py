#!/usr/bin/env python
"""Verify Prompt 10 result trees remain byte/hash-identical to the Phase 0 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_risk_fs.experiments.atomic_io import sha256_file, write_json_atomic


def _tree_identity(directory: Path) -> tuple[int, int, str]:
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        rows.append(
            (
                path.relative_to(directory).as_posix(),
                path.stat().st_size,
                sha256_file(path),
            )
        )
    payload = "\n".join(
        "\0".join((relative, str(size), digest))
        for relative, size, digest in rows
    ) + "\n"
    return len(rows), sum(item[1] for item in rows), hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify(repository_root: Path, snapshot_path: Path) -> dict[str, object]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    expected = {str(item[1]): tuple(item[2:]) for item in snapshot["cells"]}
    runs = repository_root / "results/full_baseline_v1/runs"
    observed = {}
    for run_id in expected:
        matches = list(runs.glob(f"*/{run_id}"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one Prompt 10 directory for {run_id}; found {len(matches)}")
        observed[run_id] = _tree_identity(matches[0])
    mismatches = [
        {
            "run_id": run_id,
            "expected": list(expected[run_id]),
            "observed": list(observed[run_id]),
        }
        for run_id in expected
        if observed[run_id] != expected[run_id]
    ]
    marker = repository_root / snapshot["success_marker"]["path"]
    marker_matches = (
        marker.is_file()
        and marker.stat().st_size == int(snapshot["success_marker"]["size_bytes"])
        and sha256_file(marker) == snapshot["success_marker"]["sha256"]
    )
    return {
        "schema_version": "prompt_10_baseline_preservation_verification_v1",
        "status": "byte_hash_identical" if not mismatches and marker_matches else "mismatch",
        "expected_cells": len(expected),
        "identical_cells": len(expected) - len(mismatches),
        "mismatches": mismatches,
        "success_marker_identical": marker_matches,
        "phase_0_aggregate_snapshot_sha256": snapshot["aggregate_snapshot_sha256"],
        "raw_dataset_paths_resolved": False,
        "prompt_10_workload_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--snapshot",
        default="cleanup/audits/prompt_11_selector_combinations/baseline_preservation_snapshot.json",
    )
    parser.add_argument(
        "--output",
        default="cleanup/audits/prompt_11_selector_combinations/baseline_preservation_verification.json",
    )
    args = parser.parse_args(argv)
    root = Path(args.repository_root).resolve()
    snapshot = Path(args.snapshot)
    snapshot = snapshot.resolve() if snapshot.is_absolute() else (root / snapshot).resolve()
    output = Path(args.output)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    result = verify(root, snapshot)
    write_json_atomic(output, result, overwrite=True)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "byte_hash_identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
