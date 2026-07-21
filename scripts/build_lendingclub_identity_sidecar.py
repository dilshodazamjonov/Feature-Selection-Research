"""Build the versioned LendingClub original-ID alignment sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.experiments.lendingclub_identity import (  # noqa: E402
    build_lendingclub_identity_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default="data/lendingclub/raw/accepted_2007_to_2018Q4.csv",
    )
    parser.add_argument(
        "--processed",
        default="data/lendingclub_v2/processed/application_train.csv",
    )
    parser.add_argument(
        "--sidecar",
        default="data/lendingclub_v2/processed/record_identity_v1.csv",
    )
    parser.add_argument(
        "--manifest",
        default="data/lendingclub_v2/processed/record_identity_v1.manifest.json",
    )
    parser.add_argument(
        "--audit",
        default="cleanup/audits/foundation_protocol_freeze/lendingclub_identity_evidence.json",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--overwrite-generated-outputs",
        action="store_true",
        help="Replace only the configured generated sidecar, manifest, and audit outputs.",
    )
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_lendingclub_identity_sidecar(
        raw_path=_resolve(args.raw),
        processed_path=_resolve(args.processed),
        sidecar_path=_resolve(args.sidecar),
        manifest_path=_resolve(args.manifest),
        audit_path=_resolve(args.audit),
        chunk_size=args.chunk_size,
        overwrite_generated_outputs=args.overwrite_generated_outputs,
    )
    print(json.dumps({
        "status": "created_and_validated",
        "identity_type": manifest["identity_type"],
        "retained_row_count": manifest["retained_row_count"],
        "splits": manifest["splits"],
        "resource_usage": manifest["resource_usage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
