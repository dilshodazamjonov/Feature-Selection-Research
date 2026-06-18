from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.clip.manifest_builder import build_training_manifest, load_training_manifest_config  # noqa: E402
from credit_risk_fs.clip.validation import validate_dataset_roles  # noqa: E402
from scripts.validate_clip_readiness import validate_manifest  # noqa: E402
from credit_risk_fs.clip.training_manifest import load_readiness_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic DEV-only CLIP dry-run training manifest.")
    parser.add_argument("--config", default="configs/clip/training_manifest.yaml")
    parser.add_argument("--readiness-config", default="configs/clip/readiness.yaml")
    parser.add_argument("--train-dataset", default="homecredit")
    parser.add_argument("--external-validation-dataset", default="lendingclub_v2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    readiness_errors = validate_manifest(load_readiness_manifest(Path(args.readiness_config)))
    if readiness_errors:
        print("CLIP readiness validation failed; manifest not built.")
        for error in readiness_errors:
            print(f"- {error}")
        return 1

    role_errors = validate_dataset_roles(args.train_dataset, args.external_validation_dataset)
    if role_errors:
        print("CLIP dataset role validation failed; manifest not built.")
        for error in role_errors:
            print(f"- {error}")
        return 1

    config = load_training_manifest_config(Path(args.config))
    if args.train_dataset != config.train_dataset or args.external_validation_dataset != config.external_validation_dataset:
        print("CLI dataset roles must match configs/clip/training_manifest.yaml for this dry-run boundary.")
        print(f"- config train_dataset={config.train_dataset}, cli train_dataset={args.train_dataset}")
        print(
            "- config external_validation_dataset="
            f"{config.external_validation_dataset}, cli external_validation_dataset={args.external_validation_dataset}"
        )
        return 1

    result = build_training_manifest(
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=True,
    )
    manifest = result.manifest

    print("CLIP dry-run training manifest built.")
    print(f"training dataset: {manifest.train_dataset}")
    print(f"external-validation dataset: {manifest.external_validation_dataset}")
    for dataset in manifest.active_datasets:
        print(f"{dataset} source: {manifest.source_files[dataset]}")
        print(f"{dataset} sha256: {manifest.source_hashes[dataset]}")
        print(
            f"{dataset} rows: allowed={manifest.allowed_row_counts[dataset]}, "
            f"blocked={manifest.blocked_row_counts[dataset]}, total={manifest.source_row_counts[dataset]}"
        )
    print(f"text fields: {len(manifest.text_fields)}")
    print(f"candidate statistical fields: {len(manifest.candidate_statistical_fields)}")
    print(
        "forbidden fields found: "
        f"{sum(len(fields) for fields in manifest.forbidden_fields_detected.values())}"
    )
    print(f"warnings: {len(manifest.validation_warnings)}")
    print(f"errors: {len(manifest.validation_errors)}")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")
    print("No model was trained. No encoder was loaded. No contrastive pairs were created.")

    return 0 if manifest.validation_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
