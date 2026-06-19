from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.statistical_baseline import build_statistical_baseline, load_statistical_baseline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DEV-only CLIP statistical-vector baseline.")
    parser.add_argument("--config", default="configs/clip/statistical_baseline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_statistical_baseline_config(args.config)
        result = build_statistical_baseline(config=config, dry_run=bool(args.dry_run))
    except Exception as exc:
        print(f"CLIP statistical baseline failed: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    mode = "dry-run" if args.dry_run else "complete"
    print(f"CLIP statistical baseline {mode}.")
    print("baseline: DEV-only statistical-vector baseline with a Home Credit training-split stable-core anchor.")
    print(f"training dataset: {config.train_dataset}")
    print(f"external-validation dataset: {config.external_validation_dataset}")
    print(f"main statistical fields: {', '.join(summary.get('main_statistical_fields', []))}")
    print(f"optional ablation fields: {', '.join(summary.get('optional_ablation_fields', []))}")
    expected = summary.get("expected", summary)
    print(f"homecredit vectors: {expected.get('homecredit_vectors')}")
    print(f"lendingclub_v2 vectors: {expected.get('lendingclub_v2_vectors')}")
    print(f"vector dimension: {expected.get('vector_dimension', summary.get('vector_dimension'))}")
    print(f"anchor count: {expected.get('train_split_anchor_count', summary.get('anchor_count'))}")
    print("model trained: False")
    print("contrastive pairs created: False")
    for key, path in result.output_paths.items():
        print(f"{key}: {path}")
    if args.dry_run:
        print("Dry-run wrote only dry-run statistical audit artifacts and did not fit or overwrite the full-run preprocessor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
