from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.clip.evaluation_aggregation import (  # noqa: E402
    aggregate_status,
    atomic_write_aggregates,
    build_all_aggregates,
)


OUTPUT_ROOT = Path("results/clip/final_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild CLIP final-evaluation aggregate tables from completed runs.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and report aggregate rebuild without writing.")
    mode.add_argument("--execute", action="store_true", help="Write rebuilt aggregate tables atomically.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before = aggregate_status(args.output_root)
    aggregates = build_all_aggregates(args.output_root)
    evaluation = aggregates["evaluation_summary.csv"]
    payload = {
        "mode": "execute" if args.execute else "dry_run",
        "production_writes": bool(args.execute),
        "completed_runs_discovered": int(len(evaluation)),
        "evaluation_summary_rows": int(len(evaluation)),
        "run_manifest_entries": int(len(aggregates["run_manifest.json"]["runs"])),
        "selected_feature_run_coverage": int(aggregates["selected_feature_summary.csv"]["run_id"].nunique()),
        "semantic_summary_run_coverage": int(aggregates["semantic_coverage_summary.csv"]["run_id"].nunique()),
        "redundancy_summary_run_coverage": int(aggregates["redundancy_summary.csv"]["run_id"].nunique()),
        "runtime_summary_run_coverage": int(aggregates["runtime_summary.csv"]["run_id"].nunique()),
        "score_psi_run_coverage": int(aggregates["score_psi_summary.csv"]["run_id"].nunique()),
        "comparison_clip_rows": int(aggregates["comparison_with_frozen_baselines.csv"]["result_origin"].eq("clip_extension").sum()),
        "significance_rows": int(len(aggregates["statistical_significance_summary.csv"])),
        "significance_status_counts": aggregates["statistical_significance_summary.csv"]["status"].value_counts(dropna=False).to_dict(),
        "aggregate_status_before": before,
    }
    if args.execute:
        atomic_write_aggregates(args.output_root, aggregates)
        payload["aggregate_status_after"] = aggregate_status(args.output_root)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
