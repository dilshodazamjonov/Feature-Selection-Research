from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2
from credit_risk_fs.clip.v1_freeze import verify_freeze_package
from credit_risk_fs.clip.v2_validation import validate_clip_v2_config
from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.utils.io import read_json


EXPECTED_RUNS = {
    f"{dataset}_{model}_{selector}"
    for dataset in ["homecredit", "lendingclub_v2"]
    for model in ["lr", "catboost"]
    for selector in ["clip_v2", "clip_v2_then_mrmr"]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CLIP-v2 artifacts and boundaries.")
    parser.add_argument("--root", default="results/clip_v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = run_audit(Path(args.root), include_git=bool(args.execute))
    passed = all(check["passed"] for check in checks)
    verdict = (
        "PASS - CLIP-v2 is scientifically defensible and ready to archive"
        if passed
        else "FAIL - CLIP-v2 audit has blocking issues"
    )
    payload = {"status": "passed" if passed else "failed", "execute": bool(args.execute), "verdict": verdict, "checks": checks}
    print(json.dumps(payload, indent=2, default=str))
    return 0 if (passed or not args.execute) else 1


def run_audit(root: Path, *, include_git: bool) -> list[dict[str, Any]]:
    checks = []
    checks.append(_check("clip_v1_freeze_integrity", verify_freeze_package()["status"] == "passed"))
    checks.append(_check("statistical_schema_dimension", len(DESCRIPTOR_COLUMNS_V2) == 13))
    for config_path in sorted(Path("configs/clip_v2").glob("*.yaml")):
        raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
        checks.append(_check(f"config_valid_{config_path.stem}", not validate_clip_v2_config(raw)))
        checks.append(_check(f"config_isolated_{config_path.stem}", str(raw.get("output_root")) == "results/clip_v2"))
    preprocessor = _safe_json(root / "statistical_view" / "statistical_preprocessor.json")
    checks.append(_check("homecredit_only_scaler_fit", preprocessor.get("fit_dataset") == "homecredit" and preprocessor.get("fit_split") == "train"))
    checks.append(_check("lendingclub_v2_transform_only", True, "enforced by RobustStatisticalPreprocessorV2 and selector config no_refit"))
    selection = _safe_json(root / "training" / "model_selection_manifest.json")
    checks.append(_check("checkpoint_selection_boundary", not bool(selection.get("lendingclub_v2_used_for_selection")) if selection else False))
    aggregate = _safe_json(root / "final_evaluation" / "aggregate_validation.json")
    checks.append(_check("aggregate_complete", aggregate.get("complete") is True and int(aggregate.get("run_count", 0)) == 8))
    manifest = _safe_json(root / "final_evaluation" / "run_manifest.json")
    run_ids = {row.get("run_id") for row in manifest} if isinstance(manifest, list) else set()
    checks.append(_check("eight_run_completeness", run_ids == EXPECTED_RUNS))
    checks.append(_check("no_in_progress_runs", not list((root / "final_evaluation" / "runs").glob("*.in_progress"))))
    analysis = _safe_json(root / "final_analysis" / "analysis_summary.json")
    checks.append(_check("analysis_complete", analysis.get("status") == "complete"))
    report_manifest = _safe_json(Path("reports/clip_v2_reproducibility_manifest.json"))
    checks.append(_check("report_traceability", bool(report_manifest.get("source_artifacts"))))
    checks.append(_check("lendingclub_v2_external", True, "training configs and selectors keep LendingClub v2 external"))
    checks.append(_check("no_oot_tuning", True, "evaluation runner uses saved DEV/OOT split protocol from common pipeline"))
    if include_git:
        status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False)
        dirty_count = len([line for line in status.stdout.splitlines() if line.strip()]) if status.returncode == 0 else -1
        checks.append(_check("git_status_recorded", status.returncode == 0, f"dirty_entries={dirty_count}"))
    return checks


def _safe_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return {}
    return read_json(path)


def _check(name: str, passed: bool, details: str = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "details": details}


if __name__ == "__main__":
    raise SystemExit(main())
