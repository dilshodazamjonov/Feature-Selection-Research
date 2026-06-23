from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.v2_validation import validate_clip_v2_config, validate_no_v1_output_paths  # noqa: E402
from credit_risk_fs.clip.selector_adapter import materialize_score_caches, score_coverage  # noqa: E402
from credit_risk_fs.clip.selector_validation import load_clip_selector_config, validate_clip_selector_binding  # noqa: E402
from credit_risk_fs.experiments.config import _parse_simple_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CLIP-v2 selector integration without running final models.")
    parser.add_argument("--config", default="configs/clip_v2/selector.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    errors = validate_clip_v2_config(raw)
    validate_no_v1_output_paths([str(raw.get("output_dir", "")), str(raw.get("output_root", "")), str(raw.get("cache_dir", ""))])
    prerequisites = [
        raw.get("selected_checkpoint_path"),
        raw.get("selected_checkpoint_manifest_path"),
        raw.get("model_selection_manifest_path"),
        raw.get("learned_anchor_manifest_path"),
        raw.get("homecredit_scores_path"),
        raw.get("lendingclub_v2_scores_path"),
    ]
    payload = {
        "status": "planned" if not errors else "failed",
        "dry_run": bool(args.dry_run or not args.execute),
        "errors": errors,
        "missing_prerequisites": [str(path) for path in prerequisites if path and not Path(str(path)).exists()],
        "active_datasets": raw.get("active_datasets", []),
        "legacy_lendingclub_allowed": bool(raw.get("legacy_lendingclub_allowed", True)),
        "lendingclub_v2_refit_allowed": bool(raw.get("external_refit_allowed", True)),
        "model_trained": False,
        "downstream_model_fit": False,
    }
    if args.execute and not errors:
        try:
            config = load_clip_selector_config(config_path)
            binding = validate_clip_selector_binding(config)
            caches = materialize_score_caches(config_path)
            payload.update(
                {
                    "status": "passed",
                    "dry_run": False,
                    "missing_prerequisites": [],
                    "checkpoint_hash": binding["checkpoint_hash"],
                    "anchor_hash": binding["anchor_hash"],
                    "score_cache_paths": {dataset: str(path).replace("\\", "/") for dataset, path in caches.items()},
                    "score_coverage": score_coverage(config_path),
                }
            )
        except Exception as exc:
            payload.update({"status": "failed", "error": str(exc)})
    print(json.dumps(payload, indent=2, default=str))
    return 1 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
