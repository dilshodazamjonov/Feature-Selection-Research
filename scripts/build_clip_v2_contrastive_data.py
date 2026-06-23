from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.pair_builder import build_contrastive_data, load_contrastive_data_config  # noqa: E402
from credit_risk_fs.clip.v2_validation import validate_clip_v2_config, validate_no_v1_output_paths  # noqa: E402
from credit_risk_fs.experiments.config import _parse_simple_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP-v2 contrastive-pair artifacts.")
    parser.add_argument("--config", default="configs/clip_v2/contrastive_data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    errors = validate_clip_v2_config(raw)
    validate_no_v1_output_paths([str(raw.get("output_dir", "")), str(raw.get("output_root", ""))])
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    if args.status:
        print(json.dumps(_status(raw), indent=2, default=str))
        return 0
    if args.dry_run or not args.execute:
        print(json.dumps(_plan(raw), indent=2, default=str))
        return 0
    try:
        config = load_contrastive_data_config(config_path)
        result = build_contrastive_data(config=config, dry_run=False)
    except Exception as exc:
        print(f"CLIP-v2 contrastive data build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", **result.summary}, indent=2, default=str))
    return 0


def _plan(config: dict) -> dict:
    output_dir = Path(str(config.get("output_dir", "results/clip_v2/contrastive_data")))
    required = [
        config.get("homecredit_text_embeddings_path"),
        config.get("lendingclub_v2_text_embeddings_path"),
        config.get("homecredit_statistical_vectors_path"),
        config.get("lendingclub_v2_statistical_vectors_path"),
        config.get("statistical_preprocessor_path"),
    ]
    return {
        "status": "planned",
        "dry_run": True,
        "output_dir": str(output_dir).replace("\\", "/"),
        "missing_prerequisites": [str(path) for path in required if path and not Path(str(path)).exists()],
        "statistical_input_dimension": int(config["statistical_input_dimension"]),
        "model_trained": False,
        "optimizer_created": False,
        "requires_execute_for_full_build": True,
    }


def _status(config: dict) -> dict:
    output_dir = Path(str(config.get("output_dir", "results/clip_v2/contrastive_data")))
    expected = [
        "contrastive_tensor_schema.json",
        "contrastive_pair_manifest.json",
        "homecredit_train_positive_pairs.parquet",
        "homecredit_validation_positive_pairs.parquet",
        "lendingclub_v2_external_pairs.parquet",
    ]
    return {"status": "complete" if all((output_dir / name).exists() for name in expected) else "incomplete", "files": {name: (output_dir / name).exists() for name in expected}}


if __name__ == "__main__":
    raise SystemExit(main())
