from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.model import SemanticStatisticalContrastiveEncoder, count_trainable_parameters  # noqa: E402
from credit_risk_fs.clip.trainer import train_seed  # noqa: E402
from credit_risk_fs.clip.training_validation import load_and_validate_training_inputs, load_training_config  # noqa: E402
from credit_risk_fs.clip.v2_validation import validate_clip_v2_config, validate_no_v1_output_paths  # noqa: E402
from credit_risk_fs.experiments.config import _parse_simple_yaml  # noqa: E402
from credit_risk_fs.utils.io import read_json  # noqa: E402
from scripts.train_clip_encoder import _write_full_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CLIP-v2 encoder. Requires --execute for training.")
    parser.add_argument("--config", default="configs/clip_v2/training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
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
    config = load_training_config(config_path)
    if args.dry_run or not args.execute:
        model = SemanticStatisticalContrastiveEncoder(config.model)
        prerequisites = [
            config.tensor_schema_path,
            config.train_pairs_path,
            config.validation_pairs_path,
            config.external_pairs_path,
            config.homecredit_statistical_vectors_path,
            config.lendingclub_v2_statistical_vectors_path,
        ]
        print(
            json.dumps(
                {
                    "status": "planned",
                    "dry_run": True,
                    "requires_execute_for_training": True,
                    "statistical_input_dimension": config.model.statistical_input_dim,
                    "seeds": list(config.seeds),
                    "parameter_count": count_trainable_parameters(model),
                    "missing_prerequisites": [str(path) for path in prerequisites if not path.exists()],
                    "model_trained": False,
                    "checkpoint_created": False,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if not (args.smoke_test or args.all_seeds):
        print("Specify --smoke-test or --all-seeds with --execute.", file=sys.stderr)
        return 2
    if args.all_seeds and not args.execute:
        print("--all-seeds requires --execute.", file=sys.stderr)
        return 2
    data = load_and_validate_training_inputs(config)
    config_text = config_path.read_text(encoding="utf-8")
    seeds = [int(args.seed)] if args.smoke_test else list(config.seeds)
    if args.smoke_test and args.seed is None:
        print("--smoke-test requires --seed.", file=sys.stderr)
        return 2
    results = []
    for seed in seeds:
        result = train_seed(
            config=config,
            data=data,
            seed=seed,
            output_dir=config.output_dir / ("smoke_test" if args.smoke_test else ""),
            config_snapshot_text=config_text,
            smoke_test=bool(args.smoke_test),
        )
        results.append(result)
        print(f"CLIP-v2 seed {seed} complete: {result.checkpoint_path}")
    if args.all_seeds:
        _write_full_outputs(config=config, data=data, results=results)
        _copy_selected_checkpoint(config.output_dir)
    return 0


def _copy_selected_checkpoint(output_dir: Path) -> None:
    selection = read_json(output_dir / "model_selection_manifest.json")
    selected_checkpoint = Path(str(selection["selected_checkpoint_path"]))
    selected_manifest = selected_checkpoint.parent / "checkpoint_manifest.json"
    shutil.copy2(selected_checkpoint, output_dir / "selected_checkpoint.pt")
    shutil.copy2(selected_manifest, output_dir / "selected_checkpoint_manifest.json")


if __name__ == "__main__":
    raise SystemExit(main())
