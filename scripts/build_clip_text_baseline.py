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

from credit_risk_fs.clip.text_baseline import build_text_baseline, load_text_baseline_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CLIP text-only semantic baseline artifacts.")
    parser.add_argument("--config", default="configs/clip/text_baseline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_text_baseline_config(args.config)
    try:
        result = build_text_baseline(config=config, dry_run=bool(args.dry_run))
    except Exception as exc:
        print(f"CLIP text baseline failed: {exc}")
        return 1

    summary = result.summary
    print("CLIP text baseline dry-run complete." if args.dry_run else "CLIP text baseline complete.")
    print(f"training dataset: {config.train_dataset}")
    print(f"external-validation dataset: {config.external_validation_dataset}")
    print(f"text template version: {config.text_template_version}")
    print(f"text fields: {', '.join(config.text_fields)}")
    print(f"encoder model: {config.encoder_model_name}")
    print(f"encoder revision: {config.encoder_revision}")
    print(f"encoder loaded: {summary.get('encoder_loaded')}")
    print(f"model trained: {summary.get('model_trained')}")
    print(f"expected/homecredit texts: {summary.get('homecredit_texts', summary.get('expected_embedding_count', {}).get('homecredit'))}")
    print(
        "expected/lendingclub_v2 texts: "
        f"{summary.get('lendingclub_v2_texts', summary.get('expected_embedding_count', {}).get('lendingclub_v2'))}"
    )
    print(f"anchor count: {summary.get('anchor_count')}")
    print(f"group split warnings: {summary.get('group_split', {}).get('warnings', [])}")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")
    if args.dry_run:
        print("Dry-run did not load the encoder and did not generate embeddings.")
    else:
        print("No contrastive model was trained. No selector was integrated into the matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
