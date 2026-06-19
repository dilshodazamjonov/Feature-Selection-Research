from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.contrastive_dataset import ContrastiveFeatureDataset
from credit_risk_fs.clip.pair_builder import build_contrastive_data, load_contrastive_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP contrastive data boundary artifacts without training.")
    parser.add_argument("--config", default="configs/clip/contrastive_data.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_contrastive_data_config(args.config)
        result = build_contrastive_data(config=config, dry_run=bool(args.dry_run))
        if not args.dry_run:
            _smoke_test_datasets(config, result.output_paths)
    except Exception as exc:
        print(f"CLIP contrastive data build failed: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(f"CLIP contrastive data {'dry-run' if args.dry_run else 'complete'}.")
    print(f"text embedding dimension: {summary['text_embedding_dimension']}")
    print(f"statistical vector dimension: {summary['statistical_vector_dimension']}")
    print(f"homecredit train pairs: {summary['homecredit_train_pair_count']}")
    print(f"homecredit validation pairs: {summary['homecredit_validation_pair_count']}")
    print(f"lendingclub_v2 external pairs: {summary['lendingclub_v2_external_pair_count']}")
    print(f"group overlap count: {summary['group_overlap_count']}")
    print(f"base-family overlap count: {summary['base_family_overlap_count']}")
    print(f"model trained: {summary['model_trained']}")
    print(f"optimizer created: {summary['optimizer_created']}")
    print(f"checkpoint created: {summary['checkpoint_created']}")
    for key, path in result.output_paths.items():
        print(f"{key}: {path}")
    if args.dry_run:
        print("Dry-run did not write full-run pair parquet files and did not train anything.")
    return 0


def _smoke_test_datasets(config, paths: dict[str, Path]) -> None:
    train = ContrastiveFeatureDataset(
        pairs_path=paths["homecredit_train_positive_pairs"],
        text_embeddings_path=config.homecredit_text_embeddings_path,
        statistical_vectors_path=config.homecredit_statistical_vectors_path,
        mode="train",
    )
    validation = ContrastiveFeatureDataset(
        pairs_path=paths["homecredit_validation_positive_pairs"],
        text_embeddings_path=config.homecredit_text_embeddings_path,
        statistical_vectors_path=config.homecredit_statistical_vectors_path,
        mode="validation",
    )
    external = ContrastiveFeatureDataset(
        pairs_path=paths["lendingclub_v2_external_pairs"],
        text_embeddings_path=config.lendingclub_v2_text_embeddings_path,
        statistical_vectors_path=config.lendingclub_v2_statistical_vectors_path,
        mode="external",
    )
    for dataset in [train, validation, external]:
        if len(dataset):
            dataset[0]


if __name__ == "__main__":
    raise SystemExit(main())
