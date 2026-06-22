from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.learned_scoring import build_learned_outputs  # noqa: E402
from credit_risk_fs.clip.model import SemanticStatisticalContrastiveEncoder, count_trainable_parameters  # noqa: E402
from credit_risk_fs.clip.trainer import SeedTrainingResult, train_seed  # noqa: E402
from credit_risk_fs.clip.training_validation import load_and_validate_training_inputs, load_training_config  # noqa: E402
from credit_risk_fs.utils.io import read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-style semantic-statistical contrastive feature encoder.")
    parser.add_argument("--config", default="configs/clip/training.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config_text = config_path.read_text(encoding="utf-8")
    try:
        config = load_training_config(config_path)
        data = load_and_validate_training_inputs(config)
        if args.dry_run:
            return _dry_run(config, data)
        if args.smoke_test:
            if args.seed is None:
                raise RuntimeError("--smoke-test requires --seed")
            result = train_seed(
                config=config,
                data=data,
                seed=int(args.seed),
                output_dir=config.output_dir / "smoke_test",
                config_snapshot_text=config_text,
                smoke_test=True,
            )
            print(f"CLIP encoder smoke test complete for seed {args.seed}.")
            print(f"checkpoint: {result.checkpoint_path}")
            return 0
        if args.all_seeds:
            results = [
                train_seed(
                    config=config,
                    data=data,
                    seed=seed,
                    output_dir=config.output_dir,
                    config_snapshot_text=config_text,
                    smoke_test=False,
                )
                for seed in config.seeds
            ]
            _write_full_outputs(config=config, data=data, results=results)
            print("CLIP encoder multi-seed training complete.")
            print(f"seeds: {', '.join(str(seed) for seed in config.seeds)}")
            print(f"output_dir: {config.output_dir}")
            return 0
        raise RuntimeError("choose one of --dry-run, --smoke-test, or --all-seeds")
    except Exception as exc:
        print(f"CLIP encoder training failed: {exc}", file=sys.stderr)
        return 1


def _dry_run(config, data) -> int:
    model = SemanticStatisticalContrastiveEncoder(config.model)
    parameter_count = count_trainable_parameters(model)
    summary = {
        "dry_run": True,
        "model_instantiated": True,
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "parameter_count": parameter_count,
        "architecture": config.model.__dict__,
        "statistical_view_scope": config.statistical_view_scope,
        "statistical_fields": data.statistical_fields,
        "statistical_view_limitation": "architectural proof of concept: aligns feature semantics primarily with DEV missingness behavior"
        if data.statistical_dim == 1
        else "approved multi-dimensional statistical view",
        "counts": {
            "homecredit_train_pairs": int(len(data.train_pairs)),
            "homecredit_validation_pairs": int(len(data.validation_pairs)),
            "lendingclub_v2_external_pairs": int(len(data.external_pairs)),
        },
        "upstream_hashes": data.upstream_hashes,
        "lendingclub_v2_used_for_training": False,
        "lendingclub_v2_used_for_model_selection": False,
        "downstream_lr_catboost_run": False,
    }
    print("CLIP encoder dry-run complete.")
    print(json.dumps(summary, indent=2, default=str))
    print(f"parameter count: {parameter_count}")
    print(f"statistical view scope: {config.statistical_view_scope}")
    print("optimizer steps: 0")
    print("checkpoint created: False")
    return 0


def _write_full_outputs(*, config, data, results: list[SeedTrainingResult]) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_epoch_frames = []
    collapse_rows = []
    for result in results:
        epoch_frame = pd.read_csv(result.epoch_metrics_path)
        best_row = epoch_frame.loc[epoch_frame["epoch"].eq(result.best_epoch)].iloc[0].to_dict()
        best_epoch_frames.append(pd.DataFrame([best_row]))
        rep = read_json(result.representation_metrics_path)
        for label, payload in rep.items():
            collapse_rows.append({"seed": result.seed, **payload})
        rows.append(
            {
                "seed": result.seed,
                "best_epoch": result.best_epoch,
                "final_epoch": result.final_epoch,
                "early_stopping_epoch": result.early_stopping_epoch,
                "best_validation_loss": result.best_validation_loss,
                "best_validation_mrr": result.best_validation_mrr,
                "checkpoint_hash": result.checkpoint_hash,
                "checkpoint_path": str(result.checkpoint_path).replace("\\", "/"),
                "parameter_count": result.parameter_count,
            }
        )
    seed_comparison = pd.DataFrame(rows).sort_values(["best_validation_loss", "seed"], kind="mergesort").reset_index(drop=True)
    selected = seed_comparison.iloc[0].to_dict()
    training_summary = seed_comparison.copy()
    training_summary["selection_metric"] = config.selection_metric
    training_summary["statistical_view_scope"] = config.statistical_view_scope
    seed_comparison.to_csv(output_dir / "seed_comparison.csv", index=False)
    training_summary.to_csv(output_dir / "training_summary.csv", index=False)
    retrieval = pd.concat(best_epoch_frames, ignore_index=True).sort_values(["validation_loss", "seed"], kind="mergesort")
    retrieval.to_csv(output_dir / "retrieval_metrics.csv", index=False)
    collapse = pd.DataFrame(collapse_rows)
    collapse.to_csv(output_dir / "collapse_audit.csv", index=False)

    selected_seed = int(selected["seed"])
    selected_dir = output_dir / "seeds" / f"seed_{selected_seed}"
    selected_checkpoint = selected_dir / "best_checkpoint.pt"
    selected_manifest = selected_dir / "checkpoint_manifest.json"
    learned = build_learned_outputs(
        config=config,
        data=data,
        selected_checkpoint_path=selected_checkpoint,
        selected_checkpoint_manifest_path=selected_manifest,
        output_dir=output_dir,
    )
    checkpoint_manifest = read_json(selected_manifest)
    model_selection = {
        "selection_rule": "lowest Home Credit validation loss; LendingClub v2 not inspected",
        "selection_metric": config.selection_metric,
        "selected_seed": selected_seed,
        "selected_checkpoint_hash": checkpoint_manifest["checkpoint_sha256"],
        "selected_checkpoint_path": str(selected_checkpoint).replace("\\", "/"),
        "lendingclub_v2_used_for_selection": False,
        "all_seed_results": rows,
    }
    write_json(output_dir / "model_selection_manifest.json", model_selection)
    manifest = {
        "method_name": "CLIP-style semantic-statistical contrastive feature encoder",
        "training_dataset": "homecredit",
        "external_validation_dataset": "lendingclub_v2",
        "seeds": list(config.seeds),
        "architecture": config.model.__dict__,
        "parameter_count": int(seed_comparison["parameter_count"].iloc[0]),
        "statistical_view_scope": config.statistical_view_scope,
        "statistical_fields": data.statistical_fields,
        "statistical_view_limitation": "architectural proof of concept: aligns feature semantics primarily with DEV missingness behavior"
        if data.statistical_dim == 1
        else "approved multi-dimensional statistical view",
        "counts": {
            "homecredit_train_pairs": int(len(data.train_pairs)),
            "homecredit_validation_pairs": int(len(data.validation_pairs)),
            "lendingclub_v2_external_pairs": int(len(data.external_pairs)),
        },
        "upstream_hashes": data.upstream_hashes,
        "negative_policy_hash": read_json(config.negative_policy_manifest_path).get("negative_policy_hash"),
        "downstream_lr_catboost_run": False,
        "llm_called": False,
    }
    write_json(output_dir / "training_manifest.json", manifest)
    aggregate = {
        "training_summary_rows": training_summary.to_dict("records"),
        "validation_loss_mean": float(seed_comparison["best_validation_loss"].mean()),
        "validation_loss_std": float(seed_comparison["best_validation_loss"].std(ddof=0)),
        "validation_mrr_mean": float(seed_comparison["best_validation_mrr"].mean()),
        "validation_mrr_std": float(seed_comparison["best_validation_mrr"].std(ddof=0)),
        "selected_checkpoint_hash": checkpoint_manifest["checkpoint_sha256"],
        "collapse_audit_status": "pass" if not collapse["warnings"].astype(str).str.contains("[a-zA-Z]", regex=True).any() else "warn",
        "learned_outputs": {key: str(value).replace("\\", "/") for key, value in learned.items() if isinstance(value, Path)},
        "baseline_comparison": _baseline_comparison(retrieval, output_dir),
        "lendingclub_v2_used_for_training": False,
        "lendingclub_v2_used_for_model_selection": False,
    }
    write_json(output_dir / "training_summary.json", aggregate)
    write_json(
        output_dir / "representation_audit.json",
        {
            "collapse_audit_path": str(output_dir / "collapse_audit.csv").replace("\\", "/"),
            "selected_seed": selected_seed,
            "selected_checkpoint_hash": checkpoint_manifest["checkpoint_sha256"],
            "learned_anchor": learned["anchor_manifest"],
            "statistical_view_scope": config.statistical_view_scope,
        },
    )


def _baseline_comparison(retrieval: pd.DataFrame, output_dir: Path) -> dict[str, str | float | None]:
    best = retrieval.sort_values(["validation_loss", "seed"], kind="mergesort").iloc[0]
    learned = pd.read_csv(output_dir / "homecredit_learned_scores.csv")
    text_rank = pd.read_csv("results/clip/text_baseline/homecredit_text_only_ranking.csv")
    stat_rank = pd.read_csv("results/clip/statistical_baseline/homecredit_statistical_only_ranking.csv")
    validation_learned = learned[learned["split"].astype(str).eq("validation")][
        ["feature_name", "learned_rank", "learned_similarity"]
    ]
    validation_text = text_rank[["feature_name", "text_rank", "cosine_similarity"]]
    validation_stat = stat_rank[["feature_name", "statistical_rank", "statistical_similarity"]]
    merged = validation_learned.merge(validation_text, on="feature_name", how="inner").merge(
        validation_stat, on="feature_name", how="inner"
    )
    return {
        "comparison_scope": "representation-level diagnostics only; no downstream OOT AUC",
        "validation_retrieval_mrr": float(best["validation_mean_reciprocal_rank"]),
        "validation_positive_minus_negative_margin": float(best["validation_positive_minus_negative_margin"]),
        "learned_vs_text_rank_spearman_validation": float(merged["learned_rank"].corr(merged["text_rank"], method="spearman"))
        if len(merged) > 1
        else None,
        "learned_vs_statistical_rank_spearman_validation": float(
            merged["learned_rank"].corr(merged["statistical_rank"], method="spearman")
        )
        if len(merged) > 1
        else None,
        "frozen_text_baseline_note": "text-only baseline is anchor ranking, not cross-modal retrieval",
        "statistical_baseline_note": "statistical baseline is missingness-only anchor ranking, not cross-modal retrieval",
        "claim": "learned encoder diagnostics are contrastive/alignment metrics, not feature-selection superiority",
    }


if __name__ == "__main__":
    raise SystemExit(main())
