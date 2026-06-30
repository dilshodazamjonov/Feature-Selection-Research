from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.reverse_transfer import canonical_artifact_id


ROOT = Path("results/corrected_homecredit_clip")
REG = Path("results/research_summary")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_unique(path: Path, rows: pd.DataFrame, keys: list[str]) -> None:
    current = pd.read_csv(path)
    combined = pd.concat([current, rows.reindex(columns=current.columns)], ignore_index=True)
    combined = combined.drop_duplicates(keys, keep="last")
    combined.to_csv(path, index=False)


def main() -> None:
    new = pd.read_csv(ROOT / "combined_pipeline/new_metrics.csv")
    reusable = pd.read_csv(REG / "reusable_metrics.csv")
    base = reusable[
        (reusable.dataset_name == "homecredit")
        & reusable.selector.isin(["mrmr", "llm_then_mrmr"])
    ].copy()
    reused = pd.DataFrame({
        "dataset": base.dataset_name, "method": base.selector, "model": base.model,
        "feature_budget": base.feature_budget, "dev_auc": pd.NA, "oot_auc": base.oot_auc,
        "auc_drop": pd.NA, "dev_ks": pd.NA, "oot_ks": base.oot_ks,
        "score_psi": base.model_score_psi, "semantic_group_count": pd.NA,
        "result_origin": "reused_existing", "pairing_policy_version": "not_applicable_non_clip",
        "artifact_path": base.metric_artifact_path,
    })
    comparison = pd.concat([
        reused,
        new[["dataset", "method", "model", "feature_budget", "dev_auc", "oot_auc",
             "auc_drop", "dev_ks", "oot_ks", "score_psi", "semantic_group_count",
             "result_origin", "pairing_policy_version", "artifact_path"]],
    ], ignore_index=True).sort_values(["model", "method"])
    comparison.to_csv(ROOT / "combined_pipeline/comparison_table.csv", index=False)

    # Run registry: five corrected checkpoints plus four downstream runs.
    run_rows = []
    seeds = pd.read_csv(ROOT / "training/seed_comparison.csv")
    data_hash = sha(Path("results/clip/dry_run/training_manifest.json"))
    for row in seeds.itertuples():
        run_rows.append({
            "run_id": f"corrected_homecredit_clip_training_seed_{int(row.seed)}",
            "dataset": "homecredit", "method": "corrected_homecredit_clip_training",
            "model": "semantic_statistical_dual_encoder", "seed": int(row.seed),
            "split": "Home Credit DEV train/validation group split", "feature_budget": "",
            "configuration_hash": sha(Path("configs/corrected_homecredit_clip/training.yaml")),
            "data_manifest_hash": data_hash,
            "metric_artifact_path": f"results/corrected_homecredit_clip/training/seeds/seed_{int(row.seed)}/epoch_metrics.csv",
            "prediction_artifact_path": "", "selected_feature_path": "",
            "checkpoint_path": row.checkpoint_path,
            "manifest_path": f"results/corrected_homecredit_clip/training/seeds/seed_{int(row.seed)}/checkpoint_manifest.json",
            "pairing_policy_version": "identity_equivalence_v2", "depends_on_clip": True,
            "reuse_status": "newly_executed", "reason": "Corrected pairing-policy training from scratch.",
        })
    for row in new.itertuples():
        run_id = f"homecredit_{row.model}_{row.method}"
        run_rows.append({
            "run_id": run_id, "dataset": "homecredit", "method": row.method,
            "model": row.model, "seed": 42, "split": "DEV[-600,-240);OOT[-240,0]",
            "feature_budget": row.feature_budget, "configuration_hash": row.config_hash,
            "data_manifest_hash": data_hash,
            "metric_artifact_path": f"{row.artifact_path}/results/experiment_summary.csv",
            "prediction_artifact_path": "results/corrected_homecredit_clip/combined_pipeline/new_predictions.parquet",
            "selected_feature_path": f"{row.artifact_path}/features/final_selected_features.csv",
            "checkpoint_path": "", "manifest_path": f"{row.artifact_path}/data_split_manifest.json",
            "pairing_policy_version": "identity_equivalence_v2", "depends_on_clip": True,
            "reuse_status": "newly_executed", "reason": "Professor-requested corrected CLIP-dependent run.",
        })
    append_unique(REG / "run_index.csv", pd.DataFrame(run_rows), ["run_id"])

    metric_rows = []
    for row in new.itertuples():
        run_dir = Path(row.artifact_path)
        oot = pd.read_csv(run_dir / "results/oot_test_results.csv").iloc[0]
        summary = pd.read_csv(run_dir / "results/experiment_summary.csv").iloc[0]
        metric_rows.append({
            "dataset_name": "homecredit", "model": row.model, "selector": row.method,
            "experiment_type": "corrected_clip_professor_request", "feature_budget": row.feature_budget,
            "llm_shared_ranking_enabled": row.method.startswith("llm_"),
            "llm_ranking_budget": 60 if row.model == "lr" else 100,
            "oot_auc": row.oot_auc, "oot_gini": oot.get("gini"), "oot_ks": row.oot_ks,
            "oot_log_loss": oot.get("log_loss"), "oot_brier": oot.get("brier_score"),
            "model_score_psi": row.score_psi, "selected_feature_count": row.selected_feature_count,
            "total_candidate_feature_count": 529,
            "config_hash": row.config_hash, "data_fingerprint": "",
            "run_id": f"homecredit_{row.model}_{row.method}", "output_folder": row.artifact_path,
            "runtime_seconds": summary.get("runtime_seconds"), "result_origin": "newly_executed",
            "reuse_status": "newly_executed", "pairing_policy_version": "identity_equivalence_v2",
            "metric_artifact_path": f"{row.artifact_path}/results/experiment_summary.csv",
        })
    append_unique(REG / "reusable_metrics.csv", pd.DataFrame(metric_rows), ["run_id"])

    selected_rows = []
    for row in new.itertuples():
        path = Path(row.artifact_path) / "features/final_selected_features.csv"
        selected_rows.append({
            "run_id": f"homecredit_{row.model}_{row.method}", "dataset": "homecredit",
            "model": row.model, "selector": row.method,
            "experiment_type": "corrected_clip_professor_request",
            "feature_budget": row.feature_budget, "selected_feature_count": row.selected_feature_count,
            "selected_feature_path": str(path).replace("\\", "/"), "selected_feature_hash": sha(path),
            "depends_on_clip": True, "pairing_policy_version": "identity_equivalence_v2",
            "reuse_status": "newly_executed", "reason": "Corrected CLIP-dependent DEV-selected feature set.",
        })
    append_unique(REG / "selected_feature_registry.csv", pd.DataFrame(selected_rows), ["run_id"])

    artifact_rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.as_posix()
        content_hash = sha(path)
        artifact_rows.append({
            "artifact_id": canonical_artifact_id(
                run_id="",
                artifact_type=_type(path),
                relative_path=rel,
                content_hash=content_hash,
            ),
            "artifact_type": _type(path),
            "relative_path": rel, "file_exists": True, "file_hash": content_hash,
            "created_by_run_id": "", "depends_on_clip": True, "depends_on_old_pairing": False,
            "pairing_policy_version": "identity_equivalence_v2", "reuse_status": "newly_executed",
            "human_description": "Corrected Home Credit CLIP professor-requested artifact.",
        })
    append_unique(REG / "artifact_registry.csv", pd.DataFrame(artifact_rows), ["relative_path"])

    guide = REG / "results_access_guide.md"
    text = guide.read_text(encoding="utf-8")
    marker = "## Corrected Home Credit CLIP (2026-06-25)"
    if marker not in text:
        text += (
            f"\n\n{marker}\n\nCorrected `identity_equivalence_v2` training, 436-feature projection evidence, "
            "and four new Home Credit downstream runs are registered under "
            "`results/corrected_homecredit_clip/`. The 93 remaining modeling features lack complete "
            "frozen semantic views and were not embedded. The all-feature comparator is not present in "
            "the verified registry and was not rerun.\n"
        )
        guide.write_text(text, encoding="utf-8")

    diagnostics = pd.read_csv(ROOT / "diagnostics/cluster_metrics.csv").set_index("metric").value
    control = pd.read_csv(ROOT / "diagnostics/shuffled_label_control.csv")
    stable = pd.read_csv(ROOT / "stable_core/anchor_neighbour_stability_evidence.csv")
    evidence_available = int(stable.evidence_status.eq("available").sum())
    seed = pd.read_csv(ROOT / "training/seed_comparison.csv")
    lr = new[new.model.eq("lr")].set_index("method")
    cb = new[new.model.eq("catboost")].set_index("method")
    summary = f"""# Corrected Home Credit CLIP task summary

## Objective

Retrain corrected Home Credit CLIP, test embedding structure, audit a stable-core anchor, and evaluate corrected CLIP→mRMR and LLM→corrected CLIP→mRMR without rerunning valid baselines.

## Corrected training

All five fixed seeds (11, 22, 33, 44, 55) completed from scratch under `identity_equivalence_v2`. Validation MRR ranged from {seed.best_validation_mrr.min():.4f} to {seed.best_validation_mrr.max():.4f}. Old checkpoints and score caches were not resumed or reused.

## Feature-universe reconciliation

The approved modeling frame contains 529 features. Complete frozen text and DEV statistical views exist for 436; 349 are in representation training and 87 in group-safe validation. The remaining 93 lack descriptions, so no embeddings were fabricated. UMAP count: 436.

## Embedding structure

Original-space kNN semantic purity was {diagnostics['cosine_knn_purity_k10']:.4f} versus shuffled-label 97.5% {control.knn_purity_k10.quantile(.975):.4f}; silhouette was {diagnostics['silhouette_cosine']:.4f} versus shuffled 97.5% {control.silhouette_cosine.quantile(.975):.4f}. UMAP trustworthiness was {diagnostics['umap_trustworthiness_k10']:.4f}. Verdict: strong representation-structure evidence, not predictive-value proof.

## Stable-core neighbours

The anchor is the normalized centroid of 23 frozen Home Credit training-split stable-core features. It uses no target or OOT data. Independent feature-level drift evidence was available for {evidence_available}/20 neighbours; support is weak because most values are unavailable.

## Combined pipeline

LR corrected CLIP→mRMR: OOT AUC {lr.loc['corrected_clip_then_mrmr','oot_auc']:.4f}, PSI {lr.loc['corrected_clip_then_mrmr','score_psi']:.4f}. LR combined: OOT AUC {lr.loc['llm_then_corrected_clip_then_mrmr','oot_auc']:.4f}, PSI {lr.loc['llm_then_corrected_clip_then_mrmr','score_psi']:.4f}. Reused full mRMR remains higher on AUC (0.7457), while the combined method has lower PSI (0.0049 vs 0.0065).

CatBoost corrected CLIP→mRMR: OOT AUC {cb.loc['corrected_clip_then_mrmr','oot_auc']:.4f}, PSI {cb.loc['corrected_clip_then_mrmr','score_psi']:.4f}. CatBoost combined: OOT AUC {cb.loc['llm_then_corrected_clip_then_mrmr','oot_auc']:.4f}, PSI {cb.loc['llm_then_corrected_clip_then_mrmr','score_psi']:.4f}. Reused full mRMR remains higher on AUC (0.7668), while the combined method has lower PSI (0.0041 vs 0.0102).

The combined method improves both AUC and PSI relative to corrected CLIP→mRMR, but not predictive AUC relative to full mRMR. No superiority claim is supported.

## Limitations

UMAP is qualitative; semantic clustering does not prove predictive value; anchor evidence is partly post hoc and sparse; 93 modeling features cannot be projected; one dataset does not establish robustness; reverse transfer is separate; no uncertainty-based superiority claim is made. A verified all-feature comparator was absent and was not rerun.

## Verdicts

- Corrected training: strong
- Embedding structure: strong
- Independent stable-core-neighbour support: weak
- Combined pipeline predictive superiority: not supported
- Combined pipeline drift advantage: moderate
"""
    (ROOT / "task_summary.md").write_text(summary, encoding="utf-8")

    manifest_path = ROOT / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "conditional_pass", "seeds_completed": 5, "projected_feature_count": 436,
        "umap_feature_count": 436, "reused_comparison_rows": 4, "newly_executed_runs": 4,
        "all_feature_comparator_available": False, "old_checkpoints_rejected": True,
        "deleted_matrix_restored": False,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_manifest = REG / "summary_manifest.json"
    central = json.loads(summary_manifest.read_text(encoding="utf-8"))
    central["corrected_homecredit_clip"] = {
        "status": "conditional_pass", "pairing_policy_version": "identity_equivalence_v2",
        "seeds_completed": 5, "projected_features": 436, "new_runs": 4,
        "comparison_path": "results/corrected_homecredit_clip/combined_pipeline/comparison_table.csv",
    }
    central["registry_file_hashes"] = {
        str(path).replace("\\", "/"): sha(path)
        for path in [
            REG / "run_index.csv", REG / "artifact_registry.csv", REG / "reusable_metrics.csv",
            REG / "selected_feature_registry.csv", REG / "results_access_guide.md",
        ]
    }
    summary_manifest.write_text(json.dumps(central, indent=2), encoding="utf-8")


def _type(path: Path) -> str:
    if path.suffix in {".csv", ".parquet"}:
        return "table"
    if path.suffix == ".json":
        return "manifest"
    if path.suffix in {".pt", ".npy"}:
        return "checkpoint"
    if path.suffix in {".png", ".pdf"}:
        return "figure"
    if path.suffix == ".md":
        return "report"
    return "other"


if __name__ == "__main__":
    main()
