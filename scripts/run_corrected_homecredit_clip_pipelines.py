from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.experiments.config import load_named_project_config, resolve_model_kwargs
from credit_risk_fs.experiments.tracking import build_data_version
from credit_risk_fs.pipelines.common import ExperimentConfig, prepare_modeling_data, run_experiment
from credit_risk_fs.selectors.fixed_rank_then_mrmr import FixedRankThenMRMRSelector
from credit_risk_fs.utils.hashing import sha256_text


OUT = Path("results/corrected_homecredit_clip/combined_pipeline")
RANKING = OUT / "corrected_consensus_clip_scores.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    llm_source = Path(
        "results/homecredit/catboost/hybrid_mrmr/"
        "catboost_hybrid_llm_then_mrmr_87fbcccf4952/features/llm_rankings_summary.csv"
    )
    llm = pd.read_csv(llm_source)
    llm = llm[llm.scope.astype(str).eq("final_dev")].sort_values("rank")
    approved_paths = {}
    for model, flag in [("lr", "candidate_for_lr_hybrid"), ("catboost", "candidate_for_catboost_hybrid")]:
        approved = llm[llm[flag].astype(bool)][["rank", "feature_name", "llm_reason", "metadata_signature", "prompt_hash"]].copy()
        path = OUT / f"frozen_llm_approved_{model}.csv"
        approved.to_csv(path, index=False)
        approved_paths[model] = path

    project = load_named_project_config("homecredit")
    prepared = None
    rows = []
    prediction_rows = []
    selected_rows = []
    pool_rows = []
    for model in ["lr", "catboost"]:
        budget = 20 if model == "lr" else 40
        pool_size = 60 if model == "lr" else 100
        specs = [
            ("corrected_clip_then_mrmr", None),
            ("llm_then_corrected_clip_then_mrmr", approved_paths[model]),
        ]
        for method, approved_path in specs:
            run_dir = OUT / "runs" / f"homecredit_{model}_{method}"
            payload = {
                "dataset": "homecredit", "model": model, "method": method,
                "feature_budget": budget, "screening_pool_size": pool_size,
                "ranking_hash": sha256_text(RANKING.read_text(encoding="utf-8")),
                "approved_features_path": str(approved_path or ""),
                "pairing_policy_version": "identity_equivalence_v2",
            }
            config = ExperimentConfig(
                experiment_name=method,
                selector_name=method,
                dataset_name="homecredit",
                model_name=model,
                model_kwargs=resolve_model_kwargs(project, model),
                data_dir=str(project["data_dir"]),
                description_path=str(project["description_path"]),
                base_output_dir=str(run_dir.parent),
                experiment_output_dir=str(run_dir),
                dev_start_day=int(project["dev_start_day"]),
                oot_start_day=int(project["oot_start_day"]),
                oot_end_day=int(project["oot_end_day"]),
                n_splits=int(project["n_splits"]),
                cv_gap_groups=int(project["cv_gap_groups"]),
                random_state=int(project["random_seed"]),
                feature_budget=budget,
                excluded_feature_columns=tuple(project["excluded_feature_columns"]),
                preprocessor_kwargs=dict(project.get("preprocessor_kwargs", {})),
                selector_cls=FixedRankThenMRMRSelector,
                selector_kwargs={
                    "ranking_path": str(RANKING),
                    "feature_budget": budget,
                    "screening_pool_size": pool_size,
                    "approved_features_path": str(approved_path) if approved_path else None,
                    "random_state": int(project["random_seed"]),
                    "selector_label": method,
                },
                experiment_type="corrected_clip_professor_request",
                config_hash=sha256_text(json.dumps(payload, sort_keys=True)),
                data_fingerprint=build_data_version(project["data_dir"]),
            )
            if prepared is None:
                prepared = prepare_modeling_data(config)
            run = run_experiment(config, prepared_data=prepared)
            summary = pd.read_csv(run.exp_dir / "results" / "experiment_summary.csv").iloc[0].to_dict()
            oot = pd.read_csv(run.exp_dir / "results" / "oot_test_results.csv").iloc[0].to_dict()
            selected = pd.read_csv(run.exp_dir / "features" / "final_selected_features.csv")
            predictions = pd.read_csv(run.exp_dir / "results" / "oot_predictions.csv")
            rows.append({
                "dataset": "homecredit", "method": method, "model": model,
                "feature_budget": budget, "dev_auc": summary.get("cv_auc_mean"),
                "oot_auc": oot.get("auc"), "auc_drop": (summary.get("cv_auc_mean") - oot.get("auc")) if pd.notna(summary.get("cv_auc_mean")) else None,
                "dev_ks": summary.get("cv_ks_mean"), "oot_ks": oot.get("ks"),
                "score_psi": oot.get("model_score_psi"), "selected_feature_count": len(selected),
                "semantic_group_count": selected.semantic_group.nunique(),
                "result_origin": "newly_executed", "pairing_policy_version": "identity_equivalence_v2",
                "artifact_path": str(run.exp_dir).replace("\\", "/"),
                "config_hash": config.config_hash,
            })
            selected["dataset"], selected["method"], selected["model"] = "homecredit", method, model
            selected_rows.append(selected)
            predictions["dataset"], predictions["method"], predictions["model"] = "homecredit", method, model
            predictions["result_origin"] = "newly_executed"
            prediction_rows.append(predictions)
            ranked = pd.read_csv(RANKING)
            if approved_path:
                approved_names = set(pd.read_csv(approved_path).feature_name.astype(str))
                ranked = ranked[ranked.feature_name.astype(str).isin(approved_names)]
            pool = ranked.head(pool_size).copy()
            pool["dataset"], pool["method"], pool["model"] = "homecredit", method, model
            pool_rows.append(pool)

    pd.DataFrame(rows).to_csv(OUT / "new_metrics.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(OUT / "selected_features.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "new_predictions.parquet", index=False)
    pd.concat(pool_rows, ignore_index=True).to_csv(OUT / "candidate_pool_manifest.csv", index=False)
    pd.DataFrame([{
        "prediction_path": str(OUT / "new_predictions.parquet"),
        "run_count": 4, "result_origin": "newly_executed",
    }]).to_csv(OUT / "comparison_predictions_registry.csv", index=False)
    (OUT / "combined_pipeline_manifest.json").write_text(json.dumps({
        "pairing_policy_version": "identity_equivalence_v2",
        "models": ["lr", "catboost"], "feature_budgets": {"lr": 20, "catboost": 40},
        "candidate_pool_sizes": {"lr": 60, "catboost": 100},
        "llm_source": str(llm_source), "llm_reexecuted": False,
        "oot_used_for_selection_or_tuning": False, "new_run_count": 4,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
