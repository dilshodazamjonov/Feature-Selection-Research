from pathlib import Path

import pandas as pd

from pipelines.comparison import (
    build_experiment_summary_row,
    build_feature_overlap_frame,
    compare_experiment_pair,
)


def _write_experiment(root: Path, fold_features: dict[int, list[str]], oot_auc: float, cv_auc: list[float]) -> Path:
    exp_dir = root
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    (exp_dir / "features").mkdir(parents=True, exist_ok=True)

    cv_rows = []
    for fold, (features, auc) in enumerate(zip(fold_features.values(), cv_auc), start=1):
        fold_dir = exp_dir / "features" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": features}).to_csv(fold_dir / "selected_features.csv", index=False)
        cv_rows.append(
            {
                "fold": fold,
                "auc": auc,
                "gini": 2 * auc - 1,
                "ks": 0.2 + fold * 0.01,
                "precision": 0.4,
                "recall": 0.5,
                "f1": 0.45,
                "accuracy": 0.6,
                "selected_features": len(features),
                "psi_feature_mean": 0.05,
                "psi_feature_max": 0.10,
                "psi_model": 0.03,
                "jaccard_similarity": 0.7,
                "fold_time_sec": 1.0,
            }
        )

    pd.DataFrame(cv_rows).to_csv(exp_dir / "results" / "cv_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "auc": oot_auc,
                "gini": 2 * oot_auc - 1,
                "ks": 0.3,
                "precision": 0.41,
                "recall": 0.52,
                "f1": 0.46,
                "accuracy": 0.61,
                "decision_threshold": 0.5,
            }
        ]
    ).to_csv(exp_dir / "results" / "oot_test_results.csv", index=False)

    final_features = next(iter(fold_features.values()))
    pd.DataFrame({"feature": final_features}).to_csv(
        exp_dir / "features" / "final_selected_features.csv",
        index=False,
    )
    return exp_dir


def test_build_experiment_summary_row_aggregates_fold_metrics(tmp_path):
    exp_dir = _write_experiment(
        tmp_path / "exp_a",
        fold_features={1: ["a", "b"], 2: ["a", "c"]},
        oot_auc=0.72,
        cv_auc=[0.70, 0.74],
    )

    summary = build_experiment_summary_row(
        exp_dir=exp_dir,
        method_name="mrmr",
        model_name="lr",
        selector_name="mrmr",
    )

    assert summary["method"] == "mrmr"
    assert summary["cv_fold_count"] == 2
    assert summary["cv_auc_mean"] == 0.72
    assert summary["oot_auc"] == 0.72
    assert summary["final_selected_feature_count"] == 2


def test_compare_experiment_pair_reports_overlap_and_performance_deltas(tmp_path):
    left_dir = _write_experiment(
        tmp_path / "left",
        fold_features={1: ["a", "b"], 2: ["a", "c"]},
        oot_auc=0.70,
        cv_auc=[0.69, 0.71],
    )
    right_dir = _write_experiment(
        tmp_path / "right",
        fold_features={1: ["a", "d"], 2: ["a", "c"]},
        oot_auc=0.75,
        cv_auc=[0.73, 0.77],
    )

    summary, overlap_df = compare_experiment_pair(
        left_label="stat",
        left_exp_dir=left_dir,
        left_model_name="lr",
        left_selector_name="mrmr",
        right_label="llm",
        right_exp_dir=right_dir,
        right_model_name="lr",
        right_selector_name="llm",
    )

    assert not overlap_df.empty
    assert summary["feature_overlap_fold_count"] == 2
    assert summary["delta_oot_auc_right_minus_left"] == 0.05
    assert summary["delta_cv_auc_mean_right_minus_left"] == 0.05

    final_overlap = overlap_df[overlap_df["stage"] == "final"].iloc[0]
    assert final_overlap["intersection_count"] == 1


def test_build_feature_overlap_frame_handles_missing_final_files(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    for exp_dir, features in [(left_dir, ["a", "b"]), (right_dir, ["a", "c"])]:
        fold_dir = exp_dir / "features" / "fold_1"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": features}).to_csv(fold_dir / "selected_features.csv", index=False)

    overlap_df = build_feature_overlap_frame(
        left_label="left",
        left_exp_dir=left_dir,
        right_label="right",
        right_exp_dir=right_dir,
    )

    assert len(overlap_df) == 1
    assert overlap_df.iloc[0]["stage"] == "fold"
