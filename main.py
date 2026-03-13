from __future__ import annotations

import os
import pandas as pd

from Models.catboost_pipeline import FeatureSelectionPipeline


DATA_DIR = "data/inputs/Master_Data_with_filtering_updated.csv"
TARGET = "TARGET"


def main():
    print(f"Data loading started at {DATA_DIR}")
    raw_df = pd.read_csv(DATA_DIR)
    print("Data read complete")

    X = raw_df.drop(columns=[TARGET])
    y = raw_df[TARGET]

    methods = ["pca", "boruta_rfe"]  # add "greedy" when implemented

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {method.upper()}")
        print(f"{'='*60}")

        pipeline = FeatureSelectionPipeline(
            method=method,
            n_splits=5,
            random_state=42,
            results_dir=f"data/output/training/catboost/{method}",
            features_dir=f"data/output/features/{method}",
            models_dir=f"data/output/models/{method}",
        )

        results_df = pipeline.run_cv(X, y)
        all_results[method] = results_df

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparison = []
    for method, df in all_results.items():
        mean_row = df[df["fold"] == "MEAN"].iloc[0]
        comparison.append(
            {
                "method": method,
                "auc": mean_row["auc"],
                "gini": mean_row["gini"],
                "ks": mean_row["ks"],
                "selected_features": mean_row["selected_features"],
                "fold_time_sec": mean_row["fold_time_sec"],
            }
        )

    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))

    os.makedirs("data/output/training/comparison", exist_ok=True)
    comparison_df.to_csv(
        "data/output/training/comparison/feature_selection_comparison.csv",
        index=False,
    )
    print("\nComparison saved to data/output/training/comparison/feature_selection_comparison.csv")


if __name__ == "__main__":
    main()
