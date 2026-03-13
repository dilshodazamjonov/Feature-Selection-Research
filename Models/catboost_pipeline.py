from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from Preprocessing.data_process import Preprocessor
from feature_methods.preselection.iv_calc import IVFilter
from feature_methods.boruta.boruta_selection import BorutaSelector
from feature_methods.boruta.rfe_selection import RFESelector
from feature_methods.pca.pca_entry import PCASelector
from Models.metrics import ks_statistic
from training.kfold_trainer import train_catboost_fold


class FeatureSelectionPipeline:
    """Unified pipeline supporting multiple feature selection methods."""

    def __init__(
        self,
        method: str = "boruta_rfe",
        n_splits: int = 5,
        random_state: int = 42,
        results_dir: str = "data/output/training/catboost",
        features_dir: str = "data/output/features",
        models_dir: str = "data/output/models",
    ):
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state
        self.results_dir = results_dir
        self.features_dir = features_dir
        self.models_dir = models_dir

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def _get_feature_selector(
        self,
    ) -> tuple[Callable, dict[str, Any]]:
        """Return selector instance and its config for the chosen method."""
        if self.method == "pca":
            selector = PCASelector(n_components=0.95)
            config = {"type": "pca", "n_components": 0.95}
            return selector, config

        if self.method == "boruta_rfe":
            boruta = BorutaSelector(max_iter=10)
            rfe = RFESelector(n_features=50, step=10)
            config = {"type": "boruta_rfe", "boruta_max_iter": 10, "rfe_n_features": 50}
            return (boruta, rfe), config

        if self.method == "greedy":
            raise NotImplementedError("Greedy selector not implemented yet.")

        raise ValueError(f"Unknown method: {self.method}")

    def _apply_feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        selector,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Apply feature selection and return transformed train/val and feature list."""
        if self.method == "pca":
            selector.fit(X_train)
            X_train_sel = selector.transform(X_train)
            X_val_sel = selector.transform(X_val)
            features = X_train_sel.columns.tolist()
            return X_train_sel, X_val_sel, features

        if self.method == "boruta_rfe":
            boruta, rfe = selector
            # Boruta
            boruta.fit(X_train, y_train)
            X_train_boruta = boruta.transform(X_train)
            X_val_boruta = boruta.transform(X_val)
            # RFE
            rfe.fit(X_train_boruta, y_train)
            X_train_sel = rfe.transform(X_train_boruta)
            X_val_sel = rfe.transform(X_val_boruta)
            features = X_train_sel.columns.tolist()
            return X_train_sel, X_val_sel, features

        raise ValueError(f"Unknown method: {self.method}")

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        catboost_params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run cross-validation experiment with the selected feature method."""
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        results = []
        start_total = time.time()
        model = None
        X_train_final = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n========== FOLD {fold + 1} | METHOD: {self.method.upper()} ==========")
            fold_start = time.time()

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Preprocessing
            preprocess = Preprocessor()
            preprocess.fit(X_train)
            X_train_proc = preprocess.transform(X_train)
            X_val_proc = preprocess.transform(X_val)

            X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan).fillna(
                X_train_proc.median()
            )
            X_val_proc = X_val_proc.replace([np.inf, -np.inf], np.nan).fillna(
                X_train_proc.median()
            )
            print("Preprocessing finished")

            # IV Filtering
            iv_filter = IVFilter()
            iv_filter.fit(X_train_proc, y_train)
            X_train_filtered = iv_filter.transform(X_train_proc)
            X_val_filtered = iv_filter.transform(X_val_proc)
            n_iv_features = len(X_train_filtered.columns)
            print(f"IV Filter finished | Features kept: {n_iv_features}")

            # Feature Selection
            selector, config = self._get_feature_selector()
            X_train_final, X_val_final, selected_features = self._apply_feature_selection(
                X_train_filtered, y_train, X_val_filtered, selector
            )

            feat_path = os.path.join(
                self.features_dir, f"fold_{fold + 1}_{self.method}_features.csv"
            )
            pd.DataFrame({"feature": selected_features}).to_csv(feat_path, index=False)
            print(f"Fold {fold + 1} selected features saved to {feat_path}")
            print(f"Final selected columns ({len(selected_features)})")

            # Model Training
            model, val_preds, best_iteration = train_catboost_fold(
                X_train_final, y_train, X_val_final, y_val, catboost_params
            )

            model_path = os.path.join(self.models_dir, f"catboost_fold_{fold + 1}.cbm")
            model.save_model(model_path)
            print(f"CatBoost model for fold {fold + 1} saved to {model_path}")

            # Metrics
            auc = roc_auc_score(y_val, val_preds)
            gini = 2 * auc - 1
            ks = ks_statistic(y_val, val_preds)
            fold_time = time.time() - fold_start

            print(f"Fold {fold + 1} AUC: {auc:.5f} | KS: {ks:.5f}")

            results.append(
                {
                    "fold": fold + 1,
                    "auc": auc,
                    "gini": gini,
                    "ks": ks,
                    "iv_features": n_iv_features,
                    "selected_features": len(selected_features),
                    "best_iteration": best_iteration,
                    "fold_time_sec": fold_time,
                }
            )

        # Summary
        end_total = time.time()
        results_df = pd.DataFrame(results)

        summary = {
            "fold": "MEAN",
            "auc": results_df["auc"].mean(),
            "gini": results_df["gini"].mean(),
            "ks": results_df["ks"].mean(),
            "iv_features": results_df["iv_features"].mean(),
            "selected_features": results_df["selected_features"].mean(),
            "best_iteration": results_df["best_iteration"].mean(),
            "fold_time_sec": results_df["fold_time_sec"].mean(),
        }

        results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

        save_path = os.path.join(self.results_dir, f"cv_results_{self.method}.csv")
        results_df.to_csv(save_path, index=False)

        print("\n========== CV RESULTS ==========")
        print(results_df)
        print(f"\nResults saved to {save_path}")
        print(f"Total Time: {(end_total - start_total) / 60:.2f} minutes")

        # Top 10 Feature Importances (last fold)
        if model is not None and X_train_final is not None:
            importances = model.get_feature_importance()
            feature_names = X_train_final.columns

            fi_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(10)
            )

            top_feat_path = os.path.join(
                self.features_dir, f"top10_features_last_fold_{self.method}.csv"
            )
            fi_df.to_csv(top_feat_path, index=False)
            print("\n========== TOP 10 FEATURE IMPORTANCES (Last Fold) ==========")
            print(fi_df.to_string(index=False))
            print(f"Top 10 features saved to {top_feat_path}")

        return results_df
