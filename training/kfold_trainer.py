# training/kfold_trainer.py

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from datetime import datetime

# training/kfold_trainer.py

def run_kfold_training(
    X,
    y,
    get_model,
    train_model,
    predict_proba,
    save_model,
    get_feature_importance,
    preprocessor_cls,
    iv_filter_cls,
    selector_cls,
    ks_statistic,
    model_name,
    base_output_dir="outputs",
    n_splits=5,
    random_state=42
):
    """
    Executes a full K-Fold cross-validation training pipeline.

    This function is model-agnostic and operates via dependency injection
    for all major components (model, preprocessing, feature selection).

    It computes the following metrics per fold:
      - AUC
      - Gini
      - KS (uses ks_statistic passed in)
      - Precision (thresholded)
      - Recall (thresholded)
      - F1-score (thresholded)
      - goods_count (y==0)
      - bads_count (y==1)
      - goods_bads_ratio (goods / bads)
      - bad_rate (bads / (goods + bads))

    The function saves per-fold selected features, per-fold models, a CSV of
    fold metrics (cv_results.csv) in results_dir, and top-10 feature importances
    from the last fold in features_dir.
    """

    # -------------------------
    # Create experiment folder
    # -------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_output_dir, f"{model_name}_{timestamp}")

    results_dir = os.path.join(exp_dir, "results")
    features_dir = os.path.join(exp_dir, "features")
    models_dir = os.path.join(exp_dir, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    start_total = time.time()

    # -------------------------
    # CV Loop
    # -------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        print(f"\n========== FOLD {fold} ==========")
        fold_start = time.time()

        # Fold-specific dirs
        fold_feat_dir = os.path.join(features_dir, f"fold_{fold}")
        fold_iv_dir = os.path.join(fold_feat_dir, "iv")
        fold_sel_dir = os.path.join(fold_feat_dir, "selected")

        os.makedirs(fold_iv_dir, exist_ok=True)
        os.makedirs(fold_sel_dir, exist_ok=True)

        # Split
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # -------------------------
        # Preprocessing
        # -------------------------
        preprocess = preprocessor_cls()
        preprocess.fit(X_train)

        X_train_proc = preprocess.transform(X_train)
        X_val_proc = preprocess.transform(X_val)

        X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan).fillna(X_train_proc.median())
        X_val_proc = X_val_proc.replace([np.inf, -np.inf], np.nan).fillna(X_train_proc.median())

        print("Preprocessing finished")

        # -------------------------
        # IV Filtering (per fold!)
        # -------------------------
        iv_filter = iv_filter_cls(output_dir=fold_iv_dir)
        iv_filter.fit(X_train_proc, y_train)

        X_train_filtered = iv_filter.transform(X_train_proc)
        X_val_filtered = iv_filter.transform(X_val_proc)

        n_iv_features = len(X_train_filtered.columns)
        print(f"IV Filter finished | Features kept: {n_iv_features}")

        # -------------------------
        # Feature Selection
        # -------------------------
        selector = selector_cls(max_iter=10)
        selector.fit(X_train_filtered, y_train)

        X_train_final = selector.transform(X_train_filtered)
        X_val_final = selector.transform(X_val_filtered)

        selected_features = X_train_final.columns.tolist()

        pd.DataFrame({"feature": selected_features}) \
            .to_csv(os.path.join(fold_sel_dir, "selected_features.csv"), index=False)

        print(f"Selected features ({len(selected_features)}) saved")

        # -------------------------
        # Model Training
        # -------------------------
        model = get_model()
        model = train_model(model, X_train_final, y_train, X_val_final, y_val)

        model_path = os.path.join(models_dir, f"{model_name}_fold_{fold}.model")
        save_model(model, model_path)

        print(f"Model saved: {model_path}")

        # -------------------------
        # Predictions
        # -------------------------
        best_iteration = getattr(model, "get_best_iteration", lambda: None)()
        val_preds = predict_proba(model, X_val_final)

        # Metrics
        auc = roc_auc_score(y_val, val_preds)
        gini = 2 * auc - 1
        ks, optimal_threshold = ks_statistic(y_val, val_preds)

        y_pred = (val_preds >= optimal_threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        goods = int((y_val == 0).sum())
        bads = int((y_val == 1).sum())

        goods_bads_ratio = goods / bads if bads > 0 else np.nan
        bad_rate = bads / (goods + bads) if (goods + bads) > 0 else np.nan

        fold_time = time.time() - fold_start

        print(f"AUC: {auc:.5f} | KS: {ks:.5f} | F1: {f1:.4f}")

        results.append({
            "fold": fold,
            "auc": auc,
            "gini": gini,
            "ks": ks,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": optimal_threshold,
            "iv_features": n_iv_features,
            "selected_features": len(selected_features),
            "goods": goods,
            "bads": bads,
            "goods_bads_ratio": goods_bads_ratio,
            "bad_rate": bad_rate,
            "best_iteration": best_iteration,
            "fold_time_sec": fold_time
        })

    # -------------------------
    # Results aggregation
    # -------------------------
    results_df = pd.DataFrame(results)

    summary = results_df.mean(numeric_only=True).to_dict()
    summary["fold"] = "MEAN"

    std = results_df.std(numeric_only=True).to_dict()
    std["fold"] = "STD"

    results_df = pd.concat([results_df, pd.DataFrame([summary, std])], ignore_index=True)

    save_path = os.path.join(results_dir, "cv_results.csv")
    results_df.to_csv(save_path, index=False)

    print("\n========== FINAL RESULTS ==========")
    print(results_df)
    print(f"\nSaved to {save_path}")
    print(f"Total Time: {(time.time() - start_total)/60:.2f} minutes")

    # -------------------------
    # Feature Importance (last fold)
    # -------------------------
    fi = get_feature_importance(model, X_train_final.columns)

    fi_df = pd.DataFrame(fi, columns=["feature", "importance"]) \
        .sort_values("importance", ascending=False) \
        .head(10)

    fi_path = os.path.join(features_dir, "top10_features_last_fold.csv")
    fi_df.to_csv(fi_path, index=False)

    print("\nTop 10 Features:")
    print(fi_df.to_string(index=False))

    return results_df
