import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from feature_methods.preselection.iv_calc import IVFilter
from feature_methods.boruta.boruta_selection import BorutaSelector
from Preprocessing.data_process import Preprocessor
from Models.metrics import ks_statistic

DATA_DIR = "data/inputs/Master_Data_with_filtering_updated.csv"
TARGET = "TARGET"
RESULTS_DIR = "data/output/training/catboost"
FEATURES_DIR = "data/output/features"
MODELS_DIR = "data/output/models"

N_SPLITS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Data loading started at {DATA_DIR}")

# ===============================
# Load Data
# ===============================

raw_df = pd.read_csv(DATA_DIR)
print("Data read complete")

X = raw_df.drop(columns=[TARGET])
y = raw_df[TARGET]

# ===============================
# Cross Validation Experiment
# ===============================

def run_cv_experiment():
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    results = []
    start_total = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== FOLD {fold+1} ==========")
        fold_start = time.time()

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # ===============================
        # Preprocessing
        # ===============================
        preprocess = Preprocessor()
        preprocess.fit(X_train)
        X_train_proc = preprocess.transform(X_train)
        X_val_proc = preprocess.transform(X_val)

        X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan).fillna(X_train_proc.median())
        X_val_proc = X_val_proc.replace([np.inf, -np.inf], np.nan).fillna(X_train_proc.median())
        print("Preprocessing finished")

        # ===============================
        # IV Filtering
        # ===============================
        iv_filter = IVFilter()
        iv_filter.fit(X_train_proc, y_train)
        X_train_filtered = iv_filter.transform(X_train_proc)
        X_val_filtered = iv_filter.transform(X_val_proc)
        n_iv_features = len(X_train_filtered.columns)
        print(f"IV Filter finished | Features kept: {n_iv_features}")

        # ===============================
        # Boruta Feature Selection
        # ===============================
        selector = BorutaSelector(max_iter=10)
        selector.fit(X_train_filtered, y_train)

        X_train_final = selector.transform(X_train_filtered)
        X_val_final = selector.transform(X_val_filtered)
        selected_features = X_train_final.columns.tolist()

        # Save selected features per fold
        feat_path = os.path.join(FEATURES_DIR, f"fold_{fold+1}_boruta_features.csv")
        pd.DataFrame({"feature": selected_features}).to_csv(feat_path, index=False)
        print(f"Fold {fold+1} selected features saved to {feat_path}")

        print(f"Final selected columns ({len(selected_features)}): {selected_features}")

        # ===============================
        # Model Training
        # ===============================
        model = CatBoostClassifier(
            depth=10,
            learning_rate=0.01,
            l2_leaf_reg=95,
            min_data_in_leaf=290,
            colsample_bylevel=0.9,
            random_strength=0.125,
            grow_policy='Depthwise',
            one_hot_max_size=21,
            leaf_estimation_method='Newton',
            bootstrap_type='Bernoulli',
            subsample=0.55,
            loss_function='Logloss',
            eval_metric='AUC',
            auto_class_weights='Balanced',
            iterations=2200,        
            early_stopping_rounds=150, 
            verbose=100
        )

        model.fit(
            X_train_final,
            y_train,
            eval_set=(X_val_final, y_val)
        )

        # Save model per fold
        model_path = os.path.join(MODELS_DIR, f"catboost_fold_{fold+1}.cbm")
        model.save_model(model_path)
        print(f"CatBoost model for fold {fold+1} saved to {model_path}")

        best_iteration = model.get_best_iteration()
        val_preds = model.predict_proba(X_val_final)[:, 1]

        # ===============================
        # Metrics
        # ===============================
        auc = roc_auc_score(y_val, val_preds)
        gini = 2 * auc - 1
        ks = ks_statistic(y_val, val_preds)
        fold_time = time.time() - fold_start

        print(f"Fold {fold+1} AUC: {auc:.5f} | KS: {ks:.5f}")

        results.append({
            "fold": fold + 1,
            "auc": auc,
            "gini": gini,
            "ks": ks,
            "iv_features": n_iv_features,
            "boruta_features": len(selected_features),
            "best_iteration": best_iteration,
            "fold_time_sec": fold_time
        })

    # ===============================
    # Summary Metrics
    # ===============================
    end_total = time.time()
    results_df = pd.DataFrame(results)

    summary = {
        "fold": "MEAN",
        "auc": results_df["auc"].mean(),
        "gini": results_df["gini"].mean(),
        "ks": results_df["ks"].mean(),
        "iv_features": results_df["iv_features"].mean(),
        "boruta_features": results_df["boruta_features"].mean(),
        "best_iteration": results_df["best_iteration"].mean(),
        "fold_time_sec": results_df["fold_time_sec"].mean()
    }

    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

    # Save CV results
    save_path = os.path.join(RESULTS_DIR, "cv_results.csv")
    results_df.to_csv(save_path, index=False)

    print("\n========== CV RESULTS ==========")
    print(results_df)
    print(f"\nResults saved to {save_path}")
    print(f"Total Time: {(end_total-start_total)/60:.2f} minutes")

    # ===============================
    # Top 10 Feature Importances (last fold)
    # ===============================
    importances = model.get_feature_importance()
    feature_names = X_train_final.columns

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(10)

    top_feat_path = os.path.join(FEATURES_DIR, "top10_features_last_fold.csv")
    fi_df.to_csv(top_feat_path, index=False)
    print("\n========== TOP 10 FEATURE IMPORTANCES (Last Fold) ==========")
    print(fi_df.to_string(index=False))
    print(f"Top 10 features saved to {top_feat_path}")

    return results_df


if __name__ == "__main__":
    results = run_cv_experiment()