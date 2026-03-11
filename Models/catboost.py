from feature_methods.pca.pca_entry import PCASelector
from feature_methods.preselection.iv_calc import IVFilter
from Preprocessing.data_process import Preprocessor
from Models.metrics import ks_statistic

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import time
import os


DATA_DIR = "data/inputs/Master_Data_with_filtering_updated.csv"
TARGET = "TARGET"
OUTPUT_DIR = "data/output/pca"
RESULTS_DIR = "data/output/training/catboost"

N_SPLITS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'Data loading started at {DATA_DIR}')

# ===============================
# Load Data
# ===============================

raw_df = pd.read_csv(DATA_DIR)

print("Data Read")

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

        X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan)
        X_val_proc = X_val_proc.replace([np.inf, -np.inf], np.nan)

        X_train_proc = X_train_proc.fillna(X_train_proc.median())
        X_val_proc = X_val_proc.fillna(X_train_proc.median())

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
        # PCA
        # ===============================

        pca = PCASelector(
            n_components=20,
            save_dir=OUTPUT_DIR
        )

        pca.fit(X_train_filtered)

        X_train_final = pca.transform(X_train_filtered)
        X_val_final = pca.transform(X_val_filtered)

        n_pca_features = X_train_final.shape[1]

        print(f"PCA finished | Components: {n_pca_features}")

        # ===============================
        # Model
        # ===============================

        model = CatBoostClassifier(
            iterations=2000,
            depth=6,
            l2_leaf_reg=10,
            early_stopping_rounds=50,
            verbose=False
        )

        model.fit(
            X_train_final,
            y_train,
            eval_set=(X_val_final, y_val)
        )

        best_iteration = model.get_best_iteration()

        val_preds = model.predict_proba(X_val_final)[:, 1]

        # ===============================
        # Metrics
        # ===============================

        auc = roc_auc_score(y_val, val_preds)

        gini = 2 * auc - 1

        ks = ks_statistic(y_val, val_preds)

        fold_time = time.time() - fold_start

        print(f"Fold {fold+1} AUC: {auc:.5f}")
        print(f"Fold {fold+1} KS: {ks:.5f}")

        # ===============================
        # Save Fold Results
        # ===============================

        results.append({
            "fold": fold + 1,
            "auc": auc,
            "gini": gini,
            "ks": ks,
            "iv_features": n_iv_features,
            "pca_components": n_pca_features,
            "best_iteration": best_iteration,
            "fold_time_sec": fold_time
        })

    end_total = time.time()

    results_df = pd.DataFrame(results)

    # ===============================
    # Summary Metrics
    # ===============================

    summary = {
        "fold": "MEAN",
        "auc": results_df["auc"].mean(),
        "gini": results_df["gini"].mean(),
        "ks": results_df["ks"].mean(),
        "iv_features": results_df["iv_features"].mean(),
        "pca_components": results_df["pca_components"].mean(),
        "best_iteration": results_df["best_iteration"].mean(),
        "fold_time_sec": results_df["fold_time_sec"].mean()
    }

    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

    # ===============================
    # Save CSV
    # ===============================

    save_path = f"{RESULTS_DIR}/cv_results.csv"

    results_df.to_csv(save_path, index=False)

    print("\n========== CV RESULTS ==========")

    print(f"Mean AUC: {summary['auc']:.5f}")
    print(f"Mean GINI: {summary['gini']:.5f}")
    print(f"Mean KS: {summary['ks']:.5f}")

    print(f"\nResults saved to {save_path}")

    print(f"Total Time: {(end_total-start_total)/60:.2f} minutes")

    return results_df


if __name__ == "__main__":

    results = run_cv_experiment()