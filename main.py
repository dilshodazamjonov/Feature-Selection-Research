import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import logging

from Preprocessing.data_process import DataLoader
from Preprocessing.preprocessing import Preprocessor
from Preprocessing.feature_engineering import build_all_features

from Models.utils import get_selector, get_model_bundle

from training.kfold_trainer import run_kfold_training
from evaluation.metrics import evaluate_model
from utils.logging_config import setup_logging

# Constants
MAX_DAYS_BACK = 600  # Keep data from -600 to 0 (20 months)
CV_MONTHS = 12  # First 12 months for CV (days -600 to -240)
OOT_MONTHS = 8  # Last 8 months for OOT test (days -240 to 0)
N_SPLITS = 5
DATA_DIR = "data/inputs"
TARGET = "TARGET"
TIME_COL = "recent_decision"  
MODEL_NAME = "lr"  # lr, rf, catboost
FEATURE_SELECTION_METHOD = "mrmr"  # boruta, boruta_rfe, mrmr, pca, none
DROP_ID_COLS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]

# Setup logging
logger = setup_logging("main", level=logging.INFO)

matplotlib.use("Agg")


def resolve_time_col(df: pd.DataFrame, preferred: str) -> str:
    """
    Resolve the time column name in a case-tolerant way.
    """
    candidates = [preferred, preferred.upper(), preferred.lower()]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Time column not found. Tried: {candidates}")


def main():
    """
    Loads raw data, builds engineered features, performs OOT split,
    then runs time-ordered CV on the train portion only.
    """
    logger.info(f"Loading datasets from {DATA_DIR}")
    loader = DataLoader(DATA_DIR)
    dfs = loader.load_all()

    if "application_train" not in dfs:
        raise ValueError("application_train.csv not found in data directory")

    app_train = dfs["application_train"].copy()

    logger.info("Building feature tables")
    feature_tables = build_all_features(dfs)

    logger.info("Merging feature tables into application_train")
    merged_train = loader.merge_features(app_train, feature_tables, on="SK_ID_CURR")

    time_col = resolve_time_col(merged_train, TIME_COL)
    logger.info(f"Using time column: {time_col}")

    # Filter: keep only data from -MAX_DAYS_BACK to 0 (last 20 months)
    # Days are negative, so we keep: -MAX_DAYS_BACK <= x <= 0
    logger.info(f"Before filter: {merged_train.shape}")
    merged_train = merged_train[
        (merged_train[time_col] >= -MAX_DAYS_BACK) & 
        (merged_train[time_col] <= 0)
    ]
    logger.info(f"After filter (days >= -600): {merged_train.shape}")

    # Split by months: first 12 months for CV (older), last 8 months for OOT (newer)
    # Days: -600 to -240 = CV (12 months, older), -240 to 0 = OOT (8 months, newer)
    # Since days are negative: more negative = older, closer to 0 = newer
    oot_cutoff = -OOT_MONTHS * 30  # -240 days
    
    cv_data = merged_train[merged_train[time_col] < oot_cutoff].copy()   # -600 to -240 = 12 months (older)
    oot_data = merged_train[merged_train[time_col] >= oot_cutoff].copy() # -240 to 0 = 8 months (newer)
    
    X_train_full = cv_data.drop(columns=[TARGET])
    y_train_full = cv_data[TARGET]
    X_oot = oot_data.drop(columns=[TARGET])
    y_oot = oot_data[TARGET]
    
    logger.info(f"CV split: {X_train_full.shape} (days < {oot_cutoff}, i.e. -600 to -240)")
    logger.info(f"OOT split: {X_oot.shape} (days >= {oot_cutoff}, i.e. -240 to 0)")

    drop_cols = [c for c in DROP_ID_COLS if c in X_train_full.columns]
    X_train = X_train_full.drop(columns=drop_cols, errors="ignore")

    selector_cls, selector_kwargs = get_selector(FEATURE_SELECTION_METHOD)
    get_model, train_model, predict_proba, save_model = get_model_bundle(MODEL_NAME)

    logger.info(f"Starting experiment: {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")

    results_df = run_kfold_training(
        X=X_train,
        y=y_train_full,
        time_col=time_col,
        get_model=get_model,
        train_model=train_model,
        predict_proba=predict_proba,
        save_model=save_model,
        preprocessor_cls=Preprocessor,
        preprocessor_kwargs={},
        selector_cls=selector_cls,
        selector_kwargs=selector_kwargs,
        model_name=f"{MODEL_NAME}_{FEATURE_SELECTION_METHOD}",
        base_output_dir="outputs",
        n_splits=N_SPLITS,
        random_state=42,
    )

    # ============================================
    # Final evaluation on OOT test set
    # ============================================
    logger.info("=== OOT TEST EVALUATION ===")
    
    # Prepare OOT test data
    X_oot_clean = X_oot.drop(columns=[c for c in DROP_ID_COLS if c in X_oot.columns], errors="ignore")
    
    # Fit final preprocessor and selector on full training data
    final_preprocessor = Preprocessor()
    X_train_full_processed = final_preprocessor.fit_transform(X_train)
    
    final_selector = selector_cls(**selector_kwargs) if selector_cls is not None else None
    if final_selector is not None:
        X_train_final = final_selector.fit_transform(X_train_full_processed, y_train_full)
        X_oot_final = final_selector.transform(final_preprocessor.transform(X_oot_clean))
    else:
        X_train_final = X_train_full_processed
        X_oot_final = final_preprocessor.transform(X_oot_clean)
    
    # Train final model on full training data
    final_model = get_model()
    final_model = train_model(final_model, X_train_final, y_train_full, None, None)
    
    # Predict on OOT test set
    oot_proba = predict_proba(final_model, X_oot_final)
    oot_metrics = evaluate_model(y_oot.values, oot_proba)
    
    logger.info("OOT Test Results:")
    logger.info(f"  AUC: {oot_metrics['auc']:.4f}")
    logger.info(f"  Gini: {oot_metrics['gini']:.4f}")
    logger.info(f"  KS: {oot_metrics['ks']:.4f}")
    logger.info(f"  Precision: {oot_metrics['precision']:.4f}")
    logger.info(f"  Recall: {oot_metrics['recall']:.4f}")
    logger.info(f"  F1: {oot_metrics['f1']:.4f}")
    
    # Find the experiment output directory (created by kfold_trainer with timestamp)
    exp_base = "outputs"
    exp_dirs = [d for d in os.listdir(exp_base) if d.startswith(f"{MODEL_NAME}_{FEATURE_SELECTION_METHOD}_")]
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment output directory found in {exp_base}")
    exp_dir = os.path.join(exp_base, sorted(exp_dirs)[-1])  # Use most recent
    results_dir = os.path.join(exp_dir, "results")
    
    # Save OOT results
    oot_results_path = os.path.join(results_dir, "oot_test_results.csv")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([oot_metrics]).to_csv(oot_results_path, index=False)
    logger.info(f"OOT results saved to {oot_results_path}")

    # ============================================
    # Plot ROC-AUC per fold (by actual time periods)
    # ============================================
    eval_file = os.path.join(results_dir, "evaluation_metrics_summary.csv")
    eval_df = pd.read_csv(eval_file)
    # Filter only numeric folds (exclude 'oof', 'mean', 'std')
    eval_df = eval_df[eval_df["fold"].astype(str).str.isnumeric()]
    eval_df["fold"] = eval_df["fold"].astype(int)
    eval_df = eval_df.sort_values("fold")

    # Load fold time info to get actual time periods
    time_file = os.path.join(results_dir, "fold_time_info.csv")
    time_df = pd.read_csv(time_file)
    
    # Merge to get time info with AUC
    eval_df = eval_df.merge(time_df, on="fold")
    
    # Create time labels (end of validation period)
    eval_df["time_label"] = eval_df["val_time_end"].apply(lambda x: f"t={int(x)}")
    
    # Plot 1: AUC over time (by validation period end)
    plt.figure(figsize=(14, 6))
    plt.plot(eval_df["val_time_end"], eval_df["auc"], marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f"ROC-AUC over Time | {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")
    plt.xlabel("Validation Period End (days)")
    plt.ylabel("ROC-AUC")
    plt.grid(True, alpha=0.3)
    
    # Add annotations for each point
    for _, row in eval_df.iterrows():
        plt.annotate(f'Fold {int(row["fold"])}', 
                    (row["val_time_end"], row["auc"]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=8)
    
    plt.tight_layout()
    plot_file = os.path.join(exp_dir, "roc_auc_over_time.png")
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-AUC over time plot saved to {plot_file}")

    # Plot 2: AUC with training size (to see if more data helps)
    plt.figure(figsize=(12, 6))
    plt.scatter(eval_df["train_size"], eval_df["auc"], s=100, c=eval_df["fold"], cmap='viridis', edgecolors='black')
    plt.colorbar(label='Fold Number')
    plt.title(f"ROC-AUC vs Training Size | {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")
    plt.xlabel("Training Set Size")
    plt.ylabel("ROC-AUC")
    plt.grid(True, alpha=0.3)
    
    for _, row in eval_df.iterrows():
        plt.annotate(f'Fold {int(row["fold"])}', 
                    (row["train_size"], row["auc"]),
                    textcoords="offset points", 
                    xytext=(5, 5), 
                    ha='left',
                    fontsize=8)
    
    plt.tight_layout()
    plot_file2 = os.path.join(exp_dir, "roc_auc_vs_train_size.png")
    plt.savefig(plot_file2, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-AUC vs training size plot saved to {plot_file2}")

    logger.info("CV experiment completed")
    return results_df


if __name__ == "__main__":
    results = main()