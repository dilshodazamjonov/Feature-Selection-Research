# main.py

import pandas as pd

from feature_methods.preselection.iv_calc import IVFilter
from feature_methods.boruta.boruta_selection import BorutaSelector
from Preprocessing.data_process import Preprocessor
from evaluation.metrics import ks_statistic
from training.kfold_trainer import run_kfold_training


# --- Experiment Configuration ---
MODEL_NAME = "catboost"                       # options: "catboost", "rf", "lr"
FEATURE_SELECTION_METHOD = "boruta"     # descriptive name for the feature selection
N_SPLITS = 5

# Dataset configuration
DATA_PATH = "data/inputs/Master_Data_with_filtering_updated.csv"
TARGET = "TARGET"
DROP_COLS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", TARGET]


# --- Model Dependency Injection ---
def get_model_module(model_name):
    """
    Returns model functions for training, prediction, saving, and feature importance
    based on the selected model name.
    """
    if model_name == "catboost":
        from Models.catboost import (
            get_catboost_model as get_model,
            train_catboost as train_model,
            predict_proba,
            save_model,
            get_feature_importance
        )
    elif model_name == "rf":
        from Models.random_forest import (
            get_rf_model as get_model,
            train_rf as train_model,
            predict_proba,
            save_model,
            get_feature_importance
        )
    elif model_name == "lr":
        from Models.logistic_regression import (
            get_lr_model as get_model,
            train_lr as train_model,
            predict_proba,
            save_model,
            get_feature_importance
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return get_model, train_model, predict_proba, save_model, get_feature_importance


# --- Load dataset ---
print(f"Loading dataset from {DATA_PATH}...")
raw_df = pd.read_csv(DATA_PATH)
X = raw_df.drop(columns=DROP_COLS)
y = raw_df[TARGET]
print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")


# --- Load model functions ---
get_model, train_model, predict_proba, save_model, get_feature_importance = get_model_module(MODEL_NAME)


# --- Main experiment runner ---
def main():
    """
    Executes the full K-Fold CV pipeline:
    preprocessing -> IV filtering -> feature selection -> model training -> metrics.
    The output directories are auto-created inside run_kfold_training.
    """
    print(f"\nStarting Experiment: {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")

    results_df = run_kfold_training(
        X=X,
        y=y,
        get_model=get_model,
        train_model=train_model,
        predict_proba=predict_proba,
        save_model=save_model,
        get_feature_importance=get_feature_importance,
        preprocessor_cls=Preprocessor,
        iv_filter_cls=IVFilter,
        selector_cls=BorutaSelector,
        ks_statistic=ks_statistic,
        model_name=f"{MODEL_NAME}_{FEATURE_SELECTION_METHOD}",
        base_output_dir="outputs",
        n_splits=N_SPLITS,
        random_state=42
    )

    print("\nCV Experiment Completed")
    return results_df


# --- Run script ---
if __name__ == "__main__":
    results = main()