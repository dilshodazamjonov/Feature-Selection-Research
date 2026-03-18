# main.py

import os
import pandas as pd

from feature_methods.preselection.iv_calc import IVFilter
from feature_methods.boruta.boruta_selection import BorutaSelector
from Preprocessing.data_process import Preprocessor
from Models.metrics import ks_statistic
from training.kfold_trainer import run_kfold_training


# Model selection - change this to switch between CatBoost, RF, LR
MODEL_NAME = "lr"   # options: "catboost", "rf", "lr"


# Get model-specific functions
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


# Directories and dataset configuration
DATA_DIR = "data/inputs/Master_Data_with_filtering_updated.csv"
TARGET = "TARGET"
BASE_OUTPUT = f"data/output/{MODEL_NAME}"
RESULTS_DIR = os.path.join(BASE_OUTPUT, "results")
FEATURES_DIR = os.path.join(BASE_OUTPUT, "features")
MODELS_DIR = os.path.join(BASE_OUTPUT, "models")
N_SPLITS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using model: {MODEL_NAME}")
print(f"Data loading started from: {DATA_DIR}")


# Load dataset
raw_df = pd.read_csv(DATA_DIR)
X = raw_df.drop(columns=[TARGET])
y = raw_df[TARGET]
print("Data read complete")


# Load model functions
get_model, train_model, predict_proba, save_model, get_feature_importance = get_model_module(MODEL_NAME)


# Run the K-Fold CV experiment
def main():
    """
    Executes the full CV training pipeline using run_kfold_training.
    Handles preprocessing, IV filtering, feature selection, model training,
    metrics computation, and saving of results, features, and models.
    """
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
        model_name=MODEL_NAME,
        results_dir=RESULTS_DIR,
        features_dir=FEATURES_DIR,
        models_dir=MODELS_DIR,
        n_splits=N_SPLITS,
        random_state=42
    )

    print("\nCV Experiment Completed")
    return results_df


# Run script
if __name__ == "__main__":
    results = main()