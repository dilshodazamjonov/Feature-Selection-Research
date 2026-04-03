import pandas as pd
import matplotlib

from Preprocessing.data_process import DataLoader
from Preprocessing.preprocessing import Preprocessor
from Preprocessing.feature_engineering import build_all_features

from Models.utils import get_selector, get_model_bundle

from training.oot_trainer import oot_split
from training.kfold_trainer import run_kfold_training

MAX_DAYS_BACK = 800
matplotlib.use("Agg")  

MODEL_NAME = "lr"  # lr, rf, catboost
FEATURE_SELECTION_METHOD = "mrmr"  # boruta -> boruta + rfe, mrmr, pca, none
N_SPLITS = 5

DATA_DIR = "data/inputs"
TARGET = "TARGET"
TIME_COL = "recent_decision"  
OOT_TEST_SIZE = 0.2

DROP_ID_COLS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]


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
    print(f"Loading datasets from {DATA_DIR}...")
    loader = DataLoader(DATA_DIR)
    dfs = loader.load_all()

    if "application_train" not in dfs:
        raise ValueError("application_train.csv not found in data directory")

    app_train = dfs["application_train"].copy()

    print("Building feature tables...")
    feature_tables = build_all_features(dfs)

    print("Merging feature tables into application_train...")
    merged_train = loader.merge_features(app_train, feature_tables, on="SK_ID_CURR")

    time_col = resolve_time_col(merged_train, TIME_COL)
    print(f"Using time column: {time_col}")

    print("Before filter:", merged_train.shape)
    merged_train = merged_train[
        merged_train[time_col] >= -MAX_DAYS_BACK
    ]

    print("After filter:", merged_train.shape)

    X_train_full, X_oot, y_train_full, y_oot = oot_split(
        df=merged_train,
        time_col=time_col,
        test_size=OOT_TEST_SIZE,
        target_col=TARGET,
    )
    print(f"OOT split done | train: {X_train_full.shape}, test: {X_oot.shape}")

    drop_cols = [c for c in DROP_ID_COLS if c in X_train_full.columns]
    X_train = X_train_full.drop(columns=drop_cols, errors="ignore")

    selector_cls, selector_kwargs = get_selector(FEATURE_SELECTION_METHOD)
    get_model, train_model, predict_proba, save_model = get_model_bundle(MODEL_NAME)

    print(f"\nStarting experiment: {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")

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

    print("\nCV experiment completed")
    return results_df


if __name__ == "__main__":
    results = main()