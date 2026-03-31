import pandas as pd
import matplotlib
from Preprocessing.data_process import DataLoader
from Preprocessing.preprocessing import Preprocessor
from Preprocessing.feature_engineering import build_all_features

from feature_selection.boruta_rfe import BorutaSelector, RFESelector
from feature_selection.mrmr import MRMR
from feature_selection.pca import PCASelector


from training.oot_trainer import oot_split
from training.kfold_trainer import run_kfold_training

from Models.catboost_model import CatBoostModel
from Models.random_forest_model import RandomForestModel
from Models.logistic_regression_model import LogisticRegressionModel


matplotlib.use("Agg")  # non-GUI backend (safe)

MODEL_NAME = "catboost" # lr, rf, catboost
FEATURE_SELECTION_METHOD = "mrmr" # boruta, rfe, mrmr, pca, none
N_SPLITS = 5

DATA_DIR = "data/inputs"
TARGET = "TARGET"
TIME_COL = "recent_decision"   # will also accept DAYS_DECISION if present
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


def get_selector(selector_name):
    """
    Returns the selector class and its default kwargs.
    """
    name = selector_name.lower()

    if name == "boruta":
        return BorutaSelector, {"max_iter": 10, "random_state": 42}
    if name == "rfe":
        return RFESelector, {"n_features": 50, "step": 10, "random_state": 42}
    if name == "mrmr":
        return MRMR, {"k": 50, "method": "mrmr", "random_state": 42}
    if name == "pca":
        return PCASelector, {"n_components": 0.95, "save_dir": None}
    if name == "none":
        return None, {}

    raise ValueError(f"Unsupported selector: {selector_name}")


def get_model_bundle(model_name):
    """
    Returns model factory and adapter functions.
    """
    name = model_name.lower()

    if name == "catboost":
        model_cls = CatBoostModel
    elif name == "rf":
        model_cls = RandomForestModel
    elif name == "lr":
        model_cls = LogisticRegressionModel
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def get_model():
        return model_cls()

    def train_model(model, X_train, y_train, X_val=None, y_val=None):
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        return model.fit(X_train, y_train, eval_set=eval_set)

    def predict_proba(model, X):
        return model.predict_proba(X)

    def save_model(model, path):
        return model.save(path)

    return get_model, train_model, predict_proba, save_model


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