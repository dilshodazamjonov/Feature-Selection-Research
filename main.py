from pathlib import Path
import logging
import shutil  # Added for moving folders

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Preprocessing.data_process import DataLoader
from Preprocessing.preprocessing import Preprocessor
from Preprocessing.feature_engineering import build_all_features

from Models.utils import get_selector, get_model_bundle
from training.kfold_trainer import run_kfold_training
from evaluation.metrics import evaluate_model
from utils.logging_config import setup_logging
from utils.feature_metadata import build_feature_metadata

from iv_woe_filter import IVWOEFilter

MAX_DAYS_BACK = 600
CV_MONTHS = 12
OOT_MONTHS = 8
N_SPLITS = 5
DATA_DIR = "data/inputs"
DESCRIPTION_PATH = "data/HomeCredit_columns_description.csv"
TARGET = "TARGET"
TIME_COL = "recent_decision"
MODEL_NAME = "lr"
FEATURE_SELECTION_METHOD = "llm"
DROP_ID_COLS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
OUTPUTS_DIR = Path("outputs")

logger = setup_logging("main", level=logging.INFO)


def resolve_time_col(df: pd.DataFrame, preferred: str) -> str:
    candidates = [preferred, preferred.upper(), preferred.lower()]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Time column not found. Tried: {candidates}")


def latest_experiment_dir(base_dir: Path, prefix: str) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base output directory not found: {base_dir}")

    candidates = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found in {base_dir} with prefix '{prefix}'")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
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

    logger.info(f"Before filter: {merged_train.shape}")
    merged_train = merged_train[
        (merged_train[time_col] >= -MAX_DAYS_BACK) &
        (merged_train[time_col] <= 0)
    ]
    logger.info(f"After filter: {merged_train.shape}")

    oot_cutoff = -OOT_MONTHS * 30

    cv_data = merged_train[merged_train[time_col] < oot_cutoff].copy()
    oot_data = merged_train[merged_train[time_col] >= oot_cutoff].copy()

    X_train_full = cv_data.drop(columns=[TARGET])
    y_train_full = cv_data[TARGET]
    X_oot = oot_data.drop(columns=[TARGET])
    y_oot = oot_data[TARGET]

    logger.info(f"CV split: {X_train_full.shape}")
    logger.info(f"OOT split: {X_oot.shape}")

    drop_cols = [c for c in DROP_ID_COLS if c in X_train_full.columns]
    X_train = X_train_full.drop(columns=drop_cols, errors="ignore")
    X_oot_clean = X_oot.drop(columns=[c for c in DROP_ID_COLS if c in X_oot.columns], errors="ignore")

    # Define a temporary path for IV reports
    temp_iv_report_dir = OUTPUTS_DIR / "temp_iv_reports"

    feature_metadata = None
    if FEATURE_SELECTION_METHOD == "llm":
        logger.info("Applying IVWOEFilter to reduce feature space before passing to LLM...")
        
        train_time_col_data = X_train[time_col].copy()
        iv_filter = IVWOEFilter(
                    min_iv=0.02,
                    encode=False,
                    output_dir=str(temp_iv_report_dir)
                )
        X_train = iv_filter.fit_transform(X_train, y_train_full)
        
        if time_col not in X_train.columns:
            X_train[time_col] = train_time_col_data
            
        X_oot_clean = X_oot_clean[X_train.columns]
        
        llm_input_features = X_train.drop(columns=[time_col], errors="ignore")
        num_features = llm_input_features.shape[1]
        logger.info(f"==> Total features successfully passed IV filter and sent to LLM: {num_features} <==")

        feature_metadata = build_feature_metadata(
            X=llm_input_features,
            description_csv_path=DESCRIPTION_PATH,
        )

    selector_cls, selector_kwargs = get_selector(FEATURE_SELECTION_METHOD)

    if FEATURE_SELECTION_METHOD == "llm":
        selector_kwargs["feature_metadata"] = feature_metadata
        selector_kwargs["cache_path"] = str(OUTPUTS_DIR / f"{MODEL_NAME}_{FEATURE_SELECTION_METHOD}_llm_selected_features.json")

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

    logger.info("=== OOT TEST EVALUATION ===")

    # Use the helper function to find the folder created by run_kfold_training
    exp_dir = latest_experiment_dir(OUTPUTS_DIR, f"{MODEL_NAME}_{FEATURE_SELECTION_METHOD}_")

    # Move IV reports from temporary folder to the actual experiment folder
    if temp_iv_report_dir.exists():
        final_iv_dir = exp_dir / "iv_reports"
        shutil.move(str(temp_iv_report_dir), str(final_iv_dir))
        logger.info(f"IV reports moved to {final_iv_dir}")

    final_preprocessor = Preprocessor()
    X_train_full_processed = final_preprocessor.fit_transform(X_train)
    X_oot_processed = final_preprocessor.transform(X_oot_clean)

    final_selector = selector_cls(**selector_kwargs) if selector_cls is not None else None
    if final_selector is not None:
        X_train_final = final_selector.fit_transform(X_train_full_processed, y_train_full)
        X_oot_final = final_selector.transform(X_oot_processed)
    else:
        X_train_final = X_train_full_processed
        X_oot_final = X_oot_processed

    final_model = get_model()
    final_model = train_model(final_model, X_train_final, y_train_full, None, None)

    oot_proba = predict_proba(final_model, X_oot_final)
    oot_metrics = evaluate_model(y_oot.values, oot_proba)

    logger.info("OOT Test Results:")
    logger.info(f"  AUC: {oot_metrics['auc']:.4f}")
    logger.info(f"  Gini: {oot_metrics['gini']:.4f}")
    logger.info(f"  KS: {oot_metrics['ks']:.4f}")
    logger.info(f"  Precision: {oot_metrics['precision']:.4f}")
    logger.info(f"  Recall: {oot_metrics['recall']:.4f}")
    logger.info(f"  F1: {oot_metrics['f1']:.4f}")

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    oot_results_path = results_dir / "oot_test_results.csv"
    pd.DataFrame([oot_metrics]).to_csv(oot_results_path, index=False)
    logger.info(f"OOT results saved to {oot_results_path}")

    eval_file = results_dir / "evaluation_metrics_summary.csv"
    time_file = results_dir / "fold_time_info.csv"

    if eval_file.exists() and time_file.exists():
        eval_df = pd.read_csv(eval_file)
        eval_df = eval_df[eval_df["fold"].astype(str).str.isnumeric()].copy()
        eval_df["fold"] = eval_df["fold"].astype(int)
        eval_df = eval_df.sort_values("fold")

        time_df = pd.read_csv(time_file)
        eval_df = eval_df.merge(time_df, on="fold", how="inner")

        plt.figure(figsize=(14, 6))
        plt.plot(eval_df["val_time_end"], eval_df["auc"], marker="o", linewidth=2, markersize=8)
        plt.title(f"ROC-AUC over Time | {MODEL_NAME}_{FEATURE_SELECTION_METHOD}")
        plt.xlabel("Validation Period End (days)")
        plt.ylabel("ROC-AUC")
        plt.grid(True, alpha=0.3)

        for _, row in eval_df.iterrows():
            plt.annotate(
                f'Fold {int(row["fold"])}',
                (row["val_time_end"], row["auc"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        plot_file = exp_dir / "roc_auc_over_time.png"
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC-AUC over time plot saved to {plot_file}")

    logger.info("CV experiment completed")
    return results_df


if __name__ == "__main__":
    results = main()