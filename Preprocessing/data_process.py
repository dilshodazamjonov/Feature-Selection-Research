# data.py
import os
import logging
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("data_loader", level=logging.INFO)

class DataLoader:
    """
    Class for loading, merging, and preparing datasets for modeling.
    """

    def __init__(self, data_dir: str):
        """
        Parameters:
            data_dir: path to folder containing all CSVs
        """
        self.data_dir = data_dir
        self.dataframes = {}
        self.load_errors = []

    def load_all(self):
        """
        Loads all CSV files in the data directory into a dictionary.
        Handles encoding issues automatically.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        
        if not files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        pbar = tqdm(files, desc="Loading CSVs")

        for f in pbar:
            pbar.set_description(f"Reading {f}")
            name = os.path.splitext(f)[0]
            path = os.path.join(self.data_dir, f)

            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, encoding="latin1")
                except Exception as e:
                    self.load_errors.append((f, str(e)))
                    logger.warning(f"Failed to load {f}: {e}")
                    continue
            except Exception as e:
                self.load_errors.append((f, str(e)))
                logger.warning(f"Failed to load {f}: {e}")
                continue

            self.dataframes[name] = df

        if self.load_errors:
            logger.warning(f"{len(self.load_errors)} files failed to load")

        return self.dataframes

    def get(self, name: str) -> pd.DataFrame:
        """
        Retrieve a loaded dataframe by name (without '.csv').
        """
        return self.dataframes.get(name)

    def merge_left(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        Perform a left merge between two dataframes on a given column.
        """
        return df1.merge(df2, on=on, how="left")

    def merge_features(self, base_df: pd.DataFrame, feature_dfs: List[pd.DataFrame], on: str) -> pd.DataFrame:
        """
        Sequentially merge multiple feature tables onto base_df.
        """
        df = base_df.copy()
        for feat_df in feature_dfs:
            df = df.merge(feat_df, on=on, how="left")
        return df

    def prepare_dataset(self, raw_df: pd.DataFrame, feature_tables: List[pd.DataFrame], target_col: str = "TARGET"):
        """
        Merge feature tables into the base application dataframe and split X, y.
        """
        df = self.merge_features(raw_df, feature_tables, on="SK_ID_CURR")
        y = df[target_col] if target_col in df.columns else None
        X = df.drop(columns=[target_col, "SK_ID_CURR"], errors='ignore')
        return X, y, df

# -----------------------------
# Aggregation helper
# -----------------------------
def build_aggregations(df: pd.DataFrame, groupby_col: str, agg_config: Dict) -> pd.DataFrame:
    """
    Generic aggregation builder.
    
    Example agg_config:
        {
            "avg_credit": ("AMT_CREDIT", "mean"),
            "total_credit": ("AMT_CREDIT", "sum")
        }
    """
    return df.groupby(groupby_col).agg(**agg_config).reset_index()


