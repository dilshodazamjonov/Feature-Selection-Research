from __future__ import annotations

from credit_risk_fs.data.schemas import DatasetConfig
from credit_risk_fs.preprocessing.lendingclub import LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS


def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        name="lendingclub",
        data_dir="data/lendingclub/processed",
        description_path="data/lendingclub/metadata/columns_description.csv",
        target="TARGET",
        time_col="recent_decision",
        drop_id_cols=(),
        excluded_feature_columns=LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS,
        mode="single_table",
        results_dir="results/lendingclub",
    )
