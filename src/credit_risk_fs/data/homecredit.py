from __future__ import annotations

from credit_risk_fs.data.schemas import DatasetConfig


def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        name="homecredit",
        data_dir="data/homecredit/raw",
        description_path="data/homecredit/metadata/columns_description.csv",
        target="TARGET",
        time_col="recent_decision",
        drop_id_cols=("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"),
        excluded_feature_columns=(
            "TARGET",
            "recent_decision",
            "PREV_recent_decision_MAX",
            "DAYS_DECISION",
            "application_time_proxy",
        ),
        mode="homecredit_multitable",
        results_dir="results/homecredit",
    )
