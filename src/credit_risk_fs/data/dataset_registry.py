from __future__ import annotations

from credit_risk_fs.data.homecredit import dataset_config as homecredit_dataset_config
from credit_risk_fs.data.lendingclub import dataset_config as lendingclub_dataset_config
from credit_risk_fs.data.schemas import DatasetConfig


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    normalized = dataset_name.lower()
    if normalized == "homecredit":
        return homecredit_dataset_config()
    if normalized == "lendingclub":
        return lendingclub_dataset_config()
    raise ValueError(f"Unsupported dataset: {dataset_name}")
