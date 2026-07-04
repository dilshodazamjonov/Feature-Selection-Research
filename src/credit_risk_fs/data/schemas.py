from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetConfig:
    name: str
    data_dir: str
    description_path: str
    target: str = "TARGET"
    time_col: str = "recent_decision"
    drop_id_cols: tuple[str, ...] = ()
    excluded_feature_columns: tuple[str, ...] = ()
    mode: str = "homecredit_multitable"
    results_dir: str = "results"
