from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_schema_v2 import (
    DESCRIPTOR_COLUMNS_V2,
    SCALED_DESCRIPTOR_COLUMNS_V2,
    UNSCALED_INDICATOR_COLUMNS_V2,
)
from credit_risk_fs.utils.hashing import sha256_text


@dataclass
class RobustStatisticalPreprocessorV2:
    scaled_columns: list[str] = field(default_factory=lambda: list(SCALED_DESCRIPTOR_COLUMNS_V2))
    unscaled_columns: list[str] = field(default_factory=lambda: list(UNSCALED_INDICATOR_COLUMNS_V2))
    fit_dataset: str = "homecredit"
    fit_split: str = "train"
    clipping_lower: float = -8.0
    clipping_upper: float = 8.0
    medians_: dict[str, float] = field(default_factory=dict)
    iqr_: dict[str, float] = field(default_factory=dict)
    zero_iqr_columns_: list[str] = field(default_factory=list)
    fit_feature_count_: int = 0
    fit_split_hash_: str = ""
    preprocessor_hash_: str = ""

    def fit(self, descriptor_frame: pd.DataFrame, *, dataset: str, split: str) -> "RobustStatisticalPreprocessorV2":
        if dataset != self.fit_dataset or split != self.fit_split:
            raise ValueError("CLIP-v2 scaler may be fitted only on Home Credit training feature vectors")
        values = self._scaled_frame(descriptor_frame)
        self.fit_feature_count_ = int(len(values))
        self.fit_split_hash_ = sha256_text(values.to_csv(index=False))
        medians = values.median(axis=0, skipna=True)
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        iqr = q75 - q25
        self.zero_iqr_columns_ = iqr[iqr.fillna(0.0).eq(0.0)].index.tolist()
        iqr = iqr.fillna(1.0).replace(0.0, 1.0)
        self.medians_ = {column: float(medians[column]) for column in self.scaled_columns}
        self.iqr_ = {column: float(iqr[column]) for column in self.scaled_columns}
        self.preprocessor_hash_ = self.compute_hash()
        return self

    def transform(self, descriptor_frame: pd.DataFrame, *, allow_refit: bool = False) -> pd.DataFrame:
        if allow_refit:
            raise ValueError("CLIP-v2 external transforms must not refit the scaler")
        if not self.preprocessor_hash_:
            raise ValueError("CLIP-v2 statistical preprocessor is not fitted")
        missing = [column for column in DESCRIPTOR_COLUMNS_V2 if column not in descriptor_frame.columns]
        if missing:
            raise ValueError(f"descriptor frame missing columns: {missing}")
        scaled = self._scaled_frame(descriptor_frame).copy()
        medians = pd.Series(self.medians_, dtype=float)
        iqr = pd.Series(self.iqr_, dtype=float).replace(0.0, 1.0)
        scaled = ((scaled - medians) / iqr).clip(lower=self.clipping_lower, upper=self.clipping_upper, axis=1)
        indicators = descriptor_frame[self.unscaled_columns].astype(float).copy()
        output = pd.concat([scaled[self.scaled_columns], indicators[self.unscaled_columns]], axis=1)
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("CLIP-v2 statistical transform produced non-finite values")
        return output[DESCRIPTOR_COLUMNS_V2].astype("float32")

    def fit_transform(self, descriptor_frame: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
        self.fit(descriptor_frame, dataset=dataset, split=split)
        return self.transform(descriptor_frame)

    def compute_hash(self) -> str:
        return sha256_text(json.dumps(self.to_state(include_hash=False), sort_keys=True))

    def to_state(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": "compact_target_free_v2",
            "field_order": list(DESCRIPTOR_COLUMNS_V2),
            "scaled_columns": list(self.scaled_columns),
            "unscaled_columns": list(self.unscaled_columns),
            "fit_dataset": self.fit_dataset,
            "fit_split": self.fit_split,
            "medians": dict(self.medians_),
            "iqr": dict(self.iqr_),
            "zero_iqr_columns": list(self.zero_iqr_columns_),
            "clipping_policy": {"lower": self.clipping_lower, "upper": self.clipping_upper},
            "fit_feature_count": int(self.fit_feature_count_),
            "fit_split_hash": self.fit_split_hash_,
        }
        if include_hash:
            payload["preprocessor_hash"] = self.preprocessor_hash_
        return payload

    def _scaled_frame(self, descriptor_frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.scaled_columns if column not in descriptor_frame.columns]
        if missing:
            raise ValueError(f"descriptor frame missing scaled columns: {missing}")
        return descriptor_frame[self.scaled_columns].apply(pd.to_numeric, errors="raise").astype(float)
