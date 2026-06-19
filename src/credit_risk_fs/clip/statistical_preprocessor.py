from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text
from credit_risk_fs.utils.io import write_json
from credit_risk_fs.utils.serialization import dump_joblib


@dataclass
class StatisticalPreprocessor:
    field_order: list[str]
    imputation_strategy: str = "median"
    scaling_strategy: str = "standard"
    clipping_enabled: bool = False
    clipping_lower_quantile: float = 0.01
    clipping_upper_quantile: float = 0.99
    fit_dataset: str = "homecredit"
    fit_split: str = "train"
    imputation_values_: dict[str, float] = field(default_factory=dict)
    center_: dict[str, float] = field(default_factory=dict)
    scale_: dict[str, float] = field(default_factory=dict)
    clip_lower_: dict[str, float] = field(default_factory=dict)
    clip_upper_: dict[str, float] = field(default_factory=dict)
    preprocessor_hash_: str = ""

    def fit(self, frame: pd.DataFrame) -> "StatisticalPreprocessor":
        values = self._numeric_frame(frame)
        if len(values) == 0:
            raise ValueError("cannot fit statistical preprocessor on zero rows")
        if self.imputation_strategy != "median":
            raise ValueError(f"unsupported imputation strategy: {self.imputation_strategy}")
        if self.scaling_strategy not in {"standard", "robust", "none"}:
            raise ValueError(f"unsupported scaling strategy: {self.scaling_strategy}")

        medians = values.median(axis=0, skipna=True)
        if medians.isna().any():
            missing = medians[medians.isna()].index.tolist()
            raise ValueError(f"cannot fit imputation for all-null fields: {missing}")
        self.imputation_values_ = {field: float(medians[field]) for field in self.field_order}

        imputed = values.fillna(medians)
        if self.clipping_enabled:
            lower = imputed.quantile(self.clipping_lower_quantile)
            upper = imputed.quantile(self.clipping_upper_quantile)
            self.clip_lower_ = {field: float(lower[field]) for field in self.field_order}
            self.clip_upper_ = {field: float(upper[field]) for field in self.field_order}
            imputed = imputed.clip(lower=lower, upper=upper, axis=1)
        else:
            self.clip_lower_ = {}
            self.clip_upper_ = {}

        if self.scaling_strategy == "none":
            center = pd.Series(0.0, index=self.field_order)
            scale = pd.Series(1.0, index=self.field_order)
        elif self.scaling_strategy == "robust":
            center = imputed.median(axis=0)
            q75 = imputed.quantile(0.75)
            q25 = imputed.quantile(0.25)
            scale = (q75 - q25).replace(0, 1.0)
        else:
            center = imputed.mean(axis=0)
            scale = imputed.std(axis=0, ddof=0).replace(0, 1.0)
        scale = scale.fillna(1.0).replace(0, 1.0)
        self.center_ = {field: float(center[field]) for field in self.field_order}
        self.scale_ = {field: float(scale[field]) for field in self.field_order}
        self.preprocessor_hash_ = self.compute_hash()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.preprocessor_hash_:
            raise ValueError("statistical preprocessor is not fitted")
        values = self._numeric_frame(frame)
        impute = pd.Series(self.imputation_values_, dtype=float)
        transformed = values.fillna(impute)
        if self.clipping_enabled:
            transformed = transformed.clip(
                lower=pd.Series(self.clip_lower_, dtype=float),
                upper=pd.Series(self.clip_upper_, dtype=float),
                axis=1,
            )
        center = pd.Series(self.center_, dtype=float)
        scale = pd.Series(self.scale_, dtype=float).replace(0, 1.0)
        transformed = (transformed - center) / scale
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError("statistical transform produced non-finite values")
        return transformed[self.field_order].astype("float32")

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def compute_hash(self) -> str:
        payload = self.to_state(include_hash=False)
        return sha256_text(json.dumps(payload, sort_keys=True, default=str))

    def to_state(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "field_order": list(self.field_order),
            "imputation_strategy": self.imputation_strategy,
            "scaling_strategy": self.scaling_strategy,
            "clipping_enabled": bool(self.clipping_enabled),
            "clipping_lower_quantile": float(self.clipping_lower_quantile),
            "clipping_upper_quantile": float(self.clipping_upper_quantile),
            "fit_dataset": self.fit_dataset,
            "fit_split": self.fit_split,
            "imputation_values": dict(self.imputation_values_),
            "center": dict(self.center_),
            "scale": dict(self.scale_),
            "clip_lower": dict(self.clip_lower_),
            "clip_upper": dict(self.clip_upper_),
        }
        if include_hash:
            payload["preprocessor_hash"] = self.preprocessor_hash_
        return payload

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {
            "statistical_preprocessor_json": write_json(out / "statistical_preprocessor.json", self.to_state()),
            "statistical_preprocessor_joblib": dump_joblib(self, out / "statistical_preprocessor.joblib"),
            "statistical_feature_order": write_json(
                out / "statistical_feature_order.json",
                {
                    "field_order": list(self.field_order),
                    "input_field_hash": input_field_hash(self.field_order),
                    "vector_dimension": len(self.field_order),
                },
            ),
        }

    def _numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [field for field in self.field_order if field not in frame.columns]
        if missing:
            raise ValueError(f"statistical frame missing fields: {missing}")
        converted = {}
        invalid_fields = []
        for field in self.field_order:
            series = frame[field]
            numeric = pd.to_numeric(series, errors="coerce")
            failed = numeric.isna() & series.notna()
            if bool(failed.any()):
                invalid_fields.append(field)
            converted[field] = numeric.astype(float)
        if invalid_fields:
            raise ValueError(f"unexpected non-numeric statistical fields: {invalid_fields}")
        return pd.DataFrame(converted, index=frame.index, columns=self.field_order)


def input_field_hash(fields: list[str]) -> str:
    return sha256_text(json.dumps(list(fields), sort_keys=False))


def build_vector_frame(
    *,
    metadata: pd.DataFrame,
    transformed: pd.DataFrame,
    preprocessor: StatisticalPreprocessor,
) -> pd.DataFrame:
    if len(metadata) != len(transformed):
        raise ValueError("metadata and statistical vectors have different row counts")
    fields_hash = input_field_hash(preprocessor.field_order)
    vector_columns = [f"stat_{idx:04d}" for idx in range(len(preprocessor.field_order))]
    vector_values = transformed[preprocessor.field_order].to_numpy(dtype=np.float32)
    vector_frame = pd.DataFrame(vector_values, columns=vector_columns, index=metadata.index)
    base_columns = [
        "dataset",
        "feature_name",
        "split",
        "group_key",
        "semantic_group",
        "source_table_or_formula",
        "source_manifest_hash",
    ]
    optional_columns = [
        "canonical_feature_family",
        "family_resolution_source",
        "family_resolution_rule",
        "family_member_count",
    ]
    base = metadata[base_columns + [col for col in optional_columns if col in metadata.columns]].copy()
    base["stable_row_index"] = range(len(base))
    base["stable_row_id"] = [
        sha256_text(f"{row.dataset}|{row.feature_name}|{row.split}") for row in base.itertuples(index=False)
    ]
    base["input_field_hash"] = fields_hash
    base["preprocessor_hash"] = preprocessor.preprocessor_hash_
    base["statistical_vector"] = [json.dumps([float(value) for value in row]) for row in vector_values]
    base["statistical_vector_hash"] = [sha256_text(value) for value in base["statistical_vector"]]
    base["vector_dimension"] = len(preprocessor.field_order)
    return pd.concat([base, vector_frame], axis=1).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)


def preprocessing_audit(
    *,
    raw_train: pd.DataFrame,
    raw_all_homecredit: pd.DataFrame,
    raw_lendingclub: pd.DataFrame,
    transformed_homecredit: pd.DataFrame,
    transformed_lendingclub: pd.DataFrame,
    preprocessor: StatisticalPreprocessor,
) -> dict[str, Any]:
    return {
        "fit_dataset": preprocessor.fit_dataset,
        "fit_split": preprocessor.fit_split,
        "field_order": list(preprocessor.field_order),
        "input_field_hash": input_field_hash(preprocessor.field_order),
        "preprocessor_hash": preprocessor.preprocessor_hash_,
        "imputation_strategy": preprocessor.imputation_strategy,
        "scaling_strategy": preprocessor.scaling_strategy,
        "clipping_enabled": preprocessor.clipping_enabled,
        "missingness_before": {
            "homecredit_train_fit": _missingness(raw_train, preprocessor.field_order),
            "homecredit_all": _missingness(raw_all_homecredit, preprocessor.field_order),
            "lendingclub_v2": _missingness(raw_lendingclub, preprocessor.field_order),
        },
        "missingness_after": {
            "homecredit_all": _missingness(transformed_homecredit, preprocessor.field_order),
            "lendingclub_v2": _missingness(transformed_lendingclub, preprocessor.field_order),
        },
        "finite_checks": {
            "homecredit_all": bool(np.isfinite(transformed_homecredit.to_numpy(dtype=float)).all()),
            "lendingclub_v2": bool(np.isfinite(transformed_lendingclub.to_numpy(dtype=float)).all()),
        },
        "dropped_or_invalid_rows": [],
    }


def _missingness(frame: pd.DataFrame, fields: list[str]) -> dict[str, float]:
    return {field: float(frame[field].isna().mean()) if field in frame.columns and len(frame) else 1.0 for field in fields}
