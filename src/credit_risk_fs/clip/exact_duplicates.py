from __future__ import annotations

import json
from typing import Any

import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text


EXACT_DUPLICATE_POLICY_VERSION = "exact_dev_duplicates_v1"


def feature_order_hash(features: list[str]) -> str:
    return sha256_text(json.dumps([str(feature) for feature in features], separators=(",", ":")))


def find_exact_dev_duplicate_pairs(
    data: pd.DataFrame,
    *,
    feature_names: list[str],
    dataset: str = "homecredit",
    split: str = "train",
) -> pd.DataFrame:
    if dataset != "homecredit" or split != "train":
        raise ValueError("exact duplicate evidence must use Home Credit DEV training rows only")
    names = [str(name) for name in feature_names]
    if len(names) != len(set(names)):
        raise ValueError("feature_names contains duplicates")
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise ValueError(f"DEV matrix is missing features: {missing[:20]}")

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(_series_hash(_canonical_series(data[name])), []).append(name)

    order_hash = feature_order_hash(names)
    rows: list[dict[str, Any]] = []
    for digest, candidates in sorted(grouped.items()):
        if len(candidates) < 2:
            continue
        canonical = {name: _canonical_series(data[name]) for name in candidates}
        for i, feature_a in enumerate(candidates):
            for feature_b in candidates[i + 1 :]:
                if not canonical[feature_a].equals(canonical[feature_b]):
                    continue
                evidence = (
                    f"exact equality across {len(data)} aligned Home Credit DEV rows, "
                    "including identical missingness positions"
                )
                for anchor, excluded in ((feature_a, feature_b), (feature_b, feature_a)):
                    rows.append(
                        {
                            "anchor_feature_name": anchor,
                            "excluded_feature_name": excluded,
                            "exclusion_reason": "exact_dev_duplicate",
                            "dataset": dataset,
                            "split": split,
                            "row_count": int(len(data)),
                            "duplicate_group_hash": digest,
                            "feature_order_hash": order_hash,
                            "policy_version": EXACT_DUPLICATE_POLICY_VERSION,
                            "evidence": evidence,
                        }
                    )
    columns = [
        "anchor_feature_name",
        "excluded_feature_name",
        "exclusion_reason",
        "dataset",
        "split",
        "row_count",
        "duplicate_group_hash",
        "feature_order_hash",
        "policy_version",
        "evidence",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["anchor_feature_name", "excluded_feature_name"], kind="mergesort"
    ).reset_index(drop=True)


def _canonical_series(series: pd.Series) -> pd.DataFrame:
    missing = series.isna()
    if pd.api.types.is_bool_dtype(series.dtype):
        values = series.astype("boolean")
        kind = "bool"
    elif pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        kind = "numeric"
    else:
        values = series.astype("string")
        kind = "text"
    return pd.DataFrame(
        {
            "kind": pd.Series(kind, index=series.index, dtype="string"),
            "missing": missing.astype(bool),
            "value": values,
        },
        index=series.index,
    )


def _series_hash(series: pd.DataFrame) -> str:
    hashes = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype="uint64")
    return sha256_text(hashes.tobytes().hex())
