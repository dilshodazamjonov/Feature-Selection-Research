from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FrozenRepresentationPolicy:
    checkpoint: str = "frozen_clip_v2_selected_checkpoint"
    text_encoder: str = "frozen_text_encoder"
    anchor: str = "frozen_homecredit_anchor"
    statistical_preprocessor: str = "frozen_homecredit_fitted_preprocessor"
    refit_on_lendingclub_v2: bool = False
    oot_enters_selection: bool = False


def construct_temporal_cutoffs(
    frame: pd.DataFrame,
    *,
    dataset: str,
    date_column: str,
    target_column: str,
    max_cutoffs: int = 3,
    min_dev_rows: int = 50,
    min_oot_rows: int = 20,
    min_positive_count: int = 1,
    label_maturity_rule: str = "chronological_dev_before_oot",
) -> pd.DataFrame:
    if date_column not in frame.columns or target_column not in frame.columns:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "cutoff_id": "",
                    "dev_start": None,
                    "dev_end": None,
                    "oot_start": None,
                    "oot_end": None,
                    "dev_rows": 0,
                    "oot_rows": 0,
                    "dev_positive_count": 0,
                    "oot_positive_count": 0,
                    "dev_bad_rate": None,
                    "oot_bad_rate": None,
                    "label_maturity_rule": "not_evaluated",
                    "maturity_rule": "not_evaluated",
                    "eligible": False,
                    "eligibility_status": "unsupported_missing_date_or_target",
                    "rejection_reason": "unsupported_missing_date_or_target",
                }
            ]
        )
    data = frame[[date_column, target_column]].dropna().copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna()
    if data.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "cutoff_id": "",
                    "dev_start": None,
                    "dev_end": None,
                    "oot_start": None,
                    "oot_end": None,
                    "dev_rows": 0,
                    "oot_rows": 0,
                    "dev_positive_count": 0,
                    "oot_positive_count": 0,
                    "dev_bad_rate": None,
                    "oot_bad_rate": None,
                    "label_maturity_rule": "not_evaluated",
                    "maturity_rule": "not_evaluated",
                    "eligible": False,
                    "eligibility_status": "unsupported_no_valid_dates",
                    "rejection_reason": "unsupported_no_valid_dates",
                }
            ]
        )
    data = data.sort_values(date_column, kind="mergesort")
    quantiles = [0.50, 0.60, 0.70][:max_cutoffs]
    rows = []
    for index, quantile in enumerate(quantiles, start=1):
        cutoff = data[date_column].quantile(quantile)
        dev = data[data[date_column] <= cutoff]
        oot = data[data[date_column] > cutoff]
        dev_positive = int(dev[target_column].sum()) if len(dev) else 0
        oot_positive = int(oot[target_column].sum()) if len(oot) else 0
        rejection = []
        if len(dev) < min_dev_rows:
            rejection.append("insufficient_dev_rows")
        if len(oot) < min_oot_rows:
            rejection.append("insufficient_oot_rows")
        if dev_positive < min_positive_count or int(len(dev) - dev_positive) < min_positive_count:
            rejection.append("insufficient_dev_classes")
        if oot_positive < min_positive_count or int(len(oot) - oot_positive) < min_positive_count:
            rejection.append("insufficient_oot_classes")
        if len(dev) and len(oot) and dev[date_column].max() >= oot[date_column].min():
            rejection.append("dev_not_strictly_before_oot")
        eligible = not rejection
        rows.append(
            {
                "dataset": dataset,
                "cutoff_id": f"{dataset}_cutoff_{index}",
                "dev_start": dev[date_column].min(),
                "dev_end": dev[date_column].max(),
                "oot_start": oot[date_column].min(),
                "oot_end": oot[date_column].max(),
                "dev_rows": int(len(dev)),
                "oot_rows": int(len(oot)),
                "dev_positive_count": dev_positive,
                "oot_positive_count": oot_positive,
                "dev_bad_rate": float(dev[target_column].mean()) if len(dev) else None,
                "oot_bad_rate": float(oot[target_column].mean()) if len(oot) else None,
                "label_maturity_rule": label_maturity_rule,
                "maturity_rule": label_maturity_rule,
                "eligible": bool(eligible),
                "eligibility_status": "eligible" if eligible else "unsupported_insufficient_rows_or_classes",
                "rejection_reason": "" if eligible else ";".join(rejection),
            }
        )
    return pd.DataFrame(rows)


def frozen_policy_manifest() -> dict[str, object]:
    policy = FrozenRepresentationPolicy()
    return policy.__dict__.copy()
