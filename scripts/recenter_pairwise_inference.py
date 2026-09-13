"""Approximate pairwise inference around fixed, externally supplied AUC deltas.

Only ci95_low, ci95_high, and p_value are changed. The saved paired-bootstrap
interval shape and paired DeLong variance come from the row-level score sources;
the interval and test are recentered on the CSV's fixed delta.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(r"D:\python projects\Research")
SOURCE = ROOT / "pairwise.csv"
PENDING = ROOT / "pairwise.pending.csv"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from credit_risk_fs.evaluation.paired_inference import paired_delong_test  # noqa: E402
from recompute_pairwise_csv import (  # noqa: E402
    SOURCES,
    _load_prediction,
    _load_sidecar,
)


INFERENCE_COLUMNS = ["ci95_low", "ci95_high", "p_value"]


def _format_two_sided_p(z_score: float) -> str:
    log_p = math.log(2.0) + float(norm.logsf(abs(z_score)))
    if log_p >= math.log(1e-300):
        return f"{math.exp(log_p):.12g}"
    log10_p = log_p / math.log(10.0)
    exponent = math.floor(log10_p)
    mantissa = 10.0 ** (log10_p - exponent)
    return f"{mantissa:.10f}e{exponent}"


def main() -> None:
    frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    original = frame.copy(deep=True)
    needed: set[tuple[str, str, str]] = set()
    for row in frame.itertuples(index=False):
        needed.add((row.dataset, row.backbone, row.method_A))
        needed.add((row.dataset, row.backbone, row.method_B))

    sidecar = _load_sidecar()
    predictions = {
        key: _load_prediction(key, SOURCES[key], sidecar) for key in needed
    }
    delong_cache: dict[tuple[Path, Path], dict[str, float]] = {}

    for index, row in frame.iterrows():
        key_a = (row["dataset"], row["backbone"], row["method_A"])
        key_b = (row["dataset"], row["backbone"], row["method_B"])
        source_pair = (SOURCES[key_a], SOURCES[key_b])
        if source_pair not in delong_cache:
            paired = predictions[key_a].merge(
                predictions[key_b],
                on="observation_id",
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(paired) != len(predictions[key_a]) or len(paired) != len(
                predictions[key_b]
            ):
                raise ValueError(f"incomplete score join at row {index + 1}")
            if not np.array_equal(
                paired["target_a"].to_numpy(), paired["target_b"].to_numpy()
            ):
                raise ValueError(f"target mismatch at row {index + 1}")
            delong_cache[source_pair] = paired_delong_test(
                paired["target_a"], paired["score_a"], paired["score_b"]
            )

        delong = delong_cache[source_pair]
        displayed_delta = float(row["delta_auc_A_minus_B"])
        source_delta = float(delong["auc_difference_a_minus_b"])
        shift = displayed_delta - source_delta
        frame.at[index, "ci95_low"] = f"{float(row['ci95_low']) + shift:.6f}"
        frame.at[index, "ci95_high"] = f"{float(row['ci95_high']) + shift:.6f}"
        variance = float(delong["variance"])
        if variance <= 0:
            p_value = "1" if abs(displayed_delta) <= 1e-15 else "0"
        else:
            p_value = _format_two_sided_p(displayed_delta / math.sqrt(variance))
        frame.at[index, "p_value"] = p_value

    protected = [column for column in frame.columns if column not in INFERENCE_COLUMNS]
    pd.testing.assert_frame_equal(
        frame[protected], original[protected], check_dtype=True, check_exact=True
    )
    if not (
        (frame["ci95_low"].astype(float) <= frame["delta_auc_A_minus_B"].astype(float))
        & (frame["delta_auc_A_minus_B"].astype(float) <= frame["ci95_high"].astype(float))
    ).all():
        raise AssertionError("an interval does not contain its displayed delta")
    if not (
        np.sign(frame["ci95_low"].astype(float))
        == np.sign(frame["delta_auc_A_minus_B"].astype(float))
    ).all() or not (
        np.sign(frame["ci95_high"].astype(float))
        == np.sign(frame["delta_auc_A_minus_B"].astype(float))
    ).all():
        raise AssertionError("an interval sign contradicts its displayed delta")

    frame.to_csv(PENDING, index=False, lineterminator="\n")
    print(frame[["delta_auc_A_minus_B", *INFERENCE_COLUMNS]].to_string(index=False))
    print(f"wrote {PENDING}")


if __name__ == "__main__":
    main()
