from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.evaluation.metrics import evaluate_model


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_prompt_16_pilot_gate.py"
SPEC = importlib.util.spec_from_file_location("prompt_16_pilot_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_forecast_uses_row_scaling_and_twenty_percent_allowance():
    split = {
        "folds": [
            {"train": {"rows": 100}, "validation": {"rows": 50}},
            {"train": {"rows": 200}, "validation": {"rows": 50}},
        ],
        "dev": {"rows": 300},
        "oot": {"rows": 75},
    }
    rows = gate._forecast_rows(
        split=split,
        selector_seconds=10.0,
        encoding_seconds=2.0,
        preprocessing_seconds=3.0,
        training_seconds=5.0,
        prediction_seconds=4.0,
        metric_seconds=1.0,
        packaging_seconds=1.0,
        total_active_seconds=30.0,
        selection_bytes=100,
        evaluation_bytes=200,
        log_bytes=50,
    )
    keyed = {(item["measure"], item["component"]): item for item in rows}
    selector = keyed[("runtime_seconds", "selector_fit")]
    prediction = keyed[("runtime_seconds", "prediction")]
    selection_disk = keyed[("disk_bytes", "selection_artifacts")]
    assert selector["dev_forecast_with_20pct"] == pytest.approx(36.0)
    assert selector["oot_forecast_with_20pct"] == pytest.approx(36.0)
    assert prediction["dev_forecast_with_20pct"] == pytest.approx(9.6)
    assert prediction["oot_forecast_with_20pct"] == pytest.approx(7.2)
    assert selection_disk["dev_forecast_with_20pct"] == 240
    assert selection_disk["oot_forecast_with_20pct"] == 120


def test_metric_reconciliation_is_exact_and_detects_drift():
    predictions = pd.DataFrame(
        {
            "case_id": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
            "score": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "decision_threshold": np.full(6, 0.5),
        }
    )
    stored = evaluate_model(
        predictions["target"].to_numpy(),
        predictions["score"].to_numpy(),
        threshold=0.5,
    )
    stored.update(
        gate._ranking_utility(
            predictions["target"].to_numpy(), predictions["score"].to_numpy()
        )
    )
    gate._metrics_reconcile(predictions, stored)
    changed = dict(stored)
    changed["auc"] = float(changed["auc"]) - 0.01
    with pytest.raises(gate.Prompt16ExecutionError, match="auc"):
        gate._metrics_reconcile(predictions, changed)
