from __future__ import annotations

import pandas as pd

from scripts.run_clip_final_evaluation import _psi_bucket


def test_psi_bucket_policy():
    assert _psi_bucket(0.01) == "low"
    assert _psi_bucket(0.10) == "moderate"
    assert _psi_bucket(0.25) == "high"
    assert _psi_bucket(float("nan")) == "unknown"


def test_gini_identity_for_saved_metric_shape():
    frame = pd.DataFrame({"oot_auc": [0.7], "oot_gini": [0.4]})

    assert abs(frame.loc[0, "oot_gini"] - (2 * frame.loc[0, "oot_auc"] - 1)) < 1e-12
